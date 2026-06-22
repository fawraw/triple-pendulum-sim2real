"""Gymnasium environment for the triple inverted pendulum on a cart.

Observation: [x, dx, theta1, dtheta1, theta2, dtheta2, theta3, dtheta3, target_id_onehot...]
Action: [u] continuous in [-1, 1], scaled to motor force by the XML actuator gear.
Reward: shaped to encourage convergence toward the target equilibrium point.

The 8 equilibrium points are encoded as 3-bit (link Up=1 / Down=0) configurations.
Bit i encodes link i+1 and names are read base->tip (see sim.equilibria):
    EP0 = DDD, EP1 = UDD, EP2 = DUD, EP3 = UUD,
    EP4 = DDU, EP5 = UDU, EP6 = DUU, EP7 = UUU
with U meaning the link points up (theta = 0) and D meaning it points down
(theta = pi) in absolute world frame.

This is a stub for Milestone 1. Reward shaping, target conditioning and randomization
will evolve in later milestones.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

# Canonical EP target angles / naming live in sim.equilibria. Re-exported here
# for backward compatibility (callers import ep_target_angles from this module).
from sim.equilibria import ep_target_angles  # noqa: E402,F401

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "triple_pendulum.xml"


class TriplePendulumEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, target_ep: int = 7, render_mode: str | None = None,
                 max_episode_steps: int = 1500,
                 init_mode: str = "near_target", init_noise: float = 0.05,
                 target_mode: str = "fixed",
                 fall_grace_steps: int = 0,
                 hard_ep_weight: float = 1.0,
                 start_grace_steps: int = 0,
                 w_down: float = 1.0,
                 progress_reward_coef: float = 0.0,
                 vel_cost_coef: float = 0.05,
                 # M4 transition params: start_ep != target_ep means a swing/transition task
                 start_ep: int | None = None,
                 transition_success_tol_rad: float = 0.2,
                 transition_success_steps: int = 200,
                 transition_bonus: float = 200.0):
        """
        init_mode:
          - "near_target": start with link angles within `init_noise` of the target EP.
            Use this for the stabilization milestones (M2, M3).
          - "bottom":      start near the natural rest configuration (DDD).
            Use this for swing-up and full transition control (M4+).
          - "random":      start with all link angles uniformly in [-pi, pi].
            Use this once the policy is robust enough.
        target_mode:
          - "fixed":      target_ep stays constant.
          - "random":     target_ep resampled uniformly 0..7 on every reset.
          - "weighted":   like random but EP4/EP6 get extra_weight× more episodes.
          - "transition": pair (start_ep, target_ep) resampled uniformly with
                          start_ep != target_ep. Init is around start_ep, the
                          goal is to drive the system to target_ep. Used in M4.
        init_noise: scale of the uniform noise applied to qpos/qvel at reset (rad / rad/s).
        start_ep: explicit start EP for "transition" mode (or None to randomise).
        transition_success_tol_rad: per-link angle tolerance to consider the
            target EP reached (default 0.2 rad ~= 11.5°).
        transition_success_steps: consecutive in-tolerance steps needed to
            declare success and trigger the transition_bonus reward.
        transition_bonus: sparse positive reward emitted on successful arrival.
        """
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.target_ep = int(target_ep)
        self.max_episode_steps = max_episode_steps
        self.init_mode = init_mode
        self.init_noise = float(init_noise)
        self.target_mode = str(target_mode)
        self.fall_grace_steps = int(fall_grace_steps)
        # start_grace_steps: first N steps are immune to falls (policy orients itself)
        self.start_grace_steps = int(start_grace_steps)
        # hard_ep_weight: EP4 and EP6 get this multiplier in "weighted" target_mode
        self.hard_ep_weight = float(hard_ep_weight)
        # Reward coefficients (configurable per pod)
        self.w_down = float(w_down)
        self.progress_reward_coef = float(progress_reward_coef)
        self.vel_cost_coef = float(vel_cost_coef)
        # M4 transition params
        self.start_ep: int | None = int(start_ep) if start_ep is not None else None
        self.transition_success_tol_rad = float(transition_success_tol_rad)
        self.transition_success_steps = int(transition_success_steps)
        self.transition_bonus = float(transition_bonus)
        self._fall_counter = 0
        self._step_count = 0
        self._prev_err: np.ndarray | None = None
        # Set when the system has been within tolerance of target_ep for
        # `transition_success_steps` consecutive steps. Once True, the angle-
        # based fall check re-engages (so the policy must actually HOLD the
        # target after reaching it).
        self._reached_target = False
        self._in_tolerance_counter = 0
        self._transition_bonus_paid = False
        self.render_mode = render_mode
        self._renderer = None

        # 8 (joint state) + 8 (one-hot target EP)
        high = np.array([np.inf] * 16, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    # --- helpers ---------------------------------------------------------
    def _joint_state(self) -> np.ndarray:
        x = self.data.qpos[0]
        dx = self.data.qvel[0]
        # Hinge angles are stored relative to parent body. Recover absolute angles
        # for clean equilibrium reasoning.
        t1 = self.data.qpos[1]
        t2 = t1 + self.data.qpos[2]
        t3 = t2 + self.data.qpos[3]
        d1 = self.data.qvel[1]
        d2 = d1 + self.data.qvel[2]
        d3 = d2 + self.data.qvel[3]
        return np.array([x, dx, t1, d1, t2, d2, t3, d3], dtype=np.float32)

    def _target_onehot(self) -> np.ndarray:
        v = np.zeros(8, dtype=np.float32)
        v[self.target_ep] = 1.0
        return v

    def _obs(self) -> np.ndarray:
        return np.concatenate([self._joint_state(), self._target_onehot()]).astype(np.float32)

    # Per-link fall thresholds based on whether the link is targeted UP or DOWN.
    # CRITICAL FIX (audit 2026-05-10): the previous global 0.6 rad threshold
    # made EP4 (DDU) / EP6 (DUU) untrainable: when a link is held vertical,
    # the cart's stabilizing motion shakes the hanging links 2-3, which
    # naturally swing well past 0.6 rad. The env was terminating with -100
    # FALL_PENALTY on every recovery attempt -> 0% success on those EPs.
    #
    # New scheme: tight threshold for links targeted UP (must stay near
    # vertical to count as success), loose threshold for links targeted DOWN
    # (free to swing while link 1 is being stabilized).
    FALL_THRESHOLD_UP_RAD   = 0.6   # ~34deg, link must stay near vertical
    FALL_THRESHOLD_DOWN_RAD = 1.5   # ~86deg, generous for hanging dynamics

    def _angle_error(self) -> np.ndarray:
        st = self._joint_state()
        target = ep_target_angles(self.target_ep)
        err = np.array([st[2], st[4], st[6]]) - target
        return np.arctan2(np.sin(err), np.cos(err))

    def _fall_thresholds(self) -> np.ndarray:
        target = ep_target_angles(self.target_ep)
        # target[i] == 0.0 means link i is targeted UP, target[i] == pi DOWN
        return np.where(np.isclose(target, 0.0), self.FALL_THRESHOLD_UP_RAD,
                                                  self.FALL_THRESHOLD_DOWN_RAD)

    # Penalty applied at the terminal step when the policy fails (cart
    # off-rail or any link tipped past its per-link fall threshold).
    FALL_PENALTY = 100.0

    def _is_fallen(self) -> bool:
        if abs(self.data.qpos[0]) > 0.95:
            return True
        # Immunity window at episode start — gives the policy time to orient
        if self._step_count < self.start_grace_steps:
            return False
        # M4 transition mode: angle-based fall is only checked AFTER the policy
        # has reached the target equilibrium at least once. Otherwise the
        # episode would always terminate at step 0 (init is far from target).
        if self.target_mode == "transition" and not self._reached_target:
            return False
        if self.init_mode == "near_target" or self.target_mode == "transition":
            err = self._angle_error()
            over = bool(np.any(np.abs(err) > self._fall_thresholds()))
            if over:
                self._fall_counter += 1
                if self._fall_counter > self.fall_grace_steps:
                    return True
            else:
                self._fall_counter = 0
        return False

    def _reward(self, fallen: bool) -> float:
        st = self._joint_state()
        err = self._angle_error()
        # Per-EP adaptive reward weighting: links targeted UP (target≈0) get
        # 5× weight, links targeted DOWN (target≈π) get 1×.
        # The original fix weighted err[0] (link 1, cart-attached) uniformly 5×,
        # but for EP4 (links 1+2 DOWN, link 3 UP) this penalises the hanging
        # link 5× while giving only 1× to the inverted top link — inverting the
        # intended gradient. The adaptive version weights each link correctly
        # based on its target orientation.
        target = ep_target_angles(self.target_ep)
        # Adaptive weighting: UP-targeted links get 5×, DOWN links get w_down×
        weights = np.where(np.isclose(target, 0.0), 5.0, self.w_down)
        ang_cost = float(np.sum(weights * err ** 2))
        vel_cost = self.vel_cost_coef * float(st[3] ** 2 + st[5] ** 2 + st[7] ** 2)
        cart_cost = 0.1 * float(st[0] ** 2)
        u = float(self.data.ctrl[0])
        ctrl_cost = 0.001 * u ** 2
        r = -(ang_cost + vel_cost + cart_cost + ctrl_cost)
        # Progress reward: dense bonus for reducing weighted error each step.
        # Provides gradient even in episodes that ultimately fail.
        if self.progress_reward_coef > 0.0 and self._prev_err is not None:
            progress = float(np.sum(weights * (self._prev_err ** 2 - err ** 2)))
            r += self.progress_reward_coef * progress
        self._prev_err = err.copy()
        if fallen:
            r -= self.FALL_PENALTY
        return r

    def _terminated(self) -> bool:
        return self._is_fallen()

    # --- gym API ---------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        if options and "target_ep" in options:
            self.target_ep = int(options["target_ep"])
        elif self.target_mode == "random":
            self.target_ep = int(self.np_random.integers(0, 8))
        elif self.target_mode == "weighted":
            # Hard EPs (4, 6) get hard_ep_weight× more training exposure.
            w = np.ones(8)
            w[4] = self.hard_ep_weight
            w[6] = self.hard_ep_weight
            p = w / w.sum()
            self.target_ep = int(self.np_random.choice(8, p=p))
        elif self.target_mode == "transition":
            # Sample a (start, target) pair with start != target.
            self.start_ep = int(self.np_random.integers(0, 8))
            tgt = int(self.np_random.integers(0, 7))
            if tgt >= self.start_ep:
                tgt += 1
            self.target_ep = tgt
        if options and "init_mode" in options:
            self.init_mode = str(options["init_mode"])
        if options and "start_ep" in options:
            self.start_ep = int(options["start_ep"])

        n = float(self.init_noise)
        self.data.qpos[0] = self.np_random.uniform(-n, n)

        # Decide which EP to initialise around. Any explicit start_ep (set via
        # constructor or `options`) overrides the default (which is target_ep).
        init_ep = self.target_ep
        if self.start_ep is not None and self.start_ep != self.target_ep:
            init_ep = self.start_ep

        if self.init_mode == "bottom":
            t1, t2, t3 = np.pi, np.pi, np.pi
        elif self.init_mode == "random":
            t1 = self.np_random.uniform(-np.pi, np.pi)
            t2 = self.np_random.uniform(-np.pi, np.pi)
            t3 = self.np_random.uniform(-np.pi, np.pi)
        else:  # "near_target" (or transition — initialised around init_ep)
            t1, t2, t3 = ep_target_angles(init_ep)

        # Convert absolute angles back to relative hinge coordinates.
        # MuJoCo hinge2 is parented to pole1, hinge3 to pole2, so:
        #   absolute_t2 = qpos[1] + qpos[2]  =>  qpos[2] = t2 - t1
        self.data.qpos[1] = t1 + self.np_random.uniform(-n, n)
        self.data.qpos[2] = (t2 - t1) + self.np_random.uniform(-n, n)
        self.data.qpos[3] = (t3 - t2) + self.np_random.uniform(-n, n)
        self.data.qvel[:] = self.np_random.uniform(-0.01, 0.01, size=self.model.nv)
        self._step_count = 0
        self._fall_counter = 0
        self._prev_err = None
        self._reached_target = False
        self._in_tolerance_counter = 0
        self._transition_bonus_paid = False
        mujoco.mj_forward(self.model, self.data)
        self._prev_err = self._angle_error()
        return self._obs(), {}

    def step(self, action):
        a = float(np.asarray(action, dtype=np.float64).reshape(-1)[0])
        a = max(-1.0, min(1.0, a))
        self.data.ctrl[0] = a
        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        # Transition success detection: angles within tolerance for N consecutive steps.
        err = self._angle_error()
        in_tol = bool(np.all(np.abs(err) < self.transition_success_tol_rad))
        if in_tol:
            self._in_tolerance_counter += 1
            if (not self._reached_target
                    and self._in_tolerance_counter >= self.transition_success_steps):
                self._reached_target = True
        else:
            self._in_tolerance_counter = 0

        obs = self._obs()
        fallen = self._is_fallen()
        reward = self._reward(fallen)

        # Sparse transition bonus: emitted ONCE per episode the first time we
        # actually arrive at the target (after `transition_success_steps`).
        if (self._reached_target
                and not self._transition_bonus_paid
                and self.transition_bonus > 0.0):
            reward += self.transition_bonus
            self._transition_bonus_paid = True

        terminated = fallen
        truncated = self._step_count >= self.max_episode_steps
        info = {
            "target_ep": self.target_ep,
            "start_ep": self.start_ep if self.start_ep is not None else self.target_ep,
            "reached_target": self._reached_target,
            "in_tolerance_steps": self._in_tolerance_counter,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=720)
        self._renderer.update_scene(self.data, camera="track" if "track" in [
            self.model.camera(i).name for i in range(self.model.ncam)
        ] else -1)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def make_env(target_ep: int = 7, **kwargs) -> TriplePendulumEnv:
    return TriplePendulumEnv(target_ep=target_ep, **kwargs)


if __name__ == "__main__":
    env = make_env(target_ep=7)
    obs, _ = env.reset(seed=0)
    print("obs shape:", obs.shape, "obs:", obs)
    for _ in range(5):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        print(f"r={r:+.3f}  term={term}  obs[:8]={obs[:8]}")
    env.close()
