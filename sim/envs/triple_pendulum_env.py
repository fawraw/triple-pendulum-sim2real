"""Gymnasium environment for the triple inverted pendulum on a cart.

Observation: [x, dx, theta1, dtheta1, theta2, dtheta2, theta3, dtheta3, target_id_onehot...]
Action: [u] continuous in [-1, 1], scaled to motor force by the XML actuator gear.
Reward: shaped to encourage convergence toward the target equilibrium point.

The 8 equilibrium points are encoded as 3-bit (link Up=1 / Down=0) configurations:
    EP0 = DDD, EP1 = DDU, EP2 = DUD, EP3 = DUU,
    EP4 = UDD, EP5 = UDU, EP6 = UUD, EP7 = UUU
where the bit order is (theta1, theta2, theta3) with U meaning the link points up
(theta = 0) and D meaning it points down (theta = pi) in absolute world frame.

This is a stub for Milestone 1. Reward shaping, target conditioning and randomization
will evolve in later milestones.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "triple_pendulum.xml"

# Equilibrium target angles in absolute world frame (theta = 0 means link points up).
# 3 bits = 8 EPs. Bit i (i in {0,1,2}) encodes link i+1.
def ep_target_angles(ep_id: int) -> np.ndarray:
    """Return target absolute angles [theta1, theta2, theta3] for equilibrium ep_id (0..7)."""
    bits = [(ep_id >> i) & 1 for i in range(3)]  # bit 0 = link 1, etc.
    return np.array([0.0 if b == 1 else np.pi for b in bits], dtype=np.float64)


class TriplePendulumEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, target_ep: int = 7, render_mode: str | None = None,
                 max_episode_steps: int = 1500,
                 init_mode: str = "near_target", init_noise: float = 0.05,
                 target_mode: str = "fixed",
                 fall_grace_steps: int = 0):
        """
        init_mode:
          - "near_target": start with link angles within `init_noise` of the target EP.
            Use this for the stabilization milestones (M2, M3).
          - "bottom":      start near the natural rest configuration (DDD).
            Use this for swing-up and full transition control (M4+).
          - "random":      start with all link angles uniformly in [-pi, pi].
            Use this once the policy is robust enough.
        target_mode:
          - "fixed":  target_ep stays at the value passed to __init__. Use for
            single-equilibrium milestones (M2).
          - "random": target_ep is resampled uniformly in 0..7 on every reset.
            Use for the conditional milestone (M3) and beyond, so the policy
            learns to read the target one-hot in its observation.
        init_noise: scale of the uniform noise applied to qpos/qvel at reset (rad / rad/s).
        """
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.target_ep = int(target_ep)
        self.max_episode_steps = max_episode_steps
        self.init_mode = init_mode
        self.init_noise = float(init_noise)
        self.target_mode = str(target_mode)
        # Soft-termination: tolerate N consecutive steps over the per-link
        # threshold before triggering a fall. 0 = strict (legacy). 20 ~= 40ms
        # at 50Hz, gives the policy time to react before episode is killed.
        self.fall_grace_steps = int(fall_grace_steps)
        self._fall_counter = 0
        self._step_count = 0
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
    # made EP4 (UDD) / EP6 (UUD) untrainable: when link 1 is held vertical,
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
        if self.init_mode == "near_target":
            err = self._angle_error()
            over = bool(np.any(np.abs(err) > self._fall_thresholds()))
            if over:
                self._fall_counter += 1
                # Strict mode (grace=0): fall on first step over threshold.
                # Soft mode (grace>0): only fall after N consecutive over-threshold steps.
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
        weights = np.where(np.isclose(target, 0.0), 5.0, 1.0)
        ang_cost = float(np.sum(weights * err ** 2))
        # Bumped 0.01 -> 0.05 to discourage flailing on hard EPs.
        vel_cost = 0.05 * float(st[3] ** 2 + st[5] ** 2 + st[7] ** 2)
        cart_cost = 0.1 * float(st[0] ** 2)
        u = float(self.data.ctrl[0])
        ctrl_cost = 0.001 * u ** 2
        r = -(ang_cost + vel_cost + cart_cost + ctrl_cost)
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
        if options and "init_mode" in options:
            self.init_mode = str(options["init_mode"])

        n = float(self.init_noise)
        self.data.qpos[0] = self.np_random.uniform(-n, n)

        if self.init_mode == "bottom":
            t1, t2, t3 = np.pi, np.pi, np.pi
        elif self.init_mode == "random":
            t1 = self.np_random.uniform(-np.pi, np.pi)
            t2 = self.np_random.uniform(-np.pi, np.pi)
            t3 = self.np_random.uniform(-np.pi, np.pi)
        else:  # "near_target"
            t1, t2, t3 = ep_target_angles(self.target_ep)

        # Convert absolute angles back to relative hinge coordinates.
        # MuJoCo hinge2 is parented to pole1, hinge3 to pole2, so:
        #   absolute_t2 = qpos[1] + qpos[2]  =>  qpos[2] = t2 - t1
        self.data.qpos[1] = t1 + self.np_random.uniform(-n, n)
        self.data.qpos[2] = (t2 - t1) + self.np_random.uniform(-n, n)
        self.data.qpos[3] = (t3 - t2) + self.np_random.uniform(-n, n)
        self.data.qvel[:] = self.np_random.uniform(-0.01, 0.01, size=self.model.nv)
        self._step_count = 0
        self._fall_counter = 0
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), {}

    def step(self, action):
        a = float(np.asarray(action, dtype=np.float64).reshape(-1)[0])
        a = max(-1.0, min(1.0, a))
        self.data.ctrl[0] = a
        mujoco.mj_step(self.model, self.data)
        self._step_count += 1
        obs = self._obs()
        fallen = self._is_fallen()
        reward = self._reward(fallen)
        terminated = fallen
        truncated = self._step_count >= self.max_episode_steps
        return obs, reward, terminated, truncated, {"target_ep": self.target_ep}

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
