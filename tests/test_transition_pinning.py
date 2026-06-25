"""target_mode='transition' randomises the (start, target) pair on reset; the
eval must pin it via reset(options=...). These tests lock that contract."""
from sim.envs.triple_pendulum_env import TriplePendulumEnv


def _env():
    return TriplePendulumEnv(target_mode="transition", init_mode="near_target",
                             max_episode_steps=100)


def test_options_pin_the_pair():
    env = _env()
    for seed in range(6):
        env.reset(seed=seed, options={"start_ep": 0, "target_ep": 1})
        assert env.target_ep == 1, f"target_ep not pinned (seed {seed})"
        assert env.start_ep == 0, f"start_ep not pinned (seed {seed})"


def test_without_options_pair_varies():
    # Without options, transition mode randomises -> over several resets the
    # target should take more than one value (this is why the eval must pin it).
    env = _env()
    targets = set()
    for seed in range(20):
        env.reset(seed=seed)
        targets.add(env.target_ep)
    assert len(targets) > 1, "transition mode should randomise target_ep without options"


def test_pinned_target_drives_observation_onehot():
    env = _env()
    obs, _ = env.reset(seed=3, options={"start_ep": 0, "target_ep": 1})
    onehot = obs[8:16]
    assert onehot[1] == 1.0 and onehot.sum() == 1.0, "obs target one-hot must match pinned target_ep"
