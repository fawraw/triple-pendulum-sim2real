"""Tests for the two-stage HandoffController switch logic (no real models)."""
import numpy as np

from sim.handoff import HandoffController, wrap_to_pi


class _StubPolicy:
    """Minimal SB3-like policy: predict returns a constant labelled action."""
    def __init__(self, tag):
        self.tag = tag

    def predict(self, obs, deterministic=True):
        return np.array([self.tag], dtype=np.float32), None


def _ctrl(**kw):
    # target = UDD (link1 up=0, link2/3 down=pi)
    return HandoffController(_StubPolicy(1.0), _StubPolicy(2.0),
                             target_angles=[0.0, np.pi, np.pi], **kw)


def test_wrap_to_pi():
    assert abs(wrap_to_pi(np.array([3 * np.pi]))[0] - np.pi) < 1e-6
    assert abs(wrap_to_pi(np.array([-3 * np.pi]))[0] + np.pi) < 1e-6 or \
           abs(wrap_to_pi(np.array([-3 * np.pi]))[0] - np.pi) < 1e-6


def test_swingup_when_far():
    c = _ctrl(capture_tol_rad=0.3)
    _, mode = c.act(obs=None, abs_angles=[np.pi, np.pi, np.pi])  # at DDD, far from UDD
    assert mode == "swing"


def test_stabilizer_when_within_capture():
    c = _ctrl(capture_tol_rad=0.3)
    _, mode = c.act(obs=None, abs_angles=[0.1, np.pi - 0.1, np.pi + 0.1])  # near UDD
    assert mode == "stab"
    assert c.handoff_step == 0


def test_velocity_gate_blocks_capture():
    c = _ctrl(capture_tol_rad=0.3, capture_vel_rad_s=1.0)
    # angles in tolerance but a link is spinning fast -> not captured
    _, mode = c.act(obs=None, abs_angles=[0.0, np.pi, np.pi], abs_vels=[5.0, 0.0, 0.0])
    assert mode == "swing"


def test_latch_keeps_stabilizer_after_capture():
    c = _ctrl(capture_tol_rad=0.3, latch=True)
    c.act(obs=None, abs_angles=[0.0, np.pi, np.pi])           # capture (step 0)
    _, mode = c.act(obs=None, abs_angles=[np.pi, np.pi, np.pi])  # now far again
    assert mode == "stab"            # latched
    assert c.handoff_step == 0


def test_no_latch_releases_when_leaving_capture():
    c = _ctrl(capture_tol_rad=0.3, latch=False)
    c.act(obs=None, abs_angles=[0.0, np.pi, np.pi])           # capture
    _, mode = c.act(obs=None, abs_angles=[np.pi, np.pi, np.pi])  # far
    assert mode == "swing"           # not latched -> back to swing-up


def test_reset_clears_capture():
    c = _ctrl(capture_tol_rad=0.3)
    c.act(obs=None, abs_angles=[0.0, np.pi, np.pi])
    c.reset()
    assert c._captured is False and c.handoff_step is None
