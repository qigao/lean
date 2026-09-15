import numpy as np
import pytest

from pose_temporal_snn.normalization import normalize_body
from pose_temporal_snn.ntu import NtuBody


def _body(*, offset=(0.0, 0.0, 0.0), scale=1.0, frames=5) -> NtuBody:
    positions = np.zeros((frames, 25, 3), dtype=np.float64)
    for t in range(frames):
        for joint in range(25):
            positions[t, joint] = np.array([joint * 0.02, joint * 0.01, 0.0])
        positions[t, 20] = np.array([0.0, 1.0, 0.0])
        positions[t, 5, 0] += 0.1 * t
    positions = positions * scale + np.asarray(offset, dtype=np.float64)
    tracking = np.full((frames, 25), 2, dtype=np.int64)
    present = np.ones((frames,), dtype=bool)
    return NtuBody(body_id=10, positions=positions, tracking_state=tracking, present=present)


def test_translation_and_uniform_scale_are_removed():
    a = normalize_body(_body(offset=(0.0, 0.0, 0.0), scale=1.0))
    b = normalize_body(_body(offset=(10.0, -3.0, 5.0), scale=7.0))
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_tracking_zero_is_interpolated_not_origin_jump():
    body = _body()
    positions = np.array(body.positions, copy=True)
    tracking = np.array(body.tracking_state, copy=True)
    positions[2, 5] = 0.0
    tracking[2, 5] = 0
    changed = NtuBody(body_id=body.body_id, positions=positions, tracking_state=tracking, present=body.present)
    out = normalize_body(changed)
    np.testing.assert_allclose(out[2, 5], (out[1, 5] + out[3, 5]) / 2.0, atol=1e-6)


def test_absent_body_frame_uses_endpoint_or_linear_interpolation():
    body = _body()
    positions = np.array(body.positions, copy=True)
    tracking = np.array(body.tracking_state, copy=True)
    present = np.array(body.present, copy=True)
    positions[2] = 0.0
    tracking[2] = 0
    present[2] = False
    changed = NtuBody(body_id=10, positions=positions, tracking_state=tracking, present=present)
    out = normalize_body(changed)
    np.testing.assert_allclose(out[2], (out[1] + out[3]) / 2.0, atol=1e-6)


def test_joint_without_any_usable_observation_fails_closed():
    body = _body()
    tracking = np.array(body.tracking_state, copy=True)
    tracking[:, 24] = 0
    changed = NtuBody(body_id=10, positions=body.positions, tracking_state=tracking, present=body.present)
    with pytest.raises(ValueError, match="joint 24"):
        normalize_body(changed)


def test_degenerate_torso_scale_fails_closed():
    body = _body()
    positions = np.array(body.positions, copy=True)
    positions[:, 20] = positions[:, 0]
    changed = NtuBody(body_id=10, positions=positions, tracking_state=body.tracking_state, present=body.present)
    with pytest.raises(ValueError, match="scale"):
        normalize_body(changed)
