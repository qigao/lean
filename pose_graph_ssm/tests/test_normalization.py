from __future__ import annotations

import numpy as np
import pytest

from pose_graph_ssm.normalization import normalize_body
from pose_graph_ssm.ntu import NtuBody


def _body(*, offset=(0.0, 0.0, 0.0), scale=1.0, frames=5) -> NtuBody:
    positions = np.zeros((frames, 25, 3), dtype=np.float64)
    tracking = np.full((frames, 25), 2, dtype=np.int64)
    present = np.ones((frames,), dtype=bool)
    off = np.asarray(offset, dtype=np.float64)
    for t in range(frames):
        for j in range(25):
            base = np.array([0.05 * t + 0.01 * j, 0.02 * j, 0.1], dtype=np.float64)
            if j == 20:
                base[1] += 1.0
            positions[t, j] = off + scale * base
    return NtuBody(body_id=10, positions=positions, tracking_state=tracking, present=present)


def test_translation_and_uniform_scale_are_removed():
    a = _body(offset=(0.0, 0.0, 0.0), scale=1.0)
    b = _body(offset=(10.0, -3.0, 5.0), scale=7.0)
    np.testing.assert_allclose(normalize_body(a), normalize_body(b), atol=1e-6)


def test_tracking_zero_is_interpolated_not_origin_jump():
    body = _body()
    positions = body.positions.copy()
    tracking = body.tracking_state.copy()
    tracking[2, 5] = 0
    positions[2, 5] = 0.0
    dropped = NtuBody(10, positions, tracking, body.present)
    out = normalize_body(dropped)
    np.testing.assert_allclose(out[2, 5], (out[1, 5] + out[3, 5]) / 2.0, atol=1e-6)


def test_missing_body_frame_uses_endpoint_or_linear_interpolation():
    body = _body()
    positions = body.positions.copy()
    tracking = body.tracking_state.copy()
    present = body.present.copy()
    present[2] = False
    positions[2] = 0.0
    tracking[2] = 0
    dropped = NtuBody(10, positions, tracking, present)
    out = normalize_body(dropped)
    np.testing.assert_allclose(out[2], (out[1] + out[3]) / 2.0, atol=1e-6)


def test_joint_with_no_usable_observation_fails_closed():
    body = _body()
    tracking = body.tracking_state.copy()
    tracking[:, 7] = 0
    with pytest.raises(ValueError, match="joint 7.*no usable"):
        normalize_body(NtuBody(10, body.positions, tracking, body.present))


def test_degenerate_torso_scale_fails_closed():
    body = _body()
    positions = body.positions.copy()
    positions[:, 20] = positions[:, 0]
    with pytest.raises(ValueError, match="torso scale"):
        normalize_body(NtuBody(10, positions, body.tracking_state, body.present))
