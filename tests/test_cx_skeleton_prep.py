from __future__ import annotations

import numpy as np
import pytest

from yolo_flywire.cx_events import KinematicEventEncoder
from yolo_flywire.cx_skeleton_prep import normalize_ntu_body
from yolo_flywire.ntu_skeleton import NtuSkeletonBody


def _body(*, missing_frame: int | None = None, scale: float = 1.0, offset=(0.0, 0.0, 0.0)) -> NtuSkeletonBody:
    frames = 5
    positions = np.zeros((frames, 25, 3), dtype=np.float64)
    base = np.asarray(offset, dtype=np.float64)
    for frame in range(frames):
        positions[frame, :, :] = base
        # Static articulated pose. Joint 20 supplies a stable torso-length scale.
        positions[frame, 20, :] = base + np.asarray([0.0, scale, 0.0])
        for joint in range(1, 25):
            if joint == 20:
                continue
            positions[frame, joint, :] = base + np.asarray([0.01 * joint * scale, 0.02 * joint * scale, 0.0])
    tracking = np.full((frames, 25), 2, dtype=np.int64)
    present = np.ones((frames,), dtype=bool)
    if missing_frame is not None:
        positions[missing_frame, :, :] = 0.0
        tracking[missing_frame, :] = 0
        present[missing_frame] = False
    return NtuSkeletonBody(body_id=7, positions=positions, tracking_state=tracking, present=present)


def test_missing_body_frame_is_interpolated_not_treated_as_motion():
    normalized = normalize_ntu_body(_body(missing_frame=2))
    encoder = KinematicEventEncoder.fixed_for_test(threshold=1e-4)
    # fixed_for_test intentionally has no bones; it still sees joint motion and
    # acceleration, so a zero-filled missing frame would generate events here.
    events = encoder.transform(normalized)
    assert np.count_nonzero(events) == 0


def test_normalization_is_translation_invariant():
    a = normalize_ntu_body(_body(offset=(0.0, 0.0, 0.0)))
    b = normalize_ntu_body(_body(offset=(100.0, -3.0, 8.0)))
    np.testing.assert_allclose(a, b, atol=1e-12, rtol=0.0)


def test_normalization_is_uniform_scale_invariant():
    a = normalize_ntu_body(_body(scale=1.0))
    b = normalize_ntu_body(_body(scale=3.5))
    np.testing.assert_allclose(a, b, atol=1e-12, rtol=0.0)


def test_each_joint_requires_at_least_one_usable_observation():
    body = _body()
    tracking = body.tracking_state.copy()
    tracking[:, 24] = 0
    broken = NtuSkeletonBody(
        body_id=body.body_id,
        positions=body.positions,
        tracking_state=tracking,
        present=body.present,
    )
    with pytest.raises(ValueError, match="joint 24.*usable"):
        normalize_ntu_body(broken)


def test_degenerate_torso_scale_fails_closed():
    body = _body()
    positions = body.positions.copy()
    positions[:, 20, :] = positions[:, 0, :]
    broken = NtuSkeletonBody(
        body_id=body.body_id,
        positions=positions,
        tracking_state=body.tracking_state,
        present=body.present,
    )
    with pytest.raises(ValueError, match="torso scale"):
        normalize_ntu_body(broken)


def test_tracking_state_one_is_admissible_but_zero_is_missing():
    body = _body()
    positions = body.positions.copy()
    tracking = body.tracking_state.copy()
    positions[1, 5, :] = positions[0, 5, :] + np.asarray([99.0, 0.0, 0.0])
    tracking[1, 5] = 0
    positions[3, 6, :] = positions[0, 6, :]
    tracking[3, 6] = 1
    mixed = NtuSkeletonBody(
        body_id=body.body_id,
        positions=positions,
        tracking_state=tracking,
        present=body.present,
    )
    normalized = normalize_ntu_body(mixed)
    assert np.isfinite(normalized).all()
    # The tracking=0 outlier must have been replaced by interpolation.
    assert abs(normalized[1, 5, 0]) < 1.0
