from __future__ import annotations

import json

import numpy as np
import pytest

from yolo_flywire.cx_events import KinematicEventEncoder, NTU25_BONES


def test_static_pose_emits_no_motion_events():
    seq = np.ones((8, 25, 3), dtype=np.float32)
    encoder = KinematicEventEncoder.fixed_for_test(threshold=0.1)
    events = encoder.transform(seq)
    assert events.shape[0] == 8
    assert np.count_nonzero(events) == 0


def test_positive_and_negative_motion_keep_sign():
    seq = np.zeros((4, 1, 1), dtype=np.float32)
    seq[:, 0, 0] = [0.0, 1.0, 0.0, 0.0]
    encoder = KinematicEventEncoder.fixed_for_test(threshold=0.5)
    events = encoder.transform(seq)
    assert +1.0 in events
    assert -1.0 in events
    assert set(np.unique(events)).issubset({-1.0, 0.0, 1.0})


def test_fit_uses_ntu_joint_motion_bone_motion_and_acceleration():
    train = []
    for scale in (1.0, 2.0):
        seq = np.zeros((5, 25, 3), dtype=np.float64)
        seq[:, 4, 0] = np.arange(5) * scale
        seq[:, 5, 0] = np.arange(5) * 0.5 * scale
        train.append(seq)
    encoder = KinematicEventEncoder.fit(train, quantile=0.75)
    assert encoder.bones == NTU25_BONES
    expected_dim = (25 + len(NTU25_BONES) + 25) * 3
    assert encoder.feature_dim == expected_dim
    assert encoder.thresholds.shape == (expected_dim,)
    assert np.all(np.isfinite(encoder.thresholds))
    assert np.all(encoder.thresholds > 0.0)


def test_transform_does_not_refit_thresholds():
    train = [np.zeros((4, 25, 3), dtype=np.float64)]
    train[0][:, 4, 0] = [0.0, 0.5, 1.0, 1.5]
    encoder = KinematicEventEncoder.fit(train, quantile=0.5)
    before = encoder.thresholds.copy()
    validation = np.zeros((4, 25, 3), dtype=np.float64)
    validation[:, 4, 0] = [0.0, 10.0, -10.0, 5.0]
    encoder.transform(validation)
    np.testing.assert_array_equal(encoder.thresholds, before)


def test_encoder_serialization_and_fingerprint_are_deterministic():
    seq = np.zeros((4, 25, 3), dtype=np.float64)
    seq[:, 4, 0] = [0.0, 1.0, 2.0, 3.0]
    a = KinematicEventEncoder.fit([seq], quantile=0.75)
    b = KinematicEventEncoder.from_json(a.to_json())
    assert json.loads(a.to_json()) == json.loads(b.to_json())
    assert a.fingerprint == b.fingerprint
    np.testing.assert_array_equal(a.thresholds, b.thresholds)


def test_nonfinite_sequence_is_rejected():
    seq = np.zeros((4, 25, 3), dtype=np.float64)
    seq[2, 4, 0] = np.nan
    encoder = KinematicEventEncoder.fixed_for_test(threshold=0.1)
    with pytest.raises(ValueError, match="finite"):
        encoder.transform(seq)


def test_fit_requires_nonempty_training_sequences():
    with pytest.raises(ValueError, match="training"):
        KinematicEventEncoder.fit([], quantile=0.75)


@pytest.mark.parametrize("quantile", [0.0, 1.0, -0.1, 1.1, float("nan")])
def test_fit_rejects_invalid_threshold_quantile(quantile):
    seq = np.zeros((4, 25, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="quantile"):
        KinematicEventEncoder.fit([seq], quantile=quantile)
