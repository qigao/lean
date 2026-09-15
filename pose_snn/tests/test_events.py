import json

import numpy as np
import pytest

from pose_temporal_snn.events import EventEncoder, NTU25_PARENT


def _moving_sequence(scale: float = 1.0) -> np.ndarray:
    seq = np.zeros((6, 25, 3), dtype=np.float64)
    seq[:, 20, 1] = 1.0
    seq[:, 5, 0] = np.arange(6) * 0.2 * scale
    seq[:, 6, 0] = np.arange(6) * 0.1 * scale
    return seq


def test_static_skeleton_emits_no_events():
    seq = np.ones((8, 25, 3), dtype=np.float64)
    out = EventEncoder.fixed_for_test(0.1).transform(seq)
    assert out.flat.shape == (8, 225)
    assert out.nodes.shape == (8, 25, 9)
    assert np.count_nonzero(out.flat) == 0


def test_root_bone_motion_channels_are_structurally_zero():
    out = EventEncoder.fixed_for_test(0.01).transform(_moving_sequence())
    assert NTU25_PARENT[0] == -1
    assert np.count_nonzero(out.nodes[:, 0, 3:6]) == 0


def test_signed_motion_preserves_direction():
    seq = _moving_sequence()
    out = EventEncoder.fixed_for_test(0.05).transform(seq)
    assert +1.0 in out.flat
    reversed_seq = seq[::-1].copy()
    reversed_out = EventEncoder.fixed_for_test(0.05).transform(reversed_seq)
    assert -1.0 in reversed_out.flat
    assert set(np.unique(out.flat)).issubset({-1.0, 0.0, 1.0})
    assert set(np.unique(reversed_out.flat)).issubset({-1.0, 0.0, 1.0})


def test_fit_uses_per_feature_q75_with_positive_floor():
    encoder = EventEncoder.fit((_moving_sequence(1.0), _moving_sequence(2.0)), quantile=0.75)
    assert encoder.thresholds.shape == (225,)
    assert np.all(np.isfinite(encoder.thresholds))
    assert np.all(encoder.thresholds >= 1e-6)
    root_bone = np.arange(3, 6)
    np.testing.assert_allclose(encoder.thresholds[root_bone], 1e-6)


def test_validation_transform_does_not_mutate_training_fitted_thresholds():
    encoder = EventEncoder.fit((_moving_sequence(1.0),), quantile=0.75)
    before = encoder.thresholds.copy()
    encoder.transform(_moving_sequence(5000.0))
    np.testing.assert_array_equal(encoder.thresholds, before)


def test_serialization_and_fingerprint_are_deterministic():
    first = EventEncoder.fit((_moving_sequence(1.0), _moving_sequence(2.0)), quantile=0.75)
    second = EventEncoder.from_json(first.to_json())
    assert json.loads(first.to_json()) == json.loads(second.to_json())
    assert first.fingerprint == second.fingerprint
    np.testing.assert_array_equal(first.thresholds, second.thresholds)


def test_fit_rejects_invalid_training_or_quantile():
    with pytest.raises(ValueError, match="training"):
        EventEncoder.fit((), quantile=0.75)
    with pytest.raises(ValueError, match="quantile"):
        EventEncoder.fit((_moving_sequence(),), quantile=1.0)


def test_transform_rejects_nonfinite_or_wrong_shape():
    encoder = EventEncoder.fixed_for_test(0.1)
    with pytest.raises(ValueError, match="shape"):
        encoder.transform(np.zeros((3, 24, 3)))
    bad = _moving_sequence()
    bad[2, 5, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        encoder.transform(bad)
