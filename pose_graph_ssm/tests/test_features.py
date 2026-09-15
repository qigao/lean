from __future__ import annotations

import json

import numpy as np
import pytest

from pose_graph_ssm.features import FeatureStandardizer, NTU25_PARENT, kinematic_features


def _sequence(scale: float = 1.0, frames: int = 6) -> np.ndarray:
    seq = np.zeros((frames, 25, 3), dtype=np.float64)
    for t in range(frames):
        for j in range(25):
            seq[t, j] = scale * np.array([0.1 * t + 0.01 * j, 0.02 * j, 0.03 * t])
    return seq


def test_features_keep_joint_axis_and_15_channels():
    x = _sequence()
    out = kinematic_features(x)
    assert out.shape == (x.shape[0], 25, 15)
    np.testing.assert_allclose(out[:, :, 0:3], x)


def test_root_bone_and_root_bone_motion_are_zero():
    out = kinematic_features(_sequence())
    assert NTU25_PARENT[0] == -1
    assert np.count_nonzero(out[:, 0, 6:12]) == 0


def test_first_frame_derivative_groups_are_zero():
    out = kinematic_features(_sequence())
    assert np.count_nonzero(out[0, :, 3:6]) == 0
    assert np.count_nonzero(out[0, :, 9:12]) == 0
    assert np.count_nonzero(out[0, :, 12:15]) == 0


def test_standardizer_fits_per_joint_channel_and_round_trips():
    features = [kinematic_features(_sequence(1.0)), kinematic_features(_sequence(2.0))]
    standardizer = FeatureStandardizer.fit(features)
    assert standardizer.mean.shape == (25, 15)
    assert standardizer.std.shape == (25, 15)
    assert np.isfinite(standardizer.mean).all()
    assert np.isfinite(standardizer.std).all()
    restored = FeatureStandardizer.from_json(standardizer.to_json())
    assert restored.fingerprint == standardizer.fingerprint
    np.testing.assert_array_equal(restored.mean, standardizer.mean)
    np.testing.assert_array_equal(restored.std, standardizer.std)


def test_validation_transform_cannot_refit_or_mutate_statistics():
    train = [kinematic_features(_sequence(1.0))]
    validation = kinematic_features(_sequence(1000.0))
    standardizer = FeatureStandardizer.fit(train)
    mean_before = standardizer.mean.copy()
    std_before = standardizer.std.copy()
    transformed = standardizer.transform(validation)
    assert transformed.shape == validation.shape
    np.testing.assert_array_equal(standardizer.mean, mean_before)
    np.testing.assert_array_equal(standardizer.std, std_before)


def test_zero_variance_channels_use_epsilon_not_nan():
    features = [np.zeros((4, 25, 15), dtype=np.float64)]
    standardizer = FeatureStandardizer.fit(features)
    transformed = standardizer.transform(features[0])
    assert np.isfinite(transformed).all()
    assert np.count_nonzero(transformed) == 0


def test_feature_input_rejects_nonfinite_values():
    seq = _sequence()
    seq[2, 4, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        kinematic_features(seq)
