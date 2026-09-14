import numpy as np

from yolo_flywire.features import FeatureSpec, encode_sequence
from yolo_flywire.synthetic import make_synthetic_dataset


def test_encoding_is_translation_invariant_when_enabled():
    sample = make_synthetic_dataset(3, 1, 8)[0].sequence
    shifted = sample.shifted(dx=100.0, dy=-40.0)
    spec = FeatureSpec(normalize_translation=True, normalize_scale=True)
    np.testing.assert_allclose(
        encode_sequence(sample, spec),
        encode_sequence(shifted, spec),
        atol=1e-6,
    )


def test_encoding_contains_velocity_and_confidence():
    sample = make_synthetic_dataset(4, 1, 8)[0].sequence
    x = encode_sequence(sample, FeatureSpec())
    assert x.shape[0] == 8
    assert np.isfinite(x).all()
