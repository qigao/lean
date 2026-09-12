"""Analytic COCO17 fixtures, not detector accuracy or recognition evidence."""
from dataclasses import FrozenInstanceError
from fractions import Fraction
import hashlib
import importlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from yolo_flywire.schema import KeypointFrame, PointSequence


XY = slice(0, 34)
VELOCITY = slice(34, 68)
CONFIDENCE = slice(68, 85)
POSITION_MASK = slice(85, 102)
VELOCITY_MASK = slice(102, 119)
PERSON = 119
DT = 120


def _module():
    return importlib.import_module("yolo_flywire.pose_features")


def _frames(count=3):
    frames = []
    for index in range(count):
        points = [[10., 14., .8] for _ in range(17)]
        points[0] = [25., 7., .9]  # Nose is deliberately NOT a torso anchor.
        points[1], points[2] = [24., 6., .8], [26., 6., .8]
        points[5], points[6] = [8., 12., .8], [12., 12., .8]
        points[11], points[12] = [8., 20., .8], [12., 20., .8]
        points[9] = [14. + 4. * index, 16., .7]
        frames.append(points)
    return frames


def _sequence(frames):
    return PointSequence(tuple(KeypointFrame(tuple(tuple(p) for p in f)) for f in frames))


def _encode(frames=None, **changes):
    module = _module()
    frames = _frames() if frames is None else frames
    count = len(frames)
    options = dict(pts=(0, 1, 3) if count == 3 else tuple(range(count)),
                   time_bases=(Fraction(1, 4),) * count,
                   detector_confidences=(.9,) * count, spec=module.PoseFeatureSpec())
    options.update(changes)
    return module.encode_timed_pose(_sequence(frames), **options)


def test_coco_anchors_layout_and_seconds_based_velocity_have_analytic_values():
    result = _encode()
    assert result.shape == (3, 121)
    assert result.dtype == np.float64 and result.flags.c_contiguous
    np.testing.assert_array_equal(result[:, 18:20], [[1., -1.], [2., -1.], [3., -1.]])
    np.testing.assert_array_equal(result[:, 52:54], [[0., 0.], [4., 0.], [2., 0.]])
    np.testing.assert_array_equal(result[0, :2], [3.75, -3.25])
    np.testing.assert_array_equal(result[0, 10:14], [-.5, -2., .5, -2.])
    np.testing.assert_array_equal(result[0, 22:26], [-.5, 0., .5, 0.])
    np.testing.assert_array_equal(result[:, DT], [0., .25, .5])
    np.testing.assert_array_equal(result[:, PERSON], [.9] * 3)
    np.testing.assert_array_equal(result[:, CONFIDENCE], np.array(_frames())[:, :, 2])
    assert np.all(result[:, POSITION_MASK] == 1.)
    assert np.all(result[0, VELOCITY_MASK] == 0.)
    assert np.all(result[1:, VELOCITY_MASK] == 1.)


def test_rational_differences_are_taken_before_float_conversion():
    bases = (Fraction(1, 120), Fraction(1, 60), Fraction(1, 120))
    small = _encode(pts=(-1, 1, 7), time_bases=bases)
    offset = 10 ** 40
    large = _encode(pts=(offset * 120 - 1, offset * 60 + 1, offset * 120 + 7), time_bases=bases)
    np.testing.assert_array_equal(small, large)
    np.testing.assert_allclose(large[:, DT], [0., 1 / 40, 1 / 24])
    np.testing.assert_allclose(large[:, 52], [0., 40., 24.])


def test_time_dilation_changes_velocity_not_geometry_or_confidence():
    normal, slow = _encode(), _encode(pts=(0, 2, 6))
    np.testing.assert_array_equal(slow[:, XY], normal[:, XY])
    np.testing.assert_array_equal(slow[:, VELOCITY], normal[:, VELOCITY] / 2)
    np.testing.assert_array_equal(slow[:, 68:120], normal[:, 68:120])
    np.testing.assert_array_equal(slow[:, DT], normal[:, DT] * 2)


def test_translation_and_positive_scale_are_nuisance_invariant():
    transformed = [[[3 * x + 100, 3 * y - 40, c] for x, y, c in f] for f in _frames()]
    np.testing.assert_array_equal(_encode(transformed), _encode())


def test_low_confidence_coordinates_cannot_affect_features_or_bridge_a_gap():
    frames = _frames(4)
    frames[1][9] = [1e308, -1e308, .01]
    result = _encode(frames)
    np.testing.assert_array_equal(result[:, 18:20], [[1., -1.], [0., 0.], [3., -1.], [4., -1.]])
    np.testing.assert_array_equal(result[:, 85 + 9], [1., 0., 1., 1.])
    np.testing.assert_array_equal(result[:, 102 + 9], [0., 0., 0., 1.])
    np.testing.assert_array_equal(result[:, 52], [0., 0., 0., 4.])
    assert result[1, 68 + 9] == .01
    frames[1][9][:2] = [0., 0.]
    np.testing.assert_array_equal(result, _encode(frames))


def test_confidence_threshold_is_inclusive_but_zero_confidence_is_never_visible():
    frames = _frames()
    frames[0][9][2] = .05
    frames[1][9][2] = np.nextafter(.05, 0.).item()
    frames[2][9][2] = 0.
    result = _encode(frames)
    np.testing.assert_array_equal(result[:, 85 + 9], [1., 0., 0.])
    assert not result[:, 102 + 9].any()


@pytest.mark.parametrize("reference", [5, 6, 11, 12, "collapsed", "epsilon"])
def test_invalid_reference_masks_whole_frame_and_adjacent_velocity(reference):
    frames = _frames(4)
    if isinstance(reference, int):
        frames[1][reference] = [1e308, -1e308, 0.]
    else:
        frames[1][5][:2], frames[1][6][:2] = [0., 0.], [0., 0.]
        if reference == "epsilon":
            frames[1][6][0] = 1e-6
    result = _encode(frames)
    assert not result[1, XY].any()
    assert not result[1, POSITION_MASK].any()
    assert not result[:3, VELOCITY_MASK].any()
    assert not result[:3, VELOCITY].any()
    assert np.all(result[3, VELOCITY_MASK] == 1.)
    np.testing.assert_array_equal(result[1, CONFIDENCE], np.array(frames)[1, :, 2])
    assert result[1, PERSON] == .9 and result[1, DT] == .25


def test_missing_person_keeps_frame_and_dt_without_inventing_motion():
    frames = _frames(4)
    frames[1] = [[0., 0., 0.] for _ in range(17)]
    result = _encode(frames, detector_confidences=(.9, 0., .9, .9))
    assert result.shape == (4, 121)
    assert not result[1, :120].any()
    assert not result[:3, VELOCITY].any()
    assert not result[:3, VELOCITY_MASK].any()
    assert np.all(result[3, VELOCITY_MASK] == 1.)
    np.testing.assert_array_equal(result[:, DT], [0., .25, .25, .25])


def test_single_frame_has_zero_velocity_and_no_invented_initial_interval():
    result = _encode(_frames(1))
    assert result.shape == (1, 121)
    assert not result[:, VELOCITY].any()
    assert not result[:, VELOCITY_MASK].any()
    assert result[0, DT] == 0.


@pytest.mark.parametrize("changes", [
    {"pts": (0, 1)}, {"pts": [0, 1, 3]}, {"pts": (False, 1, 3)},
    {"pts": (0., 1, 3)}, {"pts": (0, 0, 3)}, {"pts": (0, -1, 3)},
    {"time_bases": (Fraction(1, 4),)}, {"time_bases": (.25,) * 3},
    {"time_bases": (Fraction(0),) * 3}, {"time_bases": (Fraction(-1),) * 3},
    {"time_bases": (Fraction(1, 10 ** 500),) * 3},
    {"time_bases": (Fraction(10 ** 500),) * 3},
    {"detector_confidences": (.9,)}, {"detector_confidences": (True, .9, .9)},
    {"detector_confidences": (1.1, .9, .9)}, {"detector_confidences": (float("nan"), .9, .9)},
    {"detector_confidences": (0., .9, .9)}, {"spec": None},
])
def test_invalid_timing_person_scores_or_spec_fail_without_fallback(changes):
    with pytest.raises(ValueError):
        _encode(**changes)


@pytest.mark.parametrize("value", [True, "1", None, float("nan"), float("inf"), 10 ** 500])
def test_invalid_coordinate_numbers_are_not_coerced_even_when_masked(value):
    frames = _frames()
    frames[1][9] = [value, 0., 0.]
    with pytest.raises(ValueError):
        _encode(frames)


@pytest.mark.parametrize("confidence", [-.01, 1.01, True, "0.9"])
def test_invalid_joint_confidence_is_rejected(confidence):
    frames = _frames()
    frames[0][9][2] = confidence
    with pytest.raises(ValueError):
        _encode(frames)


@pytest.mark.parametrize("count", [0, 16, 18])
def test_non_coco17_schema_is_rejected(count):
    frames = [[(0., 0., .8)] * count for _ in range(3)]
    with pytest.raises(ValueError):
        _encode(frames)


def test_finite_inputs_that_overflow_geometry_or_velocity_are_rejected():
    frames = _frames()
    frames[0][9][0], frames[1][9][0] = -1e308, 1e308
    with pytest.raises(ValueError):
        _encode(frames, time_bases=(Fraction(1, 100),) * 3)
    frames = _frames()
    frames[0][5][0], frames[0][6][0] = -1e308, 1e308
    with pytest.raises(ValueError):
        _encode(frames)


@pytest.mark.parametrize("field,value", [
    ("confidence_threshold", 0), ("confidence_threshold", -1.), ("confidence_threshold", 1.1),
    ("confidence_threshold", True), ("confidence_threshold", "0.05"),
    ("confidence_threshold", float("nan")), ("scale_epsilon", 0.),
    ("scale_epsilon", -1.), ("scale_epsilon", True), ("scale_epsilon", float("inf")),
])
def test_spec_rejects_implicit_or_invalid_numeric_parameters(field, value):
    with pytest.raises(ValueError):
        _module().PoseFeatureSpec(**{field: value})


def test_pure_encoder_has_no_metadata_input_or_mutation_and_returns_fresh_arrays():
    module = _module()
    assert set(inspect.signature(module.encode_timed_pose).parameters) == {
        "sequence", "pts", "time_bases", "detector_confidences", "spec"}
    sequence, spec = _sequence(_frames()), module.PoseFeatureSpec()
    options = dict(pts=(0, 1, 3), time_bases=(Fraction(1, 4),) * 3,
                   detector_confidences=(.9,) * 3, spec=spec)
    before = repr((sequence, options))
    first = module.encode_timed_pose(sequence, **options)
    expected = first.copy()
    first[:] = -1.
    np.testing.assert_array_equal(module.encode_timed_pose(sequence, **options), expected)
    assert repr((sequence, options)) == before
    with pytest.raises(FrozenInstanceError):
        spec.confidence_threshold = .2


def test_descriptor_and_encoder_hash_bind_layout_parameters_and_actual_source():
    module = _module()
    spec = module.PoseFeatureSpec()
    descriptor = spec.descriptor()
    assert descriptor["feature_dim"] == 121
    assert descriptor["anchor_keypoints"] == [11, 12]
    assert descriptor["scale_keypoints"] == [5, 6]
    assert descriptor["layout"][-1] == {"name": "dt_seconds", "start": 120, "stop": 121}
    code_dir = Path(module.__file__).parent
    source_hashes = {name: hashlib.sha256((code_dir / name).read_bytes()).hexdigest()
                     for name in ("pose_features.py", "schema.py")}
    payload = {"descriptor": descriptor, "source_sha256": source_hashes}
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                       allow_nan=False).encode("utf-8")).hexdigest()
    assert module.pose_encoder_hash(spec) == expected
    assert module.pose_encoder_hash(module.PoseFeatureSpec(confidence_threshold=.1)) != expected
    assert module.pose_encoder_hash(module.PoseFeatureSpec(scale_epsilon=.01)) != expected
    descriptor["anchor_keypoints"][0] = 0
    assert spec.descriptor()["anchor_keypoints"] == [11, 12]
    assert module.pose_encoder_hash(spec) == expected
