"""Variable-length tensor collation, not recognition or provenance evidence."""
from dataclasses import FrozenInstanceError, fields
import importlib
import inspect

import numpy as np
import pytest
import torch


def _module():
    return importlib.import_module("yolo_flywire.pose_batches")


def _features(count=3, offset=0.):
    """Independent analytic 121-channel fixture with nonuniform intervals."""
    result = np.zeros((count, 121), dtype=np.float64)
    result[:, :34] = np.arange(34)[None, :] / 10. + offset
    result[:, 68:85] = .8
    result[:, 85:102] = 1.
    result[:, 119] = .9
    if count > 1:
        result[1:, 120] = np.arange(1, count) / 40.
        result[:, 18] += np.arange(count) / 2.
        result[1:, 52] = .5 / result[1:, 120]
        result[1:, 102:119] = 1.
    return result


def _missing_frame(sequence, index):
    sequence[index, :120] = 0.
    sequence[index, 34:68] = 0.
    if index + 1 < len(sequence):
        sequence[index + 1, 34:68] = 0.
        sequence[index + 1, 102:119] = 0.


def test_ragged_batch_preserves_order_values_and_exact_lengths():
    sequences = (_features(2, 5.), _features(4, -1.), _features(1, 9.))
    batch = _module().collate_pose_features(sequences)
    assert batch.features.shape == (3, 4, 121)
    assert batch.features.dtype == torch.float32
    assert batch.lengths.dtype == torch.int64
    assert batch.time_mask.dtype == torch.bool
    assert batch.lengths.tolist() == [2, 4, 1]  # Not length-sorted.
    assert batch.time_mask.tolist() == [
        [True, True, False, False], [True] * 4, [True, False, False, False],
    ]
    for index, sequence in enumerate(sequences):
        np.testing.assert_array_equal(batch.features[index, :len(sequence)].numpy(),
                                      sequence.astype(np.float32))
    assert torch.count_nonzero(batch.features[~batch.time_mask]).item() == 0
    for tensor in (batch.features, batch.lengths, batch.time_mask):
        assert tensor.device.type == "cpu" and tensor.is_contiguous()
        assert not tensor.requires_grad


def test_observed_all_missing_frames_are_not_padding_even_when_both_are_zero():
    single = np.zeros((1, 121), dtype=np.float64)
    sequence = _features(4)
    _missing_frame(sequence, 1)
    batch = _module().collate_pose_features((single, sequence))
    assert torch.equal(batch.features[0, 0], batch.features[0, 1])
    assert batch.time_mask[0].tolist() == [True, False, False, False]
    assert batch.time_mask[1].all().item()
    assert not batch.features[1, 1, :120].any().item()
    assert batch.features[1, 1, 120].item() == np.float32(1 / 40)
    assert not batch.features[1, 2, 34:68].any().item()
    assert not batch.features[1, 2, 102:119].any().item()
    assert batch.features[1, 3, 102:119].all().item()


def test_masks_are_not_rederived_from_rounded_confidences():
    sequence = _features(3)
    # Two distinct float64 confidences round to the same float32 value.
    sequence[0, 77] = .05
    sequence[1, 77] = np.nextafter(.05, 0.)
    sequence[1, 18:20] = 0.
    sequence[1, 94] = 0.
    sequence[1:, 52:54] = 0.
    sequence[1:, 111] = 0.
    batch = _module().collate_pose_features((sequence,))
    assert batch.features[0, 0, 77] == batch.features[0, 1, 77]
    assert batch.features[0, :, 94].tolist() == [1., 0., 1.]
    assert batch.features[0, :, 111].tolist() == [0., 0., 0.]


def test_collation_is_repeatable_copies_storage_and_never_mutates_inputs():
    sequence = _features()
    before = sequence.copy()
    module = _module()
    first = module.collate_pose_features((sequence,))
    second = module.collate_pose_features((sequence,))
    for name in ("features", "lengths", "time_mask"):
        assert torch.equal(getattr(first, name), getattr(second, name))
        assert getattr(first, name).data_ptr() != getattr(second, name).data_ptr()
    first.features.fill_(7)
    first.lengths.fill_(1)
    first.time_mask.fill_(False)
    np.testing.assert_array_equal(sequence, before)
    sequence.fill_(0)
    np.testing.assert_array_equal(second.features[0].numpy(), before.astype(np.float32))
    assert second.lengths.tolist() == [3] and second.time_mask.all().item()
    with pytest.raises(FrozenInstanceError):
        second.features = first.features


def test_read_only_strided_and_fortran_arrays_are_accepted_without_mutation():
    reference = _features(3)
    storage = np.zeros((3, 242), dtype=np.float64)
    storage[:, ::2] = reference
    strided = storage[:, ::2]
    strided.setflags(write=False)
    fortran = np.asfortranarray(reference)
    batch = _module().collate_pose_features((strided, fortran))
    for index in (0, 1):
        np.testing.assert_array_equal(batch.features[index].numpy(), reference.astype(np.float32))
    np.testing.assert_array_equal(strided, reference)
    np.testing.assert_array_equal(fortran, reference)


def test_explicit_cpu_float32_is_independent_of_torch_defaults():
    module = _module()
    original_dtype = torch.get_default_dtype()
    original_device = torch.get_default_device()
    try:
        torch.set_default_dtype(torch.float64)
        torch.set_default_device("meta")
        batch = module.collate_pose_features((_features(),))
    finally:
        torch.set_default_device(original_device)
        torch.set_default_dtype(original_dtype)
    assert batch.features.device.type == "cpu"
    assert batch.lengths.device.type == "cpu" and batch.time_mask.device.type == "cpu"
    assert batch.features.dtype == torch.float32
    assert batch.lengths.tolist() == [3]


def test_observation_only_api_has_no_labels_ids_paths_split_or_fallback_options():
    module = _module()
    assert tuple(inspect.signature(module.collate_pose_features).parameters) == ("sequences",)
    assert [field.name for field in fields(module.PoseBatch)] == ["features", "lengths", "time_mask"]


@pytest.mark.parametrize("sequences", [None, (), [], [_features()], np.zeros((1, 3, 121))])
def test_explicit_nonempty_tuple_of_sequences_is_required(sequences):
    with pytest.raises(ValueError):
        _module().collate_pose_features(sequences)


@pytest.mark.parametrize("sequence", [
    None, [[0.] * 121], np.zeros(121), np.zeros((1, 1, 121)),
    np.zeros((0, 121)), np.zeros((2, 102)), np.zeros((2, 120)), np.zeros((2, 122)),
    _features().astype(np.float32), _features().astype(np.float16),
    _features().astype(np.int64), _features().astype(object),
])
def test_wrong_schema_shape_or_source_dtype_is_rejected(sequence):
    with pytest.raises(ValueError):
        _module().collate_pose_features((sequence,))


@pytest.mark.parametrize("row,column,value", [
    (0, 0, float("nan")), (1, 20, float("inf")), (1, 60, -float("inf")),
    (1, 68, -.01), (1, 84, 1.01), (1, 119, -.1), (1, 119, 1.1),
    (1, 85, .5), (1, 102, -1.), (1, 118, 2.),
    (0, 120, .01), (1, 120, 0.), (1, 120, -.1),
    (0, 102, 1.), (1, 102, 0.),  # Adjacent masks must match, not be rederived.
    (0, 85, 0.),  # A masked position cannot keep nonzero coordinates.
    (0, 34, .5),  # A masked first velocity cannot keep a value.
    (1, 119, 0.),  # Missing person cannot retain geometry/confidence.
    (1, 68, 0.),  # A valid position cannot have zero joint confidence.
])
def test_invalid_feature_contract_is_rejected_before_padding(row, column, value):
    sequence = _features()
    sequence[row, column] = value
    with pytest.raises(ValueError):
        _module().collate_pose_features((sequence,))


@pytest.mark.parametrize("column,value", [
    (0, 1e39), (18, -1e39), (52, 1e39),
    (0, 1e-50), (68, 1e-50), (119, 1e-50), (120, 1e-50),
])
def test_float32_overflow_and_nonzero_underflow_are_not_clipped_or_erased(column, value):
    sequence = _features()
    sequence[1, column] = value
    with pytest.raises(ValueError):
        _module().collate_pose_features((sequence,))


def test_representable_float32_extremes_remain_finite_and_nonzero():
    sequence = _features()
    sequence[1, 0] = float(np.finfo(np.float32).max)
    sequence[1, 18] = float(np.nextafter(np.float32(0), np.float32(1)))
    batch = _module().collate_pose_features((sequence,))
    assert torch.isfinite(batch.features).all().item()
    assert batch.features[0, 1, 0].item() == np.finfo(np.float32).max
    assert batch.features[0, 1, 18].item() > 0


def test_bad_later_sequence_is_not_silently_dropped():
    bad = _features(2)
    bad[1, 120] = 0.
    with pytest.raises(ValueError):
        _module().collate_pose_features((_features(4), bad))
