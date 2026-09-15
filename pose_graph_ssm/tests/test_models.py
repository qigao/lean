from __future__ import annotations

import torch

from pose_graph_ssm.models.ssm_only import SSMOnly


def test_ssm_only_forward_matches_manual_streaming():
    torch.manual_seed(7)
    model = SSMOnly(num_classes=10, width=64, blocks=2)
    sequence = torch.randn(2, 12, 25, 15)
    state = model.initial_state(2, sequence.device, sequence.dtype)
    for index in range(sequence.shape[1]):
        state = model.step(sequence[:, index], state)
    torch.testing.assert_close(model(sequence), model.logits(state))


def test_ssm_only_prefix_does_not_depend_on_future_frames():
    torch.manual_seed(11)
    model = SSMOnly(num_classes=10, width=64, blocks=2)
    sequence = torch.randn(1, 10, 25, 15)
    changed = sequence.clone()
    changed[:, 4:] = changed[:, 4:] * -1000.0

    def prefix_logits(value):
        state = model.initial_state(1, value.device, value.dtype)
        for index in range(4):
            state = model.step(value[:, index], state)
        return model.logits(state)

    torch.testing.assert_close(prefix_logits(sequence), prefix_logits(changed))


def test_ssm_only_frame_entry_projection_is_bias_free():
    model = SSMOnly(num_classes=10, width=64, blocks=2)
    assert model.input_projection.in_features == 375
    assert model.input_projection.out_features == 64
    assert model.input_projection.bias is None


def test_ssm_only_stays_below_frozen_parameter_ceiling():
    model = SSMOnly(num_classes=10, width=64, blocks=2)
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    assert count <= 120_000


def test_ssm_only_rejects_nonfinite_input():
    model = SSMOnly(num_classes=10, width=64, blocks=2)
    sequence = torch.zeros(1, 3, 25, 15)
    sequence[0, 1, 4, 2] = float("nan")
    try:
        model(sequence)
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("nonfinite input must fail closed")
