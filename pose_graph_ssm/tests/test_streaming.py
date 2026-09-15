from __future__ import annotations

import pytest
import torch

from pose_graph_ssm.models.graph_ssm import GraphSSM
from pose_graph_ssm.models.graph_tcn import CausalTemporalConvBlock, GraphTCN
from pose_graph_ssm.models.gru import StreamingGRU
from pose_graph_ssm.models.ssm_only import SSMOnly


def _models():
    return (
        StreamingGRU(num_classes=10, width=64),
        SSMOnly(num_classes=10, width=64, blocks=2),
        GraphTCN(num_classes=10, width=64),
        GraphSSM(num_classes=10, width=64),
    )


@pytest.mark.parametrize("index", range(4))
def test_forward_matches_manual_streaming(index):
    torch.manual_seed(7)
    model = _models()[index]
    sequence = torch.randn(2, 9, 25, 15)
    state = model.initial_state(2, sequence.device, sequence.dtype)
    for step in range(sequence.shape[1]):
        state = model.step(sequence[:, step], state)
    torch.testing.assert_close(model(sequence), model.logits(state))


@pytest.mark.parametrize("index", range(4))
def test_prefix_logits_do_not_depend_on_future_frames(index):
    torch.manual_seed(11)
    model = _models()[index]
    sequence = torch.randn(1, 10, 25, 15)
    changed = sequence.clone()
    changed[:, 4:] = changed[:, 4:] * -1000.0

    def prefix(value):
        state = model.initial_state(1, value.device, value.dtype)
        for step in range(4):
            state = model.step(value[:, step], state)
        return model.logits(state)

    torch.testing.assert_close(prefix(sequence), prefix(changed))


@pytest.mark.parametrize("index", range(4))
def test_all_model_arms_stay_below_parameter_ceiling(index):
    model = _models()[index]
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    assert count <= 120_000


def test_graph_models_use_bias_free_frame_entry_projection():
    graph_tcn = GraphTCN(num_classes=10, width=64)
    graph_ssm = GraphSSM(num_classes=10, width=64)
    assert graph_tcn.input_projection.bias is None
    assert graph_ssm.input_projection.bias is None


def test_graph_ssm_keeps_independent_joint_state_shape():
    model = GraphSSM(num_classes=10, width=64)
    state = model.initial_state(2, torch.device("cpu"), torch.float32)
    assert len(state.recurrent) == 2
    assert all(item.shape == (2, 25, 64) for item in state.recurrent)


def test_causal_temporal_conv_has_fixed_dilation_cache_and_zero_drive():
    block = CausalTemporalConvBlock(64, kernel_size=3, dilation=2)
    state = block.initial_state(2, torch.device("cpu"), torch.float32)
    assert state.shape == (2, 4, 25, 64)
    output, next_state = block.step(torch.zeros(2, 25, 64), state)
    torch.testing.assert_close(output, torch.zeros_like(output))
    torch.testing.assert_close(next_state, torch.zeros_like(next_state))
    assert block.current_projection.bias is None
    assert block.lag1_projection.bias is None
    assert block.lag2_projection.bias is None


@pytest.mark.parametrize("index", range(4))
def test_models_reject_nonfinite_sequence(index):
    model = _models()[index]
    sequence = torch.zeros(1, 3, 25, 15)
    sequence[0, 1, 4, 2] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(sequence)
