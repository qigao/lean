from __future__ import annotations

import torch

from pose_graph_ssm.models.selective_ssm import SelectiveSSMBlock, ssm_spec_fingerprint


def test_zero_input_from_zero_state_has_no_external_drive():
    block = SelectiveSSMBlock(64)
    state = block.initial_state((2,), torch.device("cpu"), torch.float32)
    output, next_state = block.step(torch.zeros(2, 64), state)
    torch.testing.assert_close(output, torch.zeros_like(output))
    torch.testing.assert_close(next_state, torch.zeros_like(next_state))
    assert block.B_projection.bias is None
    assert block.C_projection.bias is None
    assert block.output_projection.bias is None
    assert block.delta_projection.bias is not None


def test_ssm_state_remains_finite_over_long_stream():
    torch.manual_seed(7)
    block = SelectiveSSMBlock(64)
    state = block.initial_state((3,), torch.device("cpu"), torch.float32)
    for _ in range(1000):
        frame = torch.randn(3, 64) * 3.0
        output, state = block.step(frame, state)
        assert torch.isfinite(output).all()
        assert torch.isfinite(state).all()


def test_transition_decay_is_bounded_between_exp_minus20_and_one():
    block = SelectiveSSMBlock(64)
    frame = torch.randn(4, 64) * 1000.0
    decay = block.decay(frame)
    assert torch.all(decay <= 1.0)
    assert torch.all(decay >= torch.exp(torch.tensor(-20.0)))


def test_block_supports_independent_joint_states_with_shared_parameters():
    block = SelectiveSSMBlock(64)
    state = block.initial_state((2, 25), torch.device("cpu"), torch.float32)
    frame = torch.randn(2, 25, 64)
    output, next_state = block.step(frame, state)
    assert output.shape == (2, 25, 64)
    assert next_state.shape == (2, 25, 64)


def test_ssm_spec_fingerprint_is_stable_sha256():
    assert ssm_spec_fingerprint() == ssm_spec_fingerprint()
    assert len(ssm_spec_fingerprint()) == 64
