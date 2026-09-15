import torch
import torch.nn.functional as F

from pose_temporal_snn.models.rsnn import RSNN, RSNNState


def test_forward_equals_manual_step_loop():
    torch.manual_seed(7)
    model = RSNN(input_dim=225, hidden_size=96, num_classes=10)
    events = torch.zeros(2, 12, 225, dtype=torch.float32)
    state = model.initial_state(2, events.device, events.dtype)
    for time_index in range(events.shape[1]):
        state = model.step(events[:, time_index], state)
    torch.testing.assert_close(model(events), model.logits(state))


def test_recurrent_weight_receives_finite_nonzero_surrogate_gradient():
    torch.manual_seed(11)
    model = RSNN(input_dim=225, hidden_size=96, num_classes=10)
    with torch.no_grad():
        model.input_projection.weight.fill_(0.01)
    events = torch.ones(2, 6, 225, dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.int64)
    loss = F.cross_entropy(model(events), targets)
    loss.backward()
    grad = model.recurrent.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all().item()
    assert torch.count_nonzero(grad).item() > 0


def test_zero_external_input_can_evolve_existing_recurrent_state():
    torch.manual_seed(19)
    model = RSNN(input_dim=225, hidden_size=96, num_classes=10)
    state = model.initial_state(1, torch.device("cpu"), torch.float32)
    spikes = torch.zeros_like(state.spikes)
    spikes[:, 0] = 1.0
    seeded = RSNNState(
        membrane=torch.zeros_like(state.membrane),
        synaptic=torch.zeros_like(state.synaptic),
        spikes=spikes,
    )
    next_state = model.step(torch.zeros(1, 225), seeded)
    assert torch.count_nonzero(next_state.synaptic).item() > 0


def test_silence_state_is_deterministic_exact_and_non_mutating():
    model = RSNN(input_dim=225, hidden_size=96, num_classes=10)
    ones = torch.ones(2, 96)
    state = RSNNState(ones.clone(), ones.clone(), ones.clone())
    first = model.silence_state(state, fraction=0.25, seed=23)
    second = model.silence_state(state, fraction=0.25, seed=23)
    torch.testing.assert_close(first.membrane, second.membrane)
    silenced = (first.membrane[0] == 0).nonzero().flatten()
    assert silenced.numel() == 24
    assert torch.all(state.membrane == 1)
    assert torch.all(first.synaptic[:, silenced] == 0)
    assert torch.all(first.spikes[:, silenced] == 0)


def test_rsnn_stays_under_frozen_parameter_ceiling():
    model = RSNN(input_dim=225, hidden_size=96, num_classes=10)
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    assert count <= 50_000
    assert model.input_projection.bias is None
    assert model.recurrent.bias is None


def test_rsnn_rejects_bad_stream_shapes_and_silence_fraction():
    model = RSNN(input_dim=225, hidden_size=96, num_classes=10)
    state = model.initial_state(1, torch.device("cpu"), torch.float32)
    try:
        model.step(torch.zeros(1, 224), state)
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("bad frame shape must fail")
    try:
        model.silence_state(state, fraction=1.0, seed=7)
    except ValueError as exc:
        assert "fraction" in str(exc)
    else:
        raise AssertionError("invalid silence fraction must fail")
