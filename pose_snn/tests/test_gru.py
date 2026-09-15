import torch

from pose_temporal_snn.models.gru import StreamingGRU


def test_gru_forward_equals_manual_streaming():
    torch.manual_seed(7)
    model = StreamingGRU(input_dim=225, hidden_size=56, num_classes=10)
    events = torch.randn(2, 10, 225)
    state = model.initial_state(2, events.device, events.dtype)
    for time_index in range(events.shape[1]):
        state = model.step(events[:, time_index], state)
    torch.testing.assert_close(model(events), model.logits(state))


def test_gru_has_no_future_frame_leakage():
    torch.manual_seed(11)
    model = StreamingGRU(input_dim=225, hidden_size=56, num_classes=10)
    events = torch.randn(1, 12, 225)
    changed = events.clone()
    changed[:, 5:] = torch.randn_like(changed[:, 5:]) * 1000.0
    state_a = model.initial_state(1, events.device, events.dtype)
    state_b = model.initial_state(1, events.device, events.dtype)
    for time_index in range(5):
        state_a = model.step(events[:, time_index], state_a)
        state_b = model.step(changed[:, time_index], state_b)
    torch.testing.assert_close(model.logits(state_a), model.logits(state_b))


def test_gru_silence_is_deterministic_exact_and_non_mutating():
    model = StreamingGRU(input_dim=225, hidden_size=56, num_classes=10)
    state = torch.ones(2, 56)
    first = model.silence_state(state, fraction=0.25, seed=19)
    second = model.silence_state(state, fraction=0.25, seed=19)
    torch.testing.assert_close(first, second)
    assert int((first[0] == 0).sum().item()) == 14
    assert torch.all(state == 1).item()


def test_gru_is_deterministic_for_fixed_seed_and_under_parameter_ceiling():
    torch.manual_seed(23)
    first = StreamingGRU(input_dim=225, hidden_size=56, num_classes=10)
    torch.manual_seed(23)
    second = StreamingGRU(input_dim=225, hidden_size=56, num_classes=10)
    events = torch.randn(1, 6, 225)
    torch.testing.assert_close(first(events), second(events))
    count = sum(parameter.numel() for parameter in first.parameters() if parameter.requires_grad)
    assert count <= 50_000


def test_gru_rejects_bad_stream_shape_and_fraction():
    model = StreamingGRU(input_dim=225, hidden_size=56, num_classes=10)
    state = model.initial_state(1, torch.device("cpu"), torch.float32)
    try:
        model.step(torch.zeros(1, 224), state)
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("bad GRU frame shape must fail")
    try:
        model.silence_state(state, fraction=0.0, seed=7)
    except ValueError as exc:
        assert "fraction" in str(exc)
    else:
        raise AssertionError("invalid silence fraction must fail")
