import numpy as np
import pytest
import torch

from pose_temporal_snn.evaluation import (
    activity_metrics,
    early_prediction_auc,
    macro_f1,
    prefix_predictions,
    recovery_metrics,
    retention_metrics,
)
from pose_temporal_snn.models.gru import StreamingGRU
from pose_temporal_snn.models.rsnn import RSNN


class _IntegratorModel:
    def initial_state(self, batch_size, device, dtype):
        return torch.zeros((batch_size, 1), device=device, dtype=dtype)

    def step(self, events_t, state):
        return state + events_t[:, :1]

    def logits(self, state):
        return torch.cat((-state, state), dim=1)

    def silence_state(self, state, *, fraction, seed):
        del fraction, seed
        return torch.zeros_like(state)


def test_macro_f1_keeps_absent_declared_classes_in_average():
    truth = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    pred = torch.tensor([0, 0, 0, 1], dtype=torch.int64)
    # class0 F1=0.8, class1 F1=2/3, class2 F1=0.
    assert macro_f1(truth, pred, num_classes=3) == pytest.approx((0.8 + 2.0 / 3.0) / 3.0)


def test_early_auc_is_normalized_trapezoid():
    ratios = (0.10, 0.20, 0.40, 0.60, 0.80, 1.00)
    scores = (0.2, 0.3, 0.5, 0.7, 0.8, 0.9)
    expected = np.trapezoid(scores, ratios) / (ratios[-1] - ratios[0])
    assert early_prediction_auc(ratios, scores) == pytest.approx(expected)


def test_prefix_predictions_consume_only_declared_prefix():
    model = _IntegratorModel()
    events = torch.ones(10, 1)
    changed = events.clone()
    changed[4:] = -1000.0
    first = prefix_predictions(model, events, ratios=(0.4,))
    second = prefix_predictions(model, changed, ratios=(0.4,))
    torch.testing.assert_close(first[0].logits, second[0].logits)
    assert first[0].observed_steps == 4


def test_zero_input_retention_measures_intrinsic_state_only():
    model = _IntegratorModel()
    events = torch.ones(10, 1)
    metrics = retention_metrics(model, events, true_label=1, observation_ratio=0.4, horizon=5)
    assert metrics.initial_margin == pytest.approx(8.0)
    assert metrics.retention_auc == pytest.approx(1.0)
    assert metrics.t50 is None


def test_recovery_replays_same_future_stream_and_recovers_after_silencing():
    model = _IntegratorModel()
    events = torch.ones(10, 1)
    metrics = recovery_metrics(
        model,
        events,
        observation_ratio=0.4,
        fraction=0.25,
        horizon=3,
        seed=7,
    )
    assert metrics.eligible is True
    assert metrics.recovered is True
    assert metrics.recovery_step == 1


def test_activity_metrics_report_spikes_for_rsnn_but_not_gru():
    rsnn = RSNN(input_dim=225, hidden_size=96, num_classes=10)
    rsnn_metrics = activity_metrics(rsnn, torch.zeros(5, 225))
    assert rsnn_metrics.total_steps == 5
    assert rsnn_metrics.total_spikes == 0
    assert rsnn_metrics.active_unit_fraction == 0.0
    assert rsnn_metrics.synaptic_events_per_frame == 0.0

    gru = StreamingGRU(input_dim=225, hidden_size=56, num_classes=10)
    gru_metrics = activity_metrics(gru, torch.zeros(5, 225))
    assert gru_metrics.total_steps == 5
    assert gru_metrics.total_spikes is None
    assert gru_metrics.active_unit_fraction is None
    assert gru_metrics.synaptic_events_per_frame is None


def test_temporal_metrics_fail_closed_on_short_or_invalid_inputs():
    model = _IntegratorModel()
    with pytest.raises(ValueError, match="ratio"):
        retention_metrics(model, torch.ones(5, 1), true_label=1, observation_ratio=0.0, horizon=2)
    with pytest.raises(ValueError, match="future"):
        recovery_metrics(model, torch.ones(5, 1), observation_ratio=0.8, fraction=0.25, horizon=3, seed=7)
