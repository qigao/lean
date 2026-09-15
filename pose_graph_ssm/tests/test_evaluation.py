from __future__ import annotations

import numpy as np
import pytest
import torch

from pose_graph_ssm.evaluation import (
    dropout_recovery_metrics,
    early_prediction_auc,
    length_quartile_boundaries,
    macro_f1,
    prefix_predictions,
    retention_metrics,
)


class _IntegratorModel:
    def initial_state(self, batch_size, device, dtype):
        return torch.zeros((batch_size, 1), device=device, dtype=dtype)

    def step(self, frame_t, state):
        return state + frame_t[:, 0, :1]

    def logits(self, state):
        return torch.cat((-state, state), dim=1)


class _CurrentFrameModel:
    def initial_state(self, batch_size, device, dtype):
        return torch.zeros((batch_size, 1), device=device, dtype=dtype)

    def step(self, frame_t, state):
        del state
        return frame_t[:, 0, :1]

    def logits(self, state):
        return torch.cat((-state, state), dim=1)


def _events(length=30, value=1.0):
    result = torch.zeros(length, 25, 15)
    result[:, 0, 0] = value
    return result


def test_macro_f1_keeps_absent_declared_classes_in_average():
    truth = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    pred = torch.tensor([0, 0, 0, 1], dtype=torch.int64)
    assert macro_f1(truth, pred, num_classes=3) == pytest.approx((0.8 + 2.0 / 3.0) / 3.0)


def test_early_auc_is_normalized_trapezoid():
    ratios = (0.10, 0.20, 0.40, 0.60, 0.80, 1.00)
    scores = (0.2, 0.3, 0.5, 0.7, 0.8, 0.9)
    expected = np.trapezoid(scores, ratios) / (ratios[-1] - ratios[0])
    assert early_prediction_auc(ratios, scores) == pytest.approx(expected)


def test_prefix_predictions_consume_only_declared_prefix():
    model = _IntegratorModel()
    events = _events(length=10)
    changed = events.clone()
    changed[4:] *= -1000.0
    first = prefix_predictions(model, events, ratios=(0.4,))
    second = prefix_predictions(model, changed, ratios=(0.4,))
    assert first[0].observed_steps == 4
    torch.testing.assert_close(first[0].logits, second[0].logits)


def test_zero_input_retention_measures_intrinsic_state_only():
    model = _IntegratorModel()
    metrics = retention_metrics(model, _events(30), true_label=1, observation_ratio=0.4, horizon=20)
    assert metrics.initial_margin == pytest.approx(24.0)
    assert metrics.retention_auc == pytest.approx(1.0)
    assert metrics.t50 is None


def test_pose_dropout_recovers_stably_on_first_restored_frame():
    model = _CurrentFrameModel()
    metrics = dropout_recovery_metrics(
        model,
        _events(30),
        observation_ratio=0.4,
        burst=8,
        recovery_horizon=10,
    )
    assert metrics.recovered is True
    assert metrics.recovery_step == 1
    assert metrics.recovery_rate == pytest.approx(1.0)
    assert metrics.dropout_steps == 8


def test_dropout_requires_enough_future_stream():
    with pytest.raises(ValueError, match="future"):
        dropout_recovery_metrics(
            _CurrentFrameModel(),
            _events(20),
            observation_ratio=0.4,
            burst=8,
            recovery_horizon=10,
        )


def test_length_quartiles_are_train_only_nearest_rank_boundaries():
    assert length_quartile_boundaries([10, 20, 30, 40, 50, 60, 70, 80]) == (20, 40, 60)


def test_length_quartiles_fail_when_boundaries_are_not_distinct():
    with pytest.raises(ValueError, match="distinct"):
        length_quartile_boundaries([10, 10, 10, 10, 10, 10, 10, 10])
