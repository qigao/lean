from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import pytest
import torch
from torch import nn

from yolo_flywire.cx_evaluation import (
    early_prediction_auc,
    macro_f1,
    recovery_metrics,
    retention_metrics,
    state_logits,
)
from yolo_flywire.models.cx_lif import CxLifState


class _ScriptedCxModel(nn.Module):
    """Small deterministic state machine implementing the Gate 1 evaluation seam."""

    def __init__(self, *, competitor_bias: float = 0.0) -> None:
        super().__init__()
        self.input_dim = 1
        self.num_nodes = 2
        self.register_buffer("output_indices", torch.tensor([1], dtype=torch.int64), persistent=False)
        self.readout = nn.Linear(1, 2, bias=True)
        with torch.no_grad():
            self.readout.weight.copy_(torch.tensor([[1.0], [0.0]]))
            self.readout.bias.copy_(torch.tensor([0.0, competitor_bias]))

    def initial_state(self, *, batch_size: int, device: torch.device, dtype: torch.dtype) -> CxLifState:
        shape = (batch_size, self.num_nodes)
        return CxLifState(
            membrane=torch.zeros(shape, device=device, dtype=dtype),
            synaptic=torch.zeros(shape, device=device, dtype=dtype),
            refractory=torch.zeros(shape, device=device, dtype=torch.int64),
            spikes=torch.zeros(shape, device=device, dtype=dtype),
        )

    def step(self, events_t: torch.Tensor, state: CxLifState) -> CxLifState:
        drive = events_t[:, 0]
        membrane = state.membrane.clone()
        membrane[:, 0] = drive
        membrane[:, 1] = 0.8 * state.membrane[:, 1] + drive
        return CxLifState(
            membrane=membrane,
            synaptic=torch.zeros_like(state.synaptic),
            refractory=torch.zeros_like(state.refractory),
            spikes=torch.zeros_like(state.spikes),
        )


def test_macro_f1_includes_every_declared_class_even_if_never_predicted_or_observed():
    y_true = torch.tensor([0, 0, 1, 1])
    y_pred = torch.tensor([0, 0, 0, 0])
    # class 0 F1 = 2/3, classes 1 and 2 = 0; macro across the frozen 3-class task.
    assert macro_f1(y_true, y_pred, num_classes=3) == pytest.approx(2.0 / 9.0)


def test_early_auc_uses_trapezoids_and_normalizes_ratio_span():
    ratios = (0.10, 0.20, 0.40, 0.60, 0.80, 1.00)
    values = (0.2, 0.3, 0.5, 0.7, 0.8, 0.9)
    expected = np.trapezoid(values, ratios) / (ratios[-1] - ratios[0])
    assert early_prediction_auc(ratios, values) == pytest.approx(float(expected))


def test_early_auc_rejects_reordered_or_out_of_range_inputs():
    with pytest.raises(ValueError, match="strictly increasing"):
        early_prediction_auc((0.2, 0.1), (0.3, 0.4))
    with pytest.raises(ValueError, match="\[0,1\]"):
        early_prediction_auc((0.1, 1.0), (0.3, 1.4))


def test_state_logits_read_only_declared_output_population():
    model = _ScriptedCxModel()
    state = model.initial_state(batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
    state = state._replace(membrane=torch.tensor([[100.0, 2.0]]))
    logits = state_logits(model, state)
    torch.testing.assert_close(logits, torch.tensor([[2.0, 0.0]]))


def test_retention_evolves_zero_input_from_observed_boundary():
    model = _ScriptedCxModel()
    observed = torch.tensor([[[1.0]]])
    metrics = retention_metrics(model, observed, true_label=0, horizon=4, epsilon=1e-6)
    # Boundary output is 1.0, then zero-input dynamics are 0.8, .64, .512, .4096.
    expected = np.trapezoid([1.0, 0.8, 0.64, 0.512, 0.4096], dx=1.0) / 4.0
    assert metrics.initial_margin == pytest.approx(1.0)
    assert metrics.retention_auc == pytest.approx(float(expected))
    assert metrics.t50 == 4
    assert metrics.stable_duration == 4


def test_retention_does_not_mutate_observed_events():
    model = _ScriptedCxModel()
    observed = torch.tensor([[[1.0]], [[2.0]]])
    before = observed.clone()
    retention_metrics(model, observed, true_label=torch.tensor([0, 0]), horizon=2, epsilon=1e-6)
    torch.testing.assert_close(observed, before)


def test_recovery_silences_frozen_nodes_and_replays_same_future_events():
    model = _ScriptedCxModel(competitor_bias=0.25)
    # At perturb_at=1 the output is 1.0 (class 0). Silencing output node 1 makes
    # class 1 win immediately; the next shared event drives both trajectories
    # back to class 0, so recovery is exactly one step.
    events = torch.tensor([[[1.0], [1.0], [0.0]]])
    metrics = recovery_metrics(
        model,
        events,
        perturb_at=1,
        silence_fraction=0.5,
        horizon=2,
        seed=7,
        candidate_node_indices=(1,),
    )
    assert metrics.selected_node_indices == (1,)
    assert metrics.eligible_samples == 1
    assert metrics.recovered_samples == 1
    assert metrics.recovery_rate == pytest.approx(1.0)
    assert metrics.median_recovery_steps == pytest.approx(1.0)


def test_recovery_selection_is_deterministic():
    model = _ScriptedCxModel(competitor_bias=0.25)
    events = torch.ones(1, 3, 1)
    kwargs = dict(
        perturb_at=1,
        silence_fraction=0.5,
        horizon=2,
        seed=23,
        candidate_node_indices=(0, 1),
    )
    first = recovery_metrics(model, events, **kwargs)
    second = recovery_metrics(model, events, **kwargs)
    assert first.selected_node_indices == second.selected_node_indices
    assert first == second


@pytest.mark.parametrize(
    "labels,predicted,num_classes",
    [
        (torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64), 2),
        (torch.tensor([0]), torch.tensor([0, 1]), 2),
        (torch.tensor([2]), torch.tensor([0]), 2),
    ],
)
def test_macro_f1_rejects_invalid_label_contract(labels, predicted, num_classes):
    with pytest.raises(ValueError):
        macro_f1(labels, predicted, num_classes=num_classes)


def test_retention_rejects_nonpositive_horizon_and_nonfinite_events():
    model = _ScriptedCxModel()
    with pytest.raises(ValueError, match="horizon"):
        retention_metrics(model, torch.ones(1, 1, 1), true_label=0, horizon=0)
    bad = torch.tensor([[[float("nan")]]])
    with pytest.raises(ValueError, match="finite"):
        retention_metrics(model, bad, true_label=0, horizon=2)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.1, float("nan")])
def test_recovery_rejects_invalid_silence_fraction(fraction):
    model = _ScriptedCxModel()
    with pytest.raises(ValueError, match="silence_fraction"):
        recovery_metrics(
            model,
            torch.ones(1, 3, 1),
            perturb_at=1,
            silence_fraction=fraction,
            horizon=1,
            seed=7,
            candidate_node_indices=(1,),
        )
