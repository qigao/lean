from __future__ import annotations

from dataclasses import dataclass
import math
import random
import statistics
from typing import Any

import torch

from .models.cx_lif import CxLifState


_INTEGER_DTYPES = frozenset({torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64})


@dataclass(frozen=True)
class RetentionMetrics:
    initial_margin: float
    retention_auc: float
    t50: float | None
    stable_duration: float


@dataclass(frozen=True)
class RecoveryMetrics:
    selected_node_indices: tuple[int, ...]
    eligible_samples: int
    recovered_samples: int
    recovery_rate: float
    median_recovery_steps: float | None


def macro_f1(y_true: torch.Tensor, y_pred: torch.Tensor, *, num_classes: int) -> float:
    if type(num_classes) is not int or num_classes <= 0:
        raise ValueError("num_classes must be a positive integer")
    if type(y_true) is not torch.Tensor or type(y_pred) is not torch.Tensor:
        raise ValueError("labels and predictions must be tensors")
    if y_true.ndim != 1 or y_pred.ndim != 1 or y_true.shape != y_pred.shape or y_true.numel() == 0:
        raise ValueError("labels and predictions must be non-empty equal-length vectors")
    if y_true.dtype not in _INTEGER_DTYPES or y_pred.dtype not in _INTEGER_DTYPES:
        raise ValueError("labels and predictions must use integer dtypes")
    if (
        torch.any(y_true < 0).item()
        or torch.any(y_true >= num_classes).item()
        or torch.any(y_pred < 0).item()
        or torch.any(y_pred >= num_classes).item()
    ):
        raise ValueError("labels and predictions must lie in the declared class range")

    values: list[float] = []
    for label in range(num_classes):
        truth = y_true == label
        predicted = y_pred == label
        true_positive = int((truth & predicted).sum().item())
        false_positive = int((~truth & predicted).sum().item())
        false_negative = int((truth & ~predicted).sum().item())
        denominator = 2 * true_positive + false_positive + false_negative
        values.append(0.0 if denominator == 0 else (2.0 * true_positive) / denominator)
    return sum(values) / num_classes


def early_prediction_auc(ratios: tuple[float, ...], macro_f1_values: tuple[float, ...]) -> float:
    if len(ratios) != len(macro_f1_values) or len(ratios) < 2:
        raise ValueError("early prediction ratios and values must have equal length >= 2")
    xs = tuple(float(value) for value in ratios)
    ys = tuple(float(value) for value in macro_f1_values)
    if any(not math.isfinite(value) for value in xs + ys):
        raise ValueError("early prediction inputs must be finite")
    if any(not 0.0 < value <= 1.0 for value in xs):
        raise ValueError("observation ratios must lie in (0,1]")
    if any(right <= left for left, right in zip(xs, xs[1:])):
        raise ValueError("observation ratios must be strictly increasing")
    if any(not 0.0 <= value <= 1.0 for value in ys):
        raise ValueError("macro F1 values must lie in [0,1]")

    area = sum(
        (right_x - left_x) * (left_y + right_y) * 0.5
        for left_x, right_x, left_y, right_y in zip(xs, xs[1:], ys, ys[1:])
    )
    span = xs[-1] - xs[0]
    if span <= 0.0:
        raise ValueError("observation-ratio span must be positive")
    return area / span


def _validate_state(state: CxLifState) -> None:
    if type(state) is not CxLifState:
        raise ValueError("CX temporal evaluation requires CxLifState")
    shape = state.membrane.shape
    if state.membrane.ndim != 2 or any(tensor.shape != shape for tensor in state):
        raise ValueError("CX temporal state tensors must share [batch,node] shape")
    if not (
        torch.isfinite(state.membrane).all().item()
        and torch.isfinite(state.synaptic).all().item()
        and torch.isfinite(state.spikes).all().item()
    ):
        raise ValueError("CX temporal state must be finite")


def state_logits(model: Any, state: CxLifState) -> torch.Tensor:
    _validate_state(state)
    output_indices = getattr(model, "output_indices", None)
    readout = getattr(model, "readout", None)
    if type(output_indices) is not torch.Tensor or output_indices.ndim != 1 or output_indices.numel() == 0:
        raise ValueError("CX model must expose non-empty output_indices")
    if not isinstance(readout, torch.nn.Module):
        raise ValueError("CX model must expose a readout module")
    if output_indices.dtype not in _INTEGER_DTYPES:
        raise ValueError("CX output indices must use an integer dtype")
    if torch.any(output_indices < 0).item() or torch.any(output_indices >= state.membrane.shape[1]).item():
        raise ValueError("CX output index is outside state node range")

    output_state = state.membrane.index_select(1, output_indices)
    output_state = output_state + state.spikes.index_select(1, output_indices)
    logits = readout(output_state)
    if logits.ndim != 2 or logits.shape[0] != state.membrane.shape[0] or logits.shape[1] < 2:
        raise ValueError("CX readout must return [batch,num_classes>=2] logits")
    if not torch.isfinite(logits).all().item():
        raise ValueError("CX state logits must be finite")
    return logits


def _validate_events(events: torch.Tensor) -> None:
    if type(events) is not torch.Tensor or events.ndim != 3:
        raise ValueError("CX temporal events must have shape [batch,time,feature]")
    if events.shape[0] <= 0 or events.shape[1] <= 0 or events.shape[2] <= 0:
        raise ValueError("CX temporal event dimensions must be positive")
    if not torch.is_floating_point(events) or not torch.isfinite(events).all().item():
        raise ValueError("CX temporal events must be finite floating point")


def _initial_state(model: Any, events: torch.Tensor) -> CxLifState:
    initial_state = getattr(model, "initial_state", None)
    if not callable(initial_state):
        raise ValueError("CX model must expose initial_state")
    state = initial_state(batch_size=events.shape[0], device=events.device, dtype=events.dtype)
    _validate_state(state)
    return state


def _advance(model: Any, events: torch.Tensor, state: CxLifState) -> CxLifState:
    step = getattr(model, "step", None)
    if not callable(step):
        raise ValueError("CX model must expose step")
    result = state
    for time_index in range(events.shape[1]):
        result = step(events[:, time_index, :], result)
        _validate_state(result)
    return result


def _labels(value: int | torch.Tensor, *, batch_size: int, device: torch.device, num_classes: int) -> torch.Tensor:
    if type(value) is int:
        result = torch.full((batch_size,), value, dtype=torch.int64, device=device)
    elif type(value) is torch.Tensor:
        if value.ndim != 1 or value.shape[0] != batch_size or value.dtype not in _INTEGER_DTYPES:
            raise ValueError("true labels must be an integer vector matching batch size")
        result = value.to(device=device, dtype=torch.int64)
    else:
        raise ValueError("true_label must be an integer or integer tensor")
    if torch.any(result < 0).item() or torch.any(result >= num_classes).item():
        raise ValueError("true label lies outside model class range")
    return result


def _true_class_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    competitors = logits.clone()
    competitors.scatter_(1, labels[:, None], float("-inf"))
    competitor_logits = competitors.max(dim=1).values
    return true_logits - competitor_logits


def retention_metrics(
    model: Any,
    observed_events: torch.Tensor,
    *,
    true_label: int | torch.Tensor,
    horizon: int,
    epsilon: float = 1e-6,
) -> RetentionMetrics:
    _validate_events(observed_events)
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("retention horizon must be a positive integer")
    if type(epsilon) not in (int, float) or isinstance(epsilon, bool) or not math.isfinite(float(epsilon)) or epsilon <= 0:
        raise ValueError("retention epsilon must be positive and finite")

    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    try:
        with torch.no_grad():
            state = _advance(model, observed_events, _initial_state(model, observed_events))
            boundary_logits = state_logits(model, state)
            labels = _labels(
                true_label,
                batch_size=observed_events.shape[0],
                device=observed_events.device,
                num_classes=boundary_logits.shape[1],
            )
            initial_margin = _true_class_margin(boundary_logits, labels)
            denominator = torch.clamp(initial_margin.abs(), min=float(epsilon))
            normalized: list[torch.Tensor] = [initial_margin / denominator]
            boundary_prediction = boundary_logits.argmax(dim=1)

            t50: list[int | None] = [0 if value <= 0.5 else None for value in normalized[0].tolist()]
            stable = [0 for _ in range(observed_events.shape[0])]
            still_stable = [True for _ in range(observed_events.shape[0])]
            zeros = observed_events.new_zeros((observed_events.shape[0], observed_events.shape[2]))

            for step_index in range(1, horizon + 1):
                state = model.step(zeros, state)
                _validate_state(state)
                logits = state_logits(model, state)
                margin = _true_class_margin(logits, labels)
                current_normalized = margin / denominator
                normalized.append(current_normalized)
                predictions = logits.argmax(dim=1)
                for sample_index, value in enumerate(current_normalized.tolist()):
                    if t50[sample_index] is None and value <= 0.5:
                        t50[sample_index] = step_index
                    if still_stable[sample_index] and int(predictions[sample_index].item()) == int(
                        boundary_prediction[sample_index].item()
                    ):
                        stable[sample_index] += 1
                    else:
                        still_stable[sample_index] = False

            values = torch.stack(normalized, dim=1)
            per_sample_area = 0.5 * (values[:, :-1] + values[:, 1:]).sum(dim=1) / float(horizon)
            observed_t50 = [float(value) for value in t50 if value is not None]
            return RetentionMetrics(
                initial_margin=float(initial_margin.mean().item()),
                retention_auc=float(per_sample_area.mean().item()),
                t50=None if not observed_t50 else sum(observed_t50) / len(observed_t50),
                stable_duration=sum(float(value) for value in stable) / len(stable),
            )
    finally:
        if was_training and hasattr(model, "train"):
            model.train()


def _clone_state(state: CxLifState) -> CxLifState:
    return CxLifState(*(tensor.clone() for tensor in state))


def _silence_state(state: CxLifState, selected: tuple[int, ...]) -> CxLifState:
    result = []
    for tensor in state:
        cloned = tensor.clone()
        cloned[:, list(selected)] = 0
        result.append(cloned)
    return CxLifState(*result)


def recovery_metrics(
    model: Any,
    events: torch.Tensor,
    *,
    perturb_at: int,
    silence_fraction: float,
    horizon: int,
    seed: int,
    candidate_node_indices: tuple[int, ...] | None = None,
) -> RecoveryMetrics:
    _validate_events(events)
    if type(perturb_at) is not int or perturb_at <= 0 or perturb_at >= events.shape[1]:
        raise ValueError("perturb_at must be inside the observed sequence")
    if type(horizon) is not int or horizon <= 0 or perturb_at + horizon > events.shape[1]:
        raise ValueError("recovery horizon must be positive and fit within the remaining events")
    if (
        type(silence_fraction) not in (int, float)
        or isinstance(silence_fraction, bool)
        or not math.isfinite(float(silence_fraction))
        or not 0.0 < float(silence_fraction) < 1.0
    ):
        raise ValueError("silence_fraction must be finite and inside (0,1)")
    if type(seed) is not int:
        raise ValueError("recovery seed must be an integer")

    num_nodes = int(getattr(model, "num_nodes", 0))
    if num_nodes <= 0:
        raise ValueError("CX model must expose a positive num_nodes")
    candidates = tuple(range(num_nodes)) if candidate_node_indices is None else candidate_node_indices
    if (
        not candidates
        or any(type(index) is not int or index < 0 or index >= num_nodes for index in candidates)
        or len(set(candidates)) != len(candidates)
    ):
        raise ValueError("candidate_node_indices must be unique valid node indices")
    selected_count = max(1, int(math.floor(len(candidates) * float(silence_fraction))))
    selected = tuple(sorted(random.Random(seed).sample(list(candidates), selected_count)))

    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    try:
        with torch.no_grad():
            prefix = events[:, :perturb_at, :]
            boundary = _advance(model, prefix, _initial_state(model, events))
            unperturbed = _clone_state(boundary)
            perturbed = _silence_state(boundary, selected)

            baseline_prediction = state_logits(model, unperturbed).argmax(dim=1)
            perturbed_prediction = state_logits(model, perturbed).argmax(dim=1)
            eligible = perturbed_prediction != baseline_prediction
            recovered_step: list[int | None] = [None] * events.shape[0]

            for step_index in range(1, horizon + 1):
                frame = events[:, perturb_at + step_index - 1, :]
                unperturbed = model.step(frame, unperturbed)
                perturbed = model.step(frame, perturbed)
                _validate_state(unperturbed)
                _validate_state(perturbed)
                reference = state_logits(model, unperturbed).argmax(dim=1)
                candidate = state_logits(model, perturbed).argmax(dim=1)
                for sample_index in range(events.shape[0]):
                    if (
                        bool(eligible[sample_index].item())
                        and recovered_step[sample_index] is None
                        and int(candidate[sample_index].item()) == int(reference[sample_index].item())
                    ):
                        recovered_step[sample_index] = step_index

            eligible_count = int(eligible.sum().item())
            recovered_values = [value for index, value in enumerate(recovered_step) if bool(eligible[index].item()) and value is not None]
            recovered_count = len(recovered_values)
            return RecoveryMetrics(
                selected_node_indices=selected,
                eligible_samples=eligible_count,
                recovered_samples=recovered_count,
                recovery_rate=0.0 if eligible_count == 0 else recovered_count / eligible_count,
                median_recovery_steps=None if not recovered_values else float(statistics.median(recovered_values)),
            )
    finally:
        if was_training and hasattr(model, "train"):
            model.train()
