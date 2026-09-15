from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Any, Iterator, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class PrefixPrediction:
    ratio: float
    observed_steps: int
    logits: torch.Tensor


@dataclass(frozen=True)
class RetentionMetrics:
    initial_margin: float
    retention_auc: float
    t50: int | None
    margins: tuple[float, ...]


@dataclass(frozen=True)
class RecoveryMetrics:
    eligible: bool
    recovered: bool
    recovery_step: int | None
    matched_steps: int
    horizon: int


@dataclass(frozen=True)
class ActivityMetrics:
    total_steps: int
    total_spikes: int | None
    active_unit_fraction: float | None
    synaptic_events_per_frame: float | None


@contextmanager
def _evaluation_mode(model: Any) -> Iterator[None]:
    training = getattr(model, "training", None)
    if hasattr(model, "eval"):
        model.eval()
    try:
        yield
    finally:
        if training is True and hasattr(model, "train"):
            model.train(True)


def _validate_events(events: torch.Tensor) -> torch.Tensor:
    if type(events) is not torch.Tensor or events.ndim < 2:
        raise ValueError("events must have shape [time,...features]")
    if events.shape[0] <= 0:
        raise ValueError("events must contain at least one time step")
    if not torch.is_floating_point(events) or not torch.isfinite(events).all().item():
        raise ValueError("events must be finite floating point")
    return events


def _validate_ratio(value: float) -> float:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise ValueError("observation ratio must be numeric")
    ratio = float(value)
    if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise ValueError("observation ratio must be in (0,1]")
    return ratio


def _validate_horizon(value: int, *, name: str = "horizon") -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _prefix_steps(total_steps: int, ratio: float) -> int:
    return min(total_steps, max(1, int(math.ceil(total_steps * ratio))))


def _initial_state(model: Any, events: torch.Tensor):
    return model.initial_state(1, events.device, events.dtype)


def _step_frame(model: Any, frame: torch.Tensor, state: Any):
    return model.step(frame.unsqueeze(0), state)


def _run_prefix(model: Any, events: torch.Tensor, steps: int):
    state = _initial_state(model, events)
    for index in range(steps):
        state = _step_frame(model, events[index], state)
    return state


def _logits_1d(model: Any, state: Any) -> torch.Tensor:
    logits = model.logits(state)
    if type(logits) is not torch.Tensor or logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError("streaming model logits must have shape [1,classes]")
    if logits.shape[1] <= 1 or not torch.is_floating_point(logits) or not torch.isfinite(logits).all().item():
        raise ValueError("streaming model logits must be finite multi-class scores")
    return logits[0]


def _true_class_margin(logits: torch.Tensor, true_label: int) -> float:
    if type(true_label) is not int or not 0 <= true_label < logits.numel():
        raise ValueError("true_label is outside the declared class range")
    true_value = logits[true_label]
    mask = torch.ones(logits.numel(), dtype=torch.bool, device=logits.device)
    mask[true_label] = False
    competitor = logits[mask].max()
    return float((true_value - competitor).detach().cpu().item())


def macro_f1(y_true: torch.Tensor, y_pred: torch.Tensor, *, num_classes: int) -> float:
    if type(num_classes) is not int or num_classes <= 1:
        raise ValueError("num_classes must be at least 2")
    truth = torch.as_tensor(y_true)
    pred = torch.as_tensor(y_pred)
    if truth.ndim != 1 or pred.ndim != 1 or truth.shape != pred.shape or truth.numel() == 0:
        raise ValueError("macro F1 labels must be non-empty matching 1D tensors")
    if truth.dtype == torch.bool or pred.dtype == torch.bool:
        raise ValueError("macro F1 labels must be integer class IDs")
    if torch.is_floating_point(truth) or torch.is_floating_point(pred):
        raise ValueError("macro F1 labels must be integer class IDs")
    truth = truth.to(dtype=torch.int64)
    pred = pred.to(dtype=torch.int64)
    if torch.any((truth < 0) | (truth >= num_classes)).item() or torch.any((pred < 0) | (pred >= num_classes)).item():
        raise ValueError("macro F1 label outside declared class range")

    scores: list[float] = []
    for label in range(num_classes):
        tp = int(((truth == label) & (pred == label)).sum().item())
        fp = int(((truth != label) & (pred == label)).sum().item())
        fn = int(((truth == label) & (pred != label)).sum().item())
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else (2.0 * tp) / denominator)
    return float(sum(scores) / num_classes)


def early_prediction_auc(ratios: Sequence[float], scores: Sequence[float]) -> float:
    if len(ratios) != len(scores) or len(ratios) < 2:
        raise ValueError("early prediction AUC requires matching ratio/score sequences of length >=2")
    parsed_ratios = tuple(_validate_ratio(value) for value in ratios)
    if any(right <= left for left, right in zip(parsed_ratios, parsed_ratios[1:])):
        raise ValueError("early prediction ratios must be strictly increasing")
    parsed_scores: list[float] = []
    for value in scores:
        if type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError("early prediction scores must be finite numeric values")
        parsed_scores.append(float(value))
    area = 0.0
    for index in range(len(parsed_ratios) - 1):
        width = parsed_ratios[index + 1] - parsed_ratios[index]
        area += width * (parsed_scores[index] + parsed_scores[index + 1]) * 0.5
    span = parsed_ratios[-1] - parsed_ratios[0]
    if span <= 0.0:
        raise ValueError("early prediction ratio span must be positive")
    return area / span


def prefix_predictions(
    model: Any,
    events: torch.Tensor,
    *,
    ratios: Sequence[float],
) -> tuple[PrefixPrediction, ...]:
    stream = _validate_events(events)
    parsed = tuple(_validate_ratio(value) for value in ratios)
    if not parsed:
        raise ValueError("at least one observation ratio is required")
    if any(right <= left for left, right in zip(parsed, parsed[1:])):
        raise ValueError("observation ratios must be strictly increasing")

    outputs: list[PrefixPrediction] = []
    with _evaluation_mode(model), torch.no_grad():
        for ratio in parsed:
            steps = _prefix_steps(stream.shape[0], ratio)
            state = _run_prefix(model, stream, steps)
            logits = _logits_1d(model, state).detach().clone()
            outputs.append(PrefixPrediction(ratio=ratio, observed_steps=steps, logits=logits))
    return tuple(outputs)


def retention_metrics(
    model: Any,
    events: torch.Tensor,
    *,
    true_label: int,
    observation_ratio: float,
    horizon: int,
    epsilon: float = 1e-6,
) -> RetentionMetrics:
    stream = _validate_events(events)
    ratio = _validate_ratio(observation_ratio)
    steps = _prefix_steps(stream.shape[0], ratio)
    horizon = _validate_horizon(horizon)
    if type(epsilon) not in (int, float) or isinstance(epsilon, bool) or not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise ValueError("retention epsilon must be positive and finite")

    with _evaluation_mode(model), torch.no_grad():
        state = _run_prefix(model, stream, steps)
        initial_margin = _true_class_margin(_logits_1d(model, state), true_label)
        margins = [initial_margin]
        zero_frame = torch.zeros_like(stream[0])
        for _ in range(horizon):
            state = _step_frame(model, zero_frame, state)
            margins.append(_true_class_margin(_logits_1d(model, state), true_label))

    scale = max(abs(initial_margin), float(epsilon))
    normalized = [margin / scale for margin in margins]
    area = 0.0
    for left, right in zip(normalized, normalized[1:]):
        area += 0.5 * (left + right)
    retention_auc = area / horizon

    t50: int | None = None
    if initial_margin > 0.0:
        threshold = 0.5 * initial_margin
        for index, margin in enumerate(margins[1:], start=1):
            if margin <= threshold:
                t50 = index
                break
    elif initial_margin == 0.0:
        t50 = 0

    return RetentionMetrics(
        initial_margin=initial_margin,
        retention_auc=float(retention_auc),
        t50=t50,
        margins=tuple(float(value) for value in margins),
    )


def recovery_metrics(
    model: Any,
    events: torch.Tensor,
    *,
    observation_ratio: float,
    fraction: float,
    horizon: int,
    seed: int,
) -> RecoveryMetrics:
    stream = _validate_events(events)
    ratio = _validate_ratio(observation_ratio)
    boundary = _prefix_steps(stream.shape[0], ratio)
    horizon = _validate_horizon(horizon)
    if boundary + horizon > stream.shape[0]:
        raise ValueError("insufficient future observed frames for recovery horizon")
    if type(fraction) not in (int, float) or isinstance(fraction, bool) or not math.isfinite(float(fraction)) or not 0.0 < float(fraction) < 1.0:
        raise ValueError("perturbation fraction must be in (0,1)")
    if type(seed) is not int:
        raise ValueError("perturbation seed must be an integer")

    with _evaluation_mode(model), torch.no_grad():
        reference_state = _run_prefix(model, stream, boundary)
        reference_class = int(_logits_1d(model, reference_state).argmax().item())
        perturbed_state = model.silence_state(reference_state, fraction=float(fraction), seed=seed)
        perturbed_class = int(_logits_1d(model, perturbed_state).argmax().item())
        eligible = perturbed_class != reference_class
        if not eligible:
            return RecoveryMetrics(False, False, None, 0, horizon)

        matched_steps = 0
        recovery_step: int | None = None
        for offset in range(horizon):
            frame = stream[boundary + offset]
            reference_state = _step_frame(model, frame, reference_state)
            perturbed_state = _step_frame(model, frame, perturbed_state)
            expected = int(_logits_1d(model, reference_state).argmax().item())
            observed = int(_logits_1d(model, perturbed_state).argmax().item())
            if expected == observed:
                matched_steps += 1
                if recovery_step is None:
                    recovery_step = offset + 1

    return RecoveryMetrics(
        eligible=True,
        recovered=recovery_step is not None,
        recovery_step=recovery_step,
        matched_steps=matched_steps,
        horizon=horizon,
    )


def _spike_tensor(state: Any) -> torch.Tensor | None:
    spikes = getattr(state, "spikes", None)
    if type(spikes) is not torch.Tensor:
        return None
    if not torch.is_floating_point(spikes) or not torch.isfinite(spikes).all().item():
        raise ValueError("spike state must be finite floating point")
    return spikes


def _synaptic_events_for_step(model: Any, spikes: torch.Tensor) -> float:
    hidden_size = getattr(model, "hidden_size", None)
    if type(hidden_size) is not int or hidden_size <= 0:
        return 0.0
    if spikes.ndim == 2:
        return float(spikes.sum().item() * hidden_size)
    if spikes.ndim == 3 and hasattr(model, "adjacency"):
        adjacency = model.adjacency
        degree = (adjacency != 0).sum(dim=1).to(device=spikes.device, dtype=spikes.dtype)
        per_node = spikes.sum(dim=(0, 2))
        # Each spike uses the shared self-recurrent transform and the graph
        # neighbor transform for every non-zero anatomical adjacency target.
        return float((per_node * hidden_size * (1.0 + degree)).sum().item())
    return 0.0


def activity_metrics(model: Any, events: torch.Tensor) -> ActivityMetrics:
    stream = _validate_events(events)
    with _evaluation_mode(model), torch.no_grad():
        state = _initial_state(model, stream)
        total_spikes = 0
        total_synaptic_events = 0.0
        active_mask: torch.Tensor | None = None
        spike_capable: bool | None = None
        for index in range(stream.shape[0]):
            state = _step_frame(model, stream[index], state)
            spikes = _spike_tensor(state)
            if spikes is None:
                if spike_capable is True:
                    raise ValueError("model spike-state availability changed during one stream")
                spike_capable = False
                continue
            if spike_capable is False:
                raise ValueError("model spike-state availability changed during one stream")
            spike_capable = True
            binary = spikes.detach() > 0.0
            total_spikes += int(binary.sum().item())
            flattened = binary.reshape(binary.shape[0], -1).any(dim=0)
            active_mask = flattened.clone() if active_mask is None else (active_mask | flattened)
            total_synaptic_events += _synaptic_events_for_step(model, spikes.detach())

    if not spike_capable:
        return ActivityMetrics(
            total_steps=stream.shape[0],
            total_spikes=None,
            active_unit_fraction=None,
            synaptic_events_per_frame=None,
        )

    if active_mask is None or active_mask.numel() == 0:
        raise ValueError("spiking model exposed an empty spike state")
    active_fraction = float(active_mask.to(dtype=torch.float32).mean().item())
    return ActivityMetrics(
        total_steps=stream.shape[0],
        total_spikes=total_spikes,
        active_unit_fraction=active_fraction,
        synaptic_events_per_frame=float(total_synaptic_events / stream.shape[0]),
    )
