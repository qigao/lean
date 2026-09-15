from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class PrefixPrediction:
    observation_ratio: float
    observed_steps: int
    logits: torch.Tensor


@dataclass(frozen=True)
class RetentionMetrics:
    initial_margin: float
    retention_auc: float
    t50: int | None


@dataclass(frozen=True)
class DropoutRecoveryMetrics:
    recovered: bool
    recovery_step: int | None
    recovery_rate: float
    dropout_steps: int


def _validate_sequence(sequence: torch.Tensor) -> torch.Tensor:
    if type(sequence) is not torch.Tensor or sequence.ndim != 3:
        raise ValueError("sequence must be a [time,joints,features] tensor")
    if sequence.shape[0] <= 0 or sequence.shape[1] <= 0 or sequence.shape[2] <= 0:
        raise ValueError("sequence dimensions must be non-empty")
    if not torch.is_floating_point(sequence) or not torch.isfinite(sequence).all().item():
        raise ValueError("sequence must contain finite floating-point values")
    return sequence


def _validated_ratio(value: float, *, name: str = "observation ratio") -> float:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    ratio = float(value)
    if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise ValueError(f"{name} must be finite and in (0,1]")
    return ratio


def _observed_steps(length: int, ratio: float) -> int:
    return max(1, min(length, int(math.ceil(length * ratio))))


def _initial_state(model, sequence: torch.Tensor):
    return model.initial_state(1, sequence.device, sequence.dtype)


def _step_frame(model, frame: torch.Tensor, state):
    return model.step(frame.unsqueeze(0), state)


def _logits(model, state) -> torch.Tensor:
    logits = model.logits(state)
    if type(logits) is not torch.Tensor or logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError("model logits must have shape [1,classes]")
    if not torch.is_floating_point(logits) or not torch.isfinite(logits).all().item():
        raise ValueError("model logits must be finite floating point")
    return logits


def macro_f1(y_true: torch.Tensor, y_pred: torch.Tensor, *, num_classes: int) -> float:
    if type(y_true) is not torch.Tensor or type(y_pred) is not torch.Tensor:
        raise ValueError("labels must be tensors")
    if y_true.ndim != 1 or y_pred.ndim != 1 or y_true.shape != y_pred.shape or y_true.numel() == 0:
        raise ValueError("labels must be non-empty one-dimensional tensors with equal shape")
    if type(num_classes) is not int or num_classes <= 0:
        raise ValueError("num_classes must be a positive integer")
    truth = y_true.to(dtype=torch.int64)
    pred = y_pred.to(dtype=torch.int64)
    if torch.any(truth < 0).item() or torch.any(truth >= num_classes).item():
        raise ValueError("true labels are outside the declared class range")
    if torch.any(pred < 0).item() or torch.any(pred >= num_classes).item():
        raise ValueError("predicted labels are outside the declared class range")

    values: list[float] = []
    for class_id in range(num_classes):
        true_positive = int(torch.sum((truth == class_id) & (pred == class_id)).item())
        false_positive = int(torch.sum((truth != class_id) & (pred == class_id)).item())
        false_negative = int(torch.sum((truth == class_id) & (pred != class_id)).item())
        denominator = 2 * true_positive + false_positive + false_negative
        values.append(0.0 if denominator == 0 else (2.0 * true_positive) / denominator)
    return float(sum(values) / num_classes)


def early_prediction_auc(ratios: Sequence[float], scores: Sequence[float]) -> float:
    if len(ratios) != len(scores) or len(ratios) < 2:
        raise ValueError("ratios and scores must have equal length of at least two")
    x = np.asarray([_validated_ratio(value, name="observation ratio") for value in ratios], dtype=np.float64)
    y = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(y)):
        raise ValueError("early-prediction scores must be finite")
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("observation ratios must be strictly increasing")
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def prefix_predictions(model, sequence: torch.Tensor, *, ratios: Sequence[float]) -> tuple[PrefixPrediction, ...]:
    seq = _validate_sequence(sequence)
    if not ratios:
        raise ValueError("at least one observation ratio is required")
    validated = tuple(_validated_ratio(value) for value in ratios)
    if any(validated[index] >= validated[index + 1] for index in range(len(validated) - 1)):
        raise ValueError("observation ratios must be strictly increasing")

    wanted = tuple(_observed_steps(seq.shape[0], ratio) for ratio in validated)
    state = _initial_state(model, seq)
    results: list[PrefixPrediction] = []
    next_index = 0
    with torch.no_grad():
        for step_index, frame in enumerate(seq, start=1):
            state = _step_frame(model, frame, state)
            while next_index < len(wanted) and wanted[next_index] == step_index:
                results.append(
                    PrefixPrediction(
                        observation_ratio=validated[next_index],
                        observed_steps=step_index,
                        logits=_logits(model, state).detach().clone().squeeze(0),
                    )
                )
                next_index += 1
            if next_index == len(wanted):
                break
    if len(results) != len(validated):
        raise RuntimeError("could not produce every requested prefix prediction")
    return tuple(results)


def _true_class_margin(logits: torch.Tensor, true_label: int) -> float:
    if type(true_label) is not int or not 0 <= true_label < logits.shape[-1]:
        raise ValueError("true_label is outside the model class range")
    row = logits.squeeze(0)
    true_value = row[true_label]
    if row.numel() == 1:
        competitor = torch.zeros_like(true_value)
    else:
        mask = torch.ones(row.numel(), dtype=torch.bool, device=row.device)
        mask[true_label] = False
        competitor = torch.max(row[mask])
    return float((true_value - competitor).item())


def retention_metrics(
    model,
    sequence: torch.Tensor,
    *,
    true_label: int,
    observation_ratio: float,
    horizon: int,
) -> RetentionMetrics:
    seq = _validate_sequence(sequence)
    ratio = _validated_ratio(observation_ratio)
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("retention horizon must be a positive integer")

    observed = _observed_steps(seq.shape[0], ratio)
    state = _initial_state(model, seq)
    with torch.no_grad():
        for frame in seq[:observed]:
            state = _step_frame(model, frame, state)
        initial_margin = _true_class_margin(_logits(model, state), true_label)
        denominator = max(abs(initial_margin), 1e-12)
        normalized = [1.0]
        t50: int | None = None
        zero = torch.zeros_like(seq[0])
        for step in range(1, horizon + 1):
            state = _step_frame(model, zero, state)
            margin = _true_class_margin(_logits(model, state), true_label)
            value = margin / denominator
            normalized.append(value)
            if t50 is None and value <= 0.5:
                t50 = step

    x = np.arange(horizon + 1, dtype=np.float64)
    retention_auc = float(np.trapezoid(np.asarray(normalized, dtype=np.float64), x) / horizon)
    return RetentionMetrics(initial_margin=initial_margin, retention_auc=retention_auc, t50=t50)


def _predicted_class(model, state) -> int:
    return int(torch.argmax(_logits(model, state), dim=1).item())


def dropout_recovery_metrics(
    model,
    sequence: torch.Tensor,
    *,
    observation_ratio: float,
    burst: int,
    recovery_horizon: int,
) -> DropoutRecoveryMetrics:
    seq = _validate_sequence(sequence)
    ratio = _validated_ratio(observation_ratio)
    if type(burst) is not int or burst <= 0:
        raise ValueError("dropout burst must be a positive integer")
    if type(recovery_horizon) is not int or recovery_horizon <= 0:
        raise ValueError("recovery horizon must be a positive integer")
    observed = _observed_steps(seq.shape[0], ratio)
    required = observed + burst + recovery_horizon
    if seq.shape[0] < required:
        raise ValueError("sequence does not contain enough future stream for dropout recovery")

    reference_state = _initial_state(model, seq)
    perturbed_state = _initial_state(model, seq)
    zero = torch.zeros_like(seq[0])
    reference_predictions: list[int] = []
    perturbed_predictions: list[int] = []

    with torch.no_grad():
        for frame in seq[:observed]:
            reference_state = _step_frame(model, frame, reference_state)
            perturbed_state = _step_frame(model, frame, perturbed_state)

        for frame in seq[observed : observed + burst]:
            reference_state = _step_frame(model, frame, reference_state)
            perturbed_state = _step_frame(model, zero, perturbed_state)

        restored = seq[observed + burst : required]
        for frame in restored:
            reference_state = _step_frame(model, frame, reference_state)
            perturbed_state = _step_frame(model, frame, perturbed_state)
            reference_predictions.append(_predicted_class(model, reference_state))
            perturbed_predictions.append(_predicted_class(model, perturbed_state))

    recovery_step: int | None = None
    for index in range(recovery_horizon):
        if perturbed_predictions[index:] == reference_predictions[index:]:
            recovery_step = index + 1
            break
    recovered = recovery_step is not None
    return DropoutRecoveryMetrics(
        recovered=recovered,
        recovery_step=recovery_step,
        recovery_rate=1.0 if recovered else 0.0,
        dropout_steps=burst,
    )


def length_quartile_boundaries(frame_counts: Iterable[int]) -> tuple[int, int, int]:
    values = sorted(int(value) for value in frame_counts)
    if not values or any(value <= 0 for value in values):
        raise ValueError("frame counts must be positive and non-empty")

    def nearest_rank(fraction: float) -> int:
        rank = max(1, int(math.ceil(fraction * len(values))))
        return values[rank - 1]

    boundaries = (nearest_rank(0.25), nearest_rank(0.50), nearest_rank(0.75))
    if len(set(boundaries)) != 3:
        raise ValueError("length quartile boundaries must be distinct")
    return boundaries
