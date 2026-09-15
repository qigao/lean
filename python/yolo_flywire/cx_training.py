from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
import statistics
from typing import Iterable

import torch
import torch.nn.functional as F

from .cx_artifact import CxArtifact
from .cx_development import (
    CxDevelopmentBinding,
    CxDevelopmentResult,
    CxScoreboard,
)
from .cx_evaluation import early_prediction_auc, macro_f1, recovery_metrics, retention_metrics
from .models.cx_lif import CxDynamics, CxLifClassifier


_OBSERVATION_RATIOS = (0.10, 0.20, 0.40, 0.60, 0.80, 1.00)


@dataclass(frozen=True)
class CxEventSample:
    sample_id: str
    label: int
    events: torch.Tensor

    def __post_init__(self) -> None:
        if type(self.sample_id) is not str or not self.sample_id:
            raise ValueError("sample_id must be a non-empty string")
        if type(self.label) is not int or self.label < 0:
            raise ValueError("sample label must be a non-negative integer")
        if type(self.events) is not torch.Tensor or self.events.ndim != 2:
            raise ValueError("sample signed events must have shape [time,feature]")
        if self.events.shape[0] <= 0 or self.events.shape[1] <= 0:
            raise ValueError("sample signed events must have positive dimensions")
        if self.events.dtype != torch.float32 or not torch.isfinite(self.events).all().item():
            raise ValueError("sample signed events must be finite float32")
        signed = (self.events == -1.0) | (self.events == 0.0) | (self.events == 1.0)
        if not signed.all().item():
            raise ValueError("sample signed events must contain only -1, 0, or +1")
        object.__setattr__(self, "events", self.events.detach().clone())


@dataclass(frozen=True)
class CxTrainingTrace:
    initial_parameter_fingerprint: str
    training_order_fingerprint: str
    best_epoch: int
    updates: int
    best_validation_macro_f1: float


@dataclass(frozen=True)
class CxTrainingOutcome:
    result: CxDevelopmentResult
    trace: CxTrainingTrace


def _hash_bytes(parts: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


def _parameter_fingerprint(model: CxLifClassifier) -> str:
    parts: list[bytes] = []
    for name, parameter in sorted(model.named_parameters()):
        value = parameter.detach().cpu().contiguous()
        parts.extend(
            (
                name.encode("utf-8"),
                str(value.dtype).encode("ascii"),
                repr(tuple(value.shape)).encode("ascii"),
                value.numpy().tobytes(order="C"),
            )
        )
    return _hash_bytes(parts)


def _order_fingerprint(sample_ids: list[str]) -> str:
    return _hash_bytes(sample_id.encode("utf-8") for sample_id in sample_ids)


def _stable_seed(seed: int, sample_id: str) -> int:
    raw = hashlib.sha256(f"cx-gate1:{seed}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(raw[:8], "big")


def _validate_samples(
    train_samples: tuple[CxEventSample, ...],
    validation_samples: tuple[CxEventSample, ...],
) -> tuple[int, int]:
    if type(train_samples) is not tuple or not train_samples:
        raise ValueError("CX training requires non-empty train samples")
    if type(validation_samples) is not tuple or not validation_samples:
        raise ValueError("CX training requires non-empty validation samples")
    if any(type(sample) is not CxEventSample for sample in train_samples + validation_samples):
        raise ValueError("CX training samples must be CxEventSample values")

    all_samples = train_samples + validation_samples
    sample_ids = [sample.sample_id for sample in all_samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("CX train/validation sample IDs must be unique")
    feature_dims = {int(sample.events.shape[1]) for sample in all_samples}
    if len(feature_dims) != 1:
        raise ValueError("CX train/validation event feature dimensions must match")
    input_dim = next(iter(feature_dims))

    train_labels = sorted({sample.label for sample in train_samples})
    if len(train_labels) < 2 or train_labels != list(range(train_labels[-1] + 1)):
        raise ValueError("CX training labels must cover a contiguous class range starting at zero")
    num_classes = train_labels[-1] + 1
    if any(sample.label >= num_classes for sample in validation_samples):
        raise ValueError("CX validation label lies outside the training class range")
    return input_dim, num_classes


def _predict(
    model: CxLifClassifier,
    samples: tuple[CxEventSample, ...],
    *,
    num_classes: int,
    ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    labels: list[int] = []
    predictions: list[int] = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for sample in samples:
                time_steps = int(sample.events.shape[0])
                prefix = max(1, min(time_steps, int(math.ceil(time_steps * ratio))))
                logits = model(sample.events[:prefix].unsqueeze(0))
                labels.append(sample.label)
                predictions.append(int(logits.argmax(dim=1).item()))
    finally:
        if was_training:
            model.train()
    y_true = torch.tensor(labels, dtype=torch.int64)
    y_pred = torch.tensor(predictions, dtype=torch.int64)
    return y_true, y_pred, macro_f1(y_true, y_pred, num_classes=num_classes)


def _validation_score(model: CxLifClassifier, samples: tuple[CxEventSample, ...], num_classes: int) -> float:
    return _predict(model, samples, num_classes=num_classes, ratio=1.0)[2]


def _clone_state_dict(model: CxLifClassifier) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _scoreboard(
    model: CxLifClassifier,
    artifact: CxArtifact,
    samples: tuple[CxEventSample, ...],
    *,
    num_classes: int,
    binding: CxDevelopmentBinding,
    seed: int,
) -> CxScoreboard:
    early_values = tuple(
        _predict(model, samples, num_classes=num_classes, ratio=ratio)[2]
        for ratio in _OBSERVATION_RATIOS
    )
    early_auc = early_prediction_auc(_OBSERVATION_RATIOS, early_values)

    retention_auc_values: list[float] = []
    retention_t50_values: list[float] = []
    recovered_steps: list[float] = []
    total_recovery_eligible = 0
    total_recovered = 0
    total_spikes = 0
    total_active = 0
    total_synaptic_events = 0
    total_frames = 0

    was_training = model.training
    model.eval()
    try:
        for sample in samples:
            time_steps = int(sample.events.shape[0])
            boundary = max(1, int(math.ceil(time_steps * 0.40)))
            observed = sample.events[:boundary].unsqueeze(0)
            retention = retention_metrics(
                model,
                observed,
                true_label=sample.label,
                horizon=binding.budget.retention_horizon,
            )
            retention_auc_values.append(retention.retention_auc)
            retention_t50_values.append(
                float(binding.budget.retention_horizon + 1) if retention.t50 is None else float(retention.t50)
            )

            if boundary >= time_steps or boundary + binding.budget.recovery_horizon > time_steps:
                raise ValueError(
                    f"validation sample {sample.sample_id!r} is too short for the frozen recovery horizon"
                )
            recovery = recovery_metrics(
                model,
                sample.events.unsqueeze(0),
                perturb_at=boundary,
                silence_fraction=binding.budget.perturbation_fraction,
                horizon=binding.budget.recovery_horizon,
                seed=_stable_seed(seed, sample.sample_id),
                candidate_node_indices=artifact.core_indices,
            )
            total_recovery_eligible += recovery.eligible_samples
            total_recovered += recovery.recovered_samples
            if recovery.median_recovery_steps is not None:
                recovered_steps.append(float(recovery.median_recovery_steps))

            with torch.no_grad():
                _, stats = model.encode(sample.events.unsqueeze(0))
            total_spikes += stats.total_spikes
            total_active += sum(stats.active_neurons_per_step)
            total_synaptic_events += stats.recurrent_synaptic_events
            total_frames += stats.steps
    finally:
        if was_training:
            model.train()

    if total_frames <= 0:
        raise ValueError("validation event frame count must be positive")
    node_frames = total_frames * artifact.graph.num_nodes
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    recovery_rate = 0.0 if total_recovery_eligible == 0 else total_recovered / total_recovery_eligible
    median_recovery = (
        float(binding.budget.recovery_horizon + 1)
        if not recovered_steps
        else float(statistics.median(recovered_steps))
    )

    return CxScoreboard(
        full_macro_f1=early_values[-1],
        early_macro_f1_by_ratio=tuple(zip(_OBSERVATION_RATIOS, early_values)),
        early_prediction_auc=early_auc,
        retention_auc=sum(retention_auc_values) / len(retention_auc_values),
        retention_t50=sum(retention_t50_values) / len(retention_t50_values),
        recovery_rate=recovery_rate,
        median_recovery_steps=median_recovery,
        spike_rate=total_spikes / node_frames,
        active_neuron_fraction=total_active / node_frames,
        synaptic_events_per_frame=total_synaptic_events / total_frames,
        parameter_count=int(parameter_count),
    )


def train_validation_arm(
    artifact: CxArtifact,
    *,
    binding: CxDevelopmentBinding,
    arm: str,
    seed: int,
    train_samples: tuple[CxEventSample, ...],
    validation_samples: tuple[CxEventSample, ...],
    dynamics: CxDynamics,
) -> CxTrainingOutcome:
    if type(artifact) is not CxArtifact:
        raise ValueError("artifact must be a CxArtifact")
    if type(binding) is not CxDevelopmentBinding:
        raise ValueError("binding must be a CxDevelopmentBinding")
    if type(dynamics) is not CxDynamics:
        raise ValueError("dynamics must be a CxDynamics")
    if type(seed) is not int or seed not in binding.seeds:
        raise ValueError("seed is not part of the frozen development binding")
    expected_fingerprint = binding.graph_fingerprint_for(arm, seed)
    if artifact.fingerprint != expected_fingerprint:
        raise ValueError(
            f"artifact fingerprint does not match binding for arm={arm!r} seed={seed}"
        )
    if not artifact.core_indices:
        raise ValueError("CX training requires non-empty recurrent core indices")

    input_dim, num_classes = _validate_samples(train_samples, validation_samples)
    budget = binding.budget

    # R0 uses CPU as the deterministic reference execution path. Resetting the
    # same torch seed before every arm gives identical trainable initialization;
    # recurrent artifact buffers consume no RNG and are not in the state dict.
    torch.manual_seed(seed)
    model = CxLifClassifier(
        input_dim,
        artifact,
        num_classes,
        dynamics=dynamics,
    )
    initial_fingerprint = _parameter_fingerprint(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=budget.learning_rate)

    updates = 0
    training_order: list[str] = []
    best_epoch = -1
    best_validation = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(budget.epochs):
        model.train()
        order = list(range(len(train_samples)))
        random.Random((seed << 32) + epoch).shuffle(order)
        for sample_index in order:
            if updates >= budget.max_updates:
                break
            sample = train_samples[sample_index]
            optimizer.zero_grad(set_to_none=True)
            logits = model(sample.events.unsqueeze(0))
            target = torch.tensor([sample.label], dtype=torch.int64)
            loss = F.cross_entropy(logits, target)
            if not torch.isfinite(loss).item():
                raise ValueError("CX training loss became non-finite")
            loss.backward()
            optimizer.step()
            updates += 1
            training_order.append(sample.sample_id)

        validation_macro_f1 = _validation_score(model, validation_samples, num_classes)
        if validation_macro_f1 > best_validation:
            best_validation = validation_macro_f1
            best_epoch = epoch
            best_state = _clone_state_dict(model)
        if updates >= budget.max_updates:
            break

    if best_state is None or best_epoch < 0 or not math.isfinite(best_validation):
        raise ValueError("CX training did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    scoreboard = _scoreboard(
        model,
        artifact,
        validation_samples,
        num_classes=num_classes,
        binding=binding,
        seed=seed,
    )

    result = CxDevelopmentResult(
        arm=arm,
        seed=seed,
        graph_fingerprint=artifact.fingerprint,
        dataset_content_hash=binding.dataset_content_hash,
        split_hash=binding.split_hash,
        event_encoder_hash=binding.event_encoder_hash,
        dynamics_hash=binding.dynamics_hash,
        input_node_fingerprint=binding.input_node_fingerprint,
        output_node_fingerprint=binding.output_node_fingerprint,
        budget=budget,
        scoreboard=scoreboard,
    )
    trace = CxTrainingTrace(
        initial_parameter_fingerprint=initial_fingerprint,
        training_order_fingerprint=_order_fingerprint(training_order),
        best_epoch=best_epoch,
        updates=updates,
        best_validation_macro_f1=best_validation,
    )
    return CxTrainingOutcome(result=result, trace=trace)
