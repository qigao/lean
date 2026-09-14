"""Explicit CPU development training for padded observations, never final-test access."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
import math
import random
from typing import Iterator

import numpy as np
import torch
from torch.nn import functional as F

from .eval import MetricBundle
from .models import GRUClassifier, GraphDiagnosticClassifier, GraphRecurrentClassifier
from .models._padded import validate_pose_batch
from .pose_batches import PoseBatch
from .train import TrainConfig, _reset_parameters, _state_hash

PaddedModel = GRUClassifier | GraphRecurrentClassifier | GraphDiagnosticClassifier
_GRAPH_MODELS = (GraphRecurrentClassifier, GraphDiagnosticClassifier)
_SUPPORTED_MODELS = (GRUClassifier, *_GRAPH_MODELS)


@dataclass(frozen=True)
class PosePartition:
    """Declared development metadata and aligned targets, not authenticated provenance.

    Construct this from independently verified samples. Frozen fields do not make
    tensor storage immutable; callers must not mutate inputs during execution.
    """

    observations: PoseBatch
    targets: torch.Tensor
    classes: tuple[str, ...]
    sample_ids: tuple[str, ...]
    subjects: tuple[int, ...]
    split: str


@dataclass(frozen=True)
class PaddedTrainedRun:
    model: PaddedModel
    state_hash: str
    best_validation_macro_f1: float
    best_epoch: int
    optimizer_steps: int
    epoch_order_hashes: tuple[str, ...]


def _positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _strings(values: tuple[str, ...], name: str) -> None:
    if (type(values) is not tuple or not values
            or any(type(v) is not str or not v.strip() for v in values)
            or len(set(values)) != len(values)):
        raise ValueError(f"{name} must be a nonempty tuple of unique nonblank strings")


def _validate_partition(model: PaddedModel, part: PosePartition, role: str) -> None:
    if type(model) not in _SUPPORTED_MODELS:
        raise ValueError("expected an explicit supported padded model")
    if type(part) is not PosePartition or type(part.split) is not str or part.split != role:
        raise ValueError(f"expected a {role} PosePartition; no final-test access")
    input_dim = (model.gru.input_size if type(model) is GRUClassifier
                 else model.input_projection.in_features)
    validate_pose_batch(part.observations, input_dim=input_dim, model=model)
    for value in (*model.parameters(), *model.buffers()):
        if value.layout != torch.strided or not torch.isfinite(value).all().item():
            raise ValueError("model parameters and buffers must be dense and finite")
    _strings(part.classes, "classes")
    _strings(part.sample_ids, "sample_ids")
    size = part.observations.features.shape[0]
    if model.readout.out_features != len(part.classes):
        raise ValueError("model outputs must match the explicit ordered class vocabulary")
    if len(part.sample_ids) != size:
        raise ValueError("sample_ids must align with observations")
    if (type(part.subjects) is not tuple or len(part.subjects) != size
            or any(type(s) is not int or s <= 0 for s in part.subjects)):
        raise ValueError("subjects must be aligned positive integers")
    y = part.targets
    if (type(y) is not torch.Tensor or y.layout != torch.strided
            or y.device.type != "cpu" or y.dtype != torch.int64 or y.shape != (size,)):
        raise ValueError("targets must be an aligned dense CPU int64 vector")
    if torch.any((y < 0) | (y >= len(part.classes))).item():
        raise ValueError("targets are outside the explicit class vocabulary")


def _validate_config(model: PaddedModel, config: TrainConfig) -> None:
    if type(config) is not TrainConfig:
        raise ValueError("expected an explicit TrainConfig")
    for name in ("epochs", "batch_size", "parameter_ceiling"):
        _positive_integer(getattr(config, name), name)
    if type(config.seed) is not int or not 0 <= config.seed < 2**32:
        raise ValueError("seed must be an integer in [0,2**32)")
    if type(config.lr) not in (int, float) or not math.isfinite(config.lr) or config.lr <= 0:
        raise ValueError("lr must be finite and positive")
    if sum(p.numel() for p in model.parameters()) > config.parameter_ceiling:
        raise ValueError("model exceeds parameter ceiling")
    if any(not parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("all learned parameters must be trainable in this training condition")


def _copy_partition(part: PosePartition) -> PosePartition:
    b = part.observations
    return replace(part, observations=PoseBatch(
        b.features.detach().clone(), b.lengths.clone(), b.time_mask.clone(),
    ), targets=part.targets.detach().clone())


def _select(part: PosePartition, indices: torch.Tensor) -> tuple[PoseBatch, torch.Tensor]:
    """Apply one row selection to every field; remove only unused trailing padding."""
    b = part.observations
    lengths = b.lengths.index_select(0, indices)
    steps = int(lengths.max().item())
    return PoseBatch(
        b.features[:, :steps].detach().index_select(0, indices), lengths,
        b.time_mask[:, :steps].index_select(0, indices),
    ), part.targets.index_select(0, indices)


def _logits(model: PaddedModel, batch: PoseBatch, classes: int) -> torch.Tensor:
    logits = model.forward_padded(batch)
    if (type(logits) is not torch.Tensor or logits.layout != torch.strided
            or logits.device.type != "cpu" or logits.dtype != torch.float32
            or logits.shape != (len(batch.lengths), classes)):
        raise ValueError("model must return dense CPU float32 [batch,classes] logits")
    if not torch.isfinite(logits).all().item():
        raise ValueError("model logits must be finite")
    return logits


def _metrics(counts: torch.Tensor) -> MetricBundle:
    matrix = counts.tolist()
    f1s, recalls = [], []
    for label in range(len(matrix)):
        tp = matrix[label][label]
        actual = sum(matrix[label])
        predicted = sum(row[label] for row in matrix)
        f1s.append(2 * tp / (actual + predicted) if actual + predicted else 0.)
        recalls.append(tp / actual if actual else 0.)
    return MetricBundle(sum(f1s) / len(matrix), sum(recalls) / len(matrix))


def evaluate_padded(
    model: PaddedModel, validation: PosePartition, *, batch_size: int,
) -> MetricBundle:
    """Aggregate validation counts over all batches and the full declared vocabulary."""
    _positive_integer(batch_size, "batch_size")
    _validate_partition(model, validation, "validation")
    classes, size = len(validation.classes), len(validation.sample_ids)
    counts = torch.zeros((classes, classes), dtype=torch.int64, device="cpu")
    modes = tuple((module, module.training) for module in model.modules())
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, size, batch_size):
                indices = torch.arange(start, min(start + batch_size, size), device="cpu")
                batch, targets = _select(validation, indices)
                predicted = _logits(model, batch, classes).argmax(dim=1)
                counts += torch.bincount(
                    targets * classes + predicted, minlength=classes * classes,
                ).reshape(classes, classes)
        return _metrics(counts)
    finally:
        for module, training in modes:
            module.training = training


@contextmanager
def _seeded_cpu(seed: int) -> Iterator[None]:
    python_state, numpy_state = random.getstate(), np.random.get_state()
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    with torch.random.fork_rng(devices=[]):
        try:
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            torch.use_deterministic_algorithms(True)
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def train_padded_model(
    model: PaddedModel, train: PosePartition, validation: PosePartition, config: TrainConfig,
) -> PaddedTrainedRun:
    """Reset and train with Adam; select the earliest best validation-only checkpoint.

    Declared split/sample/subject metadata is checked, not authenticated. Both
    partitions must come from the verified upstream development pipeline. On a
    compute failure no result is returned; prior optimizer updates are not undone.
    """
    _validate_partition(model, train, "train")
    _validate_partition(model, validation, "validation")
    if train.classes != validation.classes:
        raise ValueError("train and validation must share the exact ordered class vocabulary")
    if set(train.sample_ids) & set(validation.sample_ids):
        raise ValueError("train and validation sample_ids overlap")
    if set(train.subjects) & set(validation.subjects):
        raise ValueError("train and validation subjects overlap")
    _validate_config(model, config)
    train, validation = _copy_partition(train), _copy_partition(validation)
    classes, size = len(train.classes), len(train.sample_ids)
    best_score, best_epoch, best_state = -1., 0, None
    updates, order_hashes = 0, []
    with _seeded_cpu(config.seed):
        _reset_parameters(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        for epoch in range(1, config.epochs + 1):
            model.train()
            order = torch.randperm(size, generator=generator, device="cpu")
            encoded = json.dumps([train.sample_ids[i] for i in order.tolist()],
                                 separators=(",", ":")).encode("utf-8")
            order_hashes.append(hashlib.sha256(encoded).hexdigest())
            for start in range(0, size, config.batch_size):
                batch, targets = _select(train, order[start:start + config.batch_size])
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(_logits(model, batch, classes), targets)
                if not torch.isfinite(loss).item():
                    raise ValueError("training loss must be finite")
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all().item()
                       for p in model.parameters()):
                    raise ValueError("training gradients must be finite")
                optimizer.step()
                updates += 1
                if any(not torch.isfinite(p).all().item() for p in model.parameters()):
                    raise ValueError("updated model parameters must be finite")
            score = evaluate_padded(model, validation, batch_size=config.batch_size).macro_f1
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("validation score must be finite and in [0,1]")
            if score > best_score:
                best_score, best_epoch = score, epoch
                best_state = deepcopy(model.state_dict())
        assert best_state is not None
        model.load_state_dict(best_state)
        model.zero_grad(set_to_none=True)
        model.eval()
        return PaddedTrainedRun(model, _state_hash(model), best_score, best_epoch,
                                updates, tuple(order_hashes))
