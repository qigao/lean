"""CPU development training over verified indexed minibatches; never final-test access."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

import torch
from torch.nn import functional as F

from .eval import MetricBundle
from .models import GRUClassifier, GraphRecurrentClassifier
from .pose_indexed_development import IndexedPoseDevelopment
from .pose_training import (
    PaddedModel, PaddedTrainedRun, _logits, _metrics, _positive_integer,
    _seeded_cpu, _validate_config, _validate_partition,
)
from .train import TrainConfig, _reset_parameters, _state_hash


def _split_rows(source: IndexedPoseDevelopment, split: str):
    return tuple(entry for entry in source.index.samples if entry.split == split)


def _validate_source_model(
    model: PaddedModel, source: IndexedPoseDevelopment, *, expected_binding_sha256: str,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    if type(source) is not IndexedPoseDevelopment:
        raise ValueError("expected an explicit IndexedPoseDevelopment source")
    if type(model) not in (GRUClassifier, GraphRecurrentClassifier):
        raise ValueError("expected an explicit supported padded model")
    source.verify(expected_binding_sha256=expected_binding_sha256)
    input_dim = (model.gru.input_size if type(model) is GRUClassifier
                 else model.input_projection.in_features)
    if input_dim != 121:
        raise ValueError("indexed pose training requires the frozen 121-feature input")
    if model.readout.out_features != len(source.classes):
        raise ValueError("model outputs must match the indexed ordered class vocabulary")
    for value in (*model.parameters(), *model.buffers()):
        if value.layout != torch.strided or not torch.isfinite(value).all().item():
            raise ValueError("model parameters and buffers must be dense and finite")
    train, validation = _split_rows(source, "train"), _split_rows(source, "validation")
    if not train or not validation:
        raise ValueError("indexed development source requires nonempty train and validation splits")
    return train, validation


def evaluate_indexed(
    model: PaddedModel, source: IndexedPoseDevelopment, *,
    expected_binding_sha256: str, batch_size: int,
) -> MetricBundle:
    """Evaluate the complete indexed validation roster one requested minibatch at a time."""
    _positive_integer(batch_size, "batch_size")
    _, validation = _validate_source_model(
        model, source, expected_binding_sha256=expected_binding_sha256,
    )
    classes = len(source.classes)
    counts = torch.zeros((classes, classes), dtype=torch.int64, device="cpu")
    modes = tuple((module, module.training) for module in model.modules())
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, len(validation), batch_size):
                indices = tuple(range(start, min(start + batch_size, len(validation))))
                part = source.read_partition(
                    split="validation", indices=indices,
                    expected_binding_sha256=expected_binding_sha256,
                )
                _validate_partition(model, part, "validation")
                predicted = _logits(model, part.observations, classes).argmax(dim=1)
                counts += torch.bincount(
                    part.targets * classes + predicted, minlength=classes * classes,
                ).reshape(classes, classes)
                del predicted, part
        result = _metrics(counts)
        source.verify(expected_binding_sha256=expected_binding_sha256)
        return result
    finally:
        for module, training in modes:
            module.training = training


def train_indexed_model(
    model: PaddedModel, source: IndexedPoseDevelopment, config: TrainConfig, *,
    expected_binding_sha256: str,
) -> PaddedTrainedRun:
    """Train from bound development minibatches with validation-only checkpoint selection.

    The complete scalar train roster remains available for deterministic ordering,
    but observation tensors are requested only for the current minibatch. There is
    no eager-partition fallback, gradient accumulation or final-test switch.
    """
    train, _ = _validate_source_model(
        model, source, expected_binding_sha256=expected_binding_sha256,
    )
    _validate_config(model, config)
    classes, size = len(source.classes), len(train)
    sample_ids = tuple(entry.sample_id for entry in train)
    best_score, best_epoch, best_state = -1.0, 0, None
    updates, order_hashes = 0, []

    with _seeded_cpu(config.seed):
        _reset_parameters(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        for epoch in range(1, config.epochs + 1):
            model.train()
            order = torch.randperm(size, generator=generator, device="cpu")
            ordered_ids = [sample_ids[i] for i in order.tolist()]
            encoded = json.dumps(ordered_ids, separators=(",", ":")).encode("utf-8")
            order_hashes.append(hashlib.sha256(encoded).hexdigest())

            for start in range(0, size, config.batch_size):
                indices = tuple(order[start:start + config.batch_size].tolist())
                part = source.read_partition(
                    split="train", indices=indices,
                    expected_binding_sha256=expected_binding_sha256,
                )
                _validate_partition(model, part, "train")
                optimizer.zero_grad(set_to_none=True)
                logits = _logits(model, part.observations, classes)
                loss = F.cross_entropy(logits, part.targets)
                if not torch.isfinite(loss).item():
                    raise ValueError("training loss must be finite")
                loss.backward()
                if any(parameter.grad is not None
                       and not torch.isfinite(parameter.grad).all().item()
                       for parameter in model.parameters()):
                    raise ValueError("training gradients must be finite")
                optimizer.step()
                updates += 1
                if any(not torch.isfinite(parameter).all().item()
                       for parameter in model.parameters()):
                    raise ValueError("updated model parameters must be finite")
                del loss, logits, part

            score = evaluate_indexed(
                model, source, expected_binding_sha256=expected_binding_sha256,
                batch_size=config.batch_size,
            ).macro_f1
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("validation score must be finite and in [0,1]")
            if score > best_score:
                best_score, best_epoch = score, epoch
                best_state = deepcopy(model.state_dict())

        assert best_state is not None
        model.load_state_dict(best_state)
        model.zero_grad(set_to_none=True)
        model.eval()
        source.verify(expected_binding_sha256=expected_binding_sha256)
        return PaddedTrainedRun(
            model, _state_hash(model), best_score, best_epoch, updates, tuple(order_hashes),
        )
