from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import io
import random

import numpy as np
import torch
from torch import nn

from .eval import evaluate


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    epochs: int
    lr: float
    batch_size: int
    parameter_ceiling: int


@dataclass(frozen=True)
class TrainedRun:
    model: nn.Module
    state_hash: str
    best_validation_macro_f1: float


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _reset_parameters(module: nn.Module) -> None:
    for child in module.children():
        _reset_parameters(child)
    reset = getattr(module, "reset_parameters", None)
    if callable(reset):
        reset()


def _state_hash(model: nn.Module) -> str:
    buffer = io.BytesIO()
    state = {name: tensor.detach().cpu() for name, tensor in sorted(model.state_dict().items())}
    torch.save(state, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def train_model(
    model: nn.Module,
    train: tuple[torch.Tensor, torch.Tensor],
    validation: tuple[torch.Tensor, torch.Tensor],
    config: TrainConfig,
) -> TrainedRun:
    """Train deterministically; validation selects checkpoints and test data is never accepted here."""
    if config.epochs <= 0 or config.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count > config.parameter_ceiling:
        raise ValueError("model exceeds parameter ceiling")

    _seed_everything(config.seed)
    _reset_parameters(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.CrossEntropyLoss()
    train_x, train_y = train

    best_score = -1.0
    best_state = deepcopy(model.state_dict())
    generator = torch.Generator().manual_seed(config.seed)

    for _ in range(config.epochs):
        model.train()
        order = torch.randperm(train_x.shape[0], generator=generator)
        for start in range(0, train_x.shape[0], config.batch_size):
            indices = order[start : start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(train_x[indices]), train_y[indices])
            loss.backward()
            optimizer.step()

        score = evaluate(model, validation).macro_f1
        if score > best_score:
            best_score = score
            best_state = deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return TrainedRun(model=model, state_hash=_state_hash(model), best_validation_macro_f1=best_score)
