"""Bound indexed development training contracts; generated fixtures, not recognition evidence."""
from __future__ import annotations

from dataclasses import replace
import gc
import hashlib
import importlib
import inspect
import json
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
import torch

from yolo_flywire.graphs import DirectedGraph
from yolo_flywire.models import GRUClassifier, GraphRecurrentClassifier
from yolo_flywire.pose_batches import collate_pose_features
from yolo_flywire.pose_indexed_development import IndexedPoseDevelopment
from yolo_flywire.pose_training import PosePartition
from yolo_flywire.train import TrainConfig


_PIN = "a" * 64
_CLASSES = ("sit", "wave", "point")


def _api():
    return importlib.import_module("yolo_flywire.pose_indexed_training")


def _model(kind: str):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(91)
        if kind == "gru":
            return GRUClassifier(121, 4, 3).float().cpu()
        return GraphRecurrentClassifier(
            121, DirectedGraph(3, (0, 0, 1, 2), (0, 1, 2, 0), (.2, .3, .4, .5)), 2, 3,
        ).float().cpu()


def _feature(marker: int, length: int) -> np.ndarray:
    x = np.zeros((length, 121), dtype=np.float64)
    x[:, 0] = marker + 1
    x[:, 68:102] = 1
    x[1:, 102:119] = 1
    x[:, 119] = 1
    x[1:, 120] = .04
    return x


def _config(**changes):
    base = TrainConfig(seed=7, epochs=2, lr=.01, batch_size=2, parameter_ceiling=10000)
    return replace(base, **changes)


def _source(monkeypatch, *, require_release: bool = False):
    train = tuple(SimpleNamespace(
        sample_id=f"train-{i}", label=_CLASSES[i % 3], subject=i + 1, split="train",
        frame_count=(3, 1, 4, 2, 1, 3)[i],
    ) for i in range(6))
    validation = tuple(SimpleNamespace(
        sample_id=f"validation-{i}", label=_CLASSES[i], subject=21 + i, split="validation",
        frame_count=(2, 4, 1)[i],
    ) for i in range(3))
    source = IndexedPoseDevelopment(
        index=SimpleNamespace(samples=train + validation), feature_spec=None,
        classes=_CLASSES, binding_json="{}", binding_sha256=_PIN,
    )
    reads: list[tuple[str, tuple[int, ...]]] = []
    previous: list[weakref.ReferenceType[PosePartition] | None] = [None]

    def verify(self, *, expected_binding_sha256: str) -> None:
        assert self is source
        assert expected_binding_sha256 == _PIN

    def read_partition(self, *, split: str, indices: tuple[int, ...],
                       expected_binding_sha256: str) -> PosePartition:
        assert self is source
        assert expected_binding_sha256 == _PIN
        if require_release and previous[0] is not None:
            gc.collect()
            assert previous[0]() is None, "previous indexed minibatch remained live across reads"
        roster = train if split == "train" else validation
        selected = tuple(roster[i] for i in indices)
        part = PosePartition(
            observations=collate_pose_features(tuple(
                _feature((0 if split == "train" else 10) + i, entry.frame_count)
                for i, entry in zip(indices, selected, strict=True)
            )),
            targets=torch.tensor([_CLASSES.index(entry.label) for entry in selected],
                                 dtype=torch.int64, device="cpu"),
            classes=_CLASSES,
            sample_ids=tuple(entry.sample_id for entry in selected),
            subjects=tuple(entry.subject for entry in selected),
            split=split,
        )
        reads.append((split, indices))
        previous[0] = weakref.ref(part)
        return part

    monkeypatch.setattr(IndexedPoseDevelopment, "verify", verify)
    monkeypatch.setattr(IndexedPoseDevelopment, "read_partition", read_partition)
    return source, reads, previous


def _expected_epoch_batches(seed: int) -> tuple[tuple[int, ...], ...]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = []
    for _ in range(2):
        order = torch.randperm(6, generator=generator, device="cpu").tolist()
        result.extend(tuple(order[start:start + 2]) for start in range(0, 6, 2))
    return tuple(result)


@pytest.mark.parametrize("kind", ("gru", "graph"))
def test_training_reads_seeded_complete_roster_minibatches_and_validation_only(kind, monkeypatch):
    api = _api()
    source, reads, _ = _source(monkeypatch)
    run = api.train_indexed_model(
        _model(kind), source, _config(), expected_binding_sha256=_PIN,
    )

    expected_train = _expected_epoch_batches(7)
    actual_train = tuple(indices for split, indices in reads if split == "train")
    assert actual_train == expected_train
    assert run.optimizer_steps == len(expected_train) == 6
    assert 1 <= run.best_epoch <= 2
    assert not run.model.training and all(parameter.grad is None for parameter in run.model.parameters())

    expected_validation = ((0, 1), (2,)) * 2
    actual_validation = tuple(indices for split, indices in reads if split == "validation")
    assert actual_validation == expected_validation
    assert all(len(indices) <= 2 for _, indices in reads)

    generator = torch.Generator(device="cpu").manual_seed(7)
    train_ids = tuple(f"train-{i}" for i in range(6))
    expected_hashes = []
    for _ in range(2):
        order = torch.randperm(6, generator=generator, device="cpu").tolist()
        encoded = json.dumps([train_ids[i] for i in order], separators=(",", ":")).encode("utf-8")
        expected_hashes.append(hashlib.sha256(encoded).hexdigest())
    assert run.epoch_order_hashes == tuple(expected_hashes)


def test_training_releases_each_indexed_partition_before_next_read(monkeypatch):
    api = _api()
    source, _, previous = _source(monkeypatch, require_release=True)
    api.train_indexed_model(
        _model("gru"), source, _config(epochs=1), expected_binding_sha256=_PIN,
    )
    gc.collect()
    assert previous[0] is not None and previous[0]() is None


@pytest.mark.parametrize("kind", ("gru", "graph"))
def test_indexed_evaluation_reads_validation_only_without_grad_and_restores_modes(kind, monkeypatch):
    api = _api()
    source, reads, _ = _source(monkeypatch)
    model = _model(kind)
    model.train()
    model.readout.eval()
    modes = tuple(module.training for module in model.modules())
    original = model.forward_padded

    def checked(batch):
        assert not torch.is_grad_enabled()
        assert all(not module.training for module in model.modules())
        return original(batch)

    monkeypatch.setattr(model, "forward_padded", checked)
    metric = api.evaluate_indexed(
        model, source, expected_binding_sha256=_PIN, batch_size=2,
    )
    assert 0 <= metric.macro_f1 <= 1 and 0 <= metric.balanced_accuracy <= 1
    assert reads == [("validation", (0, 1)), ("validation", (2,))]
    assert tuple(module.training for module in model.modules()) == modes


def test_indexed_training_api_has_no_eager_partition_or_final_test_switches():
    api = _api()
    assert tuple(inspect.signature(api.train_indexed_model).parameters) == (
        "model", "source", "config", "expected_binding_sha256",
    )
    assert tuple(inspect.signature(api.evaluate_indexed).parameters) == (
        "model", "source", "expected_binding_sha256", "batch_size",
    )
    for function in (api.train_indexed_model, api.evaluate_indexed):
        assert not ({"train", "validation", "split", "final_test", "fallback"}
                    & set(inspect.signature(function).parameters))
