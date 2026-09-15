from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import math
import random
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .data import PreparedData, PreparationBinding
from .evaluation import (
    dropout_recovery_metrics,
    early_prediction_auc,
    macro_f1,
    prefix_predictions,
    retention_metrics,
)
from .graph import adjacency_fingerprint
from .models.graph_ssm import GraphSSM
from .models.graph_tcn import GraphTCN
from .models.gru import StreamingGRU
from .models.selective_ssm import ssm_spec_fingerprint
from .models.ssm_only import SSMOnly
from .protocol import ExperimentProtocol


@dataclass(frozen=True)
class RunResult:
    model_kind: str
    seed: int
    architecture_fingerprint: str
    sample_order_fingerprint: str
    parameter_count: int
    update_count: int
    best_epoch: int
    best_validation_macro_f1: float
    full_sequence_macro_f1: float
    early_macro_f1_by_ratio: tuple[tuple[float, float], ...]
    early_prediction_auc: float
    retention_auc: float
    dropout_recovery_rate: float


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def architecture_fingerprint(model_kind: str, protocol: ExperimentProtocol) -> str:
    if model_kind not in protocol.model_kinds:
        raise ValueError(f"model kind is outside frozen roster: {model_kind!r}")
    config = protocol.model(model_kind)
    common: dict[str, Any] = {
        "id": "pose-graph-ssm-v1-architecture",
        "kind": model_kind,
        "num_classes": len(protocol.actions),
        "input_shape": [25, 15],
        "hidden_size": config.hidden_size,
        "blocks": config.blocks,
    }
    if model_kind == "gru":
        common.update({"cell": "GRUCell", "input_width": 375, "standard_biases": True})
    elif model_kind == "ssm_only":
        common.update(
            {
                "input_projection": "Linear(375,64,bias=False)",
                "ssm_spec_hash": ssm_spec_fingerprint(),
                "pooling": "none",
            }
        )
    elif model_kind == "graph_tcn":
        common.update(
            {
                "adjacency_hash": adjacency_fingerprint(),
                "input_projection": "Linear(15,64,bias=False)",
                "graph_block": "human-graph-v1",
                "temporal": {"kind": "causal-tcn", "kernel_size": 3, "dilations": [1, 2]},
                "pooling": "joint-attention-v1",
            }
        )
    elif model_kind == "graph_ssm":
        common.update(
            {
                "adjacency_hash": adjacency_fingerprint(),
                "input_projection": "Linear(15,64,bias=False)",
                "graph_block": "human-graph-v1",
                "ssm_spec_hash": ssm_spec_fingerprint(),
                "state_shape": "per-joint",
                "pooling": "joint-attention-v1",
            }
        )
    else:  # pragma: no cover - roster check above is exhaustive.
        raise ValueError(f"unsupported model kind: {model_kind!r}")
    return _hash_json(common)


def _build_model(model_kind: str, protocol: ExperimentProtocol):
    config = protocol.model(model_kind)
    num_classes = len(protocol.actions)
    if model_kind == "gru":
        return StreamingGRU(num_classes=num_classes, width=config.hidden_size)
    if model_kind == "ssm_only":
        return SSMOnly(num_classes=num_classes, width=config.hidden_size, blocks=config.blocks)
    if model_kind == "graph_tcn":
        return GraphTCN(num_classes=num_classes, width=config.hidden_size)
    if model_kind == "graph_ssm":
        return GraphSSM(num_classes=num_classes, width=config.hidden_size)
    raise ValueError(f"unsupported model kind: {model_kind!r}")


def _validate_binding(prepared: PreparedData, protocol: ExperimentProtocol) -> None:
    binding = prepared.binding
    if binding.feature_stats_hash != prepared.standardizer.fingerprint:
        raise ValueError("feature-statistics binding mismatch")
    expected_label_map = tuple((action, index) for index, action in enumerate(protocol.actions))
    if binding.label_map != expected_label_map:
        raise ValueError("label-map binding mismatch")
    if binding.label_map_hash != PreparationBinding.hash_label_map(expected_label_map):
        raise ValueError("label-map hash mismatch")
    if binding.adjacency_hash != adjacency_fingerprint():
        raise ValueError("adjacency binding mismatch")
    if binding.ssm_spec_hash != ssm_spec_fingerprint():
        raise ValueError("SSM specification binding mismatch")
    if protocol.dataset_content_hash is not None and protocol.dataset_content_hash != binding.dataset_content_hash:
        raise ValueError("dataset-content binding mismatch")
    if protocol.split_hash is not None and protocol.split_hash != binding.split_hash:
        raise ValueError("split binding mismatch")
    if protocol.feature_stats_hash is not None and protocol.feature_stats_hash != binding.feature_stats_hash:
        raise ValueError("feature-statistics binding mismatch")
    if protocol.label_map_hash is not None and protocol.label_map_hash != binding.label_map_hash:
        raise ValueError("label-map binding mismatch")
    if protocol.adjacency_hash is not None and protocol.adjacency_hash != binding.adjacency_hash:
        raise ValueError("adjacency protocol pin mismatch")
    if protocol.ssm_spec_hash is not None and protocol.ssm_spec_hash != binding.ssm_spec_hash:
        raise ValueError("SSM protocol pin mismatch")
    if protocol.length_quartiles is not None and protocol.length_quartiles != binding.length_quartiles:
        raise ValueError("length-quartile binding mismatch")
    if not prepared.train_samples or not prepared.validation_samples:
        raise ValueError("training requires non-empty development splits")


def _planned_order(prepared: PreparedData, seed: int, protocol: ExperimentProtocol) -> tuple[int, ...]:
    result: list[int] = []
    for epoch in range(protocol.training.epochs):
        order = list(range(len(prepared.train_samples)))
        random.Random((seed << 32) + epoch).shuffle(order)
        for index in order:
            if len(result) >= protocol.training.max_updates:
                return tuple(result)
            result.append(index)
    return tuple(result)


def _sample_order_fingerprint(prepared: PreparedData, seed: int, protocol: ExperimentProtocol) -> str:
    order = _planned_order(prepared, seed, protocol)
    return _hash_json(
        {
            "seed": seed,
            "epochs": protocol.training.epochs,
            "max_updates": protocol.training.max_updates,
            "sample_ids": [prepared.train_samples[index].sample_id for index in order],
        }
    )


def _tensor(features: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(features, dtype=np.float32)).unsqueeze(0)


def _validation_predictions(model, prepared: PreparedData, *, num_classes: int) -> tuple[float, torch.Tensor, torch.Tensor]:
    truth: list[int] = []
    predicted: list[int] = []
    with torch.no_grad():
        for sample in prepared.validation_samples:
            logits = model(_tensor(sample.features))
            if not torch.isfinite(logits).all().item():
                raise ValueError("validation produced nonfinite logits")
            truth.append(sample.label)
            predicted.append(int(torch.argmax(logits, dim=1).item()))
    truth_tensor = torch.tensor(truth, dtype=torch.int64)
    pred_tensor = torch.tensor(predicted, dtype=torch.int64)
    return macro_f1(truth_tensor, pred_tensor, num_classes=num_classes), truth_tensor, pred_tensor


def _secondary_metrics(model, prepared: PreparedData, protocol: ExperimentProtocol) -> tuple[
    float,
    tuple[tuple[float, float], ...],
    float,
    float,
    float,
]:
    num_classes = len(protocol.actions)
    truths = torch.tensor([sample.label for sample in prepared.validation_samples], dtype=torch.int64)
    predictions_by_ratio: dict[float, list[int]] = {ratio: [] for ratio in protocol.observation_ratios}
    retention_values: list[float] = []
    recovery_values: list[float] = []

    for sample in prepared.validation_samples:
        sequence = torch.from_numpy(np.asarray(sample.features, dtype=np.float32))
        prefixes = prefix_predictions(model, sequence, ratios=protocol.observation_ratios)
        for item in prefixes:
            predictions_by_ratio[item.observation_ratio].append(int(torch.argmax(item.logits).item()))
        retention = retention_metrics(
            model,
            sequence,
            true_label=sample.label,
            observation_ratio=protocol.retention_ratio,
            horizon=protocol.retention_horizon,
        )
        recovery = dropout_recovery_metrics(
            model,
            sequence,
            observation_ratio=protocol.retention_ratio,
            burst=protocol.dropout_burst,
            recovery_horizon=protocol.recovery_horizon,
        )
        retention_values.append(retention.retention_auc)
        recovery_values.append(recovery.recovery_rate)

    ratio_scores: list[tuple[float, float]] = []
    for ratio in protocol.observation_ratios:
        predictions = torch.tensor(predictions_by_ratio[ratio], dtype=torch.int64)
        ratio_scores.append((ratio, macro_f1(truths, predictions, num_classes=num_classes)))
    early_auc = early_prediction_auc(
        tuple(ratio for ratio, _ in ratio_scores),
        tuple(score for _, score in ratio_scores),
    )
    full_score = dict(ratio_scores)[1.0]
    return (
        full_score,
        tuple(ratio_scores),
        early_auc,
        float(sum(retention_values) / len(retention_values)),
        float(sum(recovery_values) / len(recovery_values)),
    )


def train_validation_arm(
    model_kind: str,
    seed: int,
    prepared: PreparedData,
    protocol: ExperimentProtocol,
) -> RunResult:
    if model_kind not in protocol.model_kinds:
        raise ValueError("model kind is outside frozen roster")
    if type(seed) is not int or seed not in protocol.seeds:
        raise ValueError("seed is outside frozen seed roster")
    _validate_binding(prepared, protocol)

    architecture_hash = architecture_fingerprint(model_kind, protocol)
    if protocol.model_fingerprints is not None:
        expected = dict(protocol.model_fingerprints)[model_kind]
        if architecture_hash != expected:
            raise ValueError("architecture fingerprint does not match frozen protocol pin")

    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        torch.manual_seed(seed)
        model = _build_model(model_kind, protocol).to(device="cpu", dtype=torch.float32)
        parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        if parameter_count > protocol.parameter_ceiling:
            raise ValueError(
                f"model exceeds shared parameter ceiling: {parameter_count} > {protocol.parameter_ceiling}"
            )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=protocol.training.learning_rate,
            weight_decay=protocol.training.weight_decay,
        )

        planned = _planned_order(prepared, seed, protocol)
        order_fingerprint = _sample_order_fingerprint(prepared, seed, protocol)
        update_count = 0
        best_score = -math.inf
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        cursor = 0
        num_classes = len(protocol.actions)

        for epoch in range(protocol.training.epochs):
            model.train()
            epoch_updates = min(len(prepared.train_samples), len(planned) - cursor)
            for _ in range(epoch_updates):
                sample = prepared.train_samples[planned[cursor]]
                cursor += 1
                optimizer.zero_grad(set_to_none=True)
                logits = model(_tensor(sample.features))
                target = torch.tensor([sample.label], dtype=torch.int64)
                loss = F.cross_entropy(logits, target)
                if not torch.isfinite(loss).item():
                    raise ValueError("training produced nonfinite loss")
                loss.backward()
                optimizer.step()
                update_count += 1
            model.eval()
            validation_score, _, _ = _validation_predictions(model, prepared, num_classes=num_classes)
            if validation_score > best_score:
                best_score = validation_score
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
            if cursor >= len(planned):
                break

        if best_state is None or not math.isfinite(best_score):
            raise ValueError("training failed to produce a validation checkpoint")
        if update_count != len(planned):
            raise ValueError("training did not consume the frozen update schedule")
        model.load_state_dict(best_state)
        model.eval()
        full_score, early_scores, early_auc, retention_auc, recovery_rate = _secondary_metrics(
            model, prepared, protocol
        )
        return RunResult(
            model_kind=model_kind,
            seed=seed,
            architecture_fingerprint=architecture_hash,
            sample_order_fingerprint=order_fingerprint,
            parameter_count=parameter_count,
            update_count=update_count,
            best_epoch=best_epoch,
            best_validation_macro_f1=float(best_score),
            full_sequence_macro_f1=float(full_score),
            early_macro_f1_by_ratio=early_scores,
            early_prediction_auc=float(early_auc),
            retention_auc=float(retention_auc),
            dropout_recovery_rate=float(recovery_rate),
        )
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
