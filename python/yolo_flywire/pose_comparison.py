"""Pinned four-arm CPU development execution; no final-test or topology claim API."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import logging
import math
from pathlib import Path
from typing import Any

import torch

from .graphs import DirectedGraph, graph_fingerprint, random_sparse_graph
from .models import GRUClassifier, GraphRecurrentClassifier
from .ntu_io import _canonical_json
from .pose_bundle import _digest, _json
from .pose_development import PreparedPoseDevelopment, _runtime, _tensor_record, load_pose_development
from .pose_extract import ExtractionSpec
from .pose_features import PoseFeatureSpec
from .pose_training import (
    PaddedModel, PosePartition, _positive_integer, _seeded_cpu,
    _validate_config, _validate_partition, evaluate_padded, train_padded_model,
)
from .provenance import _context, verify_controls
from .train import TrainConfig, _state_hash

_LOGGER = logging.getLogger(__name__)
_FAMILIES = ("gru", "random_graph", "rewired", "flywire")
_INPUT_DIM = 121
_SOURCE_FILES = (
    "__init__.py", "pose_comparison.py", "pose_development.py", "pose_bundle.py",
    "pose_extract.py", "pose_backend.py", "ntu_io.py", "schema.py",
    "pose_features.py", "pose_batches.py", "pose_training.py", "train.py", "eval.py",
    "graphs.py", "flywire.py", "provenance.py", "models/__init__.py",
    "models/gru.py", "models/graph_rnn.py", "models/_padded.py",
)


@dataclass(frozen=True)
class PoseComparisonSpec:
    """One explicit development condition for every arm; no validation-driven defaults."""

    seeds: tuple[int, ...]
    epochs: int
    lr: float
    batch_size: int
    max_updates: int
    parameter_ceiling: int
    gru_hidden_dim: int
    graph_node_dim: int


def _validate_spec(config: PoseComparisonSpec) -> None:
    if type(config) is not PoseComparisonSpec:
        raise ValueError("an explicit PoseComparisonSpec is required")
    if (type(config.seeds) is not tuple or not config.seeds
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in config.seeds)
            or len(set(config.seeds)) != len(config.seeds)):
        raise ValueError("seeds must be an ordered tuple of unique integers in [0,2**32)")
    for name in ("epochs", "batch_size", "max_updates", "parameter_ceiling",
                 "gru_hidden_dim", "graph_node_dim"):
        _positive_integer(getattr(config, name), name)
    if type(config.lr) not in (int, float):
        raise ValueError("lr must be finite and positive")
    try:
        valid_lr = math.isfinite(config.lr) and config.lr > 0
    except OverflowError as exc:
        raise ValueError("lr exceeds finite range") from exc
    if not valid_lr:
        raise ValueError("lr must be finite and positive")
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("CPU float32 construction is required; no implicit dtype repair")


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _path(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)):
        raise ValueError("graph inputs require file paths, not constructed objects")
    path = Path(value)
    if path.is_symlink() or not path.is_file():
        raise ValueError("graph input must be a regular non-symlink file")
    return path


def _read_pinned(path: str | Path, expected: str, name: str) -> bytes:
    pin = _digest(expected, name)
    raw = _path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin:
        raise ValueError(f"{name} does not match the independent pin")
    return raw


def _graph_record(data: Any) -> DirectedGraph:
    if type(data) is not dict or set(data) != {"num_nodes", "src", "dst", "weight"}:
        raise ValueError("malformed control graph fields")
    if type(data["num_nodes"]) is not int:
        raise ValueError("control node count must be an integer")
    if any(type(data[name]) is not list for name in ("src", "dst", "weight")):
        raise ValueError("control edges must be JSON arrays")
    if any(type(v) is not int for name in ("src", "dst") for v in data[name]):
        raise ValueError("control endpoints must be integers")
    if any(type(v) not in (int, float) for v in data["weight"]):
        raise ValueError("control weights must be numeric, not coerced")
    return DirectedGraph(data["num_nodes"], tuple(data["src"]),
                         tuple(data["dst"]), tuple(data["weight"]))


def _graph_inputs(protocol_path: str | Path, source: str | Path, controls: str | Path,
                  protocol_pin: str, control_pin: str, config: PoseComparisonSpec):
    protocol = _json(_read_pinned(protocol_path, protocol_pin, "topology protocol SHA-256"))
    bundle = _json(_read_pinned(controls, control_pin, "control bundle SHA-256"))
    if type(protocol) is not dict or type(bundle) is not dict:
        raise ValueError("topology protocol and control bundle must be JSON objects")
    budget = {name: getattr(config, name) for name in ("epochs", "max_updates", "parameter_ceiling")}
    # Canonical comparison also rejects bool/int and float/int aliases.
    if (_canonical_json(protocol.get("seeds")) != _canonical_json(config.seeds)
            or _canonical_json(protocol.get("budget")) != _canonical_json(budget)):
        raise ValueError("execution seeds and budget must match the pinned topology protocol")
    csv_path = _path(source)
    _read_pinned(csv_path, protocol.get("flywire_connectivity_sha256"), "connectivity SHA-256")
    selection, identity = _context(protocol, csv_path)
    _read_pinned(csv_path, protocol["flywire_connectivity_sha256"], "connectivity SHA-256")
    if _canonical_json(bundle.get("identity")) != _canonical_json(identity):
        raise ValueError("controls require the exact source/protocol/generator/runtime identity")
    if type(bundle.get("format_version")) is not int or bundle["format_version"] != 1:
        raise ValueError("unsupported control bundle format")
    try:
        graphs = {}
        for seed in config.seeds:
            control = bundle["controls"][str(seed)]
            if (type(control["successful_swaps"]) is not int
                    or control["successful_swaps"] != identity["swaps"]):
                raise ValueError("control successful-swap budget mismatch")
            graphs[seed] = _graph_record(control["graph"])
        fingerprints = verify_controls(bundle, selection.graph, identity)
    except (KeyError, TypeError) as exc:
        raise ValueError("malformed or incomplete control bundle") from exc
    if fingerprints != protocol["rewired_graph_fingerprints"]:
        raise ValueError("control graphs do not match the frozen fingerprints")
    return selection, identity, graphs


def _execution_identity() -> dict[str, Any]:
    package = Path(__file__).parent
    return {"source_sha256": {name: hashlib.sha256((package / name).read_bytes()).hexdigest()
                               for name in _SOURCE_FILES}, "runtime": _runtime(),
            "device": "cpu", "dtype": "float32", "input_dim": _INPUT_DIM,
            "random_graph_policy": "loop-free-uniform-edge-sample; same-N-E; unit-weights; run-seed",
            "graph_weight_policy": "original-recurrence; no-weight-normalization"}


def _adjacency_hash(model: PaddedModel) -> str | None:
    if type(model) is GRUClassifier:
        return None
    return _tensor_record(model.adjacency, torch.float32, "<f4")["sha256"]


def _expected_orders(part: PosePartition, seed: int, epochs: int) -> tuple[str, ...]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return tuple(_hash_json([part.sample_ids[i] for i in torch.randperm(
        len(part.sample_ids), generator=generator, device="cpu").tolist()]) for _ in range(epochs))


@dataclass(frozen=True)
class _ArmPlan:
    """Preflight evidence and reconstruction inputs; never model/tensor ownership."""

    seed: int
    family: str
    graph: DirectedGraph | None
    training: TrainConfig
    orders: tuple[str, ...]
    parameter_count: int
    topology_fingerprint: str
    adjacency_sha256: str | None
    initial_state_hash: str


def _construct_arm_model(family: str, graph: DirectedGraph | None,
                         config: PoseComparisonSpec, classes: int, seed: int) -> PaddedModel:
    with _seeded_cpu(seed), torch.device("cpu"):
        if family == "gru" and graph is None:
            return GRUClassifier(_INPUT_DIM, config.gru_hidden_dim, classes)
        if family in _FAMILIES[1:] and type(graph) is DirectedGraph:
            return GraphRecurrentClassifier(_INPUT_DIM, graph, config.graph_node_dim, classes)
    raise ValueError("unsupported preflight arm construction")


def _validate_arm_model(model: PaddedModel, prepared: PreparedPoseDevelopment,
                        training: TrainConfig, parameter_count: int) -> None:
    for role in ("train", "validation"):
        _validate_partition(model, getattr(prepared, role), role)
    _validate_config(model, training)
    if sum(p.numel() for p in model.parameters()) != parameter_count:
        raise ValueError("constructed model parameter count differs from the preflight architecture")


def _preflight_models(prepared: PreparedPoseDevelopment, graph: DirectedGraph,
                      rewired: dict[int, DirectedGraph], config: PoseComparisonSpec) -> tuple[_ArmPlan, ...]:
    size, classes = len(prepared.train.sample_ids), len(prepared.train.classes)
    updates = config.epochs * ((size + config.batch_size - 1) // config.batch_size)
    if config.max_updates != updates:
        raise ValueError("max_updates must equal the complete-roster epoch/minibatch budget")
    h, d, n = config.gru_hidden_dim, config.graph_node_dim, graph.num_nodes
    # Reject oversized widths before allocation; check constructed counts below too.
    counts = {"gru": 3 * h * (_INPUT_DIM + h + 2) + classes * (h + 1),
              "graph": n * d * (_INPUT_DIM + 1) + 2 * d * d + classes * (d + 1)}
    if max(counts.values()) > config.parameter_ceiling:
        raise ValueError("an arm exceeds the common parameter ceiling")
    plans = []
    for seed in config.seeds:
        variants = {"random_graph": random_sparse_graph(n, graph.num_edges, seed),
                    "rewired": rewired[seed], "flywire": graph}
        training = TrainConfig(seed=seed, epochs=config.epochs, lr=config.lr,
                               batch_size=config.batch_size, parameter_ceiling=config.parameter_ceiling)
        orders = _expected_orders(prepared.train, seed, config.epochs)
        for family in _FAMILIES:
            topology = None if family == "gru" else variants[family]
            model = _construct_arm_model(family, topology, config, classes, seed)
            count = counts["gru" if family == "gru" else "graph"]
            _validate_arm_model(model, prepared, training, count)
            fingerprint = "none" if family == "gru" else graph_fingerprint(topology)
            plans.append(_ArmPlan(seed, family, topology, training, orders, count, fingerprint,
                                  _adjacency_hash(model), _state_hash(model)))
            # Do not keep the previous model alive while allocating the next one.
            del model
    return tuple(plans)


def _execute_arm(plan: _ArmPlan, prepared: PreparedPoseDevelopment,
                 config: PoseComparisonSpec, pin: str) -> dict[str, Any]:
    """Keep each model and trained-run handle local to one completed arm."""
    seed, family, training, orders = plan.seed, plan.family, plan.training, plan.orders
    count, fingerprint, adjacency = (plan.parameter_count, plan.topology_fingerprint,
                                      plan.adjacency_sha256)
    model = _construct_arm_model(family, plan.graph, config, len(prepared.train.classes), seed)
    _validate_arm_model(model, prepared, training, count)
    if _state_hash(model) != plan.initial_state_hash or _adjacency_hash(model) != adjacency:
        raise ValueError("reconstructed model differs from its preflight state or adjacency")
    _LOGGER.info("seed=%s family=%s start epochs=%s updates=%s", seed, family,
                 config.epochs, config.max_updates)
    prepared.verify()
    run = train_padded_model(model, prepared.train, prepared.validation, training)
    prepared.verify()
    if (run.model is not model or type(run.optimizer_steps) is not int
            or run.optimizer_steps != config.max_updates or run.epoch_order_hashes != orders):
        raise ValueError("arm execution differs from the required model/update/sample-order condition")
    if type(run.best_epoch) is not int or not 1 <= run.best_epoch <= config.epochs:
        raise ValueError("selected epoch is outside the full training budget")
    if _state_hash(model) != run.state_hash or _adjacency_hash(model) != adjacency:
        raise ValueError("selected learned state or fixed adjacency changed")
    metrics = asdict(evaluate_padded(model, prepared.validation, batch_size=config.batch_size))
    if (any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
            for v in metrics.values()) or metrics["macro_f1"] != run.best_validation_macro_f1):
        raise ValueError("selected checkpoint validation metrics are invalid or inconsistent")
    prepared.verify()
    if _state_hash(model) != run.state_hash or _adjacency_hash(model) != adjacency:
        raise ValueError("validation changed selected learned state or fixed adjacency")
    row = {
        "seed": seed, "family": family, "input_binding_sha256": pin,
        "parameter_count": count, "topology_fingerprint": fingerprint,
        "adjacency_sha256": adjacency, "state_hash": run.state_hash,
        "best_epoch": run.best_epoch, "validation_metrics": metrics,
        "optimizer_steps": run.optimizer_steps, "epoch_order_hashes": list(run.epoch_order_hashes),
        "training_config": asdict(training), "training_config_hash": _hash_json(asdict(training)),
    }
    _LOGGER.info("seed=%s family=%s completed updates=%s", seed, family, run.optimizer_steps)
    return row


def run_pose_comparison(
    bundle: str | Path, *, root: str | Path, inventory: dict[str, Any],
    extraction_spec: ExtractionSpec, feature_spec: PoseFeatureSpec, classes: tuple[str, ...],
    expected_manifest_sha256: str, expected_encoder_hash: str, expected_binding_sha256: str,
    topology_protocol: str | Path, connectivity: str | Path, controls: str | Path,
    expected_topology_sha256: str, expected_controls_sha256: str,
    config: PoseComparisonSpec,
) -> dict[str, Any]:
    """Execute all four development arms after checking every input and budget.

    Requires independently retained pins and private read-only source trees. This
    materializes full partitions and is not hostile-Python or filesystem isolation.
    No partial report, file publication, automatic retry or final-test access.
    """
    _validate_spec(config)
    pin = _digest(expected_binding_sha256, "prepared binding SHA-256")
    execution = _execution_identity()
    selection, identity, rewired = _graph_inputs(
        topology_protocol, connectivity, controls, expected_topology_sha256,
        expected_controls_sha256, config,
    )
    prepared = load_pose_development(
        bundle, root=root, inventory=inventory, extraction_spec=extraction_spec,
        feature_spec=feature_spec, classes=classes,
        expected_manifest_sha256=expected_manifest_sha256, expected_encoder_hash=expected_encoder_hash,
    )
    prepared.verify()
    if prepared.binding_sha256 != pin:
        raise ValueError("prepared binding does not match the independent pin")
    plans = _preflight_models(prepared, selection.graph, rewired, config)
    rows = [_execute_arm(plan, prepared, config, pin) for plan in plans]
    if _execution_identity() != execution:
        raise ValueError("execution source or runtime identity changed during comparison")
    paired = []
    for seed in config.seeds:
        scores = {row["family"]: row["validation_metrics"]["macro_f1"]
                  for row in rows if row["seed"] == seed}
        paired.append({"seed": seed, "flywire_macro_f1": scores["flywire"],
                       "rewired_macro_f1": scores["rewired"],
                       "difference": scores["flywire"] - scores["rewired"]})
    report = {
        "format_version": 1, "evidence_scope": "development_validation_only",
        "final_test_evaluated": False, "topology_claim_evaluated": False,
        "prepared_binding_sha256": pin, "input_binding": prepared.descriptor(),
        "topology_protocol_sha256": expected_topology_sha256,
        "control_bundle_sha256": expected_controls_sha256,
        "graph_provenance": {"identity": identity, "node_types": list(selection.node_types)},
        "execution": execution, "config": asdict(config), "config_hash": _hash_json(asdict(config)),
        "arms": rows, "paired_validation": paired,
    }
    # Return an independent JSON-compatible record, not model or tensor handles.
    return _json(_canonical_json(report).encode("utf-8"))
