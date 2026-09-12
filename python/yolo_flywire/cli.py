from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .eval import evaluate
from .features import FeatureSpec, encode_sequence
from .graphs import DirectedGraph, graph_fingerprint, random_sparse_graph, rewire_degree_preserving
from .manifests import RunManifest, aggregate_topology_evidence
from .models import GRUClassifier, GraphRecurrentClassifier
from .synthetic import make_synthetic_dataset
from .train import TrainConfig, train_model


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(_canonical_json(value) + "\n", encoding="utf-8")


def _synthetic_tensors(seed: int) -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
    str,
    str,
]:
    samples = make_synthetic_dataset(seed=seed, samples_per_class=6, frames=8)
    labels = tuple(sorted({sample.label for sample in samples}))
    label_to_index = {label: index for index, label in enumerate(labels)}
    spec = FeatureSpec()

    buckets: dict[str, list[tuple[np.ndarray, int, str]]] = {"train": [], "validation": [], "test": []}
    for sample in samples:
        sample_index = int(sample.sample_id.rsplit("-", 1)[1])
        split = "train" if sample_index < 4 else "validation" if sample_index == 4 else "test"
        encoded = encode_sequence(sample.sequence, spec).astype(np.float32)
        buckets[split].append((encoded, label_to_index[sample.label], sample.sample_id))

    def tensor_pair(name: str) -> tuple[torch.Tensor, torch.Tensor]:
        rows = buckets[name]
        x = torch.from_numpy(np.stack([row[0] for row in rows], axis=0))
        y = torch.tensor([row[1] for row in rows], dtype=torch.long)
        return x, y

    split_payload = {
        name: [sample_id for _, _, sample_id in rows]
        for name, rows in sorted(buckets.items())
    }
    schema_payload = {
        "feature_spec": asdict(spec),
        "feature_dim": int(buckets["train"][0][0].shape[-1]),
        "labels": labels,
    }
    return (
        tensor_pair("train"),
        tensor_pair("validation"),
        tensor_pair("test"),
        _hash_json(split_payload),
        _hash_json(schema_payload),
    )


def synthetic_run(seed: int, output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    train, validation, test, split_hash, schema_hash = _synthetic_tensors(seed)

    input_dim = int(train[0].shape[-1])
    num_classes = int(torch.unique(train[1]).numel())
    model = GRUClassifier(input_dim=input_dim, hidden_dim=12, num_classes=num_classes)
    config = TrainConfig(seed=seed, epochs=20, lr=0.03, batch_size=12, parameter_ceiling=50_000)
    trained = train_model(model, train, validation, config)
    metrics = evaluate(trained.model, test)
    metric_values = asdict(metrics)

    updates_per_epoch = math.ceil(train[0].shape[0] / config.batch_size)
    config_payload = asdict(config)
    manifest = RunManifest(
        code_commit=os.environ.get("GITHUB_SHA", "working-tree"),
        dataset_id="synthetic-v0",
        split_hash=split_hash,
        observation_schema_hash=schema_hash,
        model_family="gru",
        topology_fingerprint="none",
        seed=seed,
        budget={
            "epochs": config.epochs,
            "max_updates": config.epochs * updates_per_epoch,
            "parameter_ceiling": config.parameter_ceiling,
        },
        training_config_hash=_hash_json(config_payload),
        metrics=metric_values,
        compared_families=("gru",),
        claim="temporal_model_useful",
    )
    manifest.validate_claim_boundary()
    (output / "manifest.json").write_text(manifest.canonical_json() + "\n", encoding="utf-8")
    _write_json(output / "metrics.json", metric_values)
    return 0


def _arm_family(arm: Any) -> str:
    if isinstance(arm, str):
        return arm
    if isinstance(arm, dict) and isinstance(arm.get("family"), str):
        return arm["family"]
    raise ValueError("each protocol arm must be a family string or an object with a family field")


def _validate_frozen_protocol(config: dict[str, Any]) -> tuple[str, ...]:
    required = (
        "dataset_id",
        "split_hash",
        "observation_schema_hash",
        "seeds",
        "budget",
        "primary_metric",
        "success_threshold",
        "final_test_used_for_selection",
    )
    missing = [name for name in required if config.get(name) is None]
    if missing:
        raise ValueError("protocol must freeze required fields before execution: " + ", ".join(missing))
    if config["final_test_used_for_selection"] is not False:
        raise ValueError("final test must remain sealed from model selection")
    seeds = config["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("protocol requires an explicit non-empty seed list")
    arms = config.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ValueError("comparison protocol requires an explicit non-empty arms list")
    families = tuple(_arm_family(arm) for arm in arms)

    claim = config.get("claim", "temporal_model_useful")
    if claim == "topology_specific_advantage":
        topology_required = ("yolo_version", "flywire_release", "selection_rule", "rewiring_algorithm")
        topology_missing = [name for name in topology_required if config.get(name) is None]
        if topology_missing:
            raise ValueError(
                "topology protocol must freeze required provenance before execution: "
                + ", ".join(topology_missing)
            )
        family_set = set(families)
        if "flywire" not in family_set or "rewired" not in family_set:
            raise ValueError("topology-specific claim requires matched flywire and rewired controls before training")
    return families


def _fixture_graph(config: dict[str, Any]) -> DirectedGraph:
    fixture = config.get("graph_fixture")
    if not isinstance(fixture, dict):
        raise ValueError("executable synthetic topology protocol requires graph_fixture")
    return DirectedGraph(
        num_nodes=int(fixture["num_nodes"]),
        src=tuple(int(value) for value in fixture["src"]),
        dst=tuple(int(value) for value in fixture["dst"]),
        weight=tuple(float(value) for value in fixture["weight"]),
    )


def _rewired_graph(graph: DirectedGraph, seed: int) -> DirectedGraph:
    target = max(1, min(5, graph.num_edges // 2))
    for swaps in range(target, 0, -1):
        try:
            candidate = rewire_degree_preserving(graph, seed=seed, swaps=swaps)
        except ValueError:
            continue
        if graph_fingerprint(candidate) != graph_fingerprint(graph):
            return candidate
    raise ValueError("could not construct a non-identical matched rewired control")


def _model_for_family(
    family: str,
    *,
    input_dim: int,
    num_classes: int,
    flywire_graph: DirectedGraph,
    rewired_graph: DirectedGraph,
    seed: int,
) -> tuple[torch.nn.Module, str]:
    if family == "gru":
        return GRUClassifier(input_dim=input_dim, hidden_dim=12, num_classes=num_classes), "none"
    if family == "flywire":
        return (
            GraphRecurrentClassifier(input_dim, flywire_graph, node_dim=4, num_classes=num_classes),
            graph_fingerprint(flywire_graph),
        )
    if family == "rewired":
        return (
            GraphRecurrentClassifier(input_dim, rewired_graph, node_dim=4, num_classes=num_classes),
            graph_fingerprint(rewired_graph),
        )
    if family == "random_graph":
        random_graph = random_sparse_graph(
            num_nodes=flywire_graph.num_nodes,
            num_edges=flywire_graph.num_edges,
            seed=seed,
        )
        return (
            GraphRecurrentClassifier(input_dim, random_graph, node_dim=4, num_classes=num_classes),
            graph_fingerprint(random_graph),
        )
    raise ValueError(f"unsupported comparison family: {family}")


def compare(config_path: Path, output: Path) -> int:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("protocol config must be a JSON object")
    families = _validate_frozen_protocol(config)
    claim = config.get("claim", "temporal_model_useful")

    # Non-executable real-data templates remain validation-only until their frozen
    # external inputs are present. Synthetic fixture protocols exercise the entire
    # evidence path without upgrading the result to real-data topology evidence.
    if config.get("dataset_id") != "synthetic-v0" or "graph_fixture" not in config:
        output.mkdir(parents=True, exist_ok=True)
        record = {
            "kind": "comparison_protocol_validation",
            "claim": claim,
            "arms": list(families),
            "protocol_hash": _hash_json(config),
            "execution_status": "protocol_validated_only",
        }
        _write_json(output / "manifest.json", record)
        _write_json(output / "metrics.json", {})
        return 0

    output.mkdir(parents=True, exist_ok=True)
    flywire_graph = _fixture_graph(config)
    budget = config["budget"]
    epochs = int(budget["epochs"])
    parameter_ceiling = int(budget["parameter_ceiling"])
    max_updates = int(budget["max_updates"])
    all_rows: list[dict[str, Any]] = []
    paired_metrics: dict[int, dict[str, float]] = {}

    for raw_seed in config["seeds"]:
        seed = int(raw_seed)
        train, validation, test, _, _ = _synthetic_tensors(seed)
        updates_per_epoch = math.ceil(train[0].shape[0] / 12)
        if epochs * updates_per_epoch > max_updates:
            raise ValueError("training budget max_updates would be exceeded")

        rewired = _rewired_graph(flywire_graph, seed)
        input_dim = int(train[0].shape[-1])
        num_classes = int(torch.unique(train[1]).numel())
        paired_metrics[seed] = {}

        for family in families:
            model, topology_fingerprint = _model_for_family(
                family,
                input_dim=input_dim,
                num_classes=num_classes,
                flywire_graph=flywire_graph,
                rewired_graph=rewired,
                seed=seed,
            )
            train_config = TrainConfig(
                seed=seed,
                epochs=epochs,
                lr=0.03,
                batch_size=12,
                parameter_ceiling=parameter_ceiling,
            )
            trained = train_model(model, train, validation, train_config)
            metrics = asdict(evaluate(trained.model, test))
            primary = float(metrics[config["primary_metric"]])
            paired_metrics[seed][family] = primary
            all_rows.append(
                {
                    "seed": seed,
                    "family": family,
                    "split_hash": config["split_hash"],
                    "observation_schema_hash": config["observation_schema_hash"],
                    "budget": budget,
                    "topology_fingerprint": topology_fingerprint,
                    "metrics": metrics,
                    "state_hash": trained.state_hash,
                }
            )

    evidence = aggregate_topology_evidence(
        seed_metrics=paired_metrics,
        success_threshold=float(config["success_threshold"]),
    )
    aggregate = {
        "kind": "synthetic_comparison_evidence",
        "claim": claim,
        "arms": list(families),
        "protocol_hash": _hash_json(config),
        "conclusion": evidence.conclusion,
        "mean_paired_difference": evidence.mean_paired_difference,
        "paired_differences": evidence.paired_differences,
        "success_threshold": evidence.success_threshold,
        "seed_results": all_rows,
        "evidence_scope": "synthetic_harness_only",
    }
    _write_json(output / "aggregate_report.json", aggregate)
    _write_json(output / "manifest.json", {"protocol_hash": _hash_json(config), "arms": list(families)})
    _write_json(output / "metrics.json", {"mean_paired_difference": evidence.mean_paired_difference})
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yolo-flywire")
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic = subparsers.add_parser("synthetic-run")
    synthetic.add_argument("--seed", type=int, required=True)
    synthetic.add_argument("--output", type=Path, required=True)

    comparison = subparsers.add_parser("compare")
    comparison.add_argument("--config", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "synthetic-run":
            return synthetic_run(args.seed, args.output)
        if args.command == "compare":
            return compare(args.config, args.output)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
