"""Development-only corrected FlyWire-vs-rewired comparison on cached real KTH pose data."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any

from .graphs import diagnostic_graph, graph_fingerprint
from .kth_development import _file_pin, _path, _read_json
from .kthreuse import _prepare_source
from .models import GraphDiagnosticClassifier
from .ntu_io import _canonical_json, _hash_json, _publish_exclusive
from .pose_comparison import _graph_inputs
from .pose_indexed_training import evaluate_indexed, train_indexed_model
from .train import TrainConfig

_FAMILIES = ("rewired", "flywire")
_WEIGHT_POLICY = "log1p_incoming_l1"
_DIAGONAL_POLICY = "drop"
_INPUT_POLICY = "dense_all_nodes"
_READOUT_POLICY = "flatten"
_NODE_DIM = 2


def _same(actual: Any, expected: Any, context: str) -> None:
    if _canonical_json(actual) != _canonical_json(expected):
        raise ValueError(f"corrected KTH topology {context} mismatch")


def _architecture(parameter_count: int) -> dict[str, Any]:
    return {
        "input_policy": _INPUT_POLICY,
        "readout_policy": _READOUT_POLICY,
        "node_dim": _NODE_DIM,
        "parameter_count": parameter_count,
    }


def _model(graph, *, classes: int) -> GraphDiagnosticClassifier:
    return GraphDiagnosticClassifier(
        121,
        graph,
        _NODE_DIM,
        classes,
        input_policy=_INPUT_POLICY,
        readout_policy=_READOUT_POLICY,
    )


def run_seed(
    protocol: str | Path,
    *,
    protocol_sha256: str,
    source_manifest: dict[str, Any],
    bundle: str | Path,
    execution_config: dict[str, Any],
    connectivity: str | Path,
    controls: str | Path,
    controls_sha256: str,
    seed: int,
    output: str | Path,
) -> dict[str, Any]:
    controls_path, controls_pin = _file_pin(controls, controls_sha256, "control bundle")
    connectivity_path = _path(connectivity, "FlyWire connectivity")
    (
        protocol_path,
        protocol_pin,
        _frozen,
        source_record,
        config,
        prepared,
        binding_pin,
        manifest_pin,
    ) = _prepare_source(
        protocol,
        protocol_sha256=protocol_sha256,
        source_manifest=source_manifest,
        bundle=bundle,
        execution_config=execution_config,
    )
    if type(seed) is not int or seed not in config.seeds:
        raise ValueError("corrected KTH topology seed not in frozen protocol")
    if config.graph_node_dim != _NODE_DIM:
        raise ValueError("corrected KTH topology requires frozen graph node_dim=2")

    selection, identity, rewired = _graph_inputs(
        protocol_path,
        connectivity_path,
        controls_path,
        protocol_pin,
        controls_pin,
        config,
    )
    raw_graphs = {"rewired": rewired[seed], "flywire": selection.graph}
    graphs = {
        family: diagnostic_graph(
            graph,
            weight_policy=_WEIGHT_POLICY,
            diagonal_policy=_DIAGONAL_POLICY,
        )
        for family, graph in raw_graphs.items()
    }
    training = TrainConfig(
        seed=seed,
        epochs=config.epochs,
        lr=config.lr,
        batch_size=config.batch_size,
        parameter_ceiling=config.parameter_ceiling,
    )

    rows: list[dict[str, Any]] = []
    expected_orders = None
    expected_parameter_count = None
    for family in _FAMILIES:
        model = _model(graphs[family], classes=len(prepared.classes))
        parameter_count = model.parameter_count()
        if parameter_count > config.parameter_ceiling:
            raise ValueError("corrected KTH topology model exceeds parameter ceiling")
        if expected_parameter_count is None:
            expected_parameter_count = parameter_count
        elif parameter_count != expected_parameter_count:
            raise ValueError("corrected KTH topology arms have different parameter counts")
        run = train_indexed_model(
            model,
            prepared,
            training,
            expected_binding_sha256=binding_pin,
        )
        metrics = asdict(evaluate_indexed(
            model,
            prepared,
            expected_binding_sha256=binding_pin,
            batch_size=config.batch_size,
        ))
        if metrics["macro_f1"] != run.best_validation_macro_f1:
            raise ValueError("corrected KTH topology selected checkpoint metric mismatch")
        if run.optimizer_steps != config.max_updates:
            raise ValueError("corrected KTH topology optimizer-step budget mismatch")
        if expected_orders is None:
            expected_orders = run.epoch_order_hashes
        elif run.epoch_order_hashes != expected_orders:
            raise ValueError("corrected KTH topology arms used different sample orders")
        rows.append({
            "seed": seed,
            "family": family,
            "parameter_count": parameter_count,
            "raw_topology_fingerprint": graph_fingerprint(raw_graphs[family]),
            "normalized_topology_fingerprint": graph_fingerprint(graphs[family]),
            "best_epoch": run.best_epoch,
            "optimizer_steps": run.optimizer_steps,
            "state_hash": run.state_hash,
            "validation_metrics": metrics,
            "epoch_order_hashes": list(run.epoch_order_hashes),
        })
        del model

    by_family = {row["family"]: row for row in rows}
    flywire = float(by_family["flywire"]["validation_metrics"]["macro_f1"])
    rewired_score = float(by_family["rewired"]["validation_metrics"]["macro_f1"])
    pair = {
        "seed": seed,
        "flywire_macro_f1": flywire,
        "rewired_macro_f1": rewired_score,
        "difference": flywire - rewired_score,
    }
    assert expected_parameter_count is not None
    record = {
        "format_version": 1,
        "kind": "kth_corrected_topology_seed",
        "evidence_scope": "kth_real_development_corrected_topology_only",
        "final_test_evaluated": False,
        "topology_claim_evaluated": False,
        "ntu_confirmatory_rule_evaluated": False,
        "seed": seed,
        "protocol_sha256": protocol_pin,
        "controls_sha256": controls_pin,
        "source_manifest_hash": source_record["source_manifest_hash"],
        "prepared_binding_sha256": binding_pin,
        "aggregate_manifest_sha256": manifest_pin,
        "execution_config_hash": _hash_json(asdict(config)),
        "control_identity_hash": _hash_json(identity),
        "normalization": {
            "weight_policy": _WEIGHT_POLICY,
            "diagonal_policy": _DIAGONAL_POLICY,
        },
        "architecture": _architecture(expected_parameter_count),
        "arms": rows,
        "paired_validation": pair,
    }
    result = json.loads(_canonical_json(record))
    destination = Path(output).absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError("corrected KTH topology output already exists")
    destination.mkdir(parents=True)
    _publish_exclusive(destination / "seed_report.json", result)
    return result


def reduce_records(
    records: tuple[dict[str, Any], ...],
    *,
    expected_seeds: tuple[int, ...],
) -> dict[str, Any]:
    if (
        type(records) is not tuple
        or type(expected_seeds) is not tuple
        or not records
        or not expected_seeds
        or any(type(seed) is not int for seed in expected_seeds)
        or len(set(expected_seeds)) != len(expected_seeds)
    ):
        raise ValueError("corrected KTH topology reducer input invalid")

    by_seed: dict[int, dict[str, Any]] = {}
    for record in records:
        if (
            type(record) is not dict
            or record.get("format_version") != 1
            or record.get("kind") != "kth_corrected_topology_seed"
        ):
            raise ValueError("corrected KTH topology seed record kind/version invalid")
        seed = record.get("seed")
        if type(seed) is not int or seed in by_seed:
            raise ValueError("corrected KTH topology duplicate/invalid seed")
        by_seed[seed] = record
    if set(by_seed) != set(expected_seeds):
        raise ValueError("corrected KTH topology seed roster mismatch")

    ordered = [by_seed[seed] for seed in expected_seeds]
    reference = ordered[0]
    common = (
        "protocol_sha256",
        "controls_sha256",
        "source_manifest_hash",
        "prepared_binding_sha256",
        "aggregate_manifest_sha256",
        "execution_config_hash",
        "control_identity_hash",
        "normalization",
        "architecture",
    )
    for record in ordered[1:]:
        for name in common:
            _same(record.get(name), reference.get(name), f"cross-seed {name}")

    scores = {family: [] for family in _FAMILIES}
    paired: list[dict[str, Any]] = []
    all_arms: list[dict[str, Any]] = []
    for seed, record in zip(expected_seeds, ordered, strict=True):
        arms = record.get("arms")
        if type(arms) is not list or len(arms) != len(_FAMILIES):
            raise ValueError("corrected KTH topology arm roster invalid")
        by_family = {row.get("family"): row for row in arms if type(row) is dict}
        if set(by_family) != set(_FAMILIES):
            raise ValueError("corrected KTH topology arm roster invalid")
        for family in _FAMILIES:
            row = by_family[family]
            if row.get("seed") != seed or row.get("optimizer_steps") != 40:
                raise ValueError("corrected KTH topology arm seed/update evidence invalid")
            macro = row.get("validation_metrics", {}).get("macro_f1")
            if type(macro) not in (int, float) or not math.isfinite(macro) or not 0 <= macro <= 1:
                raise ValueError("corrected KTH topology macro-F1 invalid")
            scores[family].append(float(macro))
            all_arms.append(row)
        expected_pair = {
            "seed": seed,
            "flywire_macro_f1": scores["flywire"][-1],
            "rewired_macro_f1": scores["rewired"][-1],
            "difference": scores["flywire"][-1] - scores["rewired"][-1],
        }
        _same(record.get("paired_validation"), expected_pair, f"seed {seed} paired validation")
        paired.append(expected_pair)

    mean_scores = {
        family: sum(values) / len(values)
        for family, values in scores.items()
    }
    mean_difference = sum(row["difference"] for row in paired) / len(paired)
    result = {name: reference[name] for name in common}
    result.update({
        "format_version": 1,
        "kind": "kth_corrected_topology_comparison",
        "evidence_scope": "kth_real_development_corrected_topology_only",
        "final_test_evaluated": False,
        "topology_claim_evaluated": False,
        "ntu_confirmatory_rule_evaluated": False,
        "seeds": list(expected_seeds),
        "arms": all_arms,
        "paired_validation": paired,
        "mean_macro_f1": mean_scores,
        "paired_mean_difference": mean_difference,
        "positive_difference_seeds": sum(row["difference"] > 0 for row in paired),
    })
    return json.loads(_canonical_json(result))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    seed_cmd = commands.add_parser("seed")
    seed_cmd.add_argument("--protocol", type=Path, required=True)
    seed_cmd.add_argument("--protocol-sha256", required=True)
    seed_cmd.add_argument("--source-manifest", type=Path, required=True)
    seed_cmd.add_argument("--bundle", type=Path, required=True)
    seed_cmd.add_argument("--execution-config", type=Path, required=True)
    seed_cmd.add_argument("--connectivity", type=Path, required=True)
    seed_cmd.add_argument("--controls", type=Path, required=True)
    seed_cmd.add_argument("--controls-sha256", required=True)
    seed_cmd.add_argument("--seed", type=int, required=True)
    seed_cmd.add_argument("--output", type=Path, required=True)

    reduce_cmd = commands.add_parser("reduce")
    reduce_cmd.add_argument("--protocol", type=Path, required=True)
    reduce_cmd.add_argument("--reports-root", type=Path, required=True)
    reduce_cmd.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "seed":
            result = run_seed(
                args.protocol,
                protocol_sha256=args.protocol_sha256,
                source_manifest=_read_json(args.source_manifest, "corrected KTH topology source manifest"),
                bundle=args.bundle,
                execution_config=_read_json(args.execution_config, "corrected KTH topology execution config"),
                connectivity=args.connectivity,
                controls=args.controls,
                controls_sha256=args.controls_sha256,
                seed=args.seed,
                output=args.output,
            )
            print(_canonical_json(result["paired_validation"]))
        else:
            protocol = _read_json(args.protocol, "corrected KTH topology frozen protocol")
            seeds = tuple(protocol.get("seeds", ()))
            reports = tuple(
                _read_json(path, "corrected KTH topology seed report")
                for path in sorted(args.reports_root.glob("**/seed_report.json"))
            )
            report = reduce_records(reports, expected_seeds=seeds)
            destination = args.output.absolute()
            if destination.exists() or destination.is_symlink():
                raise ValueError("corrected KTH topology reduction output already exists")
            destination.mkdir(parents=True)
            _publish_exclusive(destination / "corrected_topology_report.json", report)
            print(_canonical_json({
                "mean_macro_f1": report["mean_macro_f1"],
                "paired_mean_difference": report["paired_mean_difference"],
                "positive_difference_seeds": report["positive_difference_seeds"],
            }))
        return 0
    except (OSError, ValueError, TypeError, RuntimeError, ImportError, UnicodeError) as exc:
        parser.exit(1, f"corrected KTH topology error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
