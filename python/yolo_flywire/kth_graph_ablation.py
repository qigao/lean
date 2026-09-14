"""Development-only KTH graph architecture ablation over a verified cached pose bundle."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .flywire import load_visual_type_graph
from .graphs import diagnostic_graph, graph_fingerprint
from .kth_development import _read_json
from .kthreuse import _prepare_source
from .models import GraphDiagnosticClassifier
from .ntu_io import _canonical_json, _hash_json, _publish_exclusive
from .pose_indexed_training import evaluate_indexed, train_indexed_model
from .train import TrainConfig

_VARIANTS = (
    "normalized_dense_mean_d2",
    "normalized_dense_flatten_d2",
    "normalized_selected_mean_d2",
    "normalized_selected_flatten_d2",
)


def _same(actual: Any, expected: Any, context: str) -> None:
    if _canonical_json(actual) != _canonical_json(expected):
        raise ValueError(f"KTH graph ablation {context} mismatch")


def _selected_graph(frozen: dict[str, Any], connectivity: Path,
                    selected_graph_record: dict[str, Any]):
    raw = connectivity.read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    if source_sha != frozen.get("flywire_connectivity_sha256"):
        raise ValueError("KTH graph ablation connectivity differs from frozen SHA-256")
    try:
        seed_types = tuple(selected_graph_record["seed_types"])
        threshold = int(selected_graph_record["min_seed_synapses"])
        expected_types = list(selected_graph_record["node_types"])
        expected_fingerprint = selected_graph_record["graph_fingerprint"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("KTH graph ablation selected-graph record malformed") from exc
    selection = load_visual_type_graph(
        connectivity,
        seed_types=seed_types,
        min_seed_synapses=threshold,
    )
    fingerprint = graph_fingerprint(selection.graph)
    if fingerprint != expected_fingerprint or fingerprint != frozen.get("selected_graph_fingerprint"):
        raise ValueError("KTH graph ablation selected graph fingerprint mismatch")
    if list(selection.node_types) != expected_types:
        raise ValueError("KTH graph ablation selected graph node roster mismatch")
    expected_counts = {
        "num_nodes": selection.graph.num_nodes,
        "num_edges": selection.graph.num_edges,
        "num_diagonal_edges": sum(a == b for a, b in zip(selection.graph.src, selection.graph.dst)),
    }
    for name, actual in expected_counts.items():
        if selected_graph_record.get(name) != actual:
            raise ValueError(f"KTH graph ablation selected graph {name} mismatch")
    return selection, seed_types, threshold, source_sha


def _variant_model(name: str, *, graph, seed_node_indices: tuple[int, ...], classes: int):
    if name not in _VARIANTS:
        raise ValueError("KTH graph ablation variant is not frozen")
    selected = "selected" in name
    flatten = "flatten" in name
    return GraphDiagnosticClassifier(
        121,
        graph,
        2,
        classes,
        input_policy="selected_nodes" if selected else "dense_all_nodes",
        input_node_indices=seed_node_indices if selected else (),
        readout_policy="flatten" if flatten else "mean",
    )


def run_seed(protocol: str | Path, *, protocol_sha256: str,
             source_manifest: dict[str, Any], bundle: str | Path,
             execution_config: dict[str, Any], connectivity: str | Path,
             selected_graph_record: dict[str, Any], seed: int,
             output: str | Path) -> dict[str, Any]:
    (protocol_path, protocol_pin, frozen, source_record, config, prepared,
     binding_pin, manifest_pin) = _prepare_source(
        protocol,
        protocol_sha256=protocol_sha256,
        source_manifest=source_manifest,
        bundle=bundle,
        execution_config=execution_config,
    )
    if type(seed) is not int or seed not in config.seeds:
        raise ValueError("KTH graph ablation seed not in frozen protocol")
    connectivity_path = Path(connectivity).absolute()
    if connectivity_path.is_symlink() or not connectivity_path.is_file():
        raise ValueError("KTH graph ablation connectivity must be a regular file")
    selection, seed_types, threshold, connectivity_sha = _selected_graph(
        frozen, connectivity_path, selected_graph_record,
    )
    normalized = diagnostic_graph(
        selection.graph,
        weight_policy="log1p_incoming_l1",
        diagonal_policy="drop",
    )
    node_types = tuple(selection.node_types)
    seed_node_indices = tuple(node_types.index(name) for name in seed_types)
    training = TrainConfig(
        seed=seed,
        epochs=config.epochs,
        lr=config.lr,
        batch_size=config.batch_size,
        parameter_ceiling=config.parameter_ceiling,
    )
    rows = []
    expected_orders = None
    for name in _VARIANTS:
        model = _variant_model(
            name,
            graph=normalized,
            seed_node_indices=seed_node_indices,
            classes=len(prepared.classes),
        )
        parameter_count = model.parameter_count()
        if parameter_count > config.parameter_ceiling:
            raise ValueError(f"KTH graph ablation {name} exceeds parameter ceiling")
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
            raise ValueError("KTH graph ablation selected checkpoint metric mismatch")
        if run.optimizer_steps != config.max_updates:
            raise ValueError("KTH graph ablation optimizer-step budget mismatch")
        if expected_orders is None:
            expected_orders = run.epoch_order_hashes
        elif run.epoch_order_hashes != expected_orders:
            raise ValueError("KTH graph ablation variants used different sample orders")
        rows.append({
            "seed": seed,
            "variant": name,
            "node_dim": model.node_dim,
            "input_policy": model.input_policy,
            "readout_policy": model.readout_policy,
            "parameter_count": parameter_count,
            "best_epoch": run.best_epoch,
            "optimizer_steps": run.optimizer_steps,
            "state_hash": run.state_hash,
            "validation_metrics": metrics,
            "epoch_order_hashes": list(run.epoch_order_hashes),
        })
        del model

    record = {
        "format_version": 1,
        "kind": "kth_real_graph_ablation_seed",
        "evidence_scope": "kth_real_development_graph_ablation_only",
        "final_test_evaluated": False,
        "topology_claim_evaluated": False,
        "seed": seed,
        "protocol_sha256": protocol_pin,
        "source_manifest_hash": source_record["source_manifest_hash"],
        "prepared_binding_sha256": binding_pin,
        "aggregate_manifest_sha256": manifest_pin,
        "execution_config": asdict(config),
        "execution_config_hash": _hash_json(asdict(config)),
        "connectivity_sha256": connectivity_sha,
        "selected_graph_fingerprint": graph_fingerprint(selection.graph),
        "normalized_graph_fingerprint": graph_fingerprint(normalized),
        "normalization": {
            "weight_policy": "log1p_incoming_l1",
            "diagonal_policy": "drop",
        },
        "seed_types": list(seed_types),
        "seed_node_indices": list(seed_node_indices),
        "min_seed_synapses": threshold,
        "variants": rows,
    }
    result = json.loads(_canonical_json(record))
    destination = Path(output).absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError("KTH graph ablation output already exists")
    destination.mkdir(parents=True)
    _publish_exclusive(destination / "seed_report.json", result)
    return result


def reduce_records(records: tuple[dict[str, Any], ...], *, expected_seeds: tuple[int, ...]) -> dict[str, Any]:
    if (type(records) is not tuple or type(expected_seeds) is not tuple
            or not records or not expected_seeds or len(set(expected_seeds)) != len(expected_seeds)):
        raise ValueError("KTH graph ablation reducer input invalid")
    by_seed = {}
    for record in records:
        if (type(record) is not dict or record.get("format_version") != 1
                or record.get("kind") != "kth_real_graph_ablation_seed"):
            raise ValueError("KTH graph ablation seed record kind/version invalid")
        seed = record.get("seed")
        if type(seed) is not int or seed in by_seed:
            raise ValueError("KTH graph ablation duplicate/invalid seed")
        by_seed[seed] = record
    if set(by_seed) != set(expected_seeds):
        raise ValueError("KTH graph ablation seed roster mismatch")
    ordered = [by_seed[seed] for seed in expected_seeds]
    reference = ordered[0]
    common = (
        "protocol_sha256", "source_manifest_hash", "prepared_binding_sha256",
        "aggregate_manifest_sha256", "execution_config", "execution_config_hash",
        "connectivity_sha256", "selected_graph_fingerprint", "normalized_graph_fingerprint",
        "normalization", "seed_types", "seed_node_indices", "min_seed_synapses",
    )
    for record in ordered[1:]:
        for name in common:
            _same(record.get(name), reference.get(name), f"cross-seed {name}")
    variant_rows = {name: [] for name in _VARIANTS}
    all_rows = []
    for seed, record in zip(expected_seeds, ordered, strict=True):
        rows = record.get("variants")
        if type(rows) is not list or len(rows) != len(_VARIANTS):
            raise ValueError("KTH graph ablation variant roster size invalid")
        by_variant = {row.get("variant"): row for row in rows if type(row) is dict}
        if set(by_variant) != set(_VARIANTS):
            raise ValueError("KTH graph ablation variant roster mismatch")
        for name in _VARIANTS:
            row = by_variant[name]
            if row.get("seed") != seed or row.get("optimizer_steps") != 40:
                raise ValueError("KTH graph ablation seed/update evidence invalid")
            macro = row.get("validation_metrics", {}).get("macro_f1")
            if type(macro) not in (int, float) or not math.isfinite(macro) or not 0 <= macro <= 1:
                raise ValueError("KTH graph ablation macro-F1 invalid")
            variant_rows[name].append(float(macro))
            all_rows.append(row)
    summary = {
        name: {
            "macro_f1_by_seed": [
                {"seed": seed, "macro_f1": score}
                for seed, score in zip(expected_seeds, variant_rows[name], strict=True)
            ],
            "mean_macro_f1": sum(variant_rows[name]) / len(variant_rows[name]),
        }
        for name in _VARIANTS
    }
    result = {name: reference[name] for name in common}
    result.update({
        "format_version": 1,
        "kind": "kth_real_graph_ablation",
        "evidence_scope": "kth_real_development_graph_ablation_only",
        "final_test_evaluated": False,
        "topology_claim_evaluated": False,
        "seeds": list(expected_seeds),
        "variants": all_rows,
        "summary": summary,
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
    seed_cmd.add_argument("--selected-graph", type=Path, required=True)
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
                source_manifest=_read_json(args.source_manifest, "KTH graph ablation source manifest"),
                bundle=args.bundle,
                execution_config=_read_json(args.execution_config, "KTH graph ablation execution config"),
                connectivity=args.connectivity,
                selected_graph_record=_read_json(args.selected_graph, "KTH selected graph"),
                seed=args.seed,
                output=args.output,
            )
            print(_canonical_json({"seed": result["seed"], "variants": len(result["variants"])}))
        else:
            protocol = _read_json(args.protocol, "KTH graph ablation frozen protocol")
            seeds = tuple(protocol.get("seeds", ()))
            paths = sorted(args.reports_root.glob("**/seed_report.json"))
            records = tuple(_read_json(path, "KTH graph ablation seed report") for path in paths)
            reduced = reduce_records(records, expected_seeds=seeds)
            destination = args.output.absolute()
            if destination.exists() or destination.is_symlink():
                raise ValueError("KTH graph ablation reduction output already exists")
            destination.mkdir(parents=True)
            _publish_exclusive(destination / "graph_ablation_report.json", reduced)
            print(_canonical_json({
                "seeds": reduced["seeds"],
                "mean_macro_f1": {name: row["mean_macro_f1"] for name, row in reduced["summary"].items()},
            }))
        return 0
    except (OSError, ValueError, TypeError, RuntimeError, ImportError, UnicodeError) as exc:
        parser.exit(1, f"KTH graph ablation error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
