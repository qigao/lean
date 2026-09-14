"""Reuse verified KTH pose shards and execute the frozen comparison one seed per runner."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

from .kth_development import (
    _comparison_spec,
    _file_pin,
    _kth_code_identity,
    _path,
    _read_json,
)
from .kth_indexed_development import load_kth_indexed_pose_development
from .kth_pose_index import verify_kth_source
from .ntu_io import _canonical_json, _hash_json, _publish_exclusive
from .pose_bundle import _digest
from .pose_comparison import (
    _FAMILIES,
    _execute_arm,
    _execution_identity,
    _graph_inputs,
    _preflight_models,
)
from .pose_features import PoseFeatureSpec


_COMMON_SEED_FIELDS = (
    "prepared_binding_sha256",
    "protocol_sha256",
    "controls_sha256",
    "config_hash",
    "aggregate_manifest_sha256",
    "dataset_content_hash",
    "split_hash",
    "source_manifest_hash",
    "execution",
    "graph_provenance",
    "source_run_id",
    "source_head_sha",
    "kth_source_code_sha256",
    "reuse_code_sha256",
)


def _reuse_code_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _same(actual: Any, expected: Any, context: str) -> None:
    if _canonical_json(actual) != _canonical_json(expected):
        raise ValueError(f"KTH reuse {context} mismatch")


def _reduce_seed_records(records: tuple[dict[str, Any], ...], *,
                         expected_seeds: tuple[int, ...]) -> dict[str, Any]:
    if type(records) is not tuple or not records:
        raise ValueError("KTH reuse seed records must be a nonempty tuple")
    if (type(expected_seeds) is not tuple or not expected_seeds
            or any(type(seed) is not int for seed in expected_seeds)
            or len(set(expected_seeds)) != len(expected_seeds)):
        raise ValueError("KTH reuse expected seed roster invalid")

    by_seed: dict[int, dict[str, Any]] = {}
    for record in records:
        if type(record) is not dict or record.get("format_version") != 1 \
                or record.get("kind") != "kth_real_seed_execution":
            raise ValueError("KTH reuse seed record kind/version invalid")
        seed = record.get("seed")
        if type(seed) is not int or seed in by_seed:
            raise ValueError("KTH reuse seed roster contains invalid/duplicate seed")
        by_seed[seed] = record
    if set(by_seed) != set(expected_seeds) or len(by_seed) != len(expected_seeds):
        raise ValueError("KTH reuse seed roster differs from frozen protocol")

    ordered = [by_seed[seed] for seed in expected_seeds]
    reference = ordered[0]
    for record in ordered[1:]:
        if record.get("prepared_binding_sha256") != reference.get("prepared_binding_sha256"):
            raise ValueError("KTH reuse prepared binding differs across seed reports")
        for name in _COMMON_SEED_FIELDS:
            if name == "prepared_binding_sha256":
                continue
            _same(record.get(name), reference.get(name), f"cross-seed {name}")

    combined_arms: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    for seed, record in zip(expected_seeds, ordered, strict=True):
        arms = record.get("arms")
        if type(arms) is not list or len(arms) != len(_FAMILIES):
            raise ValueError("KTH reuse seed report must contain exactly four arms")
        by_family: dict[str, dict[str, Any]] = {}
        for arm in arms:
            if type(arm) is not dict or arm.get("seed") != seed:
                raise ValueError("KTH reuse arm seed differs from seed report")
            family = arm.get("family")
            if family not in _FAMILIES or family in by_family:
                raise ValueError("KTH reuse arm family roster invalid")
            if arm.get("input_binding_sha256") != reference.get("prepared_binding_sha256"):
                raise ValueError("KTH reuse arm prepared binding mismatch")
            if type(arm.get("optimizer_steps")) is not int or arm["optimizer_steps"] <= 0:
                raise ValueError("KTH reuse arm optimizer-step evidence invalid")
            metrics = arm.get("validation_metrics")
            if type(metrics) is not dict or type(metrics.get("macro_f1")) not in (int, float):
                raise ValueError("KTH reuse arm validation metrics invalid")
            by_family[family] = arm
        if set(by_family) != set(_FAMILIES):
            raise ValueError("KTH reuse seed report four-arm roster incomplete")
        combined_arms.extend(by_family[family] for family in _FAMILIES)
        expected_pair = {
            "seed": seed,
            "flywire_macro_f1": by_family["flywire"]["validation_metrics"]["macro_f1"],
            "rewired_macro_f1": by_family["rewired"]["validation_metrics"]["macro_f1"],
            "difference": (
                by_family["flywire"]["validation_metrics"]["macro_f1"]
                - by_family["rewired"]["validation_metrics"]["macro_f1"]
            ),
        }
        _same(record.get("paired_validation"), expected_pair, f"seed {seed} paired validation")
        paired.append(expected_pair)

    result = {name: reference.get(name) for name in _COMMON_SEED_FIELDS}
    result["arms"] = combined_arms
    result["paired_validation"] = paired
    return result


def _prepare_source(protocol: str | Path, *, protocol_sha256: str,
                    source_manifest: dict[str, Any], bundle: str | Path,
                    execution_config: dict[str, Any]):
    protocol_path, protocol_pin = _file_pin(protocol, protocol_sha256, "frozen KTH protocol")
    frozen = _read_json(protocol_path, "frozen KTH protocol")
    if frozen.get("protocol_id") != "v1-kth-yolo11n-real-ci":
        raise ValueError("wrong protocol for KTH cached reuse")
    if (frozen.get("final_test_decoded") is not False
            or frozen.get("classifier_evaluated") is not False
            or frozen.get("final_test_used_for_selection") is not False):
        raise ValueError("KTH cached reuse crossed final-test boundary")
    source_record = verify_kth_source(source_manifest)
    for name in ("dataset_content_hash", "split_hash", "input_inventory_hash", "source_manifest_hash"):
        if frozen.get(name) != source_record[name]:
            raise ValueError(f"KTH cached reuse frozen protocol {name} differs from source manifest")
    config = _comparison_spec(frozen, execution_config)
    try:
        feature_spec = PoseFeatureSpec(**frozen["pose_feature_spec"])
        classes = tuple(frozen["task_labels"])
        encoder_pin = _digest(frozen["pose_encoder_hash"], "KTH pose encoder SHA-256")
    except (KeyError, TypeError) as exc:
        raise ValueError("KTH cached reuse protocol missing feature/class/encoder fields") from exc
    bundle_path = Path(bundle).absolute()
    manifest_path = bundle_path / "manifest.json"
    if bundle_path.is_symlink() or not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("KTH cached aggregate pose bundle invalid")
    manifest_pin = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    prepared = load_kth_indexed_pose_development(
        bundle_path,
        source_manifest=source_record,
        protocol=frozen,
        feature_spec=feature_spec,
        classes=classes,
        expected_manifest_sha256=manifest_pin,
        expected_encoder_hash=encoder_pin,
    )
    binding_pin = prepared.binding_sha256
    prepared.verify(expected_binding_sha256=binding_pin)
    return protocol_path, protocol_pin, frozen, source_record, config, prepared, binding_pin, manifest_pin


def run_seed(protocol: str | Path, *, protocol_sha256: str,
             source_manifest: dict[str, Any], bundle: str | Path,
             execution_config: dict[str, Any], connectivity: str | Path,
             controls: str | Path, controls_sha256: str, seed: int,
             source_run_id: int, source_head_sha: str,
             output: str | Path) -> dict[str, Any]:
    if type(seed) is not int or type(source_run_id) is not int or source_run_id <= 0:
        raise ValueError("KTH cached reuse seed/source run id invalid")
    if type(source_head_sha) is not str or len(source_head_sha) != 40:
        raise ValueError("KTH cached reuse source head SHA invalid")
    controls_path, controls_pin = _file_pin(controls, controls_sha256, "control bundle")
    connectivity_path = _path(connectivity, "FlyWire connectivity")
    (protocol_path, protocol_pin, frozen, source_record, config, prepared,
     binding_pin, manifest_pin) = _prepare_source(
        protocol,
        protocol_sha256=protocol_sha256,
        source_manifest=source_manifest,
        bundle=bundle,
        execution_config=execution_config,
    )
    if seed not in config.seeds:
        raise ValueError("KTH cached reuse seed not in frozen protocol")

    execution = _execution_identity()
    selection, identity, rewired = _graph_inputs(
        protocol_path,
        connectivity_path,
        controls_path,
        protocol_pin,
        controls_pin,
        config,
    )
    prepared.verify(expected_binding_sha256=binding_pin)
    plans = _preflight_models(prepared, selection.graph, rewired, config, binding_pin)
    selected = tuple(plan for plan in plans if plan.seed == seed)
    if len(selected) != len(_FAMILIES) or tuple(plan.family for plan in selected) != _FAMILIES:
        raise ValueError("KTH cached reuse seed preflight did not produce exact four-arm roster")
    rows = [_execute_arm(plan, prepared, config, binding_pin) for plan in selected]
    if _execution_identity() != execution:
        raise ValueError("KTH cached reuse comparison core identity changed during seed execution")
    scores = {row["family"]: row["validation_metrics"]["macro_f1"] for row in rows}
    pair = {
        "seed": seed,
        "flywire_macro_f1": scores["flywire"],
        "rewired_macro_f1": scores["rewired"],
        "difference": scores["flywire"] - scores["rewired"],
    }
    record = {
        "format_version": 1,
        "kind": "kth_real_seed_execution",
        "evidence_scope": "kth_real_development_validation_only",
        "final_test_evaluated": False,
        "topology_claim_evaluated": False,
        "seed": seed,
        "source_run_id": source_run_id,
        "source_head_sha": source_head_sha,
        "prepared_binding_sha256": binding_pin,
        "protocol_sha256": protocol_pin,
        "controls_sha256": controls_pin,
        "config_hash": _hash_json(asdict(config)),
        "aggregate_manifest_sha256": manifest_pin,
        "dataset_content_hash": source_record["dataset_content_hash"],
        "split_hash": source_record["split_hash"],
        "source_manifest_hash": source_record["source_manifest_hash"],
        "kth_source_code_sha256": _kth_code_identity(),
        "reuse_code_sha256": _reuse_code_hash(),
        "execution": execution,
        "graph_provenance": {"identity": identity, "node_types": list(selection.node_types)},
        "arms": rows,
        "paired_validation": pair,
    }
    result = json.loads(_canonical_json(record))
    destination = Path(output).absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError("KTH cached seed output already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    _publish_exclusive(destination / "seed_report.json", result)
    return result


def reduce_seed_reports(protocol: str | Path, *, protocol_sha256: str,
                        source_manifest: dict[str, Any], execution_config: dict[str, Any],
                        reports: tuple[dict[str, Any], ...], output: str | Path) -> dict[str, Any]:
    protocol_path, protocol_pin = _file_pin(protocol, protocol_sha256, "frozen KTH protocol")
    frozen = _read_json(protocol_path, "frozen KTH protocol")
    source_record = verify_kth_source(source_manifest)
    for name in ("dataset_content_hash", "split_hash", "source_manifest_hash"):
        if frozen.get(name) != source_record[name]:
            raise ValueError(f"KTH cached reducer protocol {name} differs from source manifest")
    config = _comparison_spec(frozen, execution_config)
    reduced = _reduce_seed_records(reports, expected_seeds=config.seeds)
    if reduced["protocol_sha256"] != protocol_pin:
        raise ValueError("KTH cached reducer protocol pin differs from seed reports")
    if reduced["config_hash"] != _hash_json(asdict(config)):
        raise ValueError("KTH cached reducer config hash differs from seed reports")
    for name in ("dataset_content_hash", "split_hash", "source_manifest_hash"):
        if reduced[name] != source_record[name]:
            raise ValueError(f"KTH cached reducer {name} differs from source manifest")

    comparison = {
        "format_version": 1,
        "evidence_scope": "development_validation_only",
        "final_test_evaluated": False,
        "topology_claim_evaluated": False,
        "prepared_binding_sha256": reduced["prepared_binding_sha256"],
        "topology_protocol_sha256": protocol_pin,
        "control_bundle_sha256": reduced["controls_sha256"],
        "graph_provenance": reduced["graph_provenance"],
        "execution": reduced["execution"],
        "config": asdict(config),
        "config_hash": reduced["config_hash"],
        "arms": reduced["arms"],
        "paired_validation": reduced["paired_validation"],
    }
    report = {
        "format_version": 1,
        "kind": "kth_real_development_execution",
        "evidence_scope": "kth_real_development_validation_only",
        "final_test_decoded": False,
        "final_test_evaluated": False,
        "classifier_evaluated_on_final_test": False,
        "topology_claim_evaluated": False,
        "ntu_confirmatory_rule_evaluated": False,
        "source_run_id": reduced["source_run_id"],
        "source_head_sha": reduced["source_head_sha"],
        "protocol_sha256": protocol_pin,
        "controls_sha256": reduced["controls_sha256"],
        "dataset_content_hash": source_record["dataset_content_hash"],
        "split_hash": source_record["split_hash"],
        "source_manifest_hash": source_record["source_manifest_hash"],
        "yolo_weights_sha256": frozen["yolo_weights_sha256"],
        "aggregate_manifest_sha256": reduced["aggregate_manifest_sha256"],
        "prepared_binding_sha256": reduced["prepared_binding_sha256"],
        "kth_source_code_sha256": reduced["kth_source_code_sha256"],
        "reuse_code_sha256": reduced["reuse_code_sha256"],
        "execution_config": asdict(config),
        "execution_config_hash": reduced["config_hash"],
        "comparison": comparison,
    }
    result = json.loads(_canonical_json(report))
    destination = Path(output).absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError("KTH cached reduced output already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    _publish_exclusive(destination / "development_report.json", result)
    return result


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
    seed_cmd.add_argument("--source-run-id", type=int, required=True)
    seed_cmd.add_argument("--source-head-sha", required=True)
    seed_cmd.add_argument("--output", type=Path, required=True)

    reduce_cmd = commands.add_parser("reduce")
    reduce_cmd.add_argument("--protocol", type=Path, required=True)
    reduce_cmd.add_argument("--protocol-sha256", required=True)
    reduce_cmd.add_argument("--source-manifest", type=Path, required=True)
    reduce_cmd.add_argument("--execution-config", type=Path, required=True)
    reduce_cmd.add_argument("--reports-root", type=Path, required=True)
    reduce_cmd.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "seed":
            report = run_seed(
                args.protocol,
                protocol_sha256=args.protocol_sha256,
                source_manifest=_read_json(args.source_manifest, "KTH source manifest"),
                bundle=args.bundle,
                execution_config=_read_json(args.execution_config, "KTH execution config"),
                connectivity=args.connectivity,
                controls=args.controls,
                controls_sha256=args.controls_sha256,
                seed=args.seed,
                source_run_id=args.source_run_id,
                source_head_sha=args.source_head_sha,
                output=args.output,
            )
            print(_canonical_json({"seed": report["seed"], "arms": len(report["arms"])}))
        else:
            paths = tuple(sorted(args.reports_root.rglob("seed_report.json"), key=lambda path: path.as_posix()))
            reports = tuple(_read_json(path, "KTH seed report") for path in paths)
            report = reduce_seed_reports(
                args.protocol,
                protocol_sha256=args.protocol_sha256,
                source_manifest=_read_json(args.source_manifest, "KTH source manifest"),
                execution_config=_read_json(args.execution_config, "KTH execution config"),
                reports=reports,
                output=args.output,
            )
            print(_canonical_json({
                "source_run_id": report["source_run_id"],
                "seeds": [row["seed"] for row in report["comparison"]["paired_validation"]],
                "final_test_evaluated": False,
            }))
        return 0
    except (OSError, ValueError, TypeError, ImportError, RuntimeError, UnicodeError) as exc:
        parser.exit(1, f"KTH cached reuse error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
