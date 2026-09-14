"""Run KTH public real-data development only; final test has no execution entry."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .kth_indexed_development import load_kth_indexed_pose_development
from .kth_pose_index import verify_kth_source
from .ntu_io import _canonical_json, _hash_json, _publish_exclusive
from .pose_bundle import _digest
from .pose_comparison import (
    PoseComparisonSpec, _execute_arm, _execution_identity, _graph_inputs,
    _preflight_models, _validate_spec,
)
from .pose_features import PoseFeatureSpec

_DIGEST = re.compile(r"[0-9a-f]{64}")
_CONFIG_FIELDS = {
    "seeds", "epochs", "lr", "batch_size", "max_updates", "parameter_ceiling",
    "gru_hidden_dim", "graph_node_dim",
}
_KTH_FILES = (
    "kth_source.py", "kth_extract.py", "kth_aggregate.py", "kth_pose_index.py",
    "kth_indexed_development.py", "kth_development.py",
)


def _path(value: str | Path, name: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{name} must be a file path")
    path = Path(value).absolute()
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError(f"{name} path must not traverse symlinks")
    if not path.is_file():
        raise ValueError(f"{name} must be an existing regular file")
    return path


def _pin(value: Any, name: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be canonical lowercase SHA-256")
    return value


def _file_pin(path: str | Path, expected: str, name: str) -> tuple[Path, str]:
    file_path = _path(path, name)
    pin = _pin(expected, f"{name} SHA-256")
    if hashlib.sha256(file_path.read_bytes()).hexdigest() != pin:
        raise ValueError(f"{name} bytes do not match independent pin")
    return file_path, pin


def _read_json(path: str | Path, name: str) -> dict[str, Any]:
    file_path = _path(path, name)
    try:
        value = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {name} JSON") from exc
    if type(value) is not dict:
        raise ValueError(f"{name} must be a JSON object")
    return value


def _comparison_spec(protocol: dict[str, Any], config: dict[str, Any]) -> PoseComparisonSpec:
    if type(protocol) is not dict or type(config) is not dict or set(config) != _CONFIG_FIELDS:
        raise ValueError("KTH execution config fields differ from exact contract")
    if _canonical_json(config["seeds"]) != _canonical_json(protocol.get("seeds")):
        raise ValueError("KTH execution seeds differ from frozen protocol")
    budget = {name: config[name] for name in ("epochs", "max_updates", "parameter_ceiling")}
    if _canonical_json(budget) != _canonical_json(protocol.get("budget")):
        raise ValueError("KTH execution budget differs from frozen protocol")
    if type(config["seeds"]) is not list:
        raise ValueError("KTH execution seeds must be JSON array")
    spec = PoseComparisonSpec(
        seeds=tuple(config["seeds"]), epochs=config["epochs"], lr=config["lr"],
        batch_size=config["batch_size"], max_updates=config["max_updates"],
        parameter_ceiling=config["parameter_ceiling"],
        gru_hidden_dim=config["gru_hidden_dim"], graph_node_dim=config["graph_node_dim"],
    )
    _validate_spec(spec)
    return spec


def _kth_code_identity() -> dict[str, str]:
    root = Path(__file__).parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in _KTH_FILES}


def _run_bound_comparison(source, *, expected_binding_sha256: str,
                          topology_protocol: str | Path, connectivity: str | Path,
                          controls: str | Path, expected_topology_sha256: str,
                          expected_controls_sha256: str,
                          config: PoseComparisonSpec) -> dict[str, Any]:
    _validate_spec(config)
    pin = _digest(expected_binding_sha256, "KTH prepared binding SHA-256")
    execution = _execution_identity()
    selection, identity, rewired = _graph_inputs(
        topology_protocol, connectivity, controls,
        expected_topology_sha256, expected_controls_sha256, config,
    )
    source.verify(expected_binding_sha256=pin)
    if source.binding_sha256 != pin:
        raise ValueError("KTH prepared binding differs from independent pin")
    plans = _preflight_models(source, selection.graph, rewired, config, pin)
    rows = [_execute_arm(plan, source, config, pin) for plan in plans]
    if _execution_identity() != execution:
        raise ValueError("KTH comparison core identity changed during execution")
    paired = []
    for seed in config.seeds:
        scores = {row["family"]: row["validation_metrics"]["macro_f1"]
                  for row in rows if row["seed"] == seed}
        paired.append({
            "seed": seed, "flywire_macro_f1": scores["flywire"],
            "rewired_macro_f1": scores["rewired"],
            "difference": scores["flywire"] - scores["rewired"],
        })
    return json.loads(_canonical_json({
        "format_version": 1, "evidence_scope": "development_validation_only",
        "final_test_evaluated": False, "topology_claim_evaluated": False,
        "prepared_binding_sha256": pin, "input_binding": source.descriptor(),
        "topology_protocol_sha256": expected_topology_sha256,
        "control_bundle_sha256": expected_controls_sha256,
        "graph_provenance": {"identity": identity, "node_types": list(selection.node_types)},
        "execution": execution, "config": asdict(config),
        "config_hash": _hash_json(asdict(config)), "arms": rows,
        "paired_validation": paired,
    }))


def run_kth_development(protocol: str | Path, *, protocol_sha256: str,
                        source_manifest: dict[str, Any], bundle: str | Path,
                        execution_config: dict[str, Any], connectivity: str | Path,
                        controls: str | Path, controls_sha256: str,
                        output: str | Path) -> dict[str, Any]:
    protocol_path, protocol_pin = _file_pin(protocol, protocol_sha256, "frozen KTH protocol")
    controls_path, controls_pin = _file_pin(controls, controls_sha256, "control bundle")
    connectivity_path = _path(connectivity, "FlyWire connectivity")
    frozen = _read_json(protocol_path, "frozen KTH protocol")
    if frozen.get("protocol_id") != "v1-kth-yolo11n-real-ci":
        raise ValueError("wrong protocol for KTH development")
    if (frozen.get("final_test_decoded") is not False
            or frozen.get("classifier_evaluated") is not False
            or frozen.get("final_test_used_for_selection") is not False):
        raise ValueError("KTH frozen protocol crossed final-test boundary")
    source_record = verify_kth_source(source_manifest)
    for name in ("dataset_content_hash", "split_hash", "input_inventory_hash", "source_manifest_hash"):
        if frozen.get(name) != source_record[name]:
            raise ValueError(f"KTH frozen protocol {name} differs from source manifest")
    config = _comparison_spec(frozen, execution_config)
    try:
        feature_spec = PoseFeatureSpec(**frozen["pose_feature_spec"])
        classes = tuple(frozen["task_labels"])
        encoder_pin = _digest(frozen["pose_encoder_hash"], "KTH pose encoder SHA-256")
    except (KeyError, TypeError) as exc:
        raise ValueError("KTH frozen protocol missing feature/class/encoder fields") from exc
    bundle_path = Path(bundle).absolute()
    manifest_path = bundle_path / "manifest.json"
    if bundle_path.is_symlink() or not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("KTH aggregate pose bundle invalid")
    manifest_pin = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    prepared = load_kth_indexed_pose_development(
        bundle_path, source_manifest=source_record, protocol=frozen,
        feature_spec=feature_spec, classes=classes,
        expected_manifest_sha256=manifest_pin, expected_encoder_hash=encoder_pin,
    )
    binding_pin = prepared.binding_sha256
    prepared.verify(expected_binding_sha256=binding_pin)
    comparison = _run_bound_comparison(
        prepared, expected_binding_sha256=binding_pin,
        topology_protocol=protocol_path, connectivity=connectivity_path, controls=controls_path,
        expected_topology_sha256=protocol_pin, expected_controls_sha256=controls_pin,
        config=config,
    )
    if (comparison.get("evidence_scope") != "development_validation_only"
            or comparison.get("final_test_evaluated") is not False
            or comparison.get("topology_claim_evaluated") is not False
            or comparison.get("prepared_binding_sha256") != binding_pin):
        raise ValueError("KTH comparison report crossed evidence boundary")
    destination = Path(output).absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError("KTH development output already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    config_record = asdict(config)
    report = {
        "format_version": 1, "kind": "kth_real_development_execution",
        "evidence_scope": "kth_real_development_validation_only",
        "final_test_decoded": False, "final_test_evaluated": False,
        "classifier_evaluated_on_final_test": False,
        "topology_claim_evaluated": False,
        "ntu_confirmatory_rule_evaluated": False,
        "protocol_sha256": protocol_pin, "controls_sha256": controls_pin,
        "dataset_content_hash": source_record["dataset_content_hash"],
        "split_hash": source_record["split_hash"],
        "source_manifest_hash": source_record["source_manifest_hash"],
        "yolo_weights_sha256": frozen["yolo_weights_sha256"],
        "aggregate_manifest_sha256": manifest_pin,
        "prepared_binding_sha256": binding_pin,
        "kth_source_code_sha256": _kth_code_identity(),
        "execution_config": config_record,
        "execution_config_hash": _hash_json(config_record),
        "comparison": comparison,
    }
    result = json.loads(_canonical_json(report))
    _publish_exclusive(destination / "development_report.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--protocol-sha256", required=True)
    run.add_argument("--source-manifest", type=Path, required=True)
    run.add_argument("--bundle", type=Path, required=True)
    run.add_argument("--execution-config", type=Path, required=True)
    run.add_argument("--connectivity", type=Path, required=True)
    run.add_argument("--controls", type=Path, required=True)
    run.add_argument("--controls-sha256", required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run_kth_development(
            args.protocol, protocol_sha256=args.protocol_sha256,
            source_manifest=_read_json(args.source_manifest, "KTH source manifest"),
            bundle=args.bundle,
            execution_config=_read_json(args.execution_config, "KTH execution config"),
            connectivity=args.connectivity, controls=args.controls,
            controls_sha256=args.controls_sha256, output=args.output,
        )
        print(_canonical_json({
            "evidence_scope": report["evidence_scope"],
            "final_test_evaluated": False,
            "prepared_binding_sha256": report["prepared_binding_sha256"],
        }))
        return 0
    except (OSError, ValueError, TypeError, ImportError, RuntimeError) as exc:
        parser.exit(1, f"KTH development error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
