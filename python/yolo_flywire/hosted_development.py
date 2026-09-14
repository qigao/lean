"""Run hosted YOLO11 remote development only; final test has no execution entry."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

from .ntu_io import _canonical_json, _hash_json, _publish_exclusive
from .pose_bundle import _digest
from .pose_comparison import (
    PoseComparisonSpec, _execute_arm, _execution_identity, _graph_inputs,
    _preflight_models, _validate_spec,
)
from .pose_features import PoseFeatureSpec
from .real_development import (
    _comparison_spec, _file_pin, _path, _read_json, _require_safe_comparison,
)
from .remote_indexed_development import load_remote_indexed_pose_development
from .remote_rgb import validate_remote_manifest
from .hosted_extract import _verify_protocol_source


def _run_bound_pose_comparison(
    source, *, expected_binding_sha256: str,
    topology_protocol: str | Path, connectivity: str | Path, controls: str | Path,
    expected_topology_sha256: str, expected_controls_sha256: str,
    config: PoseComparisonSpec,
) -> dict[str, Any]:
    """Reuse the existing four-arm preflight/execution core from an already-bound source."""
    _validate_spec(config)
    pin = _digest(expected_binding_sha256, "prepared binding SHA-256")
    execution = _execution_identity()
    selection, identity, rewired = _graph_inputs(
        topology_protocol, connectivity, controls,
        expected_topology_sha256, expected_controls_sha256, config,
    )
    source.verify(expected_binding_sha256=pin)
    if source.binding_sha256 != pin:
        raise ValueError("hosted prepared binding differs from independent pin")
    plans = _preflight_models(source, selection.graph, rewired, config, pin)
    rows = [_execute_arm(plan, source, config, pin) for plan in plans]
    if _execution_identity() != execution:
        raise ValueError("hosted comparison source/runtime identity changed during execution")
    paired = []
    for seed in config.seeds:
        scores = {
            row["family"]: row["validation_metrics"]["macro_f1"]
            for row in rows if row["seed"] == seed
        }
        paired.append({
            "seed": seed,
            "flywire_macro_f1": scores["flywire"],
            "rewired_macro_f1": scores["rewired"],
            "difference": scores["flywire"] - scores["rewired"],
        })
    report = {
        "format_version": 1,
        "evidence_scope": "development_validation_only",
        "final_test_evaluated": False,
        "topology_claim_evaluated": False,
        "prepared_binding_sha256": pin,
        "input_binding": source.descriptor(),
        "topology_protocol_sha256": expected_topology_sha256,
        "control_bundle_sha256": expected_controls_sha256,
        "graph_provenance": {"identity": identity, "node_types": list(selection.node_types)},
        "execution": execution,
        "config": asdict(config),
        "config_hash": _hash_json(asdict(config)),
        "arms": rows,
        "paired_validation": paired,
    }
    return json.loads(_canonical_json(report))


def run_hosted_development(
    protocol: str | Path, *, protocol_sha256: str,
    remote_manifest: dict[str, Any], bundle: str | Path,
    execution_config: dict[str, Any], connectivity: str | Path,
    controls: str | Path, controls_sha256: str, output: str | Path,
) -> dict[str, Any]:
    """Bind aggregated remote pose evidence and execute the existing four-arm core."""
    protocol_path, protocol_pin = _file_pin(protocol, protocol_sha256, "frozen hosted protocol")
    controls_path, controls_pin = _file_pin(controls, controls_sha256, "control bundle")
    connectivity_path = _path(connectivity, "FlyWire connectivity")
    frozen = _read_json(protocol_path, "frozen hosted protocol")
    config = _comparison_spec(frozen, execution_config)
    remote = validate_remote_manifest(remote_manifest)
    _verify_protocol_source(frozen, remote)
    try:
        feature_spec = PoseFeatureSpec(**frozen["pose_feature_spec"])
        classes = tuple(frozen["task_labels"])
        encoder_pin = _digest(frozen["pose_encoder_hash"], "pose encoder SHA-256")
    except (KeyError, TypeError) as exc:
        raise ValueError("frozen hosted protocol missing feature/class/encoder fields") from exc

    bundle_path = Path(bundle).absolute()
    for component in (bundle_path, *bundle_path.parents):
        if component.is_symlink():
            raise ValueError("hosted pose bundle path must not traverse symlinks")
    manifest_path = bundle_path / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("hosted pose bundle manifest missing")
    manifest_pin = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    source = load_remote_indexed_pose_development(
        bundle_path,
        remote_manifest=remote_manifest,
        protocol=frozen,
        feature_spec=feature_spec,
        classes=classes,
        expected_manifest_sha256=manifest_pin,
        expected_encoder_hash=encoder_pin,
    )
    binding_pin = source.binding_sha256
    source.verify(expected_binding_sha256=binding_pin)
    comparison = _require_safe_comparison(_run_bound_pose_comparison(
        source,
        expected_binding_sha256=binding_pin,
        topology_protocol=protocol_path,
        connectivity=connectivity_path,
        controls=controls_path,
        expected_topology_sha256=protocol_pin,
        expected_controls_sha256=controls_pin,
        config=config,
    ), binding_pin)

    destination = Path(output).absolute()
    for component in (destination, *destination.parents):
        if component.is_symlink():
            raise ValueError("hosted development output must not traverse symlinks")
    if destination.exists():
        raise ValueError("hosted development output already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    config_record = asdict(config)
    report = {
        "format_version": 1,
        "kind": "hosted_real_development_execution",
        "evidence_scope": "hosted_real_development_validation_only",
        "final_test_decoded": False,
        "final_test_evaluated": False,
        "classifier_evaluated_on_final_test": False,
        "topology_claim_evaluated": False,
        "protocol_sha256": protocol_pin,
        "controls_sha256": controls_pin,
        "dataset_content_hash": remote.dataset_content_hash,
        "split_hash": remote.split_hash,
        "source_manifest_hash": remote.source_manifest_hash,
        "transport_manifest_hash": remote.transport_manifest_hash,
        "yolo_weights_sha256": frozen["yolo_weights_sha256"],
        "aggregate_manifest_sha256": manifest_pin,
        "prepared_binding_sha256": binding_pin,
        "execution_config": config_record,
        "execution_config_hash": _hash_json(config_record),
        "comparison": comparison,
    }
    result = json.loads(_canonical_json(report))
    _publish_exclusive(destination / "development_report.json", result)
    return result


def _read_private_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {name} JSON") from exc
    if type(value) is not dict:
        raise ValueError(f"{name} must be a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--protocol-sha256", required=True)
    run.add_argument("--remote-manifest", type=Path, required=True)
    run.add_argument("--bundle", type=Path, required=True)
    run.add_argument("--execution-config", type=Path, required=True)
    run.add_argument("--connectivity", type=Path, required=True)
    run.add_argument("--controls", type=Path, required=True)
    run.add_argument("--controls-sha256", required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run_hosted_development(
            args.protocol,
            protocol_sha256=args.protocol_sha256,
            remote_manifest=_read_private_json(args.remote_manifest, "remote manifest"),
            bundle=args.bundle,
            execution_config=_read_private_json(args.execution_config, "execution config"),
            connectivity=args.connectivity,
            controls=args.controls,
            controls_sha256=args.controls_sha256,
            output=args.output,
        )
        print(_canonical_json({
            "evidence_scope": report["evidence_scope"],
            "final_test_evaluated": False,
            "prepared_binding_sha256": report["prepared_binding_sha256"],
        }))
        return 0
    except (OSError, ValueError, TypeError, ImportError, RuntimeError) as exc:
        parser.exit(1, f"Hosted development error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
