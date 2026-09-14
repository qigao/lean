"""Run only the frozen real-data development path; final test has no entry here.

This command revalidates the byte-backed NTU/YOLO freeze, extracts train and
validation observations, binds the indexed development source, and executes the
pinned four-arm validation comparison. It never selects, decodes or evaluates a
final-test sample and does not authorize a topology-specific final claim.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .ntu_io import _canonical_json, _hash_json, _publish_exclusive
from .pose_comparison import PoseComparisonSpec, _validate_spec, run_pose_comparison
from .pose_extract import ExtractionSpec, extract_development
from .pose_features import PoseFeatureSpec
from .pose_indexed_development import load_indexed_pose_development
from .real_freeze import freeze_real_inputs

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_DIGEST = re.compile(r"[0-9a-f]{64}")
_CONFIG_FIELDS = {
    "seeds", "epochs", "lr", "batch_size", "max_updates", "parameter_ceiling",
    "gru_hidden_dim", "graph_node_dim",
}


def _snapshot(value: Any, context: str) -> Any:
    try:
        return json.loads(_canonical_json(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context} must contain canonical JSON values") from exc


def _digest(value: Any, name: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


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


def _file_pin(path: str | Path, expected: str, name: str) -> tuple[Path, str]:
    file_path = _path(path, name)
    pin = _digest(expected, f"{name} SHA-256")
    actual = hashlib.sha256(file_path.read_bytes()).hexdigest()
    if actual != pin:
        raise ValueError(f"{name} bytes do not match the independent SHA-256 pin")
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
    """Bind immutable protocol seeds/budget to explicit development hyperparameters."""
    if type(protocol) is not dict or type(config) is not dict:
        raise ValueError("protocol and development execution config must be JSON objects")
    if set(config) != _CONFIG_FIELDS:
        missing = sorted(_CONFIG_FIELDS - set(config))
        extra = sorted(set(config) - _CONFIG_FIELDS)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise ValueError("development execution config fields differ: " + "; ".join(details))
    raw_seeds = config["seeds"]
    if type(raw_seeds) is not list:
        raise ValueError("development execution seeds must be an explicit JSON array")
    if _canonical_json(raw_seeds) != _canonical_json(protocol.get("seeds")):
        raise ValueError("development execution seeds differ from the frozen protocol")
    budget = {
        "epochs": config["epochs"],
        "max_updates": config["max_updates"],
        "parameter_ceiling": config["parameter_ceiling"],
    }
    if _canonical_json(budget) != _canonical_json(protocol.get("budget")):
        raise ValueError("development execution budget differs from the frozen protocol")
    spec = PoseComparisonSpec(
        seeds=tuple(raw_seeds), epochs=config["epochs"], lr=config["lr"],
        batch_size=config["batch_size"], max_updates=config["max_updates"],
        parameter_ceiling=config["parameter_ceiling"],
        gru_hidden_dim=config["gru_hidden_dim"], graph_node_dim=config["graph_node_dim"],
    )
    _validate_spec(spec)
    return spec


def _require_safe_comparison(report: Any, binding_pin: str) -> dict[str, Any]:
    if type(report) is not dict:
        raise ValueError("development comparison must return a JSON object")
    if report.get("evidence_scope") != "development_validation_only":
        raise ValueError("comparison evidence scope is not development validation only")
    if report.get("final_test_evaluated") is not False:
        raise ValueError("development comparison attempted or claimed final-test evaluation")
    if report.get("topology_claim_evaluated") is not False:
        raise ValueError("development comparison must not emit a topology-specific final claim")
    if report.get("prepared_binding_sha256") != binding_pin:
        raise ValueError("development comparison used a different prepared binding")
    return _snapshot(report, "development comparison")


def run_real_development(
    protocol: str | Path, *, protocol_sha256: str, root: str | Path,
    inventory: dict[str, Any], weights: str | Path, execution_config: dict[str, Any],
    connectivity: str | Path, controls: str | Path, controls_sha256: str,
    output: str | Path,
) -> dict[str, Any]:
    """Execute the real development path after exact byte revalidation.

    The supplied frozen protocol is both the topology protocol and the real-input
    contract. Its bytes and the control bundle are independently pinned. The
    output report is published last; a partial directory is not completed evidence.
    """
    protocol_path, protocol_pin = _file_pin(
        protocol, protocol_sha256, "frozen real protocol"
    )
    controls_path, controls_pin = _file_pin(controls, controls_sha256, "control bundle")
    connectivity_path = _path(connectivity, "FlyWire connectivity")
    frozen = _read_json(protocol_path, "frozen real protocol")
    config = _comparison_spec(frozen, execution_config)

    # Recompute every local byte/runtime-derived binding. A template with null
    # hashes or a stale frozen record necessarily differs and fails here.
    reverified = freeze_real_inputs(
        frozen, root=root, inventory=inventory, weights=weights,
    )
    if _canonical_json(reverified) != _canonical_json(frozen):
        raise ValueError("frozen real protocol does not exactly match reverified local input bytes")

    try:
        extraction_identity = frozen["extraction_spec"]["identity"]
        extraction_spec = ExtractionSpec(**extraction_identity)
        feature_spec = PoseFeatureSpec(**frozen["pose_feature_spec"])
        classes = tuple(frozen["task_labels"])
        encoder_pin = _digest(frozen["pose_encoder_hash"], "pose encoder SHA-256")
    except (KeyError, TypeError) as exc:
        raise ValueError("frozen real protocol is missing executable extraction/encoder fields") from exc

    destination = Path(output).absolute()
    for component in (destination, *destination.parents):
        if component.is_symlink():
            raise ValueError("development output must not traverse symlinks")
    if destination.exists():
        raise ValueError("development output already exists; refusing to overwrite evidence")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    extraction_dir = destination / "extraction"

    extraction = extract_development(
        root, inventory, weights, extraction_spec, extraction_dir,
    )
    expected_extraction = {
        "dataset_content_hash": frozen["dataset_content_hash"],
        "split_hash": frozen["split_hash"],
        "input_inventory_hash": frozen["input_inventory_hash"],
        "extraction_spec_hash": frozen["extraction_spec_hash"],
        "observation_schema_hash": frozen["observation_schema_hash"],
    }
    for name, expected in expected_extraction.items():
        if _canonical_json(extraction.get(name)) != _canonical_json(expected):
            raise ValueError(f"development extraction {name} differs from the frozen protocol")
    if extraction.get("final_test_decoded") is not False or extraction.get("classifier_evaluated") is not False:
        raise ValueError("development extraction crossed the sealed final-test boundary")

    manifest_path = extraction_dir / "manifest.json"
    manifest_pin = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    bound = load_indexed_pose_development(
        extraction_dir, root=root, inventory=inventory, extraction_spec=extraction_spec,
        feature_spec=feature_spec, classes=classes,
        expected_manifest_sha256=manifest_pin, expected_encoder_hash=encoder_pin,
    )
    binding_pin = bound.binding_sha256
    bound.verify(expected_binding_sha256=binding_pin)

    comparison = _require_safe_comparison(run_pose_comparison(
        extraction_dir, root=root, inventory=inventory, extraction_spec=extraction_spec,
        feature_spec=feature_spec, classes=classes,
        expected_manifest_sha256=manifest_pin, expected_encoder_hash=encoder_pin,
        expected_binding_sha256=binding_pin,
        topology_protocol=protocol_path, connectivity=connectivity_path, controls=controls_path,
        expected_topology_sha256=protocol_pin, expected_controls_sha256=controls_pin,
        config=config,
    ), binding_pin)

    config_record = asdict(config)
    report = {
        "format_version": 1,
        "kind": "real_development_execution",
        "evidence_scope": "real_development_validation_only",
        "final_test_decoded": False,
        "final_test_evaluated": False,
        "classifier_evaluated_on_final_test": False,
        "topology_claim_evaluated": False,
        "protocol_sha256": protocol_pin,
        "controls_sha256": controls_pin,
        "dataset_content_hash": frozen["dataset_content_hash"],
        "split_hash": frozen["split_hash"],
        "yolo_weights_sha256": frozen["yolo_weights_sha256"],
        "extraction_manifest_sha256": manifest_pin,
        "prepared_binding_sha256": binding_pin,
        "execution_config": config_record,
        "execution_config_hash": _hash_json(config_record),
        "comparison": comparison,
    }
    result = _snapshot(report, "real development report")
    _publish_exclusive(destination / "development_report.json", result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m yolo_flywire.real_development", description=__doc__
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="execute train/validation development only")
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--protocol-sha256", required=True)
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--inventory", type=Path, required=True)
    run.add_argument("--weights", type=Path, required=True)
    run.add_argument("--execution-config", type=Path, required=True)
    run.add_argument("--connectivity", type=Path, required=True)
    run.add_argument("--controls", type=Path, required=True)
    run.add_argument("--controls-sha256", required=True)
    run.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_real_development(
            args.protocol, protocol_sha256=args.protocol_sha256, root=args.root,
            inventory=_read_json(args.inventory, "RGB inventory"), weights=args.weights,
            execution_config=_read_json(args.execution_config, "development execution config"),
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
        parser.exit(1, f"Real development error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
