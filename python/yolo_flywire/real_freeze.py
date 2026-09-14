"""Bind verified real NTU/YOLO input bytes to the frozen Phase 2 protocol.

This module inventories and hashes only. It never decodes video, constructs a
predictor, trains a classifier, evaluates a model, or authorizes final-test use.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .ntu_io import _canonical_json, _hash_file, _hash_json, _publish_exclusive, verify_rgb_manifest
from .pose_backend import runtime_versions
from .pose_extract import ExtractionSpec, _check_weights, _schema
from .pose_features import PoseFeatureSpec, pose_encoder_hash
from .provenance import validate_rewiring_protocol

_DATASET_ID = "NTU-RGB+D-120:official-rosem-lab-release:RGB:10-class-motion-subset"
_LABELS = (
    "A8:sitting_down", "A9:standing_up", "A22:cheer_up", "A23:hand_waving",
    "A26:hopping", "A27:jump_up", "A31:pointing", "A34:rub_two_hands_together",
    "A35:nod_head_or_bow", "A36:shake_head",
)
_SPLIT_RULE = "official-xsub120-outer-boundary; validation=deterministic-subject-hash-from-training-side-only"
_SEEDS = [7, 11, 19, 23, 31]
_BUDGET = {"epochs": 20, "max_updates": 40, "parameter_ceiling": 50000}
_FEATURE_SPEC = {"confidence_threshold": 0.05, "scale_epsilon": 1e-6}
_ULTRALYTICS_VERSION = "8.4.146"
_GRAPH = {
    "flywire_release": "FAFB-v783",
    "flywire_source": "murthylab/visual-system-parts-list",
    "flywire_source_commit": "0d8574d46627ce7fadd968a3c5d602e837325373",
    "flywire_connectivity_path": "data/type_to_type_connection_and_synapse_counts.csv",
    "flywire_connectivity_git_blob_sha1": "5183755ecbb41d5c8cee1a4a2d99b8eecba75c52",
    "flywire_connectivity_sha256": "215cf7a65895f9f84768db34052964542f7ebbe986ce9864b7e9ed2976c55e38",
    "selected_graph_num_nodes": 187,
    "selected_graph_num_edges": 14542,
    "selected_graph_num_diagonal_edges": 126,
    "selected_graph_fingerprint": "a7088c8590aa10d6b204c2b11ad51d5f7889dc10ea25b68ffcf24794b06d692d",
}


def _snapshot(value: Any, context: str) -> Any:
    try:
        return json.loads(_canonical_json(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context} must contain canonical JSON values") from exc


def _same(actual: Any, expected: Any, name: str) -> None:
    if _canonical_json(actual) != _canonical_json(expected):
        raise ValueError(f"{name} differs from the frozen Phase 2 design")


def _validate_base(protocol: dict[str, Any]) -> PoseFeatureSpec:
    if type(protocol) is not dict:
        raise ValueError("real protocol must be a JSON object")
    fixed = {
        "claim": "topology_specific_advantage",
        "dataset_id": _DATASET_ID,
        "task_labels": list(_LABELS),
        "split_rule": _SPLIT_RULE,
        "yolo_version": "ultralytics-yolo26n-pose",
        "ultralytics_package_version": _ULTRALYTICS_VERSION,
        "yolo_weights": "yolo26n-pose.pt",
        "seeds": _SEEDS,
        "budget": _BUDGET,
        "pose_feature_spec": _FEATURE_SPEC,
        "primary_metric": "macro_f1",
        "success_threshold": 0.02,
    }
    for name, expected in fixed.items():
        if name not in protocol:
            raise ValueError(f"real protocol is missing frozen design field {name}")
        _same(protocol[name], expected, name)
    for name, expected in _GRAPH.items():
        if name not in protocol:
            raise ValueError(f"real protocol is missing frozen graph field {name}")
        _same(protocol[name], expected, name)
    if protocol.get("rewiring_successful_swaps_per_offdiagonal_edge") != 10:
        raise ValueError("rewiring budget differs from the frozen Phase 2 design")
    if protocol.get("final_test_used_for_selection") is not False:
        raise ValueError("final test must remain sealed from model selection")
    if protocol.get("final_test_decoded") is True or protocol.get("classifier_evaluated") is True:
        raise ValueError("a freeze input cannot claim final-test decoding or classifier evaluation")
    validate_rewiring_protocol(protocol)
    feature = protocol.get("pose_feature_spec")
    if type(feature) is not dict or set(feature) != {"confidence_threshold", "scale_epsilon"}:
        raise ValueError("pose_feature_spec must explicitly contain the two frozen parameters")
    return PoseFeatureSpec(**feature)


def _bind(record: dict[str, Any], name: str, value: Any) -> None:
    existing = record.get(name)
    if existing is not None and _canonical_json(existing) != _canonical_json(value):
        raise ValueError(f"{name} conflicts with verified local input")
    record[name] = value


def freeze_real_inputs(
    protocol: dict[str, Any], *, root: str | Path, inventory: dict[str, Any],
    weights: str | Path,
) -> dict[str, Any]:
    """Return one byte-backed real-input snapshot without decoding or training."""
    source = _snapshot(protocol, "real protocol")
    feature_spec = _validate_base(source)

    verified = verify_rgb_manifest(root, inventory)
    versions = runtime_versions()
    required_versions = {"ultralytics", "torch", "numpy", "av", "opencv-python"}
    if type(versions) is not dict or set(versions) != required_versions:
        raise ValueError("runtime version record is incomplete or contains unexpected packages")
    if versions["ultralytics"] != source["ultralytics_package_version"]:
        raise ValueError("installed ultralytics version differs from the frozen protocol")

    checkpoint = Path(weights).absolute()
    weight_sha256 = _hash_file(checkpoint)[1]
    extraction = ExtractionSpec(
        weights_sha256=weight_sha256,
        ultralytics_version=versions["ultralytics"], torch_version=versions["torch"],
        numpy_version=versions["numpy"], av_version=versions["av"],
        opencv_version=versions["opencv-python"],
    )
    _check_weights(checkpoint, extraction)
    extraction_record = extraction.descriptor()

    _bind(source, "dataset_content_hash", verified["dataset_content_hash"])
    _bind(source, "split_hash", verified["split_hash"])
    _bind(source, "input_inventory_hash", _hash_json(verified))
    _bind(source, "yolo_weights_sha256", weight_sha256)
    _bind(source, "observation_schema_hash", _hash_json(_schema()))
    _bind(source, "pose_encoder_hash", pose_encoder_hash(feature_spec))
    _bind(source, "extraction_spec", extraction_record)
    _bind(source, "extraction_spec_hash", _hash_json(extraction_record))
    source["evidence_scope"] = "real_input_frozen_no_final_test_execution"
    source["final_test_decoded"] = False
    source["classifier_evaluated"] = False
    source["freeze_record"] = {
        "kind": "verified-real-input-freeze-v1",
        "input_inventory_hash": source["input_inventory_hash"],
        "extraction_spec_hash": source["extraction_spec_hash"],
        "pose_encoder_hash": source["pose_encoder_hash"],
        "media_decoding_performed": False,
        "classifier_execution_performed": False,
    }
    return _snapshot(source, "frozen real protocol")


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {name} JSON") from exc
    if type(value) is not dict:
        raise ValueError(f"{name} must be a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="bind verified local bytes without decoding or training")
    freeze.add_argument("--protocol", type=Path, required=True)
    freeze.add_argument("--root", type=Path, required=True)
    freeze.add_argument("--inventory", type=Path, required=True)
    freeze.add_argument("--weights", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("output already exists; refusing to overwrite frozen protocol")
        result = freeze_real_inputs(
            _read_json(args.protocol, "protocol"), root=args.root,
            inventory=_read_json(args.inventory, "inventory"), weights=args.weights,
        )
        _publish_exclusive(args.output, result)
        return 0
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Real input freeze error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
