"""Freeze YOLO11 hosted-CI real input identity from a verified remote NTU source."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .ntu_io import _canonical_json, _hash_file, _hash_json, _publish_exclusive
from .pose_backend import prediction_options, runtime_versions
from .pose_extract import _schema
from .pose_features import PoseFeatureSpec, pose_encoder_hash
from .provenance import validate_rewiring_protocol
from .real_freeze import (
    _BUDGET, _DATASET_ID, _FEATURE_SPEC, _GRAPH, _LABELS, _SEEDS, _SPLIT_RULE,
    _bind, _same, _snapshot,
)
from .remote_rgb import validate_remote_manifest

_ULTRALYTICS_VERSION = "8.4.146"
_MODEL = "yolo11n-pose.pt"
_DIGEST = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class HostedExtractionSpec:
    weights_sha256: str
    ultralytics_version: str
    torch_version: str
    numpy_version: str
    av_version: str
    opencv_version: str
    model: str = _MODEL

    def __post_init__(self) -> None:
        if type(self.weights_sha256) is not str or _DIGEST.fullmatch(self.weights_sha256) is None:
            raise ValueError("weights_sha256 must be a lowercase SHA-256")
        if self.model != _MODEL:
            raise ValueError("hosted extraction requires exact yolo11n-pose.pt model identity")
        for name, version in self.expected_versions().items():
            if type(version) is not str or not re.fullmatch(
                r"[0-9]+(?:\.[0-9]+)+(?:[a-zA-Z0-9.+-]*)", version
            ):
                raise ValueError(f"{name} must have an explicit exact package version")

    def expected_versions(self) -> dict[str, str]:
        return {
            "ultralytics": self.ultralytics_version,
            "torch": self.torch_version,
            "numpy": self.numpy_version,
            "av": self.av_version,
            "opencv-python": self.opencv_version,
        }

    def descriptor(self) -> dict[str, Any]:
        identity = asdict(self)
        identity.pop("model")
        return {
            "identity": identity,
            "model": self.model,
            "predict": prediction_options(),
            "decoder": "pyav-avi-single-thread-all-frames",
            "partitions": ["train", "validation"],
            "person_policy": "zero-mask-or-single-else-error",
        }


def _validate_base(protocol: dict[str, Any]) -> PoseFeatureSpec:
    if type(protocol) is not dict:
        raise ValueError("hosted real protocol must be a JSON object")
    fixed = {
        "protocol_id": "v1-hosted-yolo11n-preflight",
        "claim": "topology_specific_advantage",
        "dataset_id": _DATASET_ID,
        "task_labels": list(_LABELS),
        "split_rule": _SPLIT_RULE,
        "yolo_version": "ultralytics-yolo11n-pose",
        "ultralytics_package_version": _ULTRALYTICS_VERSION,
        "yolo_weights": _MODEL,
        "seeds": _SEEDS,
        "budget": _BUDGET,
        "pose_feature_spec": _FEATURE_SPEC,
        "primary_metric": "macro_f1",
        "success_threshold": 0.02,
    }
    for name, expected in fixed.items():
        if name not in protocol:
            raise ValueError(f"hosted real protocol is missing field {name}")
        _same(protocol[name], expected, name)
    for name, expected in _GRAPH.items():
        if name not in protocol:
            raise ValueError(f"hosted real protocol is missing graph field {name}")
        _same(protocol[name], expected, name)
    if protocol.get("rewiring_successful_swaps_per_offdiagonal_edge") != 10:
        raise ValueError("hosted rewiring budget differs from frozen Phase 2 design")
    if protocol.get("final_test_used_for_selection") is not False:
        raise ValueError("hosted final test must remain sealed from model selection")
    if protocol.get("final_test_decoded") is True or protocol.get("classifier_evaluated") is True:
        raise ValueError("hosted freeze input cannot claim final-test/classifier execution")
    validate_rewiring_protocol(protocol)
    feature = protocol.get("pose_feature_spec")
    if type(feature) is not dict or set(feature) != {"confidence_threshold", "scale_epsilon"}:
        raise ValueError("hosted pose_feature_spec must contain the two frozen parameters")
    return PoseFeatureSpec(**feature)


def _check_weights(path: Path, spec: HostedExtractionSpec) -> None:
    checkpoint = path.absolute()
    if checkpoint.name != spec.model:
        raise ValueError(f"expected explicit local {spec.model}; no alternate model")
    for component in (checkpoint, *checkpoint.parents):
        if component.is_symlink():
            raise ValueError("hosted weight path must not traverse symlinks")
    if _hash_file(checkpoint)[1] != spec.weights_sha256:
        raise ValueError("hosted weight bytes do not match frozen SHA-256")


def freeze_hosted_inputs(
    protocol: dict[str, Any], *, remote_manifest: dict[str, Any], weights: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind remote scientific source + exact YOLO11 bytes without decoding media."""
    source = _snapshot(protocol, "hosted real protocol")
    feature_spec = _validate_base(source)
    remote = validate_remote_manifest(remote_manifest)
    versions = runtime_versions()
    required_versions = {"ultralytics", "torch", "numpy", "av", "opencv-python"}
    if type(versions) is not dict or set(versions) != required_versions:
        raise ValueError("hosted runtime version record is incomplete or contains unexpected packages")
    if versions["ultralytics"] != source["ultralytics_package_version"]:
        raise ValueError("installed ultralytics version differs from hosted protocol")

    checkpoint = Path(weights).absolute()
    weight_sha256 = _hash_file(checkpoint)[1]
    extraction = HostedExtractionSpec(
        weights_sha256=weight_sha256,
        ultralytics_version=versions["ultralytics"],
        torch_version=versions["torch"],
        numpy_version=versions["numpy"],
        av_version=versions["av"],
        opencv_version=versions["opencv-python"],
    )
    _check_weights(checkpoint, extraction)
    descriptor = extraction.descriptor()

    _bind(source, "dataset_content_hash", remote.dataset_content_hash)
    _bind(source, "split_hash", remote.split_hash)
    _bind(source, "input_inventory_hash", remote.input_inventory_hash)
    _bind(source, "source_manifest_hash", remote.source_manifest_hash)
    _bind(source, "yolo_weights_sha256", weight_sha256)
    _bind(source, "observation_schema_hash", _hash_json(_schema()))
    _bind(source, "pose_encoder_hash", pose_encoder_hash(feature_spec))
    _bind(source, "extraction_spec", descriptor)
    _bind(source, "extraction_spec_hash", _hash_json(descriptor))
    source["evidence_scope"] = "hosted_real_input_frozen_no_final_test_execution"
    source["final_test_decoded"] = False
    source["classifier_evaluated"] = False
    source["freeze_record"] = {
        "kind": "verified-hosted-real-input-freeze-v1",
        "input_inventory_hash": source["input_inventory_hash"],
        "source_manifest_hash": source["source_manifest_hash"],
        "extraction_spec_hash": source["extraction_spec_hash"],
        "pose_encoder_hash": source["pose_encoder_hash"],
        "media_decoding_performed": False,
        "classifier_execution_performed": False,
    }
    transport = {
        "format_version": 1,
        "kind": "hosted_remote_transport_evidence",
        "dataset_content_hash": remote.dataset_content_hash,
        "split_hash": remote.split_hash,
        "source_manifest_hash": remote.source_manifest_hash,
        "transport_manifest_hash": remote.transport_manifest_hash,
        "locator_contents_recorded": False,
    }
    return _snapshot(source, "frozen hosted protocol"), _snapshot(transport, "transport evidence")


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
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--protocol", type=Path, required=True)
    freeze.add_argument("--remote-manifest", type=Path, required=True)
    freeze.add_argument("--weights", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--transport-evidence", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        for path in (args.output, args.transport_evidence):
            if path.exists() or path.is_symlink():
                raise ValueError("hosted freeze output already exists; refusing overwrite")
        protocol, transport = freeze_hosted_inputs(
            _read_json(args.protocol, "hosted protocol"),
            remote_manifest=_read_json(args.remote_manifest, "remote manifest"),
            weights=args.weights,
        )
        _publish_exclusive(args.output, protocol)
        _publish_exclusive(args.transport_evidence, transport)
        print(_canonical_json({
            "evidence_scope": protocol["evidence_scope"],
            "dataset_content_hash": protocol["dataset_content_hash"],
            "source_manifest_hash": protocol["source_manifest_hash"],
            "transport_manifest_hash": transport["transport_manifest_hash"],
            "final_test_decoded": False,
        }))
        return 0
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, f"Hosted real input freeze error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
