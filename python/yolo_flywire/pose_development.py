"""Bind verified pose files to complete development partitions, never final-test access."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
import torch

from .ntu_io import _canonical_json
from .pose_batches import PoseBatch, collate_pose_features
from .pose_bundle import _digest, _json, load_development_bundle
from .pose_extract import ExtractionSpec
from .pose_features import PoseFeatureSpec, encode_timed_pose, pose_encoder_hash
from .pose_training import PosePartition

_SOURCE_FIELDS = (
    "manifest_sha256", "dataset_content_hash", "split_hash", "input_inventory_hash",
    "extraction_spec_hash", "observation_schema_hash", "extractor_code_hash",
)
_PREPARATION_FILES = (
    "pose_development.py", "pose_bundle.py", "pose_extract.py", "ntu_io.py",
    "pose_features.py", "schema.py", "pose_batches.py", "pose_training.py",
)


def _strings(values: tuple[str, ...], name: str) -> None:
    if (type(values) is not tuple or not values
            or any(type(v) is not str or not v.strip() for v in values)
            or len(set(values)) != len(values)):
        raise ValueError(f"{name} must be a nonempty tuple of unique nonblank strings")


def _preparation_identity() -> dict[str, Any]:
    directory = Path(__file__).parent
    return {
        "id": "verified-pose-development-v1",
        "source_sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in _PREPARATION_FILES
        },
        "tensor_hash_rule": "sha256(canonical-json(shape,dtype)+NUL+C-order-little-endian-values)",
        "partition_policy": "complete-verified-roster; original-order; separate-development-splits",
    }


def _runtime() -> dict[str, str]:
    # Descriptive identity, not a claim to freeze all native libraries or hardware.
    return {
        "python": platform.python_version(), "implementation": platform.python_implementation(),
        "numpy": np.__version__, "torch": str(torch.__version__),
        "platform": platform.platform(), "machine": platform.machine(), "byteorder": sys.byteorder,
    }


def _tensor_record(value: torch.Tensor, dtype: torch.dtype, wire_dtype: str) -> dict[str, Any]:
    if (type(value) is not torch.Tensor or value.layout != torch.strided
            or value.device.type != "cpu" or value.dtype != dtype):
        raise ValueError(f"binding requires a dense CPU {dtype} tensor")
    # Hash logical values, not storage addresses, strides or unused storage.
    array = np.ascontiguousarray(value.detach().resolve_neg().numpy(), dtype=np.dtype(wire_dtype))
    header = {"shape": list(value.shape), "dtype": np.dtype(wire_dtype).str}
    digest = hashlib.sha256(_canonical_json(header).encode("utf-8") + b"\0")
    digest.update(memoryview(array).cast("B"))
    return {**header, "sha256": digest.hexdigest()}


def _partition_record(part: PosePartition, role: str) -> dict[str, Any]:
    if (type(part) is not PosePartition or type(part.split) is not str or part.split != role
            or type(part.observations) is not PoseBatch):
        raise ValueError(f"binding requires an explicit {role} PosePartition")
    _strings(part.classes, "classes")
    _strings(part.sample_ids, "sample_ids")
    if (type(part.subjects) is not tuple or len(part.subjects) != len(part.sample_ids)
            or any(type(subject) is not int or subject <= 0 for subject in part.subjects)):
        raise ValueError("binding subjects must be aligned positive integers")
    b = part.observations
    tensors = {
        "features": _tensor_record(b.features, torch.float32, "<f4"),
        "lengths": _tensor_record(b.lengths, torch.int64, "<i8"),
        "time_mask": _tensor_record(b.time_mask, torch.bool, "|b1"),
        "targets": _tensor_record(part.targets, torch.int64, "<i8"),
    }
    return {
        "split": role, "classes": list(part.classes), "sample_ids": list(part.sample_ids),
        "subjects": list(part.subjects), "tensors": tensors,
    }


def _partition_records(train: PosePartition, validation: PosePartition) -> dict[str, Any]:
    records = {"train": _partition_record(train, "train"),
               "validation": _partition_record(validation, "validation")}
    if train.classes != validation.classes:
        raise ValueError("development partitions must share the ordered class vocabulary")
    if set(train.sample_ids) & set(validation.sample_ids):
        raise ValueError("development sample IDs overlap")
    if set(train.subjects) & set(validation.subjects):
        raise ValueError("development subjects overlap")
    return records


@dataclass(frozen=True)
class PreparedPoseDevelopment:
    """File-verified materialization with a saved integrity record, not an access token.

    Frozen fields contain mutable tensors. Verify before/after use. This detects
    changes against the saved digest, not deliberate fabrication of both data and
    digest; the experiment runner must retain an independent binding pin.
    """

    train: PosePartition
    validation: PosePartition
    binding_json: str
    binding_sha256: str

    def descriptor(self) -> dict[str, Any]:
        if type(self.binding_json) is not str:
            raise ValueError("binding_json must be a JSON string")
        record = _json(self.binding_json.encode("utf-8"))
        if type(record) is not dict:
            raise ValueError("binding record must be an object")
        return record

    def verify(self) -> None:
        """Check materialized tensors/metadata, without rereading source files."""
        pin = _digest(self.binding_sha256, "development binding SHA-256")
        record = self.descriptor()
        if (_canonical_json(record) != self.binding_json
                or hashlib.sha256(self.binding_json.encode("utf-8")).hexdigest() != pin):
            raise ValueError("development binding record differs from its saved SHA-256")
        if (type(record.get("format_version")) is not int or record["format_version"] != 1
                or record.get("evidence_scope") != "development_preparation_only"):
            raise ValueError("unsupported development binding record")
        current = _partition_records(self.train, self.validation)
        if (_canonical_json(record.get("partitions")) != _canonical_json(current)
                or _canonical_json(record.get("classes")) != _canonical_json(list(self.train.classes))):
            raise ValueError("development partition tensors or metadata changed after preparation")


def load_pose_development(
    bundle: str | Path, *, root: str | Path, inventory: dict[str, Any],
    extraction_spec: ExtractionSpec, feature_spec: PoseFeatureSpec, classes: tuple[str, ...],
    expected_manifest_sha256: str, expected_encoder_hash: str,
) -> PreparedPoseDevelopment:
    """Verify complete source files, then encode and bind both development splits.

    Expected pins and the ordered vocabulary must be supplied independently. No
    object-based shortcut, source decoding, inference, sample filtering or final
    test is accepted. The verified bundle and padded partitions occupy RAM.
    """
    if not isinstance(bundle, (str, Path)):
        raise ValueError("bundle must be a file path, not a constructed verified object")
    if type(extraction_spec) is not ExtractionSpec or type(feature_spec) is not PoseFeatureSpec:
        raise ValueError("explicit extraction and feature specifications are required")
    _strings(classes, "classes")
    _digest(expected_manifest_sha256, "manifest SHA-256")
    encoder_pin = _digest(expected_encoder_hash, "encoder SHA-256")
    if pose_encoder_hash(feature_spec) != encoder_pin:
        raise ValueError("encoder hash does not match the independent pin")
    identity, runtime = _preparation_identity(), _runtime()

    # The actual reader finishes all byte and roster checks before encoding starts.
    verified = load_development_bundle(
        bundle, root=root, inventory=inventory, spec=extraction_spec,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    if any(sample.split not in ("train", "validation") for sample in verified.samples):
        raise ValueError("only verified development samples may be prepared")
    rosters = {role: tuple(s for s in verified.samples if s.split == role)
               for role in ("train", "validation")}
    for role, samples in rosters.items():
        if not samples or {sample.label for sample in samples} != set(classes):
            raise ValueError(f"{role} must cover exactly the explicit class vocabulary")

    class_index = {label: index for index, label in enumerate(classes)}
    partitions = {}
    for role, samples in rosters.items():
        sequences = tuple(encode_timed_pose(
            sample.sequence, pts=sample.pts, time_bases=sample.time_bases,
            detector_confidences=sample.detector_confidences, spec=feature_spec,
        ) for sample in samples)
        partitions[role] = PosePartition(
            observations=collate_pose_features(sequences),
            targets=torch.tensor([class_index[s.label] for s in samples], dtype=torch.int64, device="cpu"),
            classes=classes, sample_ids=tuple(s.sample_id for s in samples),
            subjects=tuple(s.subject for s in samples), split=role,
        )
    train, validation = partitions["train"], partitions["validation"]
    record = {
        "format_version": 1, "evidence_scope": "development_preparation_only",
        "source": {**{name: getattr(verified, name) for name in _SOURCE_FIELDS},
                   "output_sha256": dict(verified.output_sha256)},
        "classes": list(classes),
        "encoder": {"hash": encoder_pin, "descriptor": feature_spec.descriptor()},
        "preparation": identity, "runtime": runtime,
        "partitions": _partition_records(train, validation),
    }
    if (pose_encoder_hash(feature_spec) != encoder_pin
            or _preparation_identity() != identity or _runtime() != runtime):
        raise ValueError("encoder, preparation source or runtime identity changed during preparation")
    binding_json = _canonical_json(record)
    result = PreparedPoseDevelopment(
        train, validation, binding_json, hashlib.sha256(binding_json.encode("utf-8")).hexdigest(),
    )
    result.verify()
    return result
