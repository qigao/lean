"""Bind complete indexed development clips to verified, on-demand feature batches."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import torch

from .ntu_io import _canonical_json
from .pose_batches import PoseBatch, collate_pose_features
from .pose_bundle import TimedPoseSample, _digest, _json, _object, _same
from .pose_development import _runtime, _strings, _tensor_record
from .pose_extract import ExtractionSpec
from .pose_features import PoseFeatureSpec, encode_timed_pose, pose_encoder_hash
from .pose_index import IndexedPoseBundle, PoseSampleIndex, _read_sample, index_development_bundle
from .pose_training import PosePartition

_PREPARATION_FILES = (
    "pose_indexed_development.py", "pose_index.py", "pose_bundle.py", "pose_development.py",
    "pose_extract.py", "pose_backend.py", "ntu_io.py", "pose_features.py", "schema.py",
    "pose_batches.py", "pose_training.py",
)


def _preparation_identity() -> dict[str, Any]:
    directory = Path(__file__).parent
    return {
        "id": "indexed-pose-development-v1",
        "source_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                          for name in _PREPARATION_FILES},
        "tensor_hash_rule": "sha256(canonical-json(shape,dtype)+NUL+C-order-little-endian-values)",
        "partition_policy": "complete-roster; split-local-order; batch-local-trailing-padding",
    }


def _roster(index: IndexedPoseBundle, classes: tuple[str, ...]) -> None:
    _strings(classes, "classes")
    if type(index) is not IndexedPoseBundle:
        raise ValueError("expected an explicit file-verified pose index")
    _strings(tuple(entry.sample_id for entry in index.samples), "sample_ids")
    if any(entry.split not in ("train", "validation") or type(entry.subject) is not int
           or entry.subject <= 0 for entry in index.samples):
        raise ValueError("only development samples with positive integer subjects are allowed")
    subjects = {}
    for role in ("train", "validation"):
        selected = tuple(entry for entry in index.samples if entry.split == role)
        if not selected or {entry.label for entry in selected} != set(classes):
            raise ValueError(f"{role} must cover exactly the explicit class vocabulary")
        subjects[role] = {entry.subject for entry in selected}
    if subjects["train"] & subjects["validation"]:
        raise ValueError("development subjects overlap")


def _metadata(entry: PoseSampleIndex, classes: tuple[str, ...]) -> dict[str, Any]:
    return {
        "sample_id": entry.sample_id, "label": entry.label, "subject": entry.subject,
        "split": entry.split, "frame_count": entry.frame_count, "target": classes.index(entry.label),
    }


def _check_sample(sample: TimedPoseSample, entry: PoseSampleIndex) -> None:
    if type(sample) is not TimedPoseSample:
        raise ValueError("index reader must return immutable timed samples")
    for name in ("sample_id", "label", "subject", "split", "width", "height"):
        _same(getattr(sample, name), getattr(entry, name), f"index reader {name}")
    if len(sample.sequence.frames) != entry.frame_count:
        raise ValueError("index reader frame count differs from the bound roster")


def _encode(sample: TimedPoseSample, spec: PoseFeatureSpec) -> np.ndarray:
    return encode_timed_pose(sample.sequence, pts=sample.pts, time_bases=sample.time_bases,
                             detector_confidences=sample.detector_confidences, spec=spec)


def _prepare_row(geometry: BinaryIO, timing: BinaryIO, entry: PoseSampleIndex,
                 spec: PoseFeatureSpec, classes: tuple[str, ...]) -> dict[str, Any]:
    # Scoped lifetime: no sample/array/tensor handle escapes to the next clip.
    sample = _read_sample(geometry, timing, entry)
    _check_sample(sample, entry)
    batch = collate_pose_features((_encode(sample, spec),))
    return {**_metadata(entry, classes),
            "features": _tensor_record(batch.features[0], torch.float32, "<f4")}


@dataclass(frozen=True)
class IndexedPoseDevelopment:
    """Process-local file binding with scalar records, not an authorization token.

    Use the file loader. Retain the binding pin independently. Files must remain
    private and read-only; observed stat checks are not atomic snapshots.
    Returned partition tensors are fresh, mutable and verified against this pin.
    """

    index: IndexedPoseBundle
    feature_spec: PoseFeatureSpec
    classes: tuple[str, ...]
    binding_json: str
    binding_sha256: str

    def descriptor(self) -> dict[str, Any]:
        if type(self.binding_json) is not str:
            raise ValueError("binding_json must be a JSON string")
        record = _json(self.binding_json.encode("utf-8"))
        return _object(record, {"format_version", "kind", "evidence_scope", "final_test_evaluated",
                                "index_sha256", "classes", "encoder", "preparation", "runtime", "rows"},
                       "indexed development binding")

    def verify(self, *, expected_binding_sha256: str) -> None:
        """Check the external pin, scalar roster, source identities and observed file state.

        Does not reencode clips, rehash unrequested spans or revisit original RGB
        media. read_partition rehashes requested raw spans and encoded features.
        """
        pin = _digest(expected_binding_sha256, "external binding SHA-256")
        if _digest(self.binding_sha256, "saved binding SHA-256") != pin:
            raise ValueError("binding SHA-256 does not match the independent pin")
        record = self.descriptor()
        if (_canonical_json(record) != self.binding_json
                or hashlib.sha256(self.binding_json.encode("utf-8")).hexdigest() != pin):
            raise ValueError("indexed binding record changed after preparation")
        if type(self.feature_spec) is not PoseFeatureSpec:
            raise ValueError("expected an explicit PoseFeatureSpec")
        _roster(self.index, self.classes)
        self.index.verify()
        expected = {
            "format_version": 1, "kind": "indexed_pose_development",
            "evidence_scope": "development_preparation_only", "final_test_evaluated": False,
            "index_sha256": self.index.index_sha256, "classes": list(self.classes),
            "encoder": {"hash": pose_encoder_hash(self.feature_spec),
                        "descriptor": self.feature_spec.descriptor()},
            "preparation": _preparation_identity(), "runtime": _runtime(),
        }
        for name, value in expected.items():
            _same(record[name], value, f"indexed binding {name}")
        rows = record["rows"]
        if type(rows) is not list or len(rows) != len(self.index.samples):
            raise ValueError("binding must contain the complete index roster")
        for row, entry in zip(rows, self.index.samples, strict=True):
            metadata = _metadata(entry, self.classes)
            _object(row, set(metadata) | {"features"}, "bound row")
            for name, value in metadata.items():
                _same(row[name], value, f"bound row {name}")
            feature = _object(row["features"], {"shape", "dtype", "sha256"}, "bound features")
            _same(feature["shape"], [entry.frame_count, 121], "feature shape")
            _same(feature["dtype"], "<f4", "feature dtype")
            _digest(feature["sha256"], "feature SHA-256")

    def _selection(self, split: str, indices: tuple[int, ...]) -> tuple[PoseSampleIndex, ...]:
        # Pure request validation must precede even source-code hash file reads.
        if type(split) is not str or split not in ("train", "validation"):
            raise ValueError("explicit train or validation split required; no final-test access")
        if (type(indices) is not tuple or not indices
                or any(type(i) is not int or i < 0 for i in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError("indices must be a nonempty tuple of unique nonnegative integers")
        if type(self.index) is not IndexedPoseBundle:
            raise ValueError("expected an explicit file-verified pose index")
        roster = tuple(entry for entry in self.index.samples if entry.split == split)
        if any(i >= len(roster) for i in indices):
            raise ValueError("index is outside the selected development split")
        return tuple(roster[i] for i in indices)

    def verify_partition(self, part: PosePartition, *, split: str, indices: tuple[int, ...],
                         expected_binding_sha256: str) -> None:
        """Check an exact requested minibatch without rereading or encoding clips."""
        selected = self._selection(split, indices)
        self.verify(expected_binding_sha256=expected_binding_sha256)
        if type(part) is not PosePartition or type(part.observations) is not PoseBatch:
            raise ValueError("expected an explicit PosePartition containing a PoseBatch")
        if (type(part.classes) is not tuple or type(part.sample_ids) is not tuple
                or type(part.subjects) is not tuple):
            raise ValueError("partition classes, sample IDs and subjects must be tuples")
        for name, value in (("classes", self.classes), ("split", split),
                            ("sample_ids", tuple(s.sample_id for s in selected)),
                            ("subjects", tuple(s.subject for s in selected))):
            _same(getattr(part, name), value, f"requested partition {name}")
        lengths = tuple(entry.frame_count for entry in selected)
        size, steps = len(selected), max(lengths)
        b = part.observations
        for value, dtype, shape in ((b.features, torch.float32, (size, steps, 121)),
                                    (b.lengths, torch.int64, (size,)),
                                    (b.time_mask, torch.bool, (size, steps)),
                                    (part.targets, torch.int64, (size,))):
            if (type(value) is not torch.Tensor or value.layout != torch.strided
                    or value.device.type != "cpu" or value.dtype != dtype or tuple(value.shape) != shape):
                raise ValueError("requested partition tensor dtype, device or shape differs from its binding")
        if tuple(b.lengths.tolist()) != lengths:
            raise ValueError("requested partition lengths differ from its binding")
        expected_mask = torch.arange(steps, device="cpu")[None, :] < b.lengths[:, None]
        if not torch.equal(b.time_mask, expected_mask):
            raise ValueError("requested time mask must mark actual frames, including missing observations")
        rows = tuple(row for row in self.descriptor()["rows"] if row["split"] == split)
        for offset, (entry, index) in enumerate(zip(selected, indices, strict=True)):
            row = rows[index]
            if part.targets[offset].item() != row["target"]:
                raise ValueError("requested targets differ from the ordered class binding")
            count = entry.frame_count
            if torch.any(b.features[offset, count:] != 0).item():
                raise ValueError("requested padding features must be zero")
            _same(_tensor_record(b.features[offset, :count], torch.float32, "<f4"),
                  row["features"], "requested feature values/hash")
        self.verify(expected_binding_sha256=expected_binding_sha256)

    def read_partition(self, *, split: str, indices: tuple[int, ...],
                       expected_binding_sha256: str) -> PosePartition:
        """Materialize only selected clips, checking all results before exposing any."""
        selected = self._selection(split, indices)
        self.verify(expected_binding_sha256=expected_binding_sha256)
        samples = self.index.read_samples(split=split, indices=indices)
        if type(samples) is not tuple or len(samples) != len(selected):
            raise ValueError("index reader returned an incomplete selection")
        for sample, entry in zip(samples, selected, strict=True):
            _check_sample(sample, entry)
        part = PosePartition(
            observations=collate_pose_features(tuple(_encode(s, self.feature_spec) for s in samples)),
            targets=torch.tensor([self.classes.index(s.label) for s in samples],
                                 dtype=torch.int64, device="cpu"),
            classes=self.classes, sample_ids=tuple(s.sample_id for s in samples),
            subjects=tuple(s.subject for s in samples), split=split,
        )
        self.verify_partition(part, split=split, indices=indices,
                              expected_binding_sha256=expected_binding_sha256)
        return part


def load_indexed_pose_development(
    bundle: str | Path, *, root: str | Path, inventory: dict[str, Any],
    extraction_spec: ExtractionSpec, feature_spec: PoseFeatureSpec, classes: tuple[str, ...],
    expected_manifest_sha256: str, expected_encoder_hash: str,
) -> IndexedPoseDevelopment:
    """Verify all raw bytes, then validate/hash features one complete clip at a time.

    No prepared source escapes until the last clip passes. Memory scales with
    scalar roster metadata and the longest clip, not total corpus frames.
    This performs no training and grants no permission to open a final test.
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
    indexed = index_development_bundle(bundle, root=root, inventory=inventory,
        spec=extraction_spec, expected_manifest_sha256=expected_manifest_sha256)
    _roster(indexed, classes)
    indexed.verify()
    # One guarded pass avoids O(N^2) index rehashing through N read_samples calls.
    with (indexed.directory / "observations.jsonl").open("rb") as geometry, \
            (indexed.directory / "timing.jsonl").open("rb") as timing:
        rows = [_prepare_row(geometry, timing, entry, feature_spec, classes)
                for entry in indexed.samples]
    indexed.verify()
    record = {
        "format_version": 1, "kind": "indexed_pose_development",
        "evidence_scope": "development_preparation_only", "final_test_evaluated": False,
        "index_sha256": indexed.index_sha256, "classes": list(classes),
        "encoder": {"hash": encoder_pin, "descriptor": feature_spec.descriptor()},
        "preparation": identity, "runtime": runtime, "rows": rows,
    }
    binding_json = _canonical_json(record)
    result = IndexedPoseDevelopment(indexed, feature_spec, classes, binding_json,
                                    hashlib.sha256(binding_json.encode("utf-8")).hexdigest())
    result.verify(expected_binding_sha256=result.binding_sha256)
    return result
