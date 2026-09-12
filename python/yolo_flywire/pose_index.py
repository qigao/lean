"""Fully verify disk-backed pose sidecars, then materialize only requested clips."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
from pathlib import Path
from typing import Any, BinaryIO

from .ntu_io import _hash_json, verify_rgb_manifest
from .pose_bundle import (
    TimedPoseSample, _bundle_state, _digest, _frame, _json, _row, _validate_report,
)
from .pose_extract import ExtractionSpec
from .schema import PointSequence


@dataclass(frozen=True)
class ByteSpan:
    """Half-open byte range and digest in an already verified JSONL sidecar."""

    start: int
    end: int
    sha256: str


@dataclass(frozen=True)
class PoseSampleIndex:
    """Scalar source metadata and offsets; contains no frame or tensor handles."""

    sample_id: str
    label: str
    subject: int
    split: str
    width: int
    height: int
    frame_count: int
    missing_person_frames: int
    source_sha256: str
    geometry: ByteSpan
    timing: ByteSpan


class _DigestPair:
    """Feed sample and complete-file digests from the same bytes parsed by _row."""

    def __init__(self, whole: Any, sample: Any) -> None:
        self.whole, self.sample = whole, sample

    def update(self, raw: bytes) -> None:
        self.whole.update(raw)
        self.sample.update(raw)


class _SpanRows:
    """Bound logical reads to one recorded span without copying its entire bytes."""

    def __init__(self, handle: BinaryIO, span: ByteSpan) -> None:
        self.handle, self.end = handle, span.end
        handle.seek(span.start)

    def readline(self) -> bytes:
        remaining = self.end - self.handle.tell()
        return self.handle.readline(remaining) if remaining > 0 else b""


def _read_sample(geometry: BinaryIO, timing: BinaryIO, entry: PoseSampleIndex) -> TimedPoseSample:
    geometry_rows, timing_rows = _SpanRows(geometry, entry.geometry), _SpanRows(timing, entry.timing)
    geometry_digest, timing_digest = hashlib.sha256(), hashlib.sha256()
    frames, pts_values, bases, scores = [], [], [], []
    source = {"sample_id": entry.sample_id, "label": entry.label}
    previous = None
    for index in range(entry.frame_count):
        frame, pts, base, score = _frame(
            _row(geometry_rows, geometry_digest, "geometry span"),
            _row(timing_rows, timing_digest, "timing span"), source, index, previous,
        )
        previous = pts * base
        frames.append(frame)
        pts_values.append(pts)
        bases.append(base)
        scores.append(score)
    for handle, span, digest in ((geometry, entry.geometry, geometry_digest),
                                 (timing, entry.timing, timing_digest)):
        if handle.tell() != span.end or digest.hexdigest() != span.sha256:
            raise ValueError("requested span length or SHA-256 hash differs from the verified index")
    if sum(score == 0.0 for score in scores) != entry.missing_person_frames:
        raise ValueError("requested sample missing-person count differs from the verified index")
    return TimedPoseSample(
        sample_id=entry.sample_id, label=entry.label, subject=entry.subject, split=entry.split,
        width=entry.width, height=entry.height, sequence=PointSequence(tuple(frames)),
        pts=tuple(pts_values), time_bases=tuple(bases), detector_confidences=tuple(scores),
    )


@dataclass(frozen=True)
class IndexedPoseBundle:
    """Process-local verified index; not a persisted cache or an access-control token.

    Construct through index_development_bundle. A digest detects accidental changes
    against the saved record, not fabrication of both that record and its digest.
    Keep bundle files private and read-only throughout this object's lifetime.
    """

    directory: Path
    samples: tuple[PoseSampleIndex, ...]
    manifest_sha256: str
    dataset_content_hash: str
    split_hash: str
    input_inventory_hash: str
    extraction_spec_hash: str
    observation_schema_hash: str
    extractor_code_hash: str
    output_sha256: tuple[tuple[str, str], ...]
    index_sha256: str
    _file_state: tuple[tuple[str, tuple[int, ...]], ...]

    def descriptor(self) -> dict[str, Any]:
        """Fresh content identity, deliberately excluding paths and inode/timestamp state."""
        if type(self.samples) is not tuple or not self.samples or any(
            type(entry) is not PoseSampleIndex for entry in self.samples
        ):
            raise ValueError("index requires an immutable nonempty sample roster")
        return {
            "format_version": 1, "kind": "verified_pose_offset_index",
            "evidence_scope": "development_index_only", "final_test_decoded": False,
            "manifest_sha256": self.manifest_sha256,
            "dataset_content_hash": self.dataset_content_hash, "split_hash": self.split_hash,
            "input_inventory_hash": self.input_inventory_hash,
            "extraction_spec_hash": self.extraction_spec_hash,
            "observation_schema_hash": self.observation_schema_hash,
            "extractor_code_hash": self.extractor_code_hash,
            "output_sha256": dict(self.output_sha256),
            "samples": [asdict(entry) for entry in self.samples],
        }

    def verify(self) -> None:
        """Check index integrity and observed file identities, not rehash all sidecars.

        Requested spans are rehashed by read_samples. Recreate this index to fully
        reverify file bytes and the RGB inventory. Stat checks are not a filesystem
        snapshot and cannot detect every adversarial or concurrent replacement.
        """
        pin = _digest(self.index_sha256, "index SHA-256")
        if _hash_json(self.descriptor()) != pin:
            raise ValueError("index metadata does not match its saved SHA-256")
        if _bundle_state(self.directory) != dict(self._file_state):
            raise ValueError("bundle file state changed after index verification")

    def read_samples(self, *, split: str, indices: tuple[int, ...]) -> tuple[TimedPoseSample, ...]:
        """Read complete clips in explicit split-local order; no cached materialization.

        Selectors are checked before file I/O. No requested sample is returned
        until every requested span and the ending file state have been verified.
        Memory for returned observations scales with the requested clips' frames.
        """
        if type(split) is not str or split not in ("train", "validation"):
            raise ValueError("explicit train or validation split required; no final-test access")
        if (type(indices) is not tuple or not indices
                or any(type(index) is not int or index < 0 for index in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError("indices must be a nonempty tuple of unique nonnegative integers")
        selected = tuple(entry for entry in self.samples if entry.split == split)
        if any(index >= len(selected) for index in indices):
            raise ValueError("index is outside the selected development split")
        self.verify()
        with (self.directory / "observations.jsonl").open("rb") as geometry, \
                (self.directory / "timing.jsonl").open("rb") as timing:
            result = tuple(_read_sample(geometry, timing, selected[index]) for index in indices)
        self.verify()
        return result


def index_development_bundle(
    bundle: str | Path, *, root: str | Path, inventory: dict[str, Any],
    spec: ExtractionSpec, expected_manifest_sha256: str,
) -> IndexedPoseBundle:
    """Verify every development row before exposing scalar metadata and byte spans.

    Uses the eager reader's exact manifest, JSON and geometry/time validators,
    not the eager reader itself. All original RGB bytes are checked by the input
    inventory verifier, including holdout integrity hashing, but no video is
    decoded. Parsing retains only a row pair, not complete clips or the corpus.
    """
    pin = _digest(expected_manifest_sha256, "manifest SHA-256")
    if type(spec) is not ExtractionSpec:
        raise ValueError("an independent ExtractionSpec is required")
    directory = Path(bundle).absolute()
    before = _bundle_state(directory)
    manifest_raw = (directory / "manifest.json").read_bytes()
    if hashlib.sha256(manifest_raw).hexdigest() != pin:
        raise ValueError("manifest SHA-256 does not match the independent pin")
    report = _json(manifest_raw)
    verified = verify_rgb_manifest(root, inventory)
    selected = _validate_report(report, verified, spec)
    entries = []
    geometry_digest, timing_digest = hashlib.sha256(), hashlib.sha256()
    with (directory / "observations.jsonl").open("rb") as geometry, \
            (directory / "timing.jsonl").open("rb") as timing:
        for source, sample in zip(selected, report["samples"], strict=True):
            geometry_start, timing_start = geometry.tell(), timing.tell()
            sample_geometry, sample_timing = hashlib.sha256(), hashlib.sha256()
            geometry_pair = _DigestPair(geometry_digest, sample_geometry)
            timing_pair = _DigestPair(timing_digest, sample_timing)
            previous, missing = None, 0
            for index in range(sample["frame_count"]):
                frame, pts, base, score = _frame(
                    _row(geometry, geometry_pair, "geometry"),
                    _row(timing, timing_pair, "timing"), source, index, previous,
                )
                previous = pts * base
                missing += score == 0.0
                del frame  # Validation is complete; never accumulate frame objects in the index.
            if missing != sample["missing_person_frames"]:
                raise ValueError("missing-person count differs from manifest")
            entries.append(PoseSampleIndex(
                sample_id=source["sample_id"], label=source["label"], subject=source["subject"],
                split=source["split"], width=sample["width"], height=sample["height"],
                frame_count=sample["frame_count"], missing_person_frames=missing,
                source_sha256=source["sha256"],
                geometry=ByteSpan(geometry_start, geometry.tell(), sample_geometry.hexdigest()),
                timing=ByteSpan(timing_start, timing.tell(), sample_timing.hexdigest()),
            ))
        if geometry.read(1) or timing.read(1):
            raise ValueError("sidecar contains extra rows or trailing bytes")
    for name, digest in (("observations.jsonl", geometry_digest), ("timing.jsonl", timing_digest)):
        if digest.hexdigest() != report["output_sha256"][name]:
            raise ValueError(f"{name} bytes do not match manifest SHA-256")
    if _bundle_state(directory) != before:
        raise ValueError("bundle changed during verification")
    indexed = IndexedPoseBundle(
        directory=directory, samples=tuple(entries), manifest_sha256=pin,
        dataset_content_hash=verified["dataset_content_hash"], split_hash=verified["split_hash"],
        input_inventory_hash=report["input_inventory_hash"],
        extraction_spec_hash=report["extraction_spec_hash"],
        observation_schema_hash=report["observation_schema_hash"],
        extractor_code_hash=report["extractor_code_hash"],
        output_sha256=tuple(sorted(report["output_sha256"].items())), index_sha256="",
        _file_state=tuple(sorted(before.items())),
    )
    return replace(indexed, index_sha256=_hash_json(indexed.descriptor()))
