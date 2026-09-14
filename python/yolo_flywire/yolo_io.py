from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .schema import KeypointFrame, LabeledSequence, PointSequence


_REQUIRED_FIELDS = {"sample_id", "frame_index", "label", "body_keypoints"}
_OPTIONAL_FIELDS = {"hand_keypoints", "bbox", "detector_confidence"}
_ALLOWED_FIELDS = _REQUIRED_FIELDS | _OPTIONAL_FIELDS


@dataclass(frozen=True)
class FrozenPoseDataset:
    samples: tuple[LabeledSequence, ...]
    observation_schema_hash: str


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_keypoints(value: Any, field: str, line_number: int) -> tuple[tuple[float, float, float], ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list at line {line_number}")
    points: list[tuple[float, float, float]] = []
    for index, point in enumerate(value):
        if not isinstance(point, list) or len(point) != 3:
            raise ValueError(f"{field}[{index}] must be [x,y,confidence] at line {line_number}")
        try:
            x, y, confidence = (float(point[0]), float(point[1]), float(point[2]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field}[{index}] contains non-numeric values at line {line_number}") from exc
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"{field}[{index}] confidence must be in [0,1] at line {line_number}")
        points.append((x, y, confidence))
    if not points and field == "body_keypoints":
        raise ValueError("body_keypoints must not be empty")
    return tuple(points)


def load_pose_jsonl(path: str | Path) -> FrozenPoseDataset:
    """Load pre-exported YOLO/Pose observations; never invokes a detector."""
    source = Path(path)
    records: list[dict[str, Any]] = []
    identities: set[tuple[str, int]] = set()
    expected_body_count: int | None = None
    expected_hand_count: int | None = None
    optional_presence: dict[str, bool] | None = None

    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at line {line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"record at line {line_number} must be an object")

            unknown = set(record) - _ALLOWED_FIELDS
            if unknown:
                if "feature_payload" in unknown:
                    raise ValueError("label leakage risk: feature_payload is not part of the frozen observation schema")
                raise ValueError(f"unknown observation fields at line {line_number}: {sorted(unknown)}")
            missing = _REQUIRED_FIELDS - set(record)
            if missing:
                raise ValueError(f"missing required fields at line {line_number}: {sorted(missing)}")

            sample_id = record["sample_id"]
            label = record["label"]
            frame_index = record["frame_index"]
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"sample_id must be a non-empty string at line {line_number}")
            if not isinstance(label, str) or not label:
                raise ValueError(f"label must be a non-empty string at line {line_number}")
            if not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index < 0:
                raise ValueError(f"frame_index must be a non-negative integer at line {line_number}")

            identity = (sample_id, frame_index)
            if identity in identities:
                raise ValueError(f"duplicate (sample_id, frame_index) record: {identity}")
            identities.add(identity)

            body = _parse_keypoints(record["body_keypoints"], "body_keypoints", line_number)
            hand = _parse_keypoints(record.get("hand_keypoints", []), "hand_keypoints", line_number)
            if expected_body_count is None:
                expected_body_count = len(body)
                expected_hand_count = len(hand)
                optional_presence = {
                    field: field in record for field in sorted(_OPTIONAL_FIELDS)
                }
            elif len(body) != expected_body_count or len(hand) != expected_hand_count:
                raise ValueError("keypoint schema drift: body/hand keypoint counts must remain constant")
            else:
                current_presence = {field: field in record for field in sorted(_OPTIONAL_FIELDS)}
                if current_presence != optional_presence:
                    raise ValueError("observation schema drift: optional field presence must remain constant")

            if "bbox" in record:
                bbox = record["bbox"]
                if not isinstance(bbox, list) or len(bbox) != 4:
                    raise ValueError(f"bbox must be a four-number list at line {line_number}")
                try:
                    [float(value) for value in bbox]
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"bbox contains non-numeric values at line {line_number}") from exc
            if "detector_confidence" in record:
                confidence = float(record["detector_confidence"])
                if not 0.0 <= confidence <= 1.0:
                    raise ValueError(f"detector_confidence must be in [0,1] at line {line_number}")

            normalized = dict(record)
            normalized["_points"] = body + hand
            records.append(normalized)

    if not records:
        raise ValueError("pose JSONL contains no records")

    records.sort(key=lambda record: (record["sample_id"], record["frame_index"]))
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_sample.setdefault(record["sample_id"], []).append(record)

    samples: list[LabeledSequence] = []
    for sample_id in sorted(by_sample):
        sample_records = by_sample[sample_id]
        labels = {record["label"] for record in sample_records}
        if len(labels) != 1:
            raise ValueError(f"sample {sample_id!r} has inconsistent target labels")
        frame_indices = [record["frame_index"] for record in sample_records]
        if frame_indices != sorted(frame_indices):
            raise AssertionError("internal frame ordering failure")
        sequence = PointSequence(
            frames=tuple(KeypointFrame(points=record["_points"]) for record in sample_records)
        )
        samples.append(
            LabeledSequence(sequence=sequence, label=next(iter(labels)), sample_id=sample_id)
        )

    assert expected_body_count is not None
    assert expected_hand_count is not None
    assert optional_presence is not None
    schema_descriptor = {
        "format": "yolo-pose-jsonl-v0",
        "required_fields": sorted(_REQUIRED_FIELDS),
        "optional_fields_present": [
            field for field, present in sorted(optional_presence.items()) if present
        ],
        "body_keypoint_count": expected_body_count,
        "hand_keypoint_count": expected_hand_count,
        "point_layout": ["x", "y", "confidence"],
        "feature_payload_fields": [],
    }
    return FrozenPoseDataset(
        samples=tuple(samples),
        observation_schema_hash=_canonical_hash(schema_descriptor),
    )
