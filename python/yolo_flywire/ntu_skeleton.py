from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any

import numpy as np


_SKELETON_NAME = re.compile(
    r"S([0-9]{3})C([0-9]{3})P([0-9]{3})R([0-9]{3})A([0-9]{3})\.skeleton"
)

# Explicit reference: Liu et al., arXiv:1905.04757v2, section 3.2.1.
# This is an independent Gate 1 copy of the official NTU120 X-Sub training side;
# it intentionally does not import or alter the Issue #68 RGB inventory module.
_OUTER_TRAIN = frozenset({
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31,
    34, 35, 38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59,
    70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95,
    97, 98, 100, 103,
})

_BODY_SELECTION_RULE = "max-fully-tracked-joints-then-lowest-body-id"


@dataclass(frozen=True)
class NtuSkeletonBody:
    body_id: int
    positions: np.ndarray
    tracking_state: np.ndarray
    present: np.ndarray

    def __post_init__(self) -> None:
        if type(self.body_id) is not int or self.body_id <= 0:
            raise ValueError("NTU body_id must be a positive integer")
        positions = np.asarray(self.positions, dtype=np.float64).copy()
        tracking = np.asarray(self.tracking_state, dtype=np.int64).copy()
        present = np.asarray(self.present, dtype=bool).copy()
        if positions.ndim != 3 or positions.shape[1:] != (25, 3):
            raise ValueError("NTU body positions must have shape [frames,25,3]")
        if tracking.shape != positions.shape[:2]:
            raise ValueError("NTU tracking state must have shape [frames,25]")
        if present.shape != (positions.shape[0],):
            raise ValueError("NTU body presence must have shape [frames]")
        if not np.isfinite(positions).all():
            raise ValueError("NTU body positions must be finite")
        if np.any((tracking < 0) | (tracking > 2)):
            raise ValueError("NTU tracking state must be in {0,1,2}")
        positions.setflags(write=False)
        tracking.setflags(write=False)
        present.setflags(write=False)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "tracking_state", tracking)
        object.__setattr__(self, "present", present)

    @property
    def tracked_joint_count(self) -> int:
        mask = self.present[:, None] & (self.tracking_state == 2)
        return int(mask.sum())


@dataclass(frozen=True)
class NtuSkeletonSample:
    sample_id: str
    setup: int
    camera: int
    subject: int
    repetition: int
    action: int
    frame_count: int
    bodies: tuple[NtuSkeletonBody, ...]


def _identity_from_name(name: str) -> tuple[str, int, int, int, int, int]:
    match = _SKELETON_NAME.fullmatch(name) if isinstance(name, str) else None
    if match is None:
        raise ValueError(f"noncanonical NTU skeleton filename: {name!r}")
    setup, camera, subject, repetition, action = (int(value) for value in match.groups())
    for field, value, maximum in (
        ("setup", setup, 32),
        ("camera", camera, 3),
        ("subject", subject, 106),
        ("repetition", repetition, 2),
        ("action", action, 120),
    ):
        if not 1 <= value <= maximum:
            raise ValueError(f"NTU {field} must be in 1..{maximum}: {name!r}")
    return name.removesuffix(".skeleton"), setup, camera, subject, repetition, action


def _positive_int(line: str, *, field: str) -> int:
    text = line.strip()
    try:
        value = int(text)
    except ValueError as exc:
        raise ValueError(f"malformed NTU {field}: {line!r}") from exc
    if value <= 0:
        raise ValueError(f"NTU {field} must be positive")
    return value


def _nonnegative_int(line: str, *, field: str) -> int:
    text = line.strip()
    try:
        value = int(text)
    except ValueError as exc:
        raise ValueError(f"malformed NTU {field}: {line!r}") from exc
    if value < 0:
        raise ValueError(f"NTU {field} must be non-negative")
    return value


def _finite_float(token: str, *, field: str) -> float:
    try:
        value = float(token)
    except ValueError as exc:
        raise ValueError(f"malformed NTU {field}: {token!r}") from exc
    if not np.isfinite(value):
        raise ValueError(f"NTU {field} must be finite")
    return value


def parse_skeleton_file(path: str | Path) -> NtuSkeletonSample:
    source = Path(path)
    sample_id, setup, camera, subject, repetition, action = _identity_from_name(source.name)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"could not read NTU skeleton file: {source}") from exc
    lines = text.splitlines()
    if not lines:
        raise ValueError("NTU skeleton file is empty")

    cursor = 0

    def take(field: str) -> str:
        nonlocal cursor
        if cursor >= len(lines):
            raise ValueError(f"NTU skeleton ended while reading {field}")
        line = lines[cursor]
        cursor += 1
        if not line.strip():
            raise ValueError(f"NTU skeleton has an empty {field} record")
        return line

    frame_count = _positive_int(take("frame count"), field="frame count")
    frame_bodies: list[dict[int, tuple[np.ndarray, np.ndarray]]] = []
    body_ids: set[int] = set()

    for frame_index in range(frame_count):
        body_count = _nonnegative_int(take(f"frame {frame_index} body count"), field="body count")
        current: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for body_index in range(body_count):
            body_tokens = take(f"frame {frame_index} body {body_index} metadata").split()
            if len(body_tokens) != 10:
                raise ValueError("NTU body metadata record must contain exactly 10 fields")
            try:
                body_id = int(body_tokens[0])
            except ValueError as exc:
                raise ValueError("NTU body id must be an integer") from exc
            if body_id <= 0:
                raise ValueError("NTU body id must be positive")
            if body_id in current:
                raise ValueError(f"duplicate NTU body id within frame: {body_id}")
            for token_index, token in enumerate(body_tokens[1:], start=1):
                _finite_float(token, field=f"body metadata field {token_index}")

            joint_count = _nonnegative_int(
                take(f"frame {frame_index} body {body_id} joint count"),
                field="joint count",
            )
            if joint_count != 25:
                raise ValueError(f"NTU body observation must contain exactly 25 joints, got {joint_count}")

            positions = np.zeros((25, 3), dtype=np.float64)
            tracking = np.zeros((25,), dtype=np.int64)
            for joint_index in range(25):
                tokens = take(
                    f"frame {frame_index} body {body_id} joint {joint_index}"
                ).split()
                if len(tokens) != 12:
                    raise ValueError("NTU joint record must contain exactly 12 fields")
                numeric = [
                    _finite_float(token, field=f"joint {joint_index} field {index}")
                    for index, token in enumerate(tokens[:11])
                ]
                try:
                    tracking_state = int(tokens[11])
                except ValueError as exc:
                    raise ValueError("NTU joint tracking state must be an integer") from exc
                if tracking_state not in (0, 1, 2):
                    raise ValueError("NTU joint tracking state must be in {0,1,2}")
                positions[joint_index] = numeric[:3]
                tracking[joint_index] = tracking_state

            current[body_id] = positions, tracking
            body_ids.add(body_id)
        frame_bodies.append(current)

    if cursor != len(lines):
        raise ValueError("NTU skeleton contains trailing records beyond declared frame count")
    if not body_ids:
        raise ValueError("NTU skeleton contains no bodies")

    bodies: list[NtuSkeletonBody] = []
    for body_id in sorted(body_ids):
        positions = np.zeros((frame_count, 25, 3), dtype=np.float64)
        tracking = np.zeros((frame_count, 25), dtype=np.int64)
        present = np.zeros((frame_count,), dtype=bool)
        for frame_index, current in enumerate(frame_bodies):
            observation = current.get(body_id)
            if observation is None:
                continue
            positions[frame_index], tracking[frame_index] = observation
            present[frame_index] = True
        bodies.append(
            NtuSkeletonBody(
                body_id=body_id,
                positions=positions,
                tracking_state=tracking,
                present=present,
            )
        )

    return NtuSkeletonSample(
        sample_id=sample_id,
        setup=setup,
        camera=camera,
        subject=subject,
        repetition=repetition,
        action=action,
        frame_count=frame_count,
        bodies=tuple(bodies),
    )


def select_primary_body(sample: NtuSkeletonSample) -> NtuSkeletonBody:
    if not sample.bodies:
        raise ValueError("NTU skeleton sample contains no bodies")
    return max(sample.bodies, key=lambda body: (body.tracked_joint_count, -body.body_id))


def split_for_subject_gate1(subject: int) -> str:
    if type(subject) is not int or not 1 <= subject <= 106:
        raise ValueError("NTU subject must be an integer in 1..106")
    return "train" if subject in _OUTER_TRAIN else "final_test"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_actions(actions: tuple[int, ...]) -> tuple[int, ...]:
    if type(actions) is not tuple or not actions:
        raise ValueError("NTU Gate 1 actions must be a non-empty tuple")
    if any(type(action) is not int or not 1 <= action <= 120 for action in actions):
        raise ValueError("NTU Gate 1 actions must be integers in 1..120")
    if len(set(actions)) != len(actions):
        raise ValueError("NTU Gate 1 actions must be unique")
    return tuple(sorted(actions))


def build_gate1_skeleton_manifest(
    root: str | Path,
    *,
    actions: tuple[int, ...],
) -> dict[str, Any]:
    selected_actions = _validate_actions(actions)
    directory = Path(root).absolute()
    if not directory.is_dir():
        raise ValueError("NTU skeleton root must be an existing directory")
    for component in (directory, *directory.parents):
        if component.is_symlink():
            raise ValueError("NTU skeleton root must not traverse a symlink")

    candidates: list[tuple[str, Path, tuple[str, int, int, int, int, int]]] = []
    for current_root, dirnames, filenames in os.walk(directory, followlinks=False):
        dirnames.sort()
        for dirname in dirnames:
            if (Path(current_root) / dirname).is_symlink():
                raise ValueError("symlink in NTU skeleton input tree")
        for name in sorted(filenames):
            if not name.lower().endswith(".skeleton"):
                continue
            path = Path(current_root) / name
            if path.is_symlink():
                raise ValueError("symlink in NTU skeleton input tree")
            try:
                identity = _identity_from_name(name)
            except ValueError:
                # A skeleton-looking file with a noncanonical identity is never
                # silently admitted as selected evidence.
                raise
            if identity[-1] not in selected_actions:
                continue
            if not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError("selected NTU skeleton input must be a regular file")
            candidates.append((identity[0], path, identity))

    candidates.sort(key=lambda item: item[0])
    if not candidates:
        raise ValueError("no selected NTU skeleton inputs")

    seen_sample_ids: set[str] = set()
    seen_content: dict[str, str] = {}
    samples: list[dict[str, Any]] = []
    for sample_id, path, identity in candidates:
        if sample_id in seen_sample_ids:
            raise ValueError(f"duplicate NTU skeleton sample ID: {sample_id}")
        seen_sample_ids.add(sample_id)
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"could not read selected NTU skeleton input: {path}") from exc
        if not raw:
            raise ValueError(f"selected NTU skeleton input is empty: {path.name}")
        digest = hashlib.sha256(raw).hexdigest()
        if digest in seen_content:
            raise ValueError(
                f"duplicate NTU skeleton content: {seen_content[digest]} and {sample_id}"
            )
        seen_content[digest] = sample_id

        sample = parse_skeleton_file(path)
        if sample.sample_id != sample_id:
            raise ValueError("NTU skeleton parsed identity changed during inventory")
        primary = select_primary_body(sample)
        samples.append(
            {
                "sample_id": sample.sample_id,
                "subject": sample.subject,
                "action": sample.action,
                "split": split_for_subject_gate1(sample.subject),
                "relative_path": path.relative_to(directory).as_posix(),
                "size_bytes": len(raw),
                "sha256": digest,
                "frame_count": sample.frame_count,
                "selected_body_id": primary.body_id,
                "selected_body_tracked_joint_count": primary.tracked_joint_count,
            }
        )

    dataset_content_hash = _hash_json(
        [
            {
                "sample_id": row["sample_id"],
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
            }
            for row in samples
        ]
    )
    split_policy = {
        "id": "ntu120-official-xsub-outer-v0",
        "reference": "arXiv:1905.04757v2:section-3.2.1",
        "training_subjects": sorted(_OUTER_TRAIN),
        "final_test_subjects": sorted(set(range(1, 107)) - _OUTER_TRAIN),
        "validation_split_defined": False,
    }
    split_hash = _hash_json(
        {
            "dataset_content_hash": dataset_content_hash,
            "split_policy": split_policy,
            "actions": list(selected_actions),
            "body_selection_rule": _BODY_SELECTION_RULE,
            "assignments": [
                {
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "selected_body_id": row["selected_body_id"],
                }
                for row in samples
            ],
        }
    )
    return {
        "format_version": 1,
        "kind": "cx_gate1_ntu120_skeleton_input_inventory",
        "evidence_scope": "input_inventory_only",
        "skeleton_parsing_verified": True,
        "body_selection_rule": _BODY_SELECTION_RULE,
        "actions": list(selected_actions),
        "split_policy": split_policy,
        "samples": samples,
        "dataset_content_hash": dataset_content_hash,
        "split_hash": split_hash,
    }
