"""Freeze and verify local NTU RGB input bytes, without decoding or training."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any


_RGB_NAME = re.compile(
    r"S([0-9]{3})C([0-9]{3})P([0-9]{3})R([0-9]{3})A([0-9]{3})_rgb\.avi"
)
_ACTION_LABELS = {
    8: "A8:sitting_down",
    9: "A9:standing_up",
    22: "A22:cheer_up",
    23: "A23:hand_waving",
    26: "A26:hopping",
    27: "A27:jump_up",
    31: "A31:pointing",
    34: "A34:rub_two_hands_together",
    35: "A35:nod_head_or_bow",
    36: "A36:shake_head",
}
# Explicit reference: Liu et al., arXiv:1905.04757v2, section 3.2.1.
# In that reference P056 trains and P106 is on the final-test side.
_OUTER_TRAIN = frozenset({
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31,
    34, 35, 38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59,
    70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95,
    97, 98, 100, 103,
})
_VALIDATION_PREFIX = "yolo-flywire:ntu120:v0:validation:"
_VALIDATION_COUNT = 11
_VALIDATION = frozenset(sorted(
    _OUTER_TRAIN,
    key=lambda subject: (
        hashlib.sha256(f"{_VALIDATION_PREFIX}P{subject:03d}".encode("ascii")).digest(),
        subject,
    ),
)[:_VALIDATION_COUNT])
_PARTITIONS = ("train", "validation", "final_test")
_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class NtuSample:
    sample_id: str
    setup: int
    camera: int
    subject: int
    repetition: int
    action: int


def parse_rgb_name(name: str) -> NtuSample:
    """Parse a canonical basename, not a path or an alternative modality."""
    match = _RGB_NAME.fullmatch(name) if isinstance(name, str) else None
    if match is None:
        raise ValueError(f"noncanonical NTU RGB filename: {name!r}")
    values = tuple(int(value) for value in match.groups())
    fields = ("setup", "camera", "subject", "repetition", "action")
    for field, value, maximum in zip(fields, values, (32, 3, 106, 2, 120)):
        if not 1 <= value <= maximum:
            raise ValueError(f"NTU {field} must be in 1..{maximum}: {name!r}")
    return NtuSample(name.removesuffix("_rgb.avi"), *values)


def split_for_subject(subject: int) -> str:
    """Assign a whole subject independently of available data and model seeds."""
    if type(subject) is not int or not 1 <= subject <= 106:
        raise ValueError("NTU subject must be an integer in 1..106")
    if subject not in _OUTER_TRAIN:
        return "final_test"
    return "validation" if subject in _VALIDATION else "train"


def _split_policy() -> dict[str, Any]:
    return {
        "id": "ntu120-paper-v2-xsub-subject-hash-validation-v0",
        "outer_reference": "arXiv:1905.04757v2:section-3.2.1",
        "outer_training_subjects": sorted(_OUTER_TRAIN),
        "training_subjects": sorted(_OUTER_TRAIN - _VALIDATION),
        "validation_subjects": sorted(_VALIDATION),
        "final_test_subjects": sorted(set(range(1, 107)) - _OUTER_TRAIN),
        "validation_rule": "first-11-by-(sha256-ascii-prefix-plus-Pddd,subject-id)",
        "validation_prefix": _VALIDATION_PREFIX,
        "validation_subject_count": _VALIDATION_COUNT,
        "validation_is_official": False,
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _signature(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def _scan(root: Path) -> list[tuple[NtuSample, Path]]:
    def walk_error(error: OSError) -> None:
        raise error  # os.walk must not silently omit unreadable directories.

    selected: dict[str, tuple[NtuSample, Path]] = {}
    for directory, dirnames, filenames in os.walk(root, followlinks=False, onerror=walk_error):
        dirnames.sort()
        for name in sorted(dirnames + filenames):
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError(f"symlink in input tree: {path.relative_to(root)}")
            if not name.lower().endswith("_rgb.avi"):
                continue
            identity = parse_rgb_name(name)
            if identity.action not in _ACTION_LABELS:
                continue
            if not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError(f"RGB input must be a regular file: {path.relative_to(root)}")
            if identity.sample_id in selected:
                raise ValueError(f"duplicate sample ID: {identity.sample_id}")
            selected[identity.sample_id] = identity, path
    return [selected[key] for key in sorted(selected)]


def _hash_file(path: Path) -> tuple[int, str, tuple[int, ...]]:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
        raise ValueError(f"RGB input must be a nonempty regular file: {path.name}")
    digest = hashlib.sha256()
    length = 0
    with path.open("rb") as handle:
        if _signature(os.fstat(handle.fileno())) != _signature(before):
            raise ValueError(f"RGB input changed before hashing: {path.name}")
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
            length += len(chunk)
        after = os.fstat(handle.fileno())
    if (length != before.st_size or _signature(after) != _signature(before)
            or _signature(path.lstat()) != _signature(before)):
        raise ValueError(f"RGB input changed while hashing: {path.name}")
    return length, digest.hexdigest(), _signature(before)


def build_rgb_manifest(root: str | Path) -> dict[str, Any]:
    """Inventory selected bytes; neither video validity nor release completeness is inferred.

    Input trees must be immutable while being inventoried. Stat and roster checks
    detect observed changes, but are not an atomic filesystem snapshot.
    """
    directory = Path(root).absolute()
    for component in (directory, *directory.parents):
        if component.is_symlink():
            raise ValueError("input root must not traverse a symlink")
    if not directory.is_dir():
        raise ValueError("input root must be an existing directory")
    selected = _scan(directory)
    if not selected:
        raise ValueError("no selected NTU RGB inputs")

    samples: list[dict[str, Any]] = []
    signatures: dict[Path, tuple[int, ...]] = {}
    content_ids: dict[str, str] = {}
    coverage: dict[str, set[int]] = {name: set() for name in _PARTITIONS}
    for identity, path in selected:
        length, digest, signature = _hash_file(path)
        if digest in content_ids:
            raise ValueError(
                f"duplicate RGB content: {content_ids[digest]} and {identity.sample_id}"
            )
        content_ids[digest] = identity.sample_id
        signatures[path] = signature
        split = split_for_subject(identity.subject)
        coverage[split].add(identity.action)
        samples.append({
            **asdict(identity), "label": _ACTION_LABELS[identity.action], "split": split,
            "relative_path": path.relative_to(directory).as_posix(),
            "size_bytes": length, "sha256": digest,
        })
    for split, actions in coverage.items():
        missing = sorted(set(_ACTION_LABELS) - actions)
        if missing:
            raise ValueError(f"incomplete {split} action coverage: {missing}")
    if _scan(directory) != selected:
        raise ValueError("selected RGB roster changed during inventory")
    for path, signature in signatures.items():
        if _signature(path.lstat()) != signature:
            raise ValueError(f"RGB input changed during inventory: {path.name}")

    content_hash = _hash_json([
        {key: row[key] for key in ("sample_id", "size_bytes", "sha256")}
        for row in samples
    ])
    labels = [_ACTION_LABELS[action] for action in sorted(_ACTION_LABELS)]
    policy = _split_policy()
    split_hash = _hash_json({
        "dataset_content_hash": content_hash, "policy": policy, "task_labels": labels,
        "assignments": [{"sample_id": row["sample_id"], "split": row["split"]}
                        for row in samples],
    })
    return {
        "format_version": 1, "kind": "ntu120_rgb_input_inventory",
        "evidence_scope": "input_inventory_only",
        "media_decoding_verified": False,
        "official_release_completeness_verified": False,
        "task_labels": labels, "split_policy": policy, "samples": samples,
        "dataset_content_hash": content_hash, "split_hash": split_hash,
    }


def verify_rgb_manifest(root: str | Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Re-enumerate explicit local inputs, never follow unverified manifest paths."""
    if not isinstance(manifest, dict):
        raise ValueError("RGB manifest must be a JSON object")
    expected = build_rgb_manifest(root)
    try:
        # Canonical text distinguishes integers from booleans/floats unlike dict ==.
        matches = _canonical_json(manifest) == _canonical_json(expected)
    except (TypeError, ValueError) as exc:
        raise ValueError("RGB manifest contains invalid JSON values") from exc
    if not matches:
        raise ValueError("RGB manifest does not match local bytes, roster or split policy")
    return expected


def _publish_exclusive(output: Path, manifest: dict[str, Any]) -> None:
    raw = (_canonical_json(manifest) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".ntu-inventory-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        # An atomic, exclusive publish: never replace an existing output or symlink.
        # Unsupported filesystems fail rather than falling back to partial writes.
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="inventory local RGB bytes without overwriting evidence")
    freeze.add_argument("--root", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify", help="rebuild and verify a local input manifest")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            if args.output.exists() or args.output.is_symlink():
                raise ValueError("output already exists; refusing to overwrite evidence")
            manifest = build_rgb_manifest(args.root)
            _publish_exclusive(args.output, manifest)
        else:
            supplied = json.loads(args.manifest.read_text(encoding="utf-8"))
            manifest = verify_rgb_manifest(args.root, supplied)
        print(_canonical_json({
            "mode": args.command, "samples": len(manifest["samples"]),
            "dataset_content_hash": manifest["dataset_content_hash"],
            "split_hash": manifest["split_hash"], "evidence_scope": "input_inventory_only",
        }))
        return 0
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, f"NTU input error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
