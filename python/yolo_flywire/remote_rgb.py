"""Validate private remote NTU RGB transport manifests without downloading media."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse

from .ntu_io import (
    _ACTION_LABELS, _canonical_json, _hash_json, _split_policy,
    parse_rgb_name, split_for_subject,
)

_DIGEST = re.compile(r"[0-9a-f]{64}")
_MANIFEST_FIELDS = {"format_version", "kind", "task_labels", "split_policy", "samples"}
_SAMPLE_FIELDS = {
    "filename", "sample_id", "setup", "camera", "subject", "repetition", "action",
    "label", "split", "size_bytes", "sha256", "locator",
}


@dataclass(frozen=True)
class VerifiedRemoteManifest:
    """Canonical remote source identities plus private transport rows.

    ``inventory`` and ``source_manifest_hash`` contain no transport locator text.
    ``transport_manifest_hash`` identifies the current locator set for this run only.
    """

    inventory: dict[str, Any]
    samples: tuple[dict[str, Any], ...]
    source_manifest_hash: str
    transport_manifest_hash: str

    @property
    def dataset_content_hash(self) -> str:
        return self.inventory["dataset_content_hash"]

    @property
    def split_hash(self) -> str:
        return self.inventory["split_hash"]

    @property
    def input_inventory_hash(self) -> str:
        return _hash_json(self.inventory)

    def public_record(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "kind": "ntu120_rgb_remote_scientific_source",
            "task_labels": self.inventory["task_labels"],
            "split_policy": self.inventory["split_policy"],
            "samples": [
                {key: value for key, value in row.items() if key != "locator"}
                for row in self.samples
            ],
            "dataset_content_hash": self.dataset_content_hash,
            "split_hash": self.split_hash,
            "input_inventory_hash": self.input_inventory_hash,
            "source_manifest_hash": self.source_manifest_hash,
        }


def _same(actual: Any, expected: Any, context: str) -> None:
    if _canonical_json(actual) != _canonical_json(expected):
        raise ValueError(f"remote manifest {context} differs from the frozen NTU contract")


def _positive_int(value: Any, context: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"remote manifest {context} must be a positive integer")
    return value


def _locator(value: Any) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError("remote manifest locator must be a nonempty string")
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("remote manifest currently supports explicit HTTPS locators only")
    return value


def validate_remote_manifest(value: dict[str, Any]) -> VerifiedRemoteManifest:
    """Validate complete frozen remote roster and derive URL-free scientific hashes."""
    if type(value) is not dict or set(value) != _MANIFEST_FIELDS:
        raise ValueError("remote manifest must contain exactly the frozen top-level fields")
    if value["format_version"] != 1 or value["kind"] != "ntu120_rgb_remote_transport":
        raise ValueError("unsupported remote manifest format/kind")
    labels = [_ACTION_LABELS[action] for action in sorted(_ACTION_LABELS)]
    _same(value["task_labels"], labels, "task labels")
    _same(value["split_policy"], _split_policy(), "split policy")
    raw_samples = value["samples"]
    if type(raw_samples) is not list or not raw_samples:
        raise ValueError("remote manifest requires a nonempty samples array")

    sample_ids: set[str] = set()
    content_ids: dict[str, str] = {}
    coverage = {name: set() for name in ("train", "validation", "final_test")}
    transport_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    for raw in raw_samples:
        if type(raw) is not dict or set(raw) != _SAMPLE_FIELDS:
            raise ValueError("remote sample contains missing or unknown fields")
        identity = parse_rgb_name(raw["filename"])
        expected = {
            "sample_id": identity.sample_id,
            "setup": identity.setup,
            "camera": identity.camera,
            "subject": identity.subject,
            "repetition": identity.repetition,
            "action": identity.action,
            "label": _ACTION_LABELS.get(identity.action),
            "split": split_for_subject(identity.subject),
        }
        if expected["label"] is None:
            raise ValueError("remote manifest contains an action outside the frozen ten-class task")
        for key, expected_value in expected.items():
            _same(raw[key], expected_value, f"sample {identity.sample_id} {key}")
        if identity.sample_id in sample_ids:
            raise ValueError(f"duplicate remote sample ID: {identity.sample_id}")
        sample_ids.add(identity.sample_id)
        size = _positive_int(raw["size_bytes"], f"sample {identity.sample_id} size")
        digest = raw["sha256"]
        if type(digest) is not str or _DIGEST.fullmatch(digest) is None:
            raise ValueError(f"sample {identity.sample_id} SHA-256 must be canonical lowercase")
        if digest in content_ids:
            raise ValueError(
                f"duplicate remote RGB content: {content_ids[digest]} and {identity.sample_id}"
            )
        content_ids[digest] = identity.sample_id
        locator = _locator(raw["locator"])
        coverage[expected["split"]].add(identity.action)
        normalized = {
            "filename": raw["filename"], **expected,
            "size_bytes": size, "sha256": digest, "locator": locator,
        }
        transport_rows.append(normalized)
        inventory_rows.append({
            "sample_id": identity.sample_id, "setup": identity.setup, "camera": identity.camera,
            "subject": identity.subject, "repetition": identity.repetition,
            "action": identity.action, "label": expected["label"], "split": expected["split"],
            "relative_path": raw["filename"], "size_bytes": size, "sha256": digest,
        })

    transport_rows.sort(key=lambda row: row["sample_id"])
    inventory_rows.sort(key=lambda row: row["sample_id"])
    required_actions = set(_ACTION_LABELS)
    for split, actions in coverage.items():
        missing = sorted(required_actions - actions)
        if missing:
            raise ValueError(f"incomplete remote {split} action coverage: {missing}")

    dataset_content_hash = _hash_json([
        {key: row[key] for key in ("sample_id", "size_bytes", "sha256")}
        for row in inventory_rows
    ])
    policy = _split_policy()
    split_hash = _hash_json({
        "dataset_content_hash": dataset_content_hash,
        "policy": policy,
        "task_labels": labels,
        "assignments": [
            {"sample_id": row["sample_id"], "split": row["split"]}
            for row in inventory_rows
        ],
    })
    inventory = {
        "format_version": 1,
        "kind": "ntu120_rgb_input_inventory",
        "evidence_scope": "input_inventory_only",
        "media_decoding_verified": False,
        "official_release_completeness_verified": False,
        "task_labels": labels,
        "split_policy": policy,
        "samples": inventory_rows,
        "dataset_content_hash": dataset_content_hash,
        "split_hash": split_hash,
    }
    scientific_payload = {
        "format_version": 1,
        "kind": "ntu120_rgb_remote_scientific_source",
        "task_labels": labels,
        "split_policy": policy,
        "samples": [
            {key: value for key, value in row.items() if key != "locator"}
            for row in transport_rows
        ],
    }
    transport_payload = {
        "format_version": 1,
        "kind": "ntu120_rgb_remote_transport",
        "task_labels": labels,
        "split_policy": policy,
        "samples": transport_rows,
    }
    return VerifiedRemoteManifest(
        inventory=inventory,
        samples=tuple(transport_rows),
        source_manifest_hash=_hash_json(scientific_payload),
        transport_manifest_hash=_hash_json(transport_payload),
    )


def shard_rows(source: VerifiedRemoteManifest, shard_index: int, shard_count: int) -> tuple[dict[str, Any], ...]:
    if type(source) is not VerifiedRemoteManifest:
        raise ValueError("explicit VerifiedRemoteManifest required")
    if type(shard_count) is not int or shard_count <= 0:
        raise ValueError("shard_count must be a positive integer")
    if type(shard_index) is not int or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    return tuple(row for index, row in enumerate(source.samples) if index % shard_count == shard_index)


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("could not read remote transport manifest JSON") from exc
    if type(value) is not dict:
        raise ValueError("remote transport manifest must be a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--public-output", type=Path)
    args = parser.parse_args(argv)
    try:
        verified = validate_remote_manifest(_read(args.manifest))
        public = verified.public_record()
        if args.public_output is not None:
            if args.public_output.exists() or args.public_output.is_symlink():
                raise ValueError("public source output already exists")
            args.public_output.write_text(_canonical_json(public) + "\n", encoding="utf-8")
        print(_canonical_json({
            "samples": len(verified.samples),
            "dataset_content_hash": verified.dataset_content_hash,
            "split_hash": verified.split_hash,
            "source_manifest_hash": verified.source_manifest_hash,
            "transport_manifest_hash": verified.transport_manifest_hash,
        }))
        return 0
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, f"Remote NTU source error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
