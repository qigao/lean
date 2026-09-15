from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .ntu import split_for_subject
from .protocol import ExperimentProtocol


_NAME = re.compile(r"S([0-9]{3})C([0-9]{3})P([0-9]{3})R([0-9]{3})A([0-9]{3})\.skeleton")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_TOP_FIELDS = frozenset({"format_version", "kind", "samples"})
_SAMPLE_FIELDS = frozenset(
    {
        "filename",
        "sample_id",
        "setup",
        "camera",
        "subject",
        "repetition",
        "action",
        "split",
        "size_bytes",
        "sha256",
        "locator",
    }
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _parse_name(filename: str) -> tuple[str, int, int, int, int, int]:
    match = _NAME.fullmatch(filename) if isinstance(filename, str) else None
    if match is None:
        raise ValueError(f"noncanonical NTU skeleton filename: {filename!r}")
    setup, camera, subject, repetition, action = (int(value) for value in match.groups())
    for field, value, maximum in (
        ("setup", setup, 32),
        ("camera", camera, 3),
        ("subject", subject, 106),
        ("repetition", repetition, 2),
        ("action", action, 120),
    ):
        if not 1 <= value <= maximum:
            raise ValueError(f"NTU {field} must be in 1..{maximum}")
    return filename.removesuffix(".skeleton"), setup, camera, subject, repetition, action


def _https_locator(value: object) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError("remote skeleton locator must be a nonempty HTTPS URL")
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("remote skeleton locator must use HTTPS")
    return value


@dataclass(frozen=True)
class VerifiedRemoteSkeletonManifest:
    samples: tuple[dict[str, Any], ...]
    source_manifest_hash: str
    transport_manifest_hash: str

    def public_record(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "kind": "ntu120_skeleton_remote_scientific_source",
            "samples": [
                {key: value for key, value in row.items() if key != "locator"}
                for row in self.samples
            ],
            "source_manifest_hash": self.source_manifest_hash,
        }


def validate_remote_skeleton_manifest(
    value: dict[str, Any],
    protocol: ExperimentProtocol,
) -> VerifiedRemoteSkeletonManifest:
    if type(protocol) is not ExperimentProtocol:
        raise ValueError("protocol must be an ExperimentProtocol")
    if type(value) is not dict or set(value) != set(_TOP_FIELDS):
        raise ValueError("remote skeleton manifest schema mismatch")
    if value["format_version"] != 1 or value["kind"] != "ntu120_skeleton_remote_transport":
        raise ValueError("unsupported remote skeleton manifest format/kind")
    raw_samples = value["samples"]
    if type(raw_samples) is not list or not raw_samples:
        raise ValueError("remote skeleton manifest requires samples")

    seen_ids: set[str] = set()
    seen_hashes: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for raw in raw_samples:
        if type(raw) is not dict or set(raw) != set(_SAMPLE_FIELDS):
            raise ValueError("remote skeleton sample schema mismatch")
        sample_id, setup, camera, subject, repetition, action = _parse_name(raw["filename"])
        if action not in protocol.actions:
            raise ValueError(f"remote skeleton action is outside frozen roster: {action}")
        expected_split = split_for_subject(subject, protocol)
        expected = {
            "sample_id": sample_id,
            "setup": setup,
            "camera": camera,
            "subject": subject,
            "repetition": repetition,
            "action": action,
            "split": expected_split,
        }
        for key, expected_value in expected.items():
            if raw[key] != expected_value:
                raise ValueError(f"remote skeleton {key} differs from canonical identity")
        if sample_id in seen_ids:
            raise ValueError(f"duplicate remote skeleton sample ID: {sample_id}")
        seen_ids.add(sample_id)
        size = raw["size_bytes"]
        if type(size) is not int or size <= 0:
            raise ValueError("remote skeleton size_bytes must be a positive integer")
        digest = raw["sha256"]
        if type(digest) is not str or _DIGEST.fullmatch(digest) is None:
            raise ValueError("remote skeleton SHA-256 must be canonical lowercase")
        if digest in seen_hashes:
            raise ValueError(f"duplicate remote skeleton content: {seen_hashes[digest]} and {sample_id}")
        seen_hashes[digest] = sample_id
        locator = _https_locator(raw["locator"])
        rows.append(
            {
                "filename": raw["filename"],
                **expected,
                "size_bytes": size,
                "sha256": digest,
                "locator": locator,
            }
        )

    rows.sort(key=lambda row: row["sample_id"])
    scientific = {
        "format_version": 1,
        "kind": "ntu120_skeleton_remote_scientific_source",
        "samples": [{key: value for key, value in row.items() if key != "locator"} for row in rows],
    }
    transport = {
        "format_version": 1,
        "kind": "ntu120_skeleton_remote_transport",
        "samples": rows,
    }
    return VerifiedRemoteSkeletonManifest(
        samples=tuple(rows),
        source_manifest_hash=_hash_json(scientific),
        transport_manifest_hash=_hash_json(transport),
    )


def require_development_coverage(
    manifest: VerifiedRemoteSkeletonManifest,
    protocol: ExperimentProtocol,
) -> None:
    required = set(protocol.actions)
    coverage = {"train": set(), "validation": set()}
    for row in manifest.samples:
        split = row["split"]
        if split in coverage:
            coverage[split].add(row["action"])
    for split, actions in coverage.items():
        missing = sorted(required - actions)
        if missing:
            raise ValueError(f"incomplete remote skeleton {split} action coverage: {missing}")


def verify_downloaded_skeleton(path: str | Path, row: dict[str, Any]) -> None:
    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise ValueError(f"could not read downloaded skeleton: {source.name}") from exc
    if len(raw) != row["size_bytes"]:
        raise ValueError(f"downloaded skeleton size mismatch: {source.name}")
    actual = hashlib.sha256(raw).hexdigest()
    if actual != row["sha256"]:
        raise ValueError(f"downloaded skeleton SHA-256 mismatch: {source.name}")


def _download(locator: str, target: Path, *, attempts: int = 5) -> None:
    last_error: Exception | None = None
    for attempt in range(attempts):
        temp = target.with_name(target.name + f".part-{os.getpid()}")
        try:
            request = Request(locator, headers={"User-Agent": "pose-graph-ssm/0.1"})
            with urlopen(request, timeout=120) as response, temp.open("wb") as handle:  # noqa: S310 - HTTPS enforced.
                final_url = response.geturl()
                if urlparse(final_url).scheme != "https":
                    raise ValueError("remote skeleton redirect left HTTPS")
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            os.replace(temp, target)
            return
        except Exception as exc:  # transport errors are retried, then fail closed.
            last_error = exc
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass
            if attempt + 1 < attempts:
                time.sleep(min(2**attempt, 8))
    raise ValueError(f"could not download remote skeleton: {target.name}") from last_error


def materialize_remote_skeletons(
    manifest: VerifiedRemoteSkeletonManifest,
    root: str | Path,
) -> Path:
    if type(manifest) is not VerifiedRemoteSkeletonManifest:
        raise ValueError("verified remote skeleton manifest required")
    output = Path(root)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"skeleton materialization root already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    for row in manifest.samples:
        target = output / row["filename"]
        _download(row["locator"], target)
        verify_downloaded_skeleton(target, row)
    return output


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("could not read remote skeleton manifest JSON") from exc
    if type(value) is not dict:
        raise ValueError("remote skeleton manifest must be a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Validate/materialize private NTU skeleton transport")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "materialize"):
        command = sub.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--protocol", type=Path, required=True)
        if name == "validate":
            command.add_argument("--public-output", type=Path)
        else:
            command.add_argument("--root", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        from .protocol import load_protocol

        protocol = load_protocol(args.protocol)
        verified = validate_remote_skeleton_manifest(_read_json(args.manifest), protocol)
        require_development_coverage(verified, protocol)
        if args.command == "validate":
            public = verified.public_record()
            if args.public_output is not None:
                if args.public_output.exists() or args.public_output.is_symlink():
                    raise FileExistsError("public skeleton source output already exists")
                args.public_output.write_text(_canonical_json(public) + "\n", encoding="utf-8")
            print(
                _canonical_json(
                    {
                        "samples": len(verified.samples),
                        "source_manifest_hash": verified.source_manifest_hash,
                        "transport_manifest_hash": verified.transport_manifest_hash,
                    }
                )
            )
        else:
            materialize_remote_skeletons(verified, args.root)
            print(_canonical_json({"samples": len(verified.samples), "root": args.root.name}))
        return 0
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, f"Remote NTU skeleton error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
