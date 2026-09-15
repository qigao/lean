from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from .cx_artifact import CxArtifact, build_cx_artifact, write_cx_artifact
from .cx_nulls import rewire_cx_block_preserving, rewire_cx_degree_preserving
from .cx_protocol import CxProtocol


@dataclass(frozen=True)
class CxProvenanceReport:
    seeds: tuple[int, ...]
    successful_swaps: int
    source_hashes: tuple[tuple[str, str], ...]
    real_artifact_fingerprint: str
    input_node_fingerprint: str
    output_node_fingerprint: str
    degree_null_fingerprints: tuple[tuple[int, str], ...]
    block_null_fingerprints: tuple[tuple[int, str], ...]
    artifact_file_hashes: tuple[tuple[str, str], ...]
    bundle_hash: str


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    try:
        return _sha256(path.read_bytes())
    except OSError as exc:
        raise ValueError(f"could not read provenance artifact file: {path}") from exc


def _node_fingerprint(artifact: CxArtifact, indices: tuple[int, ...], role: str) -> str:
    payload = {
        "role": role,
        "nodes": [
            {
                "index": index,
                "root_id": artifact.nodes[index].root_id,
                "primary_type": artifact.nodes[index].primary_type,
                "family": artifact.nodes[index].family,
                "role": artifact.nodes[index].role,
            }
            for index in indices
        ],
    }
    return _sha256(_canonical_json(payload))


def _expected_swaps(protocol: CxProtocol, artifact: CxArtifact) -> int:
    off_diagonal = sum(1 for src, dst in zip(artifact.graph.src, artifact.graph.dst) if src != dst)
    if off_diagonal <= 0:
        raise ValueError("CX provenance requires at least one off-diagonal recurrent edge")
    return off_diagonal * protocol.rewiring_successful_swaps_per_offdiagonal_edge


def _check_optional_pin(name: str, expected: str | None, actual: str) -> None:
    if expected is not None and expected != actual:
        raise ValueError(f"{name} mismatch: expected {expected}, got {actual}")


def _check_optional_seed_pins(
    name: str,
    expected: tuple[tuple[str, str], ...] | None,
    actual: tuple[tuple[int, str], ...],
) -> None:
    if expected is None:
        return
    expected_map = {int(seed): digest for seed, digest in expected}
    actual_map = dict(actual)
    if expected_map != actual_map:
        raise ValueError(f"{name} mismatch")


def _report_payload(
    *,
    protocol: CxProtocol,
    real: CxArtifact,
    input_node_fingerprint: str,
    output_node_fingerprint: str,
    successful_swaps: int,
    degree: tuple[tuple[int, CxArtifact], ...],
    block: tuple[tuple[int, CxArtifact], ...],
    artifact_file_hashes: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "protocol_id": protocol.protocol_id,
        "dataset": protocol.dataset,
        "release": protocol.release,
        "connection_threshold": protocol.connection_threshold,
        "degree_rewiring_algorithm": protocol.degree_rewiring_algorithm,
        "block_rewiring_algorithm": protocol.block_rewiring_algorithm,
        "successful_swaps": successful_swaps,
        "seeds": list(protocol.seeds),
        "nt_signs": dict(protocol.nt_signs),
        "source_hashes": dict(real.source_hashes),
        "real_artifact_fingerprint": real.fingerprint,
        "input_node_fingerprint": input_node_fingerprint,
        "output_node_fingerprint": output_node_fingerprint,
        "degree_null_fingerprints": {str(seed): artifact.fingerprint for seed, artifact in degree},
        "block_null_fingerprints": {str(seed): artifact.fingerprint for seed, artifact in block},
        "artifact_file_hashes": dict(artifact_file_hashes),
    }


def _report_from_payload(payload: dict[str, Any], bundle_hash: str) -> CxProvenanceReport:
    try:
        seeds = tuple(int(seed) for seed in payload["seeds"])
        source_hashes = tuple(sorted((str(name), str(digest)) for name, digest in payload["source_hashes"].items()))
        degree = tuple((seed, str(payload["degree_null_fingerprints"][str(seed)])) for seed in seeds)
        block = tuple((seed, str(payload["block_null_fingerprints"][str(seed)])) for seed in seeds)
        files = tuple(sorted((str(name), str(digest)) for name, digest in payload["artifact_file_hashes"].items()))
        return CxProvenanceReport(
            seeds=seeds,
            successful_swaps=int(payload["successful_swaps"]),
            source_hashes=source_hashes,
            real_artifact_fingerprint=str(payload["real_artifact_fingerprint"]),
            input_node_fingerprint=str(payload["input_node_fingerprint"]),
            output_node_fingerprint=str(payload["output_node_fingerprint"]),
            degree_null_fingerprints=degree,
            block_null_fingerprints=block,
            artifact_file_hashes=files,
            bundle_hash=bundle_hash,
        )
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError("CX provenance bundle is missing or has malformed required fields") from exc


def _write_artifacts(
    root: Path,
    real: CxArtifact,
    degree: tuple[tuple[int, CxArtifact], ...],
    block: tuple[tuple[int, CxArtifact], ...],
) -> tuple[tuple[str, str], ...]:
    hashes: dict[str, str] = {}
    for relative, artifact in (("real", real),):
        for name, digest in write_cx_artifact(artifact, root / relative).items():
            hashes[f"{relative}/{name}"] = digest
    for kind, rows in (("degree", degree), ("block", block)):
        for seed, artifact in rows:
            relative = f"{kind}/{seed}"
            for name, digest in write_cx_artifact(artifact, root / relative).items():
                hashes[f"{relative}/{name}"] = digest
    return tuple(sorted(hashes.items()))


def build_cx_provenance(
    protocol: CxProtocol,
    cell_types_path: str | Path,
    connections_path: str | Path,
    output_dir: str | Path,
) -> CxProvenanceReport:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"CX provenance output already exists: {output}")

    # Build and validate against any frozen source pins before creating output.
    real = build_cx_artifact(cell_types_path, connections_path, protocol)
    input_fingerprint = _node_fingerprint(real, real.input_indices, "input")
    output_fingerprint = _node_fingerprint(real, real.output_indices, "output")
    _check_optional_pin("cx_artifact_fingerprint", protocol.cx_artifact_fingerprint, real.fingerprint)
    _check_optional_pin("input_node_fingerprint", protocol.input_node_fingerprint, input_fingerprint)
    _check_optional_pin("output_node_fingerprint", protocol.output_node_fingerprint, output_fingerprint)

    swaps = _expected_swaps(protocol, real)
    degree = tuple(
        (seed, rewire_cx_degree_preserving(real, seed=seed, swaps=swaps))
        for seed in protocol.seeds
    )
    block = tuple(
        (seed, rewire_cx_block_preserving(real, seed=seed, swaps=swaps))
        for seed in protocol.seeds
    )
    degree_fingerprints = tuple((seed, artifact.fingerprint) for seed, artifact in degree)
    block_fingerprints = tuple((seed, artifact.fingerprint) for seed, artifact in block)
    _check_optional_seed_pins("degree_null_fingerprints", protocol.degree_null_fingerprints, degree_fingerprints)
    _check_optional_seed_pins("block_null_fingerprints", protocol.block_null_fingerprints, block_fingerprints)

    temporary = output.with_name(output.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"CX provenance temporary output already exists: {temporary}")
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        artifact_hashes = _write_artifacts(temporary, real, degree, block)
        payload = _report_payload(
            protocol=protocol,
            real=real,
            input_node_fingerprint=input_fingerprint,
            output_node_fingerprint=output_fingerprint,
            successful_swaps=swaps,
            degree=degree,
            block=block,
            artifact_file_hashes=artifact_hashes,
        )
        bundle_hash = _sha256(_canonical_json(payload))
        manifest = {**payload, "bundle_hash": bundle_hash}
        (temporary / "provenance.json").write_bytes(_canonical_json(manifest))
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return _report_from_payload(payload, bundle_hash)


def _read_manifest(bundle: Path) -> tuple[dict[str, Any], str]:
    path = bundle / "provenance.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("could not read CX provenance bundle manifest") from exc
    if not isinstance(raw, dict):
        raise ValueError("CX provenance bundle manifest must be an object")
    try:
        stored_hash = raw["bundle_hash"]
    except KeyError as exc:
        raise ValueError("CX provenance bundle is missing bundle_hash") from exc
    if not isinstance(stored_hash, str):
        raise ValueError("CX provenance bundle hash is malformed")
    payload = dict(raw)
    payload.pop("bundle_hash", None)
    actual_hash = _sha256(_canonical_json(payload))
    if stored_hash != actual_hash:
        raise ValueError("CX provenance bundle hash mismatch")
    return payload, stored_hash


def _verify_artifact_file_hashes(bundle: Path, payload: dict[str, Any]) -> None:
    value = payload.get("artifact_file_hashes")
    if not isinstance(value, dict) or not value:
        raise ValueError("CX provenance bundle has no artifact file hashes")
    for relative, expected in value.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("CX provenance artifact file hash entry is malformed")
        actual = _file_sha256(bundle / relative)
        if actual != expected:
            raise ValueError(f"CX provenance artifact file hash mismatch: {relative}")


def verify_cx_provenance(
    protocol: CxProtocol,
    cell_types_path: str | Path,
    connections_path: str | Path,
    bundle_dir: str | Path,
) -> CxProvenanceReport:
    bundle = Path(bundle_dir)
    payload, bundle_hash = _read_manifest(bundle)
    required_scalar = {
        "format_version": 1,
        "protocol_id": protocol.protocol_id,
        "dataset": protocol.dataset,
        "release": protocol.release,
        "connection_threshold": protocol.connection_threshold,
        "degree_rewiring_algorithm": protocol.degree_rewiring_algorithm,
        "block_rewiring_algorithm": protocol.block_rewiring_algorithm,
        "seeds": list(protocol.seeds),
        "nt_signs": dict(protocol.nt_signs),
    }
    for name, expected in required_scalar.items():
        if payload.get(name) != expected:
            if name == "nt_signs":
                raise ValueError("CX provenance sign policy mismatch")
            raise ValueError(f"CX provenance bundle {name} mismatch")

    real = build_cx_artifact(cell_types_path, connections_path, protocol)
    swaps = _expected_swaps(protocol, real)
    if payload.get("successful_swaps") != swaps:
        raise ValueError("CX provenance swap budget mismatch")
    input_fingerprint = _node_fingerprint(real, real.input_indices, "input")
    output_fingerprint = _node_fingerprint(real, real.output_indices, "output")

    degree = tuple(
        (seed, rewire_cx_degree_preserving(real, seed=seed, swaps=swaps))
        for seed in protocol.seeds
    )
    block = tuple(
        (seed, rewire_cx_block_preserving(real, seed=seed, swaps=swaps))
        for seed in protocol.seeds
    )
    expected_payload = _report_payload(
        protocol=protocol,
        real=real,
        input_node_fingerprint=input_fingerprint,
        output_node_fingerprint=output_fingerprint,
        successful_swaps=swaps,
        degree=degree,
        block=block,
        artifact_file_hashes=tuple(sorted((payload.get("artifact_file_hashes") or {}).items())),
    )
    for name in (
        "source_hashes",
        "real_artifact_fingerprint",
        "input_node_fingerprint",
        "output_node_fingerprint",
        "degree_null_fingerprints",
        "block_null_fingerprints",
    ):
        if payload.get(name) != expected_payload[name]:
            raise ValueError(f"CX provenance {name} mismatch")

    _check_optional_pin("cx_artifact_fingerprint", protocol.cx_artifact_fingerprint, real.fingerprint)
    _check_optional_pin("input_node_fingerprint", protocol.input_node_fingerprint, input_fingerprint)
    _check_optional_pin("output_node_fingerprint", protocol.output_node_fingerprint, output_fingerprint)
    _check_optional_seed_pins(
        "degree_null_fingerprints",
        protocol.degree_null_fingerprints,
        tuple((seed, artifact.fingerprint) for seed, artifact in degree),
    )
    _check_optional_seed_pins(
        "block_null_fingerprints",
        protocol.block_null_fingerprints,
        tuple((seed, artifact.fingerprint) for seed, artifact in block),
    )
    _verify_artifact_file_hashes(bundle, payload)
    return _report_from_payload(payload, bundle_hash)
