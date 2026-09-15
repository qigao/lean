from __future__ import annotations

from dataclasses import dataclass
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Iterable

from .cx_protocol import CxProtocol
from .graphs import DirectedGraph, graph_fingerprint


_CELL_TYPE_FIELDS = {"root_id", "primary_type"}
_CONNECTION_FIELDS = {"pre_root_id", "post_root_id", "neuropil", "syn_count", "nt_type"}


@dataclass(frozen=True)
class CxNode:
    root_id: int
    primary_type: str
    family: str
    role: str


@dataclass(frozen=True)
class CxArtifact:
    nodes: tuple[CxNode, ...]
    graph: DirectedGraph
    input_indices: tuple[int, ...]
    core_indices: tuple[int, ...]
    output_indices: tuple[int, ...]
    edge_signs: tuple[int, ...]
    source_hashes: tuple[tuple[str, str], ...]
    matched_primary_types: tuple[str, ...]
    fingerprint: str

    def __post_init__(self) -> None:
        if len(self.nodes) != self.graph.num_nodes:
            raise ValueError("CX artifact node count does not match graph")
        if len(self.edge_signs) != self.graph.num_edges:
            raise ValueError("CX artifact edge signs do not align with graph")
        if any(sign not in (-1, 1) for sign in self.edge_signs):
            raise ValueError("CX artifact edge signs must be +/-1")
        role_indices = self.input_indices + self.core_indices + self.output_indices
        if len(role_indices) != len(set(role_indices)):
            raise ValueError("CX artifact role indices must be disjoint")
        if set(role_indices) != set(range(self.graph.num_nodes)):
            raise ValueError("every CX artifact node must have exactly one role")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: Any) -> bytes:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return (text + "\n").encode("utf-8")


def _read_csv(path: str | Path) -> tuple[str, tuple[str, ...], list[dict[str, str]]]:
    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise ValueError(f"could not read CX source: {source}") from exc
    digest = _sha256(raw)
    try:
        payload = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
        text = payload.decode("utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"could not decode CX CSV source: {source}") from exc
    try:
        reader = csv.DictReader(io.StringIO(text, newline=""))
        if reader.fieldnames is None:
            raise ValueError(f"CX CSV source has no header: {source}")
        fields = tuple(reader.fieldnames)
        rows = [dict(row) for row in reader]
    except csv.Error as exc:
        raise ValueError(f"could not parse CX CSV source: {source}") from exc
    return digest, fields, rows


def _require_fields(fields: tuple[str, ...], required: set[str], source: str) -> None:
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def _nonnegative_int(value: str | None, *, field: str, source: str) -> int:
    if value is None or not value.strip():
        raise ValueError(f"{source} has empty {field}")
    try:
        result = int(value.strip())
    except ValueError as exc:
        raise ValueError(f"{source} has malformed {field}: {value!r}") from exc
    if result < 0:
        raise ValueError(f"{source} has negative {field}: {result}")
    return result


def _root_id(value: str | None, *, field: str, source: str) -> int:
    result = _nonnegative_int(value, field=field, source=source)
    if result == 0:
        raise ValueError(f"{source} has invalid zero {field}")
    return result


def _family(primary_type: str, protocol: CxProtocol) -> tuple[str, str] | None:
    matches = [family for family in protocol.all_families if primary_type.startswith(family)]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"ambiguous CX family for {primary_type!r}: {matches}")
    family = matches[0]
    return family, protocol.role_for_family(family)


def _verify_source_hash(protocol: CxProtocol, product: str, actual: str) -> None:
    expected = dict(protocol.source_hashes).get(product)
    if expected is not None and expected != actual:
        raise ValueError(f"source hash mismatch for {product}: expected {expected}, got {actual}")


def build_cx_artifact(
    cell_types_path: str | Path,
    connections_path: str | Path,
    protocol: CxProtocol,
) -> CxArtifact:
    cell_digest, cell_fields, cell_rows = _read_csv(cell_types_path)
    conn_digest, conn_fields, conn_rows = _read_csv(connections_path)
    _require_fields(cell_fields, _CELL_TYPE_FIELDS, "consolidated_cell_types")
    _require_fields(conn_fields, _CONNECTION_FIELDS, "connections_princeton")
    _verify_source_hash(protocol, protocol.cell_types_product, cell_digest)
    _verify_source_hash(protocol, protocol.connections_product, conn_digest)

    seen: set[int] = set()
    selected: list[CxNode] = []
    for row in cell_rows:
        root_id = _root_id(row.get("root_id"), field="root_id", source="consolidated_cell_types")
        if root_id in seen:
            raise ValueError(f"duplicate root_id in consolidated_cell_types: {root_id}")
        seen.add(root_id)
        primary_type = (row.get("primary_type") or "").strip()
        if not primary_type:
            continue
        match = _family(primary_type, protocol)
        if match is None:
            continue
        family, role = match
        selected.append(CxNode(root_id, primary_type, family, role))

    selected.sort(key=lambda node: node.root_id)
    if not selected:
        raise ValueError("CX selection produced no neurons")
    index_by_root = {node.root_id: index for index, node in enumerate(selected)}
    input_indices = tuple(i for i, node in enumerate(selected) if node.role == "input")
    core_indices = tuple(i for i, node in enumerate(selected) if node.role == "core")
    output_indices = tuple(i for i, node in enumerate(selected) if node.role == "output")
    for role, indices in (("input", input_indices), ("core", core_indices), ("output", output_indices)):
        if not indices:
            raise ValueError(f"CX selection produced an empty {role} role")

    pair_counts: dict[tuple[int, int], int] = {}
    pair_nt: dict[tuple[int, int], str] = {}
    for row in conn_rows:
        pre = _root_id(row.get("pre_root_id"), field="pre_root_id", source="connections_princeton")
        post = _root_id(row.get("post_root_id"), field="post_root_id", source="connections_princeton")
        count = _nonnegative_int(row.get("syn_count"), field="syn_count", source="connections_princeton")
        if pre not in index_by_root or post not in index_by_root:
            continue
        key = (pre, post)
        nt_type = (row.get("nt_type") or "").strip()
        previous = pair_nt.get(key)
        if previous is None:
            pair_nt[key] = nt_type
        elif previous != nt_type:
            raise ValueError(
                f"inconsistent transmitter for retained CX pair {pre}->{post}: {previous!r} vs {nt_type!r}"
            )
        pair_counts[key] = pair_counts.get(key, 0) + count

    retained: list[tuple[int, int, float, int]] = []
    for (pre, post), count in sorted(pair_counts.items()):
        if count < protocol.connection_threshold:
            continue
        nt_type = pair_nt[(pre, post)]
        try:
            sign = protocol.transmitter_sign(nt_type)
        except ValueError as exc:
            raise ValueError(f"unsupported transmitter for retained CX pair {pre}->{post}: {nt_type!r}") from exc
        retained.append((index_by_root[pre], index_by_root[post], float(count), sign))
    if not retained:
        raise ValueError("CX selection produced no retained recurrent edges")

    graph = DirectedGraph(
        num_nodes=len(selected),
        src=tuple(src for src, _, _, _ in retained),
        dst=tuple(dst for _, dst, _, _ in retained),
        weight=tuple(weight for _, _, weight, _ in retained),
    )
    edge_signs = tuple(sign for _, _, _, sign in retained)
    source_hashes = (
        (protocol.cell_types_product, cell_digest),
        (protocol.connections_product, conn_digest),
    )
    matched_primary_types = tuple(sorted({node.primary_type for node in selected}))
    nodes = tuple(selected)
    identity = {
        "protocol_id": protocol.protocol_id,
        "dataset": protocol.dataset,
        "release": protocol.release,
        "connection_threshold": protocol.connection_threshold,
        "source_hashes": dict(source_hashes),
        "nodes": [node.__dict__ for node in nodes],
        "graph_fingerprint": graph_fingerprint(graph),
        "edge_signs": list(edge_signs),
        "matched_primary_types": list(matched_primary_types),
        "input_indices": list(input_indices),
        "core_indices": list(core_indices),
        "output_indices": list(output_indices),
    }
    fingerprint = _sha256(_canonical_json(identity))
    return CxArtifact(
        nodes=nodes,
        graph=graph,
        input_indices=input_indices,
        core_indices=core_indices,
        output_indices=output_indices,
        edge_signs=edge_signs,
        source_hashes=source_hashes,
        matched_primary_types=matched_primary_types,
        fingerprint=fingerprint,
    )


def _csv_bytes(header: tuple[str, ...], rows: Iterable[tuple[Any, ...]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def write_cx_artifact(artifact: CxArtifact, output_dir: str | Path) -> dict[str, str]:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"CX artifact output already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    root_ids = tuple(node.root_id for node in artifact.nodes)

    payloads = {
        "nodes.csv": _csv_bytes(
            ("index", "root_id", "primary_type", "family", "role"),
            ((i, n.root_id, n.primary_type, n.family, n.role) for i, n in enumerate(artifact.nodes)),
        ),
        "edges.csv": _csv_bytes(
            ("src_index", "dst_index", "pre_root_id", "post_root_id", "synapse_count", "sign"),
            (
                (src, dst, root_ids[src], root_ids[dst], weight, sign)
                for src, dst, weight, sign in zip(
                    artifact.graph.src, artifact.graph.dst, artifact.graph.weight, artifact.edge_signs
                )
            ),
        ),
    }
    roles = {
        "input_indices": list(artifact.input_indices),
        "input_root_ids": [root_ids[i] for i in artifact.input_indices],
        "core_indices": list(artifact.core_indices),
        "core_root_ids": [root_ids[i] for i in artifact.core_indices],
        "output_indices": list(artifact.output_indices),
        "output_root_ids": [root_ids[i] for i in artifact.output_indices],
    }
    payloads["roles.json"] = _canonical_json(roles)
    payloads["metadata.json"] = _canonical_json(
        {
            "artifact_fingerprint": artifact.fingerprint,
            "graph_fingerprint": graph_fingerprint(artifact.graph),
            "num_nodes": artifact.graph.num_nodes,
            "num_edges": artifact.graph.num_edges,
            "source_hashes": dict(artifact.source_hashes),
            "matched_primary_types": list(artifact.matched_primary_types),
            "input_root_ids": roles["input_root_ids"],
            "core_root_ids": roles["core_root_ids"],
            "output_root_ids": roles["output_root_ids"],
        }
    )

    hashes: dict[str, str] = {}
    for name, data in payloads.items():
        (output / name).write_bytes(data)
        hashes[name] = _sha256(data)
    return hashes
