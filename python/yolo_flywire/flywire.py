from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path

from .graphs import DirectedGraph


@dataclass(frozen=True)
class FlyWireEdge:
    pre_id: int
    post_id: int
    synapse_count: int
    region: str | None = None
    cell_type: str | None = None

    def __post_init__(self) -> None:
        if self.pre_id == self.post_id:
            raise ValueError("FlyWire self-loop edges are not supported in V0")
        if self.synapse_count <= 0:
            raise ValueError("synapse_count must be positive")


@dataclass(frozen=True)
class FlyWireSelection:
    release_id: str
    selection_rule: str
    neuron_ids: tuple[int, ...]
    edges: tuple[FlyWireEdge, ...]

    def __post_init__(self) -> None:
        if not self.release_id.strip() or not self.selection_rule.strip():
            raise ValueError("FlyWire provenance requires non-empty release_id and selection_rule")
        if not self.neuron_ids:
            raise ValueError("FlyWire selection must contain at least one neuron")
        known = set(self.neuron_ids)
        if len(known) != len(self.neuron_ids):
            raise ValueError("neuron_ids must be unique")
        for edge in self.edges:
            if edge.pre_id not in known or edge.post_id not in known:
                raise ValueError("edge references neuron outside selection")


def load_flywire_csv(
    path: str | Path,
    release_id: str,
    selection_rule: str,
) -> FlyWireSelection:
    if not release_id.strip() or not selection_rule.strip():
        raise ValueError("FlyWire provenance requires non-empty release_id and selection_rule")

    source = Path(path)
    edges: list[FlyWireEdge] = []
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"pre_id", "post_id", "synapse_count"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("FlyWire CSV requires pre_id, post_id, and synapse_count columns")
        for row in reader:
            edges.append(
                FlyWireEdge(
                    pre_id=int(row["pre_id"]),
                    post_id=int(row["post_id"]),
                    synapse_count=int(row["synapse_count"]),
                    region=(row.get("region") or None),
                    cell_type=(row.get("cell_type") or None),
                )
            )

    if not edges:
        raise ValueError("FlyWire CSV contains no edges")
    neuron_ids = tuple(sorted({edge.pre_id for edge in edges} | {edge.post_id for edge in edges}))
    return FlyWireSelection(
        release_id=release_id,
        selection_rule=selection_rule,
        neuron_ids=neuron_ids,
        edges=tuple(edges),
    )


def selection_to_graph(selection: FlyWireSelection) -> DirectedGraph:
    remap = {neuron_id: index for index, neuron_id in enumerate(sorted(selection.neuron_ids))}
    aggregate: dict[tuple[int, int], float] = {}
    for edge in selection.edges:
        key = (remap[edge.pre_id], remap[edge.post_id])
        aggregate[key] = aggregate.get(key, 0.0) + float(edge.synapse_count)
    ordered = sorted((src, dst, weight) for (src, dst), weight in aggregate.items())
    return DirectedGraph(
        num_nodes=len(remap),
        src=tuple(src for src, _, _ in ordered),
        dst=tuple(dst for _, dst, _ in ordered),
        weight=tuple(weight for _, _, weight in ordered),
    )
