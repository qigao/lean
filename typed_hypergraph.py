from __future__ import annotations

from dataclasses import dataclass

from typed_graph import NodeKind, TypedNode


@dataclass(frozen=True)
class HyperedgeSignature:
    premise_kinds: tuple[NodeKind, ...]
    conclusion_kind: NodeKind


@dataclass(frozen=True)
class TypedHyperedge:
    signature: HyperedgeSignature
    premises: tuple[TypedNode, ...]
    conclusion: TypedNode


def make_hyperedge(
    signature: HyperedgeSignature,
    premises: tuple[TypedNode, ...],
    conclusion: TypedNode,
) -> TypedHyperedge:
    """Runtime mirror of Lean's dependent typed-hyperedge construction."""
    actual_premise_kinds = tuple(node.kind for node in premises)
    if actual_premise_kinds != signature.premise_kinds:
        raise ValueError(
            "hyperedge premises do not match signature: "
            f"expected {signature.premise_kinds}, got {actual_premise_kinds}"
        )
    if conclusion.kind != signature.conclusion_kind:
        raise ValueError(
            "hyperedge conclusion does not match signature: "
            f"expected {signature.conclusion_kind}, got {conclusion.kind}"
        )
    return TypedHyperedge(signature, premises, conclusion)
