"""Reviewable, deterministic V19 real social-memory checkpoint evidence.

Every value comes from the existing V19 real runtime model, immutable social
state and original projector, not a parallel graph model. A recorded SHA is
a reproducibility binding, never an actor signature or authorization proof.
"""
from __future__ import annotations

import json
from tempfile import TemporaryDirectory
from pathlib import Path

from narrative_dynamics.abm.mesh_contracts import MeshRelation, MeshSnapshot, MeshSociety
from narrative_dynamics.abm.mesh_queries import find_mesh_path, mesh_path_is_valid
from narrative_dynamics.abm.mesh_topology_events import observe_topology
from narrative_dynamics.abm.situated_mesh import mesh_edges
from narrative_dynamics.abm.situated_network import (
    initialize_situated_network_runtime, project_situated_network_snapshot,
)
from narrative_dynamics.abm.situated_social_topology import (
    SocialTrustOperation, SocialTrustCheckpointIntervention,
    admit_social_trust_checkpoint, verify_social_trust_checkpoint,
)
from tests.test_network_abm_situated_network import initial_runtime_case

SCHEMA = "v19-social-topology-checkpoint-v1"


def _event(runtime, event_id: str, operation: SocialTrustOperation, trust: float):
    return SocialTrustCheckpointIntervention(
        event_id, runtime.content_hash, runtime.snapshot.content_hash,
        runtime.social_state.content_hash, "alice", "bob",
        "model-review-fixture", "explicit trust checkpoint admission",
        operation, trust,
    )


def build_vector_manifest() -> dict[str, object]:
    model, story, cognition, social = initial_runtime_case()
    societies = (MeshSociety("office", ("alice", "bob")),)
    with TemporaryDirectory(prefix="v19-social-trust-review-") as folder:
        db = str(Path(folder) / "memory.sqlite3")
        prior = initialize_situated_network_runtime(db, model, story, cognition, social)
        before = MeshSnapshot(prior.snapshot, societies)
        old_path = find_mesh_path(
            before, "alice", "bob", relation=MeshRelation.RELATIONSHIP, max_hops=1,
        )
        if old_path is None:
            raise AssertionError("real V19 source fixture must include alice -> bob")
        deactivate = admit_social_trust_checkpoint(
            db, model, prior,
            _event(prior, "deactivate-alice-to-bob", SocialTrustOperation.DEACTIVATE, 0.0),
        )
        if not verify_social_trust_checkpoint(db, model, prior, deactivate):
            raise AssertionError("real trust change failed authoritative reexecution")
        middle = deactivate.next_runtime_state
        mid_mesh = MeshSnapshot(middle.snapshot, societies)
        if (
            middle.snapshot != project_situated_network_snapshot(
                model, story, cognition, middle.social_state
            )
            or mesh_path_is_valid(
                mid_mesh, old_path, "alice", "bob",
                relation=MeshRelation.RELATIONSHIP, max_hops=1,
            )
        ):
            raise AssertionError("canonical social-memory trust change failed to reproject")
        activate = admit_social_trust_checkpoint(
            db, model, middle,
            _event(middle, "activate-alice-to-bob", SocialTrustOperation.ACTIVATE, 1.0),
        )
        if not verify_social_trust_checkpoint(db, model, middle, activate):
            raise AssertionError("real trust reactivation failed authoritative reexecution")
        last = activate.next_runtime_state
        final_mesh = MeshSnapshot(last.snapshot, societies)
        if ("alice", "bob") not in mesh_edges(final_mesh, MeshRelation.RELATIONSHIP):
            raise AssertionError("authoritative reactivation did not restore edge")
        relations = {}
        for relation in (MeshRelation.DIRECT_INTERACTION, MeshRelation.TRANSMISSION):
            s = [list(x) for x in mesh_edges(before, relation)]
            if s != [list(x) for x in mesh_edges(mid_mesh, relation)] or s != [
                list(x) for x in mesh_edges(final_mesh, relation)
            ]:
                raise AssertionError("social trust intervention altered unrelated selected relation")
            relations[relation.value] = s
        return {
            "schema": SCHEMA,
            "notice": (
                "Source-derived administrative checkpoint intervention, "
                "not actor authentication or ordinary social-evidence learning"
            ),
            "model_hash": model.content_hash,
            "story_hash": story.content_hash,
            "cognitive_state_hash": cognition.content_hash,
            "relationship_threshold": model.relationship_trust_threshold,
            "original_round_index": prior.round_index,
            "original": {
                "runtime_hash": prior.content_hash,
                "source_hash": prior.snapshot.content_hash,
                "social_hash": prior.social_state.content_hash,
                "directed_edges": [list(x) for x in mesh_edges(before, MeshRelation.RELATIONSHIP)],
                "observation": observe_topology(before, relation=MeshRelation.RELATIONSHIP),
            },
            "deactivate": {
                "intervention": deactivate.intervention.to_dict(),
                "receipt": deactivate.receipt.to_dict(),
                "receipt_hash": deactivate.receipt.content_hash,
                "next_runtime_hash": middle.content_hash,
                "next_source_hash": middle.snapshot.content_hash,
                "next_social_hash": middle.social_state.content_hash,
                "directed_edges": [list(x) for x in mesh_edges(mid_mesh, MeshRelation.RELATIONSHIP)],
                "old_path_invalid": True,
                "observation": observe_topology(mid_mesh, relation=MeshRelation.RELATIONSHIP),
            },
            "activate": {
                "intervention": activate.intervention.to_dict(),
                "receipt": activate.receipt.to_dict(),
                "receipt_hash": activate.receipt.content_hash,
                "next_runtime_hash": last.content_hash,
                "next_source_hash": last.snapshot.content_hash,
                "next_social_hash": last.social_state.content_hash,
                "directed_edges": [list(x) for x in mesh_edges(final_mesh, MeshRelation.RELATIONSHIP)],
                "observation": observe_topology(final_mesh, relation=MeshRelation.RELATIONSHIP),
            },
            "unchanged_selected_relations": relations,
        }


def render_vector_corpus() -> str:
    return json.dumps(
        build_vector_manifest(), sort_keys=True, ensure_ascii=False, indent=2,
    ) + "\n"


def main() -> None:
    print(render_vector_corpus(), end="")


if __name__ == "__main__":
    main()
