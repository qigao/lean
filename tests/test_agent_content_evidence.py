"""R2.M3: actual V19 TELL/percept/cognition/claim/verification provenance."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from narrative_dynamics.abm.mesh_agent_content_evidence import (
    AgentContentEvidenceTrace, observe_agent_content_evidence,
)
from narrative_dynamics.abm.mesh_contracts import (
    MeshRelation, MeshSnapshot, MeshSociety, source_relation_edges,
)
from narrative_dynamics.abm.situated_network import (
    project_situated_network_snapshot,
)
from narrative_dynamics.abm.situated_network_contracts import (
    SituatedNetworkRuntimeModel,
)
from narrative_dynamics.abm.situated_mesh import mesh_edges
from narrative_dynamics.abm.situated import SituatedActionIntent, SituatedActionKind, ObservationChannel
from narrative_dynamics.abm.situated_cognition_contracts import SituatedObservationRule
from narrative_dynamics.abm.situated_memory_cognition_contracts import SituatedMemoryRecallCue
from narrative_dynamics.abm.situated_percept_memory_cognition import (
    initialize_situated_percept_memory_cognition,
)
from narrative_dynamics.abm.situated_percept_social_cognition import (
    simulate_situated_percept_social_cognitive_round,
)
from narrative_dynamics.abm.situated_social_memory_contracts import (
    SituatedClaimStatus, initialize_situated_social_memory,
)
from narrative_dynamics.abm.situated_story import advance_situated_story, initialize_situated_story
from tests.situated_cognition_fixtures import cognitive_office_model
from tests.test_network_abm_situated_percept_cognition import (
    SECRET, initial_state, perception_model,
)
from tests.test_network_abm_situated_percept_memory_cognition import (
    inspection_checkpoint, recall_model,
)
from tests.test_network_abm_situated_percept_social_cognition import (
    bound_social_model, tell_checkpoint,
)


def _run(memory_model, social_model, story, name):
    checkpoint = initialize_situated_percept_memory_cognition(memory_model, story)
    social_state = initialize_situated_social_memory(social_model, checkpoint)
    with TemporaryDirectory(prefix="v19-content-evidence-" + name + "-") as tmp:
        result = simulate_situated_percept_social_cognitive_round(
            str(Path(tmp) / "memory.sqlite3"),
            memory_model, social_model, story, checkpoint, social_state,
        )
    trace = observe_agent_content_evidence(
        memory_model, social_model, story, checkpoint, social_state, result,
    )
    return memory_model, social_model, story, checkpoint, social_state, result, trace


def _tell_case(name, *, door_open, visual=False, kind_only=False, cue_text=SECRET):
    perception, cognition, story = tell_checkpoint(
        door_open=door_open, visual=visual, kind_only=kind_only,
    )
    cue = SituatedMemoryRecallCue(
        "tell", cue_text,
        event_kinds=(SituatedActionKind.TELL,) if door_open or visual else (),
        channels=(ObservationChannel.AUDITORY,),
    )
    memory_model = recall_model(
        {"bob": (cue,)}, cognition=cognition, perception=perception,
    )
    return _run(memory_model, bound_social_model(memory_model), story, name)


def _private_verification():
    perception = perception_model()
    cognition, story, _ = inspection_checkpoint(perception)
    cue = SituatedMemoryRecallCue(
        "inspect", "restructuring",
        event_kinds=(SituatedActionKind.INSPECT,),
        channels=(ObservationChannel.INSPECTION,),
    )
    memory_model = recall_model(
        {"alice": (cue,)}, cognition=cognition, perception=perception,
    )
    return _run(memory_model, bound_social_model(memory_model), story, "verification")


def _contradiction():
    """Use the existing real five-round V19 private-perception/social fixture."""
    perception = perception_model()
    cognition = cognitive_office_model()
    bob = next(a for a in cognition.agents if a.agent_id == "bob")
    bob = replace(bob, observation_rules=bob.observation_rules + (
        SituatedObservationRule(
            "tell-denied", "denied", "tell", SituatedActionKind.TELL,
            "told", "message", "The restructuring is denied.",
        ),
    ))
    cognition = replace(
        cognition, agents=tuple(bob if a.agent_id=="bob" else a for a in cognition.agents),
    )
    story = initialize_situated_story(
        cognition.world_model, initial_state(perception, door_open=True),
        perception_model=perception,
    )
    story = advance_situated_story(cognition.world_model, story, (
        SituatedActionIntent("alice-inspect", "alice", SituatedActionKind.INSPECT, "memo"),
    ))
    inspection = story.rounds[-1].events[0]
    story = advance_situated_story(cognition.world_model, story, (
        SituatedActionIntent("alice-move", "alice", SituatedActionKind.MOVE, "records-open"),
    ))
    story = advance_situated_story(cognition.world_model, story, (
        SituatedActionIntent(
            "alice-denied", "alice", SituatedActionKind.TELL,
            message="The restructuring is denied.",
            source_event_ids=(inspection.event_id,),
        ),
    ))
    story = advance_situated_story(cognition.world_model, story, (
        SituatedActionIntent("bob-move", "bob", SituatedActionKind.MOVE, "open-records"),
    ))
    story = advance_situated_story(cognition.world_model, story, (
        SituatedActionIntent("bob-inspect", "bob", SituatedActionKind.INSPECT, "memo"),
    ))
    cues = (
        SituatedMemoryRecallCue(
            "tell", "restructuring", event_kinds=(SituatedActionKind.TELL,),
        ),
        SituatedMemoryRecallCue(
            "inspect", "restructuring", event_kinds=(SituatedActionKind.INSPECT,),
        ),
    )
    memory_model = recall_model(
        {"bob": cues}, cognition=cognition, perception=perception,
    )
    return _run(memory_model, bound_social_model(memory_model), story, "contradiction")


def build_content_cases():
    """Actual V19 cognitive/social simulation, not synthesized trust mutation."""
    return (
        ("exact_observed_tell", _tell_case("exact", door_open=True)),
        ("detected_sound_no_testimony", _tell_case(
            "detected", door_open=False, cue_text="detected",
        )),
        ("identified_action_no_content", _tell_case(
            "identified", door_open=False, visual=True,
            kind_only=True, cue_text="tell",
        )),
        ("private_inspection_verification", _private_verification()),
        ("contradicted_claim_directed_trust", _contradiction()),
    )


class AgentContentEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = dict(build_content_cases())

    def test_exact_tell_admits_one_testimony_not_unverified_trust(self):
        model, social, story, prior_cog, prior_soc, result, trace = self.cases[
            "exact_observed_tell"
        ]
        x = trace.to_dict()
        self.assertEqual(x["counts"]["actual_tell_events"], 1)
        self.assertEqual(x["counts"]["admitted_testimony"], 1)
        self.assertEqual(x["counts"]["admitted_verification"], 0)
        self.assertEqual(x["counts"]["claim_revisions"], 1)
        self.assertEqual(x["counts"]["trust_revisions"], 0)
        self.assertEqual(x["source_tell_events"][0]["other_agent_perception_by_fidelity"]
                         ["exact"], 2)
        self.assertEqual(x["admitted_evidence"][0]["kind"], "testimony")
        self.assertEqual(x["admitted_evidence"][0]["source_agent_id"], "alice")
        self.assertEqual(x["admitted_evidence"][0]["observer_agent_id"], "bob")
        self.assertEqual(x["claim_revisions"][0]["status_after"], "active")
        self.assertEqual(result.next_social_state.claims[0].status,
                         SituatedClaimStatus.ACTIVE)
        self.assertEqual(prior_soc.relationships, result.next_social_state.relationships)

    def test_detected_sound_and_identified_action_cannot_create_testimony(self):
        _, _, _, _, _, detected, trace = self.cases[
            "detected_sound_no_testimony"
        ]
        d=trace.to_dict()
        self.assertEqual(d["source_tell_events"][0]["other_agent_perception_by_fidelity"]
                         ["detected"], 2)
        self.assertEqual(d["counts"]["admitted_testimony"], 0)
        self.assertEqual(d["counts"]["claim_revisions"], 0)
        self.assertEqual(d["counts"]["trust_revisions"], 0)
        self.assertEqual(detected.social_update.admitted_evidence, ())
        _, _, _, _, _, identified, report = self.cases[
            "identified_action_no_content"
        ]
        q=report.to_dict()
        self.assertEqual(q["source_tell_events"][0]["other_agent_perception_by_fidelity"]
                         ["identified"], 2)
        self.assertEqual(q["counts"]["admitted_testimony"], 0)
        self.assertEqual(q["counts"]["admitted_verification"], 0)
        self.assertEqual(identified.social_update.admitted_evidence, ())

    def test_private_inspection_verifies_only_own_exact_observation(self):
        *_, result, trace = self.cases["private_inspection_verification"]
        report=trace.to_dict()
        self.assertEqual(report["counts"]["actual_tell_events"],0)
        self.assertEqual(report["counts"]["admitted_testimony"],0)
        self.assertEqual(report["counts"]["admitted_verification"],1)
        self.assertEqual(report["admitted_evidence"][0]["kind"],"verification")
        self.assertIsNone(report["admitted_evidence"][0]["source_agent_id"])
        self.assertEqual(report["admitted_evidence"][0]["observer_agent_id"],"alice")
        self.assertEqual(result.next_social_state.claims, ())

    def test_real_contradiction_revises_claim_and_one_directed_trust(self):
        *_, result, trace = self.cases["contradicted_claim_directed_trust"]
        data=trace.to_dict()
        self.assertEqual(data["counts"]["admitted_testimony"],1)
        self.assertEqual(data["counts"]["admitted_verification"],1)
        self.assertEqual(data["counts"]["trust_revisions"],1)
        self.assertEqual(data["claim_revisions"][0]["status_after"],"contradicted")
        trust=data["directed_trust_revisions"][0]
        self.assertEqual((trust["observer_agent_id"],trust["source_agent_id"]),
                         ("bob","alice"))
        self.assertEqual(trust["trust_before"],0.5)
        self.assertEqual(trust["trust_after"],0.3)
        self.assertEqual(result.next_social_state.claims[0].status,
                         SituatedClaimStatus.CONTRADICTED)

    def test_actual_claim_contradiction_retires_only_v19_source_relationship_edge(self):
        model, social_model, story, cognitive, social, outcome, trace = self.cases[
            "contradicted_claim_directed_trust"
        ]
        network = SituatedNetworkRuntimeModel(
            "r2-content-verification", "1", model, social_model,
            "approved", 0.7, 0.5,
        )
        source_before = project_situated_network_snapshot(
            network, story, cognitive, social,
        )
        source_after = project_situated_network_snapshot(
            network, outcome.next_story,
            outcome.next_cognitive_state, outcome.next_social_state,
        )
        members = tuple(n.agent_id for n in source_before.nodes)
        societies = (MeshSociety("office", members),)
        before = MeshSnapshot(source_before, societies)
        after = MeshSnapshot(source_after, societies)
        b = set(source_relation_edges(source_before, MeshRelation.RELATIONSHIP))
        a = set(source_relation_edges(source_after, MeshRelation.RELATIONSHIP))
        self.assertEqual(b - a, {("alice", "bob")})
        self.assertEqual(a - b, set())
        self.assertIn(("bob", "alice"), b)
        self.assertIn(("bob", "alice"), a)
        self.assertEqual(
            set(mesh_edges(before, MeshRelation.RELATIONSHIP)) -
            set(mesh_edges(after, MeshRelation.RELATIONSHIP)),
            {("alice", "bob")},
        )
        self.assertEqual(trace.to_dict()["counts"]["admitted_verification"], 1)
        self.assertEqual(trace.to_dict()["counts"]["trust_revisions"], 1)
        self.assertEqual(
            source_after.social_state_hash, outcome.next_social_state.content_hash,
        )
        self.assertNotEqual(source_before.content_hash, source_after.content_hash)
        # This reprojects V19 from the actual social owner: we do not
        # fabricate the edge by mutating V24 or a separate graph.

    def test_exact_testimony_without_verification_does_not_change_v19_edges(self):
        model, social_model, story, cognitive, social, outcome, trace = self.cases[
            "exact_observed_tell"
        ]
        network = SituatedNetworkRuntimeModel(
            "r2-content-observed", "1", model, social_model,
            "approved", 0.7, 0.5,
        )
        before = project_situated_network_snapshot(network, story, cognitive, social)
        after = project_situated_network_snapshot(
            network, outcome.next_story,
            outcome.next_cognitive_state, outcome.next_social_state,
        )
        self.assertEqual(
            source_relation_edges(before, MeshRelation.RELATIONSHIP),
            source_relation_edges(after, MeshRelation.RELATIONSHIP),
        )
        self.assertEqual(trace.to_dict()["counts"]["admitted_testimony"], 1)
        self.assertEqual(trace.to_dict()["counts"]["admitted_verification"], 0)
        self.assertEqual(trace.to_dict()["counts"]["trust_revisions"], 0)
        self.assertNotEqual(before.content_hash, after.content_hash)
        # A content/claim revision has taken place, but trust and effective
        # relationship topology have not automatically changed.

    def test_forged_social_evidence_and_wrong_source_fail_closed(self):
        model,social,story,prior_cog,prior_soc,result,trace=self.cases[
            "exact_observed_tell"
        ]
        admitted=result.social_update.admitted_evidence[0]
        forged=replace(admitted,event_id="r9999:e9999")
        altered=replace(result.social_update,admitted_evidence=(forged,))
        corrupted=replace(result,social_update=altered)
        with self.assertRaisesRegex(ValueError,"not source-reproducible"):
            observe_agent_content_evidence(
                model,social,story,prior_cog,prior_soc,corrupted,
            )
        with self.assertRaisesRegex(ValueError,"stale V19"):
            observe_agent_content_evidence(
                model,social,story,prior_cog,replace(prior_soc,processed_evidence_ids=("fake",)),
                result,
            )
        with self.assertRaises(TypeError):
            observe_agent_content_evidence(
                model,social,story,prior_cog,prior_soc,object(),
            )

    def test_payload_redaction_and_immutable_trace_source(self):
        for name,(_,_,_,_,_,_,trace) in self.cases.items():
            with self.subTest(name=name):
                raw=json.dumps(trace.to_dict(),sort_keys=True)
                self.assertNotIn(SECRET,raw)
                self.assertNotIn("The restructuring is denied.",raw)
                self.assertNotIn('"message":',raw)
                self.assertNotIn('"symbol_id":',raw)
                old=trace.content_hash
                copied=trace.to_dict()
                copied["causal_attribution"]="PROVED"
                copied["admitted_evidence"].append({"message":SECRET})
                self.assertEqual(trace.content_hash,old)
                self.assertEqual(trace.to_dict()["causal_attribution"],"UNATTRIBUTED")
                with self.assertRaises(FrozenInstanceError):
                    trace._canonical_json="{}"
        trace=self.cases["exact_observed_tell"][-1]
        with self.assertRaisesRegex(ValueError,"bind exact source and scope"):
            replace(trace,_canonical_json=json.dumps(
                trace.to_dict()|{"causal_attribution":"PROVED"},
            ))
        with self.assertRaises(TypeError):
            AgentContentEvidenceTrace(trace.prior_story_hash,trace.next_social_state_hash,{})

    def test_reviewed_content_vectors_match_exact_v19_source(self):
        from tools.render_agent_content_evidence_vectors import (
            SCHEMA,build_vector_manifest,render_vector_corpus,
        )
        manifest=build_vector_manifest()
        self.assertEqual(manifest["schema"],SCHEMA)
        self.assertEqual(len(manifest["cases"]),5)
        rendered=render_vector_corpus()
        self.assertNotIn(SECRET,rendered)
        self.assertNotIn("The restructuring is denied.",rendered)
        pin=Path(__file__).resolve().parents[1]/"conformance/agent_content_evidence_v1.json"
        # Required, reviewed real-source JSON: never regenerate it in place.
        self.assertEqual(pin.read_bytes(),rendered.encode("utf-8"))


if __name__=="__main__":
    unittest.main()
