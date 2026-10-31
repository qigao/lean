"""R2.M5g0: source-owned neutral training testimony changes trust, not target belief.

One V19 world, one *new declared* cognitive/social model, two original
history Stories differing only in prior Alice audit TELL content. The
original Bob private INSPECT confirms/disconfirms the audit testimony.
Audit likelihood is EXACTLY equal under both target hypotheses, so
learning source credibility does not supply evidence about the target
approved/denied belief. This is a prerequisite for later matched
native-decision design, not trust-only causal identification.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.situated import (
    SituatedActionIntent, SituatedActionKind,
)
from narrative_dynamics.abm.situated_cognition_contracts import (
    SituatedObservationLikelihood, SituatedObservationRule,
    SituatedObservationSymbol,
)
from narrative_dynamics.abm.situated_perception_contracts import (
    SituatedPerceptFidelity,
)
from narrative_dynamics.abm.situated_percept_cognition import perceptual_timeline
from narrative_dynamics.abm.situated_percept_memory_cognition import (
    initialize_situated_percept_memory_cognition,
)
from narrative_dynamics.abm.situated_percept_social_cognition import (
    simulate_situated_percept_social_cognitive_round,
)
from narrative_dynamics.abm.situated_social_memory_contracts import (
    SituatedClaimTopic, initialize_situated_social_memory,
)
from narrative_dynamics.abm.situated_story import replay_situated_story

_SCHEMA = "r2-m5g0-v19-neutral-audit-trust-learning-v1"
_SOURCE_HASH = re.compile(r"sha256:[0-9a-f]{64}")
_AUDIT_SYMBOLS = ("audit-denied", "audit-verified")
_AUDIT_TRUTH = "audit-verified"
_AUDIT_ROWS = (
    ("confirmed_audit", "The audit marker confirms.", "audit-verified", "confirmed", 0.6),
    ("contradicted_audit", "The audit marker disputes.", "audit-denied", "contradicted", 0.3),
)
_FIELDS = frozenset({
    "schema", "source_owner", "source_world_model_hash",
    "source_perception_model_hash", "source_cognitive_model_hash",
    "source_memory_model_hash", "source_social_model_hash",
    "source_social_policy_hash", "source_initial_world_hash",
    "target_hypothesis", "audit_topic", "target_topic",
    "audit_symbols", "audit_likelihood_per_target_hypothesis",
    "target_tell_likelihood_ratio_still_informative",
    "source_action_ids_identical", "source_physical_world_snapshots_equal",
    "native_social_learning_round", "source_training_contrast",
    "original_social_owner", "target_belief_match",
    "other_cognitive_inputs_match", "trust_only_causal_effect",
    "post_training_target_action_effect", "sample_size_per_arm",
    "confidence_interval", "external_validation",
    "cross_mechanism_ranking", "arms",
})
_ARM = frozenset({
    "source_historical_story_hash", "source_historical_world_hash",
    "prior_audit_tell_event_id", "prior_audit_tell_event_hash",
    "bob_original_exact_tell_percept_hash",
    "bob_own_inspect_event_hash", "source_audit_symbol",
    "source_bob_prior_approved_belief",
    "source_bob_posterior_approved_belief",
    "source_bob_approved_belief_unchanged",
    "original_social_trust_before",
    "original_social_trust_after",
    "original_social_claim_status",
    "original_social_evidence_testimony_count",
    "original_social_evidence_private_verify_count",
    "original_social_round_hash",
    "original_next_cognitive_state_hash",
    "original_next_social_state_hash",
    "native_original_action_round",
    "original_bob_decision_hash",
    "native_physical_positions",
    "native_passage_states",
})


def _hash(x: object) -> bool:
    return type(x) is str and _SOURCE_HASH.fullmatch(x) is not None


@dataclass(frozen=True)
class NeutralAuditSourceReport:
    """Redacted exact-hash evidence of source-neutral trust learning."""
    _canonical_json: str

    def __post_init__(self) -> None:
        if type(self._canonical_json) is not str:
            raise TypeError("M5g0 source report must be canonical JSON text")
        d = json.loads(self._canonical_json)
        if (
            type(d) is not dict or set(d) != _FIELDS
            or d["schema"] != _SCHEMA
            or d["source_owner"] != "ONE_NATIVE_V19_SQLITE_COGNITION_AND_SOCIAL_OWNER"
            or d["target_hypothesis"] != "approved"
            or d["audit_topic"] != "source-audit"
            or d["target_topic"] != "restructuring"
            or d["audit_symbols"] != list(_AUDIT_SYMBOLS)
            or d["audit_likelihood_per_target_hypothesis"] != {
                "approved": {"audit-denied": 0.1, "audit-verified": 0.1},
                "denied": {"audit-denied": 0.1, "audit-verified": 0.1},
            }
            or d["target_tell_likelihood_ratio_still_informative"] is not True
            or d["source_action_ids_identical"] is not True
            or d["source_physical_world_snapshots_equal"] is not True
            or d["native_social_learning_round"] != 8
            or d["source_training_contrast"] != "PRIOR_AUDIT_TELL_CONTENT_ONLY"
            or d["original_social_owner"] != "V19_ADVANCE_SITUATED_SOCIAL_MEMORY"
            or d["target_belief_match"] != "EXACT_NATIVE_PRIOR_AND_POSTERIOR"
            or d["other_cognitive_inputs_match"] !=
               "NOT_PROVEN_DIFFERENT_PRIVATE_MEMORIES_AND_CLAIMS"
            or d["trust_only_causal_effect"] != "NOT_IDENTIFIED"
            or d["post_training_target_action_effect"] != "NOT_TESTED"
            or d["sample_size_per_arm"] != 1
            or d["confidence_interval"] != "NOT_ESTIMABLE"
            or d["external_validation"] != "NO_EXTERNAL_VALIDATION"
            or d["cross_mechanism_ranking"] != "FORBIDDEN"
            or any(not _hash(d.get(k)) for k in (
                "source_world_model_hash", "source_perception_model_hash",
                "source_cognitive_model_hash", "source_memory_model_hash",
                "source_social_model_hash", "source_social_policy_hash",
                "source_initial_world_hash",
            ))
            or type(d["arms"]) is not dict
            or set(d["arms"]) != {x[0] for x in _AUDIT_ROWS}
        ):
            raise ValueError("M5g0 requires a single neutral original cognitive/social owner")
        report_refs = []
        for name, _, symbol, status, trust in _AUDIT_ROWS:
            x = d["arms"][name]
            if (
                type(x) is not dict or set(x) != _ARM
                or any(not _hash(x.get(key)) for key in (
                    "source_historical_story_hash",
                    "source_historical_world_hash",
                    "prior_audit_tell_event_hash",
                    "bob_original_exact_tell_percept_hash",
                    "bob_own_inspect_event_hash",
                    "original_social_round_hash",
                    "original_next_cognitive_state_hash",
                    "original_next_social_state_hash",
                    "original_bob_decision_hash",
                ))
                or x["source_audit_symbol"] != symbol
                or x["original_social_claim_status"] != status
                or x["original_social_trust_before"] != 0.5
                or x["original_social_trust_after"] != trust
                or x["source_bob_prior_approved_belief"] != 0.5
                or x["source_bob_posterior_approved_belief"] != 0.5
                or x["source_bob_approved_belief_unchanged"] is not True
                or x["original_social_evidence_testimony_count"] != 1
                or x["original_social_evidence_private_verify_count"] != 1
                or x["native_original_action_round"] != 8
                or type(x["prior_audit_tell_event_id"]) is not str
                or not x["prior_audit_tell_event_id"]
                or type(x["native_physical_positions"]) is not dict
                or type(x["native_passage_states"]) is not dict
                or any(type(a) is not str or type(b) is not str
                       for a,b in x["native_physical_positions"].items())
                or any(type(a) is not str or type(b) is not bool
                       for a,b in x["native_passage_states"].items())
            ):
                raise ValueError("neutral audit cannot manufacture trust or alter target belief")
            report_refs.append(
                (x["prior_audit_tell_event_id"],
                 x["prior_audit_tell_event_hash"],
                 x["source_historical_story_hash"],
                 x["native_physical_positions"],
                 x["native_passage_states"])
            )
        a,b = report_refs
        if (a[0] != b[0] or a[1] == b[1] or a[2] == b[2]
                or a[3] != b[3] or a[4] != b[4]):
            raise ValueError("audit contrast must share actor IDs and world but differ in source story")

    def to_dict(self) -> dict[str, object]:
        return json.loads(self._canonical_json)

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _neutral_source_models():
    """Declare one new V19 model with a separate audit topic.

    Every action/hypothesis likelihood row retains 80% of its original
    target evidence distribution; the two audit symbols each receive 10%
    independently of the approved/denied target hypothesis.
    Therefore target likelihood ratios are unchanged, audit symbols are
    neutral, and row normalization still exactly holds.
    """
    from tests.test_network_epistemic_influence import source_epistemic_case
    net, root, _, _ = source_epistemic_case()
    old_cognition = net.percept_memory_model.cognitive_model
    old_social = net.social_memory_model
    symbol_objects = (
        SituatedObservationSymbol("audit-verified", "audit source confirmed by inspection"),
        SituatedObservationSymbol("audit-denied", "audit source contradicted by inspection"),
    )
    cognitive_agents = []
    for actor in old_cognition.agents:
        old_likelihood = tuple(
            replace(item, probability=item.probability * 0.8)
            for item in actor.likelihoods
        )
        neutral_rows = tuple(
            SituatedObservationLikelihood(
                action.action_id, hypothesis.hypothesis_id, symbol.symbol_id, 0.1
            )
            for action in actor.actions
            for hypothesis in actor.hypotheses
            for symbol in symbol_objects
        )
        rules = actor.observation_rules
        if actor.agent_id == "bob":
            replaced = []
            for rule in rules:
                if rule.rule_id == "inspect-approved":
                    rule = replace(rule, symbol_id=_AUDIT_TRUTH)
                replaced.append(rule)
            rules = tuple(replaced) + (
                SituatedObservationRule(
                    "audit-bob-confirmed",
                    "audit-verified", "tell", SituatedActionKind.TELL,
                    "told", "message", "The audit marker confirms.",
                ),
                SituatedObservationRule(
                    "audit-bob-contradicted",
                    "audit-denied", "tell", SituatedActionKind.TELL,
                    "told", "message", "The audit marker disputes.",
                ),
            )
        actor = replace(
            actor,
            observation_symbols=actor.observation_symbols + symbol_objects,
            likelihoods=old_likelihood + neutral_rows,
            observation_rules=rules,
        )
        cognitive_agents.append(actor)
    cognition = replace(old_cognition, agents=tuple(cognitive_agents))
    memory = replace(net.percept_memory_model, cognitive_model=cognition,
        agents=tuple(
            replace(policy, cues=tuple(
                replace(cue, text="audit")
                if cue.cue_id == "tell" else cue
                for cue in policy.cues
            )) if policy.agent_id == "bob" else policy
            for policy in net.percept_memory_model.agents
        ),
    )
    social = replace(
        old_social,
        memory_cognitive_model=replace(
            old_social.memory_cognitive_model, cognitive_model=cognition,
        ),
        topics=(
            SituatedClaimTopic("restructuring", ("approved", "denied")),
            SituatedClaimTopic("source-audit", _AUDIT_SYMBOLS),
        ),
        policy=replace(old_social.policy, trust_weight_direct_testimony=True),
    )
    # Proven native likelihood normalization and neutrality at the MODEL
    # boundary, not merely an assertion in the report JSON.
    for agent in cognition.agents:
        for action in agent.actions:
            for hypothesis in agent.hypotheses:
                entries = {
                    p.symbol_id: p.probability for p in agent.likelihoods
                    if p.action_id == action.action_id
                    and p.hypothesis_id == hypothesis.hypothesis_id
                }
                if (not math.isclose(math.fsum(entries.values()), 1.0,
                                     abs_tol=1e-12, rel_tol=0)
                        or any(entries[s] != 0.1 for s in _AUDIT_SYMBOLS)):
                    raise ValueError("native audit likelihood row not neutral/normalized")
        if agent.agent_id == "bob":
            informative = {
                h: next(p.probability for p in agent.likelihoods
                        if p.action_id == "tell"
                        and p.hypothesis_id == h
                        and p.symbol_id == "approved")
                for h in ("approved", "denied")
            }
            if informative["approved"] <= informative["denied"]:
                raise ValueError("audit protocol erased informative target testimony")
    if social.policy.trust_weight_direct_testimony is not True:
        raise ValueError("neutral training must preserve original source direct-trust policy")
    return memory, social, root


def observe_neutral_training_trust() -> NeutralAuditSourceReport:
    """Reexecute original V19 actions and single private SQLite/social round."""
    memory, social, root = _neutral_source_models()
    world = memory.cognitive_model.world_model
    perception = memory.perception_model
    author_inspect = SituatedActionIntent(
        "m5g-alice-inspect", "alice", SituatedActionKind.INSPECT,
        target_id="memo",
    )
    author_open = SituatedActionIntent(
        "m5g-alice-authorized-open", "alice", SituatedActionKind.OPEN_PASSAGE,
        target_id="records-open",
    )
    author_move = SituatedActionIntent(
        "m5g-alice-move", "alice", SituatedActionKind.MOVE,
        target_id="records-open",
    )
    first_three = replay_situated_story(
        world, root.initial_state,
        ((author_inspect,), (author_open,), (author_move,)),
        perception_model=perception,
    )
    inspected = next(e for e in first_three.rounds[0].events
                     if e.action_id == author_inspect.action_id and e.success)
    bob_entry = SituatedActionIntent(
        "m5g-bob-enter-records", "bob", SituatedActionKind.MOVE,
        target_id="open-records",
    )
    bob_inspect = SituatedActionIntent(
        "m5g-bob-own-inspect", "bob", SituatedActionKind.INSPECT,
        target_id="memo",
    )
    bob_return = SituatedActionIntent(
        "m5g-bob-return-open", "bob", SituatedActionKind.MOVE,
        target_id="records-open",
    )
    results = {}
    source_physical = []
    authored_ids = None
    for label, text, symbol, expected_status, expected_trust in _AUDIT_ROWS:
        training_tell = SituatedActionIntent(
            "m5g-alice-audit-training", "alice", SituatedActionKind.TELL,
            message=text, source_event_ids=(inspected.event_id,),
        )
        schedule = (
            (author_inspect,), (author_open,), (author_move,),
            (training_tell,), (bob_entry,), (bob_inspect,), (bob_return,),
        )
        if authored_ids is None:
            authored_ids = [x[0].action_id for x in schedule]
        elif authored_ids != [x[0].action_id for x in schedule]:
            raise ValueError("neutral training branches changed authored actions")
        story = replay_situated_story(
            world, root.initial_state, schedule, perception_model=perception,
        )
        if not all(
            next(e for e in round_result.events
                 if e.action_id == one[0].action_id).success
            for round_result,one in zip(story.rounds,schedule,strict=True)
        ):
            raise ValueError("neutral training source world action failed")
        previous_tell = next(e for e in story.rounds[3].events
                             if e.action_id == training_tell.action_id)
        original_bob_inspect = next(e for e in story.rounds[5].events
                                    if e.action_id == bob_inspect.action_id)
        BobTells = [
            p for p in perceptual_timeline(perception, story, "bob")
            if p.source_event_id == previous_tell.event_id
            and p.source_event_hash == previous_tell.content_hash
            and p.actor_agent_id == "alice"
            and p.fidelity is SituatedPerceptFidelity.EXACT
            and p.kind is SituatedActionKind.TELL
        ]
        if len(BobTells) != 1:
            raise ValueError("M5g0 Bob training TELL must be a real EXACT private percept")
        prior = initialize_situated_percept_memory_cognition(memory, story)
        prior_social = initialize_situated_social_memory(social, prior)
        with TemporaryDirectory(prefix=f"v19-m5g-neutral-{label}-") as tmp:
            native = simulate_situated_percept_social_cognitive_round(
                str(Path(tmp) / "memory.sqlite3"),
                memory, social, story, prior, prior_social,
            )
        bob_decision = next(x for x in native.decisions if x.agent_id == "bob")
        after_social = next(x.trust for x in native.next_social_state.relationships
                            if x.observer_agent_id == "bob"
                            and x.source_agent_id == "alice")
        before_social = next(x.trust for x in prior_social.relationships
                             if x.observer_agent_id == "bob"
                             and x.source_agent_id == "alice")
        observed = tuple(
            e for e in native.social_update.admitted_evidence
            if e.observer_agent_id == "bob"
        )
        claims = [
            c for c in native.next_social_state.claims
            if c.observer_agent_id == "bob" and c.source_agent_id == "alice"
            and c.topic_id == "source-audit"
        ]
        audit_count = sum(e.kind.value == "testimony"
                          and e.source_agent_id == "alice"
                          and e.topic_id == "source-audit" for e in observed)
        verify_count = sum(e.kind.value == "verification"
                           and e.source_agent_id is None
                           and e.topic_id == "source-audit" for e in observed)
        target_evidence = [
            e for e in observed if e.topic_id == "restructuring"
        ]
        if (
            before_social != 0.5 or after_social != expected_trust
            or audit_count != 1 or verify_count != 1
            or len(claims) != 1
            or claims[0].status.value != expected_status
            or target_evidence
            or bob_decision.prior_belief.probabilities["approved"] != 0.5
            or bob_decision.posterior_belief.probabilities["approved"] != 0.5
        ):
            raise ValueError("original neutral audit did not isolate target belief and trust")
        placements = {
            a.agent_id: a.place_id for a in story.current_state.agents
        }
        gates = {p.passage_id:p.open for p in story.current_state.passages}
        source_physical.append((placements, gates))
        results[label] = {
            "source_historical_story_hash": story.content_hash,
            "source_historical_world_hash": story.current_state.content_hash,
            "prior_audit_tell_event_id": previous_tell.event_id,
            "prior_audit_tell_event_hash": previous_tell.content_hash,
            "bob_original_exact_tell_percept_hash": BobTells[0].content_hash,
            "bob_own_inspect_event_hash": original_bob_inspect.content_hash,
            "source_audit_symbol": symbol,
            "source_bob_prior_approved_belief":
                bob_decision.prior_belief.probabilities["approved"],
            "source_bob_posterior_approved_belief":
                bob_decision.posterior_belief.probabilities["approved"],
            "source_bob_approved_belief_unchanged": True,
            "original_social_trust_before": before_social,
            "original_social_trust_after": after_social,
            "original_social_claim_status": claims[0].status.value,
            "original_social_evidence_testimony_count": audit_count,
            "original_social_evidence_private_verify_count": verify_count,
            "original_social_round_hash": native.content_hash,
            "original_next_cognitive_state_hash": native.next_cognitive_state.content_hash,
            "original_next_social_state_hash": native.next_social_state.content_hash,
            "native_original_action_round": native.next_story.current_state.round_index,
            "original_bob_decision_hash": bob_decision.content_hash,
            "native_physical_positions": placements,
            "native_passage_states": gates,
        }
    if source_physical[0] != source_physical[1]:
        raise ValueError("neutral audit created unequal physical world before learning")
    report = {
        "schema": _SCHEMA,
        "source_owner": "ONE_NATIVE_V19_SQLITE_COGNITION_AND_SOCIAL_OWNER",
        "source_world_model_hash": world.content_hash,
        "source_perception_model_hash": perception.content_hash,
        "source_cognitive_model_hash": memory.cognitive_model.content_hash,
        "source_memory_model_hash": memory.content_hash,
        "source_social_model_hash": social.content_hash,
        "source_social_policy_hash": social.policy.content_hash,
        "source_initial_world_hash": root.initial_state.content_hash,
        "target_hypothesis": "approved",
        "audit_topic": "source-audit",
        "target_topic": "restructuring",
        "audit_symbols": list(_AUDIT_SYMBOLS),
        "audit_likelihood_per_target_hypothesis": {
            h: {s: 0.1 for s in _AUDIT_SYMBOLS}
            for h in ("approved","denied")
        },
        "target_tell_likelihood_ratio_still_informative": True,
        "source_action_ids_identical": True,
        "source_physical_world_snapshots_equal": True,
        "native_social_learning_round": 8,
        "source_training_contrast": "PRIOR_AUDIT_TELL_CONTENT_ONLY",
        "original_social_owner": "V19_ADVANCE_SITUATED_SOCIAL_MEMORY",
        "target_belief_match": "EXACT_NATIVE_PRIOR_AND_POSTERIOR",
        "other_cognitive_inputs_match": "NOT_PROVEN_DIFFERENT_PRIVATE_MEMORIES_AND_CLAIMS",
        "trust_only_causal_effect": "NOT_IDENTIFIED",
        "post_training_target_action_effect": "NOT_TESTED",
        "sample_size_per_arm": 1,
        "confidence_interval": "NOT_ESTIMABLE",
        "external_validation": "NO_EXTERNAL_VALIDATION",
        "cross_mechanism_ranking": "FORBIDDEN",
        "arms": results,
    }
    return NeutralAuditSourceReport(
        json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )
