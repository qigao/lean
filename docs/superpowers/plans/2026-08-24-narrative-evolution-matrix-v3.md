# Narrative Evolution Matrix V3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic temporal evolution analysis for validated Narrative Testimony V2 scenarios, including baseline world/direct/testimony state snapshots, minimal single-variable counterfactuals, first-divergence evidence, validation-boundary rejections, and an explicit demonstration that final action alone does not uniquely identify the information mechanism.

**Architecture:** Add `narrative_dynamics.story.evolution_v2` above the existing V1/V2 replay semantics. First extract one pure testimony action resolver into the story layer so the existing adapter and the new analysis share one decision rule; then build immutable baseline trajectories, evaluate four single-variable intervention families through ordinary V2 validation, compare valid trajectories on canonical state only, and expose exactly six V3 names from `narrative_dynamics.story` without changing runtime, registry, metrics, Lean semantics, V1 identities, or V2 runtime identity.

**Tech Stack:** Python 3 standard library (`dataclasses`, `types.MappingProxyType`, `unittest`, `json`), existing Narrative Dynamics schema/replay/scenario modules, existing GitHub Actions `proof` workflow, Lean 4 regression gates unchanged.

**Spec:** `docs/superpowers/specs/2026-08-24-narrative-evolution-matrix-v3-design.md`

## Global Constraints

- Implementation branch: `work/narrative-evolution-matrix-v3`.
- Exact pre-V3 research SHA: `29a15a49e3191e3d6fde995064642630a4f8ea2a` on `proof/narrative-dynamics-v0`.
- `master` must not move.
- Every production task uses strict RED -> GREEN: commit and observe a focused test-only RED on the exact feature head before writing that task's production code.
- After every GREEN implementation, require a fresh exact-head GitHub Actions `proof` success before starting the next production task.
- V3 accepts only validated `NarrativeScenarioV2`; no raw fixture, oracle, source text, metadata, or gold label enters the analysis API.
- V3 analyzes exactly `story.decision.object`.
- Tracked agents are exactly the decision actor followed by unique target-report speakers in report order.
- Snapshot times are every relocation event time, every report time, and decision time, sorted ascending and deduplicated.
- Reception has no independent V2 timestamp; a declared reception becomes available at the report's `logical_time`.
- Missing direct/testimony support is represented by `None`; never fall back to objective truth.
- Counterfactuals are one-change-at-a-time only; no compound or power-set search.
- Supported intervention kinds are exactly `change_report_content`, `remove_direct_observation`, `remove_reception`, `remove_support_observation`.
- First divergence compares canonical objective/direct/testimony/provenance/action fields only. Trigger descriptions, fixture metadata, source text, oracle data, and labels are not comparison inputs.
- A rejected counterfactual has no alternate trajectory and no `first_divergence`; `rejection_stage` is exactly `scenario_validation` or `action_resolution`.
- `mechanism_uniqueness_claimed` is structurally fixed to `False`.
- Public V3 records are frozen and must defensively convert mutable sequence/mapping inputs to immutable tuples/mapping proxies.
- V1 fixture hashes and V1 runtime IDs remain exact.
- V2 runtime payload remains exactly six keys: `entities`, `events`, `observations`, `reports`, `receptions`, `decision`.
- Do not modify `SimulationRunner`, runtime contracts, model registry, `story_choice_metrics`, prison adapters, V1 schema/replay/scenario semantics, package-root exports, or Lean theorem semantics.
- `TestimonySearchModel` may change only to reuse the shared story-layer action resolver; its name, parameter contract, events, outcome, scores, policy, determinism, and runtime behavior remain exact.
- Do not claim general Theory of Mind, unique hidden-mechanism identification, deception intent, empirical human validity, population validity, or causal effect estimation from observational data.
- Report standalone `python3 -m compileall -q narrative_dynamics` only if that command is actually run independently.

## Planned File Surface

Production/test files planned in addition to this spec and plan:

- Modify `narrative_dynamics/story/replay_v2.py` — add non-exported `resolve_testimony_action`.
- Modify `narrative_dynamics/adapters/story_testimony_search.py` — reuse the shared resolver with no observable contract change.
- Create `narrative_dynamics/story/evolution_v2.py` — all V3 records, baseline timeline, intervention generation/evaluation, divergence comparison, serialization, primary API.
- Modify `narrative_dynamics/story/__init__.py` — export exactly six approved V3 names.
- Create `tests/test_story_testimony_action_resolution.py` — resolver and adapter-regression boundary.
- Create `tests/test_story_evolution_v2.py` — baseline, counterfactual, rejection, divergence, serialization, ordering, mechanism-safety behavior.
- Modify `tests/test_story_testimony_runtime.py` — exact V1+V2+V3 story package surface and root isolation.

If implementation appears to require another production file, stop and review the approved design before adding it.

---

### Task 1: Shared Testimony Action Resolution Boundary

**Files:**
- Create: `tests/test_story_testimony_action_resolution.py`
- Modify: `narrative_dynamics/story/replay_v2.py`
- Modify: `narrative_dynamics/adapters/story_testimony_search.py`

**Interfaces:**
- Consumes: `NarrativeScenarioV2`, `testimony_state`, `latest_epistemic_location`, declared decision actions.
- Produces: `resolve_testimony_action(story: NarrativeScenarioV2) -> str` in `narrative_dynamics.story.replay_v2`.
- The helper is not added to `narrative_dynamics.story.__all__`.

- [ ] **Step 1: Write the test-only RED**

Create `tests/test_story_testimony_action_resolution.py`:

```python
from __future__ import annotations

import random
import unittest

from narrative_dynamics.adapters.story_testimony_search import TestimonySearchModel
from narrative_dynamics.story.replay_v2 import resolve_testimony_action
from narrative_dynamics.story.scenario_v2 import NarrativeScenarioV2, project_testimony_scenario
from narrative_dynamics.story.schema import SearchActionV1, SearchDecisionV1, StoryEntitiesV1
from narrative_dynamics.story.schema_v2 import load_narrative_case_v2

_TRUTHFUL = "fixtures/stories/key_location_truthful_testimony_v2.json"
_STALE = "fixtures/stories/key_location_stale_testimony_v2.json"


class TestimonyActionResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.truthful_case = load_narrative_case_v2(_TRUTHFUL)
        self.stale_case = load_narrative_case_v2(_STALE)
        self.truthful = NarrativeScenarioV2.from_case(self.truthful_case)
        self.stale = NarrativeScenarioV2.from_case(self.stale_case)

    def test_shared_resolver_matches_committed_testimony_decisions(self):
        self.assertEqual(resolve_testimony_action(self.truthful), "search_box")
        self.assertEqual(resolve_testimony_action(self.stale), "search_drawer")

    def test_shared_resolver_fails_closed_without_exact_action_match(self):
        entities = StoryEntitiesV1(
            agents=self.stale.entities.agents,
            objects=self.stale.entities.objects,
            locations=("drawer", "box", "shelf"),
        )
        decision = SearchDecisionV1(
            id="d1",
            time=4,
            actor="bob",
            object="key",
            actions=(
                SearchActionV1(id="search_box", location="box"),
                SearchActionV1(id="search_shelf", location="shelf"),
            ),
        )
        story = NarrativeScenarioV2(
            entities=entities,
            events=self.stale.events,
            observations=self.stale.observations,
            reports=self.stale.reports,
            receptions=self.stale.receptions,
            decision=decision,
        )
        with self.assertRaisesRegex(ValueError, "exactly one decision action"):
            resolve_testimony_action(story)

    def test_adapter_contract_remains_exact_after_resolver_extraction(self):
        run = TestimonySearchModel().simulate(
            project_testimony_scenario(self.truthful_case), {}, random.Random(7)
        )
        self.assertEqual(
            tuple((event.tick, event.kind, dict(event.data)) for event in run.events),
            (
                (0, "epistemic_basis_resolved", {
                    "kind": "testimony",
                    "target_object": "key",
                    "resolved_location": "box",
                    "supporting_id": "r1",
                    "source_agent": "alice",
                }),
                (1, "action_policy_computed", {
                    "policy": {"search_drawer": 0.0, "search_box": 1.0}
                }),
                (2, "action_selected", {"action": "search_box"}),
            ),
        )
        self.assertEqual(
            dict(run.outcome),
            {
                "epistemic_basis": {
                    "kind": "testimony",
                    "target_object": "key",
                    "resolved_location": "box",
                    "supporting_id": "r1",
                    "source_agent": "alice",
                },
                "action_scores": {"search_drawer": 0.0, "search_box": 1.0},
                "policy": {"search_drawer": 0.0, "search_box": 1.0},
                "selected_action": "search_box",
            },
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Observe focused RED**

Run when a checkout is available:

```bash
python3 -m unittest tests.test_story_testimony_action_resolution -v
```

Expected: import/discovery error because `resolve_testimony_action` does not exist.

- [ ] **Step 3: Commit test-only RED**

```bash
git add tests/test_story_testimony_action_resolution.py
git commit -m "test: add testimony action resolver RED"
```

Record the exact RED SHA.

- [ ] **Step 4: Open draft V3 PR and observe exact-head CI RED**

Open draft PR `work/narrative-evolution-matrix-v3` -> `proof/narrative-dynamics-v0` titled:

```text
RED: add Narrative Evolution Matrix V3
```

The PR body states that current head intentionally contains Task 1 RED only and that `master` must not move.

Require the `proof` run associated with the exact RED SHA to show:

- dependency/conformance/Lean gates before Python remain green;
- Python fails because `resolve_testimony_action` is missing;
- no unrelated existing failure is accepted as RED evidence.

- [ ] **Step 5: Implement minimal shared resolver**

Append to `narrative_dynamics/story/replay_v2.py`:

```python
def resolve_testimony_action(story: NarrativeScenarioV2) -> str:
    """Resolve the unique decision action supported by the actor's latest evidence."""

    if not isinstance(story, NarrativeScenarioV2):
        raise TypeError(
            "testimony action resolution requires a validated NarrativeScenarioV2"
        )
    decision = story.decision
    location = latest_epistemic_location(
        testimony_state(story, decision.actor, at_time=decision.time),
        decision.object,
    )
    matching = tuple(
        action.id for action in decision.actions
        if action.location == location.location
    )
    if len(matching) != 1:
        raise ValueError(
            "testimony-aware target location must match exactly one decision action"
        )
    return matching[0]
```

- [ ] **Step 6: Refactor adapter to reuse the matching rule only**

Import `resolve_testimony_action` in `narrative_dynamics/adapters/story_testimony_search.py`. Keep the existing `testimony_state` / `latest_epistemic_location` calculation for the provenance basis. Delete only the local `matching` block and set:

```python
selected = resolve_testimony_action(story)
```

Do not change model name, parameter validation, scores, policy, event order, event kinds, basis keys, outcome keys, or RNG behavior.

- [ ] **Step 7: Verify focused GREEN**

```bash
python3 -m unittest \
  tests.test_story_testimony_action_resolution \
  tests.test_story_testimony_models \
  tests.test_story_testimony_runtime -v
```

Expected: all pass.

- [ ] **Step 8: Commit Task 1 GREEN**

```bash
git add \
  narrative_dynamics/story/replay_v2.py \
  narrative_dynamics/adapters/story_testimony_search.py
git commit -m "refactor: share testimony action resolution"
```

- [ ] **Step 9: Require fresh exact-head full CI GREEN**

Require full `proof` success on exact Task 1 GREEN SHA. Record run number, run ID, exact feature `head_sha`, Python count, full Lean build, StoryState gate, and Testimony gate before Task 2.

---

### Task 2: Baseline Temporal Evolution Matrix

**Files:**
- Create: `tests/test_story_evolution_v2.py`
- Create: `narrative_dynamics/story/evolution_v2.py`

**Interfaces:**
- Consumes: `NarrativeScenarioV2`, `objective_state`, `subjective_state`, `testimony_state`, `resolve_testimony_action`.
- Produces module symbols: `EvolutionSnapshotV2`, `EvolutionTrajectoryV2`, `EvolutionInterventionV2`, `EvolutionCounterfactualV2`, `EvolutionAnalysisV2`, `analyze_testimony_evolution`.
- Produces internal records: `_EvolutionAgentStateV2`, `_EvolutionDivergenceV2`.
- Package exports remain unchanged until Task 5.
- At Task 2 GREEN, analysis has a complete baseline and `counterfactuals=()`.

- [ ] **Step 1: Add baseline-only RED tests**

Create `tests/test_story_evolution_v2.py`:

```python
from __future__ import annotations

import unittest

from narrative_dynamics.story.evolution_v2 import (
    EvolutionCounterfactualV2,
    EvolutionInterventionV2,
    analyze_testimony_evolution,
)
from narrative_dynamics.story.scenario_v2 import NarrativeScenarioV2
from narrative_dynamics.story.schema_v2 import load_narrative_case_v2

_TRUTHFUL = "fixtures/stories/key_location_truthful_testimony_v2.json"
_STALE = "fixtures/stories/key_location_stale_testimony_v2.json"


class NarrativeEvolutionBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.truthful = NarrativeScenarioV2.from_case(load_narrative_case_v2(_TRUTHFUL))
        self.stale = NarrativeScenarioV2.from_case(load_narrative_case_v2(_STALE))

    def test_truthful_baseline_reconstructs_world_direct_testimony_and_action(self):
        analysis = analyze_testimony_evolution(self.truthful)
        baseline = analysis.baseline
        self.assertEqual(baseline.target_object, "key")
        self.assertEqual(baseline.tracked_agents, ("bob", "alice"))
        self.assertEqual(tuple(s.logical_time for s in baseline.snapshots), (1, 2, 3, 4))
        self.assertEqual(
            tuple(s.objective_location for s in baseline.snapshots),
            ("drawer", "box", "box", "box"),
        )
        self.assertEqual(
            tuple(s.agents["bob"].direct_location for s in baseline.snapshots),
            ("drawer", "drawer", "drawer", "drawer"),
        )
        self.assertEqual(
            tuple(s.agents["alice"].direct_location for s in baseline.snapshots),
            (None, "box", "box", "box"),
        )
        self.assertEqual(
            tuple(s.agents["bob"].testimony_location for s in baseline.snapshots),
            ("drawer", "drawer", "box", "box"),
        )
        report_state = baseline.snapshots[2].agents["bob"]
        self.assertEqual(
            (report_state.evidence_kind, report_state.supporting_id, report_state.source_agent),
            ("testimony", "r1", "alice"),
        )
        self.assertEqual(
            tuple((s.trigger_kind, s.trigger_ids) for s in baseline.snapshots),
            (
                ("relocation", ("e1",)),
                ("relocation", ("e2",)),
                ("report", ("r1",)),
                ("decision", ("d1",)),
            ),
        )
        self.assertEqual(baseline.selected_action, "search_box")
        self.assertEqual(baseline.snapshots[-1].selected_action, "search_box")
        self.assertEqual(analysis.counterfactuals, ())
        self.assertFalse(analysis.mechanism_uniqueness_claimed)

    def test_stale_baseline_changes_provenance_at_report_time_without_location_change(self):
        baseline = analyze_testimony_evolution(self.stale).baseline
        bob = tuple(snapshot.agents["bob"] for snapshot in baseline.snapshots)
        self.assertEqual(
            tuple(item.testimony_location for item in bob),
            ("drawer", "drawer", "drawer", "drawer"),
        )
        self.assertEqual((bob[1].evidence_kind, bob[1].supporting_id),
                         ("direct_perception", "e1"))
        self.assertEqual((bob[2].evidence_kind, bob[2].supporting_id, bob[2].source_agent),
                         ("testimony", "r1", "alice"))
        self.assertEqual(baseline.selected_action, "search_drawer")

    def test_analysis_requires_validated_v2_scenario(self):
        with self.assertRaisesRegex(TypeError, "validated NarrativeScenarioV2"):
            analyze_testimony_evolution(object())

    def test_public_records_freeze_nested_values_and_reject_invalid_enums(self):
        analysis = analyze_testimony_evolution(self.truthful)
        with self.assertRaises(TypeError):
            analysis.baseline.snapshots[0].agents["bob"] = object()
        with self.assertRaisesRegex(ValueError, "intervention kind"):
            EvolutionInterventionV2(
                kind="compound",
                subject_id="x",
                agent=None,
                from_value=None,
                to_value=None,
                logical_time=None,
            )
        with self.assertRaisesRegex(ValueError, "counterfactual status"):
            EvolutionCounterfactualV2(
                intervention=EvolutionInterventionV2(
                    kind="remove_reception",
                    subject_id="r1",
                    agent="bob",
                    from_value="received",
                    to_value=None,
                    logical_time=3,
                ),
                status="unknown",
                trajectory=None,
                first_divergence=None,
                rejection_stage=None,
                rejection_reason=None,
                rejection_logical_time=None,
            )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Observe focused RED**

```bash
python3 -m unittest tests.test_story_evolution_v2 -v
```

Expected: missing `narrative_dynamics.story.evolution_v2`.

- [ ] **Step 3: Commit test-only RED and observe exact-head CI RED**

```bash
git add tests/test_story_evolution_v2.py
git commit -m "test: add evolution baseline RED"
```

The exact RED head `proof` run must fail only because the new evolution module is absent.

- [ ] **Step 4: Implement immutable record layer with exact enum/coherence guards**

Create `narrative_dynamics/story/evolution_v2.py` beginning with:

```python
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

from narrative_dynamics.story.replay import objective_state, subjective_state
from narrative_dynamics.story.replay_v2 import resolve_testimony_action, testimony_state
from narrative_dynamics.story.scenario_v2 import NarrativeScenarioV2

_ALLOWED_TRIGGERS = frozenset({"relocation", "report", "decision", "mixed"})
_ALLOWED_INTERVENTIONS = frozenset({
    "change_report_content",
    "remove_direct_observation",
    "remove_reception",
    "remove_support_observation",
})
_ALLOWED_REJECTION_STAGES = frozenset({"scenario_validation", "action_resolution"})


@dataclass(frozen=True)
class _EvolutionAgentStateV2:
    agent: str
    direct_location: str | None
    direct_supporting_id: str | None
    testimony_location: str | None
    evidence_kind: str | None
    supporting_id: str | None
    source_agent: str | None
    evidence_logical_time: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "agent": self.agent,
            "direct_location": self.direct_location,
            "direct_supporting_id": self.direct_supporting_id,
            "testimony_location": self.testimony_location,
            "evidence_kind": self.evidence_kind,
            "supporting_id": self.supporting_id,
            "source_agent": self.source_agent,
            "evidence_logical_time": self.evidence_logical_time,
        }


@dataclass(frozen=True)
class EvolutionSnapshotV2:
    logical_time: int
    trigger_kind: str
    trigger_ids: tuple[str, ...]
    objective_location: str | None
    agents: Mapping[str, _EvolutionAgentStateV2]
    selected_action: str | None

    def __post_init__(self) -> None:
        if self.trigger_kind not in _ALLOWED_TRIGGERS:
            raise ValueError("evolution trigger kind is not supported")
        trigger_ids = tuple(self.trigger_ids)
        agents = dict(self.agents)
        if any(not isinstance(value, _EvolutionAgentStateV2) for value in agents.values()):
            raise TypeError("evolution agents must contain canonical agent states")
        object.__setattr__(self, "trigger_ids", trigger_ids)
        object.__setattr__(self, "agents", MappingProxyType(agents))

    def to_dict(self) -> dict[str, object]:
        return {
            "logical_time": self.logical_time,
            "trigger_kind": self.trigger_kind,
            "trigger_ids": list(self.trigger_ids),
            "objective_location": self.objective_location,
            "agents": {name: value.to_dict() for name, value in self.agents.items()},
            "selected_action": self.selected_action,
        }


@dataclass(frozen=True)
class EvolutionTrajectoryV2:
    target_object: str
    tracked_agents: tuple[str, ...]
    snapshots: tuple[EvolutionSnapshotV2, ...]
    selected_action: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "tracked_agents", tuple(self.tracked_agents))
        object.__setattr__(self, "snapshots", tuple(self.snapshots))

    def to_dict(self) -> dict[str, object]:
        return {
            "target_object": self.target_object,
            "tracked_agents": list(self.tracked_agents),
            "snapshots": [snapshot.to_dict() for snapshot in self.snapshots],
            "selected_action": self.selected_action,
        }


@dataclass(frozen=True)
class EvolutionInterventionV2:
    kind: str
    subject_id: str
    agent: str | None
    from_value: str | None
    to_value: str | None
    logical_time: int | None

    def __post_init__(self) -> None:
        if self.kind not in _ALLOWED_INTERVENTIONS:
            raise ValueError("evolution intervention kind is not supported")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "subject_id": self.subject_id,
            "agent": self.agent,
            "from_value": self.from_value,
            "to_value": self.to_value,
            "logical_time": self.logical_time,
        }


@dataclass(frozen=True)
class _EvolutionDivergenceV2:
    logical_time: int
    changed_fields: tuple[str, ...]
    action_changed: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "changed_fields", tuple(self.changed_fields))

    def to_dict(self) -> dict[str, object]:
        return {
            "logical_time": self.logical_time,
            "changed_fields": list(self.changed_fields),
            "action_changed": self.action_changed,
        }


@dataclass(frozen=True)
class EvolutionCounterfactualV2:
    intervention: EvolutionInterventionV2
    status: str
    trajectory: EvolutionTrajectoryV2 | None
    first_divergence: _EvolutionDivergenceV2 | None
    rejection_stage: str | None
    rejection_reason: str | None
    rejection_logical_time: int | None

    def __post_init__(self) -> None:
        if self.status not in {"valid", "rejected"}:
            raise ValueError("evolution counterfactual status must be valid or rejected")
        if self.status == "valid":
            if self.trajectory is None:
                raise ValueError("valid evolution counterfactual requires a trajectory")
            if any(value is not None for value in (
                self.rejection_stage,
                self.rejection_reason,
                self.rejection_logical_time,
            )):
                raise ValueError("valid evolution counterfactual cannot contain rejection data")
        else:
            if self.trajectory is not None or self.first_divergence is not None:
                raise ValueError("rejected evolution counterfactual cannot contain a trajectory")
            if self.rejection_stage not in _ALLOWED_REJECTION_STAGES:
                raise ValueError("rejected evolution counterfactual requires a rejection stage")
            if not isinstance(self.rejection_reason, str) or not self.rejection_reason:
                raise ValueError("rejected evolution counterfactual requires a rejection reason")

    def to_dict(self) -> dict[str, object]:
        return {
            "intervention": self.intervention.to_dict(),
            "status": self.status,
            "trajectory": None if self.trajectory is None else self.trajectory.to_dict(),
            "first_divergence": (
                None if self.first_divergence is None else self.first_divergence.to_dict()
            ),
            "rejection_stage": self.rejection_stage,
            "rejection_reason": self.rejection_reason,
            "rejection_logical_time": self.rejection_logical_time,
        }


@dataclass(frozen=True)
class EvolutionAnalysisV2:
    baseline: EvolutionTrajectoryV2
    counterfactuals: tuple[EvolutionCounterfactualV2, ...]
    mechanism_uniqueness_claimed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "counterfactuals", tuple(self.counterfactuals))

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline": self.baseline.to_dict(),
            "counterfactuals": [item.to_dict() for item in self.counterfactuals],
            "mechanism_uniqueness_claimed": self.mechanism_uniqueness_claimed,
        }
```

`Callable` and `replace` are imported now because later Tasks use them; they do not alter Task 2 behavior.

- [ ] **Step 5: Implement deterministic baseline helpers with no unresolved fields**

Add:

```python
def _tracked_agents(story: NarrativeScenarioV2) -> tuple[str, ...]:
    target = story.decision.object
    ordered = [story.decision.actor]
    seen = {story.decision.actor}
    for report in story.reports:
        if report.object == target and report.speaker not in seen:
            ordered.append(report.speaker)
            seen.add(report.speaker)
    return tuple(ordered)


def _snapshot_times(story: NarrativeScenarioV2) -> tuple[int, ...]:
    values = {event.logical_time for event in story.events}
    values.update(report.logical_time for report in story.reports)
    values.add(story.decision.time)
    return tuple(sorted(values))


def _trigger_at(
    story: NarrativeScenarioV2,
    logical_time: int,
) -> tuple[str, tuple[str, ...]]:
    kinds: list[str] = []
    ids: list[str] = []
    event_ids = tuple(
        event.id for event in story.events if event.logical_time == logical_time
    )
    if event_ids:
        kinds.append("relocation")
        ids.extend(event_ids)
    report_ids = tuple(
        report.id for report in story.reports if report.logical_time == logical_time
    )
    if report_ids:
        kinds.append("report")
        ids.extend(report_ids)
    if story.decision.time == logical_time:
        kinds.append("decision")
        ids.append(story.decision.id)
    if not kinds:
        raise RuntimeError("evolution snapshot time has no canonical trigger")
    return (kinds[0] if len(kinds) == 1 else "mixed", tuple(ids))


def _agent_state(
    story: NarrativeScenarioV2,
    agent: str,
    target: str,
    logical_time: int,
) -> _EvolutionAgentStateV2:
    direct = subjective_state(
        story.events, story.observations, agent, at_time=logical_time
    ).get(target)
    testimony = testimony_state(story, agent, at_time=logical_time).get(target)
    return _EvolutionAgentStateV2(
        agent=agent,
        direct_location=None if direct is None else direct.location,
        direct_supporting_id=None if direct is None else direct.supporting_event_id,
        testimony_location=None if testimony is None else testimony.location,
        evidence_kind=None if testimony is None else testimony.evidence_kind,
        supporting_id=None if testimony is None else testimony.supporting_id,
        source_agent=None if testimony is None else testimony.source_agent,
        evidence_logical_time=None if testimony is None else testimony.logical_time,
    )


def _build_trajectory(story: NarrativeScenarioV2) -> EvolutionTrajectoryV2:
    selected = resolve_testimony_action(story)
    target = story.decision.object
    tracked = _tracked_agents(story)
    snapshots: list[EvolutionSnapshotV2] = []
    for logical_time in _snapshot_times(story):
        objective = objective_state(story.events, at_time=logical_time).get(target)
        trigger_kind, trigger_ids = _trigger_at(story, logical_time)
        agents = MappingProxyType({
            agent: _agent_state(story, agent, target, logical_time)
            for agent in tracked
        })
        snapshots.append(EvolutionSnapshotV2(
            logical_time=logical_time,
            trigger_kind=trigger_kind,
            trigger_ids=trigger_ids,
            objective_location=None if objective is None else objective.location,
            agents=agents,
            selected_action=(
                selected if logical_time == story.decision.time else None
            ),
        ))
    return EvolutionTrajectoryV2(
        target_object=target,
        tracked_agents=tracked,
        snapshots=tuple(snapshots),
        selected_action=selected,
    )
```

- [ ] **Step 6: Implement baseline-only primary API**

```python
def analyze_testimony_evolution(story: NarrativeScenarioV2) -> EvolutionAnalysisV2:
    if not isinstance(story, NarrativeScenarioV2):
        raise TypeError("evolution analysis requires a validated NarrativeScenarioV2")
    return EvolutionAnalysisV2(
        baseline=_build_trajectory(story),
        counterfactuals=(),
    )
```

Do not change `narrative_dynamics/story/__init__.py` in Task 2.

- [ ] **Step 7: Verify focused GREEN**

```bash
python3 -m unittest \
  tests.test_story_evolution_v2 \
  tests.test_story_testimony_action_resolution \
  tests.test_story_replay_v2 -v
```

Expected: all pass.

- [ ] **Step 8: Commit Task 2 GREEN**

```bash
git add narrative_dynamics/story/evolution_v2.py
git commit -m "feat: add baseline narrative evolution matrix"
```

- [ ] **Step 9: Require fresh exact-head full CI GREEN**

Require full `proof` success on exact Task 2 GREEN SHA and record exact head/run/Python/Lean evidence.

---

### Task 3: Valid Minimal Counterfactuals and Canonical First Divergence

**Files:**
- Modify: `tests/test_story_evolution_v2.py`
- Modify: `narrative_dynamics/story/evolution_v2.py`

**Interfaces:**
- Consumes: Task 2 records and `_build_trajectory`.
- Produces: valid `change_report_content` and `remove_reception` counterfactuals, generic validation/action rejection evaluator, canonical first-divergence comparison, deterministic counterfactual order.

- [ ] **Step 1: Add valid-counterfactual RED tests**

Extend `tests/test_story_evolution_v2.py` and import the internal comparator for its identity invariant:

```python
from narrative_dynamics.story.evolution_v2 import _first_divergence


class NarrativeEvolutionValidCounterfactualTests(unittest.TestCase):
    def setUp(self) -> None:
        self.truthful = NarrativeScenarioV2.from_case(load_narrative_case_v2(_TRUTHFUL))
        self.stale = NarrativeScenarioV2.from_case(load_narrative_case_v2(_STALE))

    @staticmethod
    def by_kind(analysis, kind: str):
        return tuple(
            item for item in analysis.counterfactuals
            if item.intervention.kind == kind
        )

    def test_remove_reception_exposes_provenance_and_action_difference(self):
        item = self.by_kind(
            analyze_testimony_evolution(self.truthful), "remove_reception"
        )[0]
        self.assertEqual(item.status, "valid")
        self.assertEqual(item.trajectory.selected_action, "search_drawer")
        self.assertEqual(item.first_divergence.logical_time, 3)
        self.assertTrue(item.first_divergence.action_changed)
        self.assertIn("agents.bob.evidence_kind", item.first_divergence.changed_fields)
        self.assertIn("agents.bob.supporting_id", item.first_divergence.changed_fields)
        bob = item.trajectory.snapshots[2].agents["bob"]
        self.assertEqual((bob.evidence_kind, bob.supporting_id),
                         ("direct_perception", "e1"))

        stale = self.by_kind(
            analyze_testimony_evolution(self.stale), "remove_reception"
        )[0]
        self.assertEqual(stale.status, "valid")
        self.assertEqual(stale.trajectory.selected_action, "search_drawer")
        self.assertEqual(stale.first_divergence.logical_time, 3)
        self.assertFalse(stale.first_divergence.action_changed)
        self.assertIn("agents.bob.evidence_kind", stale.first_divergence.changed_fields)

    def test_change_report_content_diverges_at_report_time_and_flips_action(self):
        item = self.by_kind(
            analyze_testimony_evolution(self.truthful), "change_report_content"
        )[0]
        self.assertEqual(
            (item.intervention.subject_id,
             item.intervention.from_value,
             item.intervention.to_value),
            ("r1", "box", "drawer"),
        )
        self.assertEqual(item.status, "valid")
        self.assertEqual(item.first_divergence.logical_time, 3)
        self.assertTrue(item.first_divergence.action_changed)
        self.assertEqual(item.trajectory.selected_action, "search_drawer")
        self.assertIn(
            "agents.bob.testimony_location",
            item.first_divergence.changed_fields,
        )

        stale = self.by_kind(
            analyze_testimony_evolution(self.stale), "change_report_content"
        )[0]
        self.assertEqual(
            (stale.intervention.from_value, stale.intervention.to_value),
            ("drawer", "box"),
        )
        self.assertEqual(stale.first_divergence.logical_time, 3)
        self.assertEqual(stale.trajectory.selected_action, "search_box")
        self.assertTrue(stale.first_divergence.action_changed)

    def test_identical_trajectory_has_no_first_divergence(self):
        baseline = analyze_testimony_evolution(self.truthful).baseline
        self.assertIsNone(_first_divergence(baseline, baseline))

    def test_counterfactual_order_and_changed_paths_are_deterministic(self):
        analysis = analyze_testimony_evolution(self.truthful)
        keys = tuple(
            (
                item.intervention.kind,
                -1 if item.intervention.logical_time is None else item.intervention.logical_time,
                item.intervention.subject_id,
                "" if item.intervention.to_value is None else item.intervention.to_value,
                "" if item.intervention.agent is None else item.intervention.agent,
            )
            for item in analysis.counterfactuals
        )
        self.assertEqual(keys, tuple(sorted(keys)))
        for item in analysis.counterfactuals:
            if item.first_divergence is not None:
                self.assertEqual(
                    item.first_divergence.changed_fields,
                    tuple(sorted(item.first_divergence.changed_fields)),
                )
```

- [ ] **Step 2: Observe focused RED**

```bash
python3 -m unittest tests.test_story_evolution_v2 -v
```

Expected: baseline tests pass; new imports/assertions fail because `_first_divergence` and counterfactual generation are absent.

- [ ] **Step 3: Commit test-only RED and observe exact-head CI RED**

```bash
git add tests/test_story_evolution_v2.py
git commit -m "test: add valid evolution counterfactual RED"
```

The exact RED `proof` run may fail at import because `_first_divergence` is missing; no unrelated failure is accepted.

- [ ] **Step 4: Implement canonical snapshot comparison**

Add:

```python
def _snapshot_fields(snapshot: EvolutionSnapshotV2) -> dict[str, object]:
    fields: dict[str, object] = {
        "objective_location": snapshot.objective_location,
        "selected_action": snapshot.selected_action,
    }
    for agent, state in snapshot.agents.items():
        prefix = f"agents.{agent}."
        fields[prefix + "direct_location"] = state.direct_location
        fields[prefix + "direct_supporting_id"] = state.direct_supporting_id
        fields[prefix + "testimony_location"] = state.testimony_location
        fields[prefix + "evidence_kind"] = state.evidence_kind
        fields[prefix + "supporting_id"] = state.supporting_id
        fields[prefix + "source_agent"] = state.source_agent
        fields[prefix + "evidence_logical_time"] = state.evidence_logical_time
    return fields


def _first_divergence(
    baseline: EvolutionTrajectoryV2,
    counterfactual: EvolutionTrajectoryV2,
) -> _EvolutionDivergenceV2 | None:
    baseline_times = tuple(item.logical_time for item in baseline.snapshots)
    counterfactual_times = tuple(item.logical_time for item in counterfactual.snapshots)
    if baseline_times != counterfactual_times:
        raise RuntimeError("evolution counterfactual changed canonical snapshot times")
    action_changed = baseline.selected_action != counterfactual.selected_action
    for baseline_snapshot, counterfactual_snapshot in zip(
        baseline.snapshots, counterfactual.snapshots
    ):
        baseline_fields = _snapshot_fields(baseline_snapshot)
        counterfactual_fields = _snapshot_fields(counterfactual_snapshot)
        if baseline_fields.keys() != counterfactual_fields.keys():
            raise RuntimeError("evolution counterfactual changed tracked state fields")
        changed = tuple(sorted(
            key for key in baseline_fields
            if baseline_fields[key] != counterfactual_fields[key]
        ))
        if changed:
            return _EvolutionDivergenceV2(
                logical_time=baseline_snapshot.logical_time,
                changed_fields=changed,
                action_changed=action_changed,
            )
    return None
```

- [ ] **Step 5: Implement one generic candidate evaluator**

```python
def _evaluate_candidate(
    baseline: EvolutionTrajectoryV2,
    intervention: EvolutionInterventionV2,
    build_story: Callable[[], NarrativeScenarioV2],
) -> EvolutionCounterfactualV2:
    try:
        candidate = build_story()
    except ValueError as error:
        return EvolutionCounterfactualV2(
            intervention=intervention,
            status="rejected",
            trajectory=None,
            first_divergence=None,
            rejection_stage="scenario_validation",
            rejection_reason=str(error),
            rejection_logical_time=intervention.logical_time,
        )
    try:
        trajectory = _build_trajectory(candidate)
    except ValueError as error:
        return EvolutionCounterfactualV2(
            intervention=intervention,
            status="rejected",
            trajectory=None,
            first_divergence=None,
            rejection_stage="action_resolution",
            rejection_reason=str(error),
            rejection_logical_time=intervention.logical_time,
        )
    return EvolutionCounterfactualV2(
        intervention=intervention,
        status="valid",
        trajectory=trajectory,
        first_divergence=_first_divergence(baseline, trajectory),
        rejection_stage=None,
        rejection_reason=None,
        rejection_logical_time=None,
    )
```

`TypeError`, `RuntimeError`, and other programming failures propagate.

- [ ] **Step 6: Generate `remove_reception` candidates**

For each target report actually received by the decision actor, build:

```python
intervention = EvolutionInterventionV2(
    kind="remove_reception",
    subject_id=report.id,
    agent=story.decision.actor,
    from_value="received",
    to_value=None,
    logical_time=report.logical_time,
)
```

Candidate construction removes only the exact reception and revalidates with `replace`:

```python
receptions = tuple(
    reception for reception in story.receptions
    if not (
        reception.report == report.id
        and reception.recipient == story.decision.actor
    )
)
build_story = lambda receptions=receptions: replace(story, receptions=receptions)
```

Evaluate through `_evaluate_candidate`.

- [ ] **Step 7: Generate `change_report_content` candidates**

For every target report and every alternate decision action location excluding its current location:

```python
intervention = EvolutionInterventionV2(
    kind="change_report_content",
    subject_id=report.id,
    agent=report.speaker,
    from_value=report.location,
    to_value=alternate_location,
    logical_time=report.logical_time,
)
changed_reports = tuple(
    replace(item, location=alternate_location) if item.id == report.id else item
    for item in story.reports
)
build_story = lambda changed_reports=changed_reports: replace(
    story, reports=changed_reports
)
```

Do not compare report content with objective truth before evaluation.

- [ ] **Step 8: Add exact sort key and wire Task 3 analysis**

```python
def _counterfactual_sort_key(item: EvolutionCounterfactualV2) -> tuple[object, ...]:
    intervention = item.intervention
    return (
        intervention.kind,
        -1 if intervention.logical_time is None else intervention.logical_time,
        intervention.subject_id,
        "" if intervention.to_value is None else intervention.to_value,
        "" if intervention.agent is None else intervention.agent,
    )
```

Update `analyze_testimony_evolution` to compute baseline once, generate Task 3 candidate records, sort them by this key, and return them. Task 4 adds two more generators without changing this ordering rule.

- [ ] **Step 9: Verify focused GREEN**

```bash
python3 -m unittest tests.test_story_evolution_v2 -v
```

Expected: baseline + valid-counterfactual tests pass.

- [ ] **Step 10: Commit Task 3 GREEN**

```bash
git add narrative_dynamics/story/evolution_v2.py
git commit -m "feat: compare valid evolution counterfactuals"
```

- [ ] **Step 11: Require fresh exact-head full CI GREEN**

Require full `proof` success on exact Task 3 GREEN SHA before Task 4.

---

### Task 4: Validation Boundaries and Same-Action/Different-Mechanism Witness

**Files:**
- Modify: `tests/test_story_evolution_v2.py`
- Modify: `narrative_dynamics/story/evolution_v2.py`

**Interfaces:**
- Consumes: Task 3 evaluator/sort/divergence pipeline.
- Produces: `remove_direct_observation`, `remove_support_observation`, explicit `scenario_validation` and `action_resolution` rejection evidence, same-action/different-provenance witness.

- [ ] **Step 1: Add validation-boundary RED tests**

Append:

```python
from narrative_dynamics.story.schema import SearchActionV1, SearchDecisionV1, StoryEntitiesV1


class NarrativeEvolutionBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.truthful = NarrativeScenarioV2.from_case(load_narrative_case_v2(_TRUTHFUL))
        self.stale = NarrativeScenarioV2.from_case(load_narrative_case_v2(_STALE))

    @staticmethod
    def one(analysis, kind: str):
        items = tuple(
            item for item in analysis.counterfactuals
            if item.intervention.kind == kind
        )
        if len(items) != 1:
            raise AssertionError((kind, len(items)))
        return items[0]

    def test_remove_actor_only_observation_is_scenario_validation_rejection(self):
        item = self.one(
            analyze_testimony_evolution(self.truthful), "remove_direct_observation"
        )
        self.assertEqual(
            (item.intervention.subject_id,
             item.intervention.agent,
             item.intervention.logical_time),
            ("e1", "bob", 1),
        )
        self.assertEqual(item.status, "rejected")
        self.assertIsNone(item.trajectory)
        self.assertIsNone(item.first_divergence)
        self.assertEqual(item.rejection_stage, "scenario_validation")
        self.assertEqual(item.rejection_logical_time, 1)
        self.assertIn(
            "decision actor must have observed a relocation of the target object",
            item.rejection_reason,
        )

    def test_remove_speaker_support_is_provenance_validation_rejection(self):
        item = self.one(
            analyze_testimony_evolution(self.truthful), "remove_support_observation"
        )
        self.assertEqual(
            (item.intervention.subject_id,
             item.intervention.agent,
             item.intervention.from_value,
             item.intervention.logical_time),
            ("r1", "alice", "e2", 2),
        )
        self.assertEqual(item.status, "rejected")
        self.assertIsNone(item.trajectory)
        self.assertIsNone(item.first_divergence)
        self.assertEqual(item.rejection_stage, "scenario_validation")
        self.assertEqual(item.rejection_logical_time, 2)
        self.assertIn(
            "report speaker must have directly observed the support_event",
            item.rejection_reason,
        )

    def test_remove_reception_can_be_action_resolution_rejection(self):
        entities = StoryEntitiesV1(
            agents=self.truthful.entities.agents,
            objects=self.truthful.entities.objects,
            locations=("drawer", "box", "shelf"),
        )
        decision = SearchDecisionV1(
            id="d1",
            time=4,
            actor="bob",
            object="key",
            actions=(
                SearchActionV1(id="search_box", location="box"),
                SearchActionV1(id="search_shelf", location="shelf"),
            ),
        )
        story = NarrativeScenarioV2(
            entities=entities,
            events=self.truthful.events,
            observations=self.truthful.observations,
            reports=self.truthful.reports,
            receptions=self.truthful.receptions,
            decision=decision,
        )
        analysis = analyze_testimony_evolution(story)
        self.assertEqual(analysis.baseline.selected_action, "search_box")
        item = self.one(analysis, "remove_reception")
        self.assertEqual(item.status, "rejected")
        self.assertEqual(item.rejection_stage, "action_resolution")
        self.assertEqual(item.rejection_logical_time, 3)
        self.assertIn("exactly one decision action", item.rejection_reason)
        self.assertIsNone(item.trajectory)
        self.assertIsNone(item.first_divergence)

    def test_same_action_can_have_different_information_mechanisms(self):
        analysis = analyze_testimony_evolution(self.stale)
        reception_removed = self.one(analysis, "remove_reception")
        self.assertEqual(analysis.baseline.selected_action, "search_drawer")
        self.assertEqual(reception_removed.trajectory.selected_action, "search_drawer")
        self.assertEqual(
            analysis.baseline.snapshots[2].agents["bob"].evidence_kind,
            "testimony",
        )
        self.assertEqual(
            reception_removed.trajectory.snapshots[2].agents["bob"].evidence_kind,
            "direct_perception",
        )
        self.assertFalse(reception_removed.first_divergence.action_changed)
        self.assertFalse(analysis.mechanism_uniqueness_claimed)

    def test_full_intervention_kind_set_is_exact(self):
        kinds = {
            item.intervention.kind
            for item in analyze_testimony_evolution(self.truthful).counterfactuals
        }
        self.assertEqual(
            kinds,
            {
                "change_report_content",
                "remove_direct_observation",
                "remove_reception",
                "remove_support_observation",
            },
        )
```

- [ ] **Step 2: Observe focused RED**

```bash
python3 -m unittest tests.test_story_evolution_v2 -v
```

Expected: Task 2/3 tests pass; observation-removal kinds are absent so Task 4 tests fail.

- [ ] **Step 3: Commit test-only RED and observe exact-head CI RED**

```bash
git add tests/test_story_evolution_v2.py
git commit -m "test: add evolution validation-boundary RED"
```

Require the exact RED `proof` run to fail only the new Task 4 assertions.

- [ ] **Step 4: Generate `remove_direct_observation`**

Build `event_by_id`. For every decision-actor observation whose event relocates the target:

```python
intervention = EvolutionInterventionV2(
    kind="remove_direct_observation",
    subject_id=observation.event,
    agent=observation.agent,
    from_value="observed",
    to_value=None,
    logical_time=event_by_id[observation.event].logical_time,
)
changed_observations = tuple(
    item for item in story.observations
    if not (
        item.event == observation.event
        and item.agent == observation.agent
    )
)
build_story = lambda changed_observations=changed_observations: replace(
    story, observations=changed_observations
)
```

Evaluate via the existing `_evaluate_candidate`. On committed fixtures Bob/e1 removal is rejected by existing V2 decision-actor observation validation.

- [ ] **Step 5: Generate `remove_support_observation`**

For every target report:

```python
intervention = EvolutionInterventionV2(
    kind="remove_support_observation",
    subject_id=report.id,
    agent=report.speaker,
    from_value=report.support_event,
    to_value=None,
    logical_time=event_by_id[report.support_event].logical_time,
)
changed_observations = tuple(
    item for item in story.observations
    if not (
        item.event == report.support_event
        and item.agent == report.speaker
    )
)
build_story = lambda changed_observations=changed_observations: replace(
    story, observations=changed_observations
)
```

Evaluate via `_evaluate_candidate`. On committed fixtures Alice/e2 removal is rejected by speaker-support validation.

- [ ] **Step 6: Collect all four families and keep one sort rule**

Generate Task 3 and Task 4 records into one list, then return:

```python
counterfactuals = tuple(sorted(records, key=_counterfactual_sort_key))
return EvolutionAnalysisV2(
    baseline=baseline,
    counterfactuals=counterfactuals,
)
```

No rejected record gets a synthetic trajectory or first divergence.

- [ ] **Step 7: Verify focused GREEN**

```bash
python3 -m unittest tests.test_story_evolution_v2 -v
```

Expected: all baseline, valid intervention, scenario rejection, action rejection, ordering, and mechanism-safety tests pass.

- [ ] **Step 8: Commit Task 4 GREEN**

```bash
git add narrative_dynamics/story/evolution_v2.py
git commit -m "feat: expose evolution validation boundaries"
```

- [ ] **Step 9: Require fresh exact-head full CI GREEN**

Require full `proof` success on exact Task 4 GREEN SHA before Task 5.

---

### Task 5: Public V3 API, Serialization Lock, and Full Story Regression

**Files:**
- Modify: `tests/test_story_testimony_runtime.py`
- Modify: `tests/test_story_evolution_v2.py`
- Modify: `narrative_dynamics/story/__init__.py`

**Interfaces:**
- Consumes: completed `evolution_v2` public module symbols.
- Produces exact `narrative_dynamics.story` export expansion with six V3 names and no root leakage.

- [ ] **Step 1: Add exact public-surface RED**

In `tests/test_story_testimony_runtime.py`, add:

```python
_V3_EXPORTS = {
    "EvolutionSnapshotV2",
    "EvolutionTrajectoryV2",
    "EvolutionInterventionV2",
    "EvolutionCounterfactualV2",
    "EvolutionAnalysisV2",
    "analyze_testimony_evolution",
}
```

Update exact package export assertion:

```python
expected = _V1_EXPORTS | _V2_EXPORTS | _V3_EXPORTS
self.assertEqual(set(getattr(story, "__all__", ())), expected)
for name in expected:
    self.assertTrue(hasattr(story, name), name)
```

Update root isolation:

```python
for name in _V2_EXPORTS | _V3_EXPORTS:
    self.assertFalse(hasattr(narrative_dynamics, name), name)
```

Lock internal names out of story exports:

```python
for name in (
    "resolve_testimony_action",
    "_EvolutionAgentStateV2",
    "_EvolutionDivergenceV2",
):
    self.assertNotIn(name, getattr(story, "__all__", ()))
```

- [ ] **Step 2: Add deterministic JSON structural serialization test**

In `tests/test_story_evolution_v2.py`, add `import json` and:

```python
def test_analysis_to_dict_is_deterministic_json_serializable(self):
    first = analyze_testimony_evolution(self.truthful).to_dict()
    second = analyze_testimony_evolution(self.truthful).to_dict()
    self.assertEqual(first, second)
    encoded = json.dumps(first, sort_keys=True, separators=(",", ":"))
    self.assertEqual(json.loads(encoded), first)
    self.assertFalse(first["mechanism_uniqueness_claimed"])
```

Place this method in `NarrativeEvolutionBaselineTests`.

- [ ] **Step 3: Observe focused RED**

```bash
python3 -m unittest \
  tests.test_story_evolution_v2 \
  tests.test_story_testimony_runtime -v
```

Expected: serialization passes; exact package export assertion fails because six V3 names are not yet in `story.__all__`.

- [ ] **Step 4: Commit test-only RED and observe exact-head CI RED**

```bash
git add tests/test_story_evolution_v2.py tests/test_story_testimony_runtime.py
git commit -m "test: add Narrative Evolution V3 public API RED"
```

Require exact RED `proof` failure only in the new export assertion.

- [ ] **Step 5: Export exactly six approved V3 names**

In `narrative_dynamics/story/__init__.py` import:

```python
from narrative_dynamics.story.evolution_v2 import (
    EvolutionAnalysisV2,
    EvolutionCounterfactualV2,
    EvolutionInterventionV2,
    EvolutionSnapshotV2,
    EvolutionTrajectoryV2,
    analyze_testimony_evolution,
)
```

Append exactly:

```python
"EvolutionSnapshotV2",
"EvolutionTrajectoryV2",
"EvolutionInterventionV2",
"EvolutionCounterfactualV2",
"EvolutionAnalysisV2",
"analyze_testimony_evolution",
```

Do not export `resolve_testimony_action`, `_EvolutionAgentStateV2`, `_EvolutionDivergenceV2`, or `TestimonySearchModel`.

- [ ] **Step 6: Run focused story regression**

```bash
python3 -m unittest \
  tests.test_story_evolution_v2 \
  tests.test_story_testimony_action_resolution \
  tests.test_story_replay_v2 \
  tests.test_story_scenario_v2 \
  tests.test_story_schema_v2 \
  tests.test_story_testimony_models \
  tests.test_story_testimony_runtime \
  tests.test_story_runtime -v
```

Expected: all pass.

- [ ] **Step 7: Run full Python discovery when local checkout exists**

```bash
python3 -m unittest discover -s tests -v
```

Expected: zero failures/errors.

- [ ] **Step 8: Commit Task 5 GREEN**

```bash
git add narrative_dynamics/story/__init__.py
git commit -m "feat: publish Narrative Evolution V3 story API"
```

- [ ] **Step 9: Require fresh exact-head full CI GREEN**

Require complete `proof` success on exact Task 5 GREEN SHA. Record exact feature SHA, run number/ID, Lean full build, Python total and `OK`, StoryState success, Testimony success. Do not treat this as independent standalone `compileall` evidence.

---

### Task 6: Final Evidence, Review, and Research-Branch Integration

**Files:**
- No production change expected.
- PR title/body metadata may be updated.
- Any review defect requires a focused test-only RED before a production fix.

**Interfaces:**
- Consumes: exact final V3 feature head and pre-V3 research SHA `29a15a49e3191e3d6fde995064642630a4f8ea2a`.
- Produces: reviewed/verified V3 research head and non-forced fast-forward of `proof/narrative-dynamics-v0` only.

- [ ] **Step 1: Re-read spec and build requirement checklist**

Confirm every item:

- baseline snapshots at every relocation/report/decision time;
- decision actor + target-report speakers only;
- objective/direct/testimony/provenance states distinct;
- no objective fallback;
- exactly four single-variable intervention kinds;
- report-content intervention never consults objective truth;
- valid divergence uses canonical state fields only;
- canonically identical trajectories return no divergence;
- rejected scenario/action candidates have no trajectory/divergence;
- both `scenario_validation` and `action_resolution` paths are tested;
- stale same-action/different-provenance witness exists;
- mechanism uniqueness is structurally false;
- public records defensively freeze nested tuple/mapping inputs;
- deterministic serialization/order;
- exact six-name V3 story public surface;
- no package-root leakage;
- shared resolver leaves adapter output contract exact;
- no runtime/registry/metrics/prison/Lean/V1 semantic changes.

If any requirement is missing, do not integrate; add a focused RED before fixing it.

- [ ] **Step 2: Review complete diff against exact pre-V3 SHA**

If using a local checkout:

```bash
FINAL_SHA=$(git rev-parse work/narrative-evolution-matrix-v3)
git diff --stat 29a15a49e3191e3d6fde995064642630a4f8ea2a "$FINAL_SHA"
git diff 29a15a49e3191e3d6fde995064642630a4f8ea2a "$FINAL_SHA"
```

Equivalent connector compare must report merge base exactly `29a15a49e3191e3d6fde995064642630a4f8ea2a`, `ahead > 0`, `behind = 0`.

Expected changed files are only:

```text
docs/superpowers/specs/2026-08-24-narrative-evolution-matrix-v3-design.md
docs/superpowers/plans/2026-08-24-narrative-evolution-matrix-v3.md
narrative_dynamics/story/replay_v2.py
narrative_dynamics/adapters/story_testimony_search.py
narrative_dynamics/story/evolution_v2.py
narrative_dynamics/story/__init__.py
tests/test_story_testimony_action_resolution.py
tests/test_story_evolution_v2.py
tests/test_story_testimony_runtime.py
```

Any additional production file blocks integration until reconciled with the approved spec.

- [ ] **Step 3: Inspect claim boundaries**

Search changed code/tests/docs/PR for positive unsupported labels/claims:

```text
unique_cause
true_mechanism
identified_mechanism
human Theory of Mind
lying intent
deception intent
empirical human validity
causal effect
```

Existing negative provenance flags are not violations.

- [ ] **Step 4: Obtain fresh exact-head final verification**

Require latest `proof` workflow for exact final feature SHA to be `completed/success`. Inspect jobs/logs and confirm full Python discovery plus StoryState/Testimony gates. If final SHA changed after the last GREEN run, require a fresh run for the new SHA before any completion claim.

- [ ] **Step 5: Independently attempt compileall**

If a usable checkout exists:

```bash
python3 -m compileall -q narrative_dynamics
```

Record success only on exit 0. If environment prevents the independent command, record that gap explicitly.

- [ ] **Step 6: Update draft PR to final V3 evidence**

Set title:

```text
feat: add Narrative Evolution Matrix V3
```

Final body includes:

- temporal baseline matrix purpose;
- four intervention families;
- valid first-divergence vs validation-rejection distinction;
- same-action/different-provenance non-identification witness;
- exact six-name public API boundary;
- exact final feature SHA;
- final `proof` run number/ID and Python/Lean evidence;
- independent compileall status only if actually observed;
- strongest allowed claim: canonical temporal evolution plus minimal single-variable counterfactual comparison, not unique hidden-mechanism identification or empirical human causal validity;
- target `proof/narrative-dynamics-v0`; `master` must not move.

- [ ] **Step 7: Lock topology immediately before integration**

Freshly verify:

1. `proof/narrative-dynamics-v0` is still exactly `29a15a49e3191e3d6fde995064642630a4f8ea2a`.
2. `work/narrative-evolution-matrix-v3` equals the exact reviewed/verified final SHA.
3. Feature vs research is ahead, not behind; merge base is `29a15a49e3191e3d6fde995064642630a4f8ea2a`.
4. Capture current `master` SHA as `MASTER_BEFORE`.

If research moved or feature diverged/behind, stop; do not force-update or silently rebase.

- [ ] **Step 8: Non-forced fast-forward research only**

With connector API, update branch `proof/narrative-dynamics-v0` to the verified final feature SHA with `force=false`.

With local Git, first set:

```bash
FINAL_SHA=$(git rev-parse work/narrative-evolution-matrix-v3)
```

Then push only when the remote research ref is still the exact pre-V3 SHA:

```bash
git push origin "$FINAL_SHA":refs/heads/proof/narrative-dynamics-v0
```

Do not update `master`.

- [ ] **Step 9: Verify post-integration identity**

Freshly verify:

- research and feature refs are identical at the exact final SHA;
- `master` is still `MASTER_BEFORE`;
- feature branch remains present.

GitHub may automatically mark the PR closed/merged after base/head become identical. Record the actual PR state and do not describe that as a merge into `master`.

- [ ] **Step 10: Final completion report**

Report only freshly observed evidence: final research SHA, exact workflow run evidence, Python total, Lean gates, research/feature identity, `master` unchanged, PR state, and independent compileall result or explicit gap.

The strongest allowed final claim is:

> Narrative Dynamics can reconstruct a canonical temporal evolution of objective, direct-perception, and testimony-aware decision state for a validated testimony scenario, and can compare that trajectory with minimal single-variable counterfactuals to identify the earliest canonical state divergence or validation boundary.

Do not claim that final behavior uniquely reveals the underlying cognitive mechanism.
