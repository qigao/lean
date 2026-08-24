# Narrative Evolution Matrix V3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic temporal evolution analysis for validated Narrative Testimony V2 scenarios, including baseline world/direct/testimony state snapshots, minimal single-variable counterfactuals, first-divergence evidence, validation-boundary rejections, and explicit non-uniqueness of mechanism identification from final action alone.

**Architecture:** Add an independent `narrative_dynamics.story.evolution_v2` analysis layer above the existing V1/V2 replay semantics. First extract one pure testimony action resolver into the story layer so both the existing adapter and the new analysis share the same decision rule; then build immutable baseline trajectories, evaluate single-variable counterfactuals through ordinary V2 validation, compare valid trajectories on canonical state only, and expose exactly six V3 public names from `narrative_dynamics.story` without changing runtime, registry, metrics, Lean semantics, V1 identities, or V2 runtime payload identity.

**Tech Stack:** Python 3 standard library (`dataclasses`, `types.MappingProxyType`, `unittest`), existing Narrative Dynamics story schema/replay/scenario modules, existing GitHub Actions `proof` workflow, Lean 4 regression gates unchanged.

**Spec:** `docs/superpowers/specs/2026-08-24-narrative-evolution-matrix-v3-design.md`

## Global Constraints

- Implementation branch is `work/narrative-evolution-matrix-v3`, forked from research SHA `29a15a49e3191e3d6fde995064642630a4f8ea2a` on `proof/narrative-dynamics-v0`.
- `master` must not move.
- Strict RED -> GREEN applies to every production task: commit and observe the focused test-only RED on the exact feature head before writing production code.
- After each GREEN implementation, obtain a fresh exact-head GitHub Actions `proof` success before starting the next production task.
- V3 accepts only validated `NarrativeScenarioV2`; no raw fixture, oracle, source text, metadata, or gold label may enter the analysis API.
- V3 analyzes exactly `story.decision.object`.
- Tracked agents are exactly the decision actor followed by unique report speakers for reports about the decision target, preserving report order.
- V3 snapshot times are every relocation event time, every report time, and decision time, sorted ascending and deduplicated.
- V2 reception has no independent time; a declared reception becomes available at the report's `logical_time`.
- Missing direct/testimony support is represented as `None`; never fall back to objective truth.
- Counterfactuals are single-variable only; no compound or power-set search.
- Supported interventions are exactly `remove_reception`, `change_report_content`, `remove_direct_observation`, and `remove_support_observation`.
- Valid counterfactual divergence compares canonical objective/direct/testimony/provenance/action fields only; trigger descriptions, fixture metadata, source text, oracle data, and labels are excluded.
- Rejected counterfactuals have no alternate trajectory and no `first_divergence`; they record `rejection_stage` as exactly `scenario_validation` or `action_resolution`.
- `mechanism_uniqueness_claimed` is always `False`.
- V1 fixture hashes and V1 runtime IDs remain exact.
- V2 runtime payload remains exactly six keys: `entities`, `events`, `observations`, `reports`, `receptions`, `decision`.
- Do not modify `SimulationRunner`, runtime contracts, model registry, `story_choice_metrics`, prison adapters, V1 schema/replay/scenario semantics, package-root exports, or Lean theorem semantics.
- `TestimonySearchModel` may change only to reuse the shared story-layer action resolver; its name, parameter contract, events, outcome, scores, policy, determinism, and runtime behavior remain exact.
- Do not claim general Theory of Mind, unique hidden-mechanism identification, deception intent, empirical human validity, population validity, or causal effect estimation from observational data.
- Standalone `python3 -m compileall -q narrative_dynamics` evidence may be reported only if that command is actually run independently.

## File Structure

The implementation should create or modify only these production/test files in addition to this spec and plan:

- Modify `narrative_dynamics/story/replay_v2.py` — add one non-exported story-layer `resolve_testimony_action` helper.
- Modify `narrative_dynamics/adapters/story_testimony_search.py` — reuse the shared resolver without changing observable adapter behavior.
- Create `narrative_dynamics/story/evolution_v2.py` — own all V3 immutable analysis records, baseline trajectory construction, intervention generation/evaluation, divergence comparison, serialization, and primary analysis entry point.
- Modify `narrative_dynamics/story/__init__.py` — export exactly the approved six V3 public names.
- Create `tests/test_story_testimony_action_resolution.py` — lock resolver behavior and adapter contract preservation.
- Create `tests/test_story_evolution_v2.py` — lock baseline timeline, valid/rejected counterfactuals, divergence, serialization, ordering, and mechanism-safety behavior.
- Modify `tests/test_story_testimony_runtime.py` — extend the exact story package export/root-isolation contract with the approved V3 surface.

No other production file is planned. If implementation appears to require another production file, stop and review the design before adding it.

---

### Task 1: Shared Testimony Action Resolution Boundary

**Files:**
- Create: `tests/test_story_testimony_action_resolution.py`
- Modify: `narrative_dynamics/story/replay_v2.py`
- Modify: `narrative_dynamics/adapters/story_testimony_search.py`

**Interfaces:**
- Consumes: `NarrativeScenarioV2`, `testimony_state`, `latest_epistemic_location`, `SearchDecisionV1` actions.
- Produces: `resolve_testimony_action(story: NarrativeScenarioV2) -> str` in `narrative_dynamics.story.replay_v2`. This helper is intentionally not exported from `narrative_dynamics.story.__all__`.

- [ ] **Step 1: Write the failing resolver/adapter-regression tests**

Create `tests/test_story_testimony_action_resolution.py` with focused tests equivalent to:

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

- [ ] **Step 2: Run the focused test locally if a checkout is available**

Run:

```bash
python3 -m unittest tests.test_story_testimony_action_resolution -v
```

Expected RED: import/discovery error because `resolve_testimony_action` does not yet exist. Existing testimony model behavior must not be changed before observing this RED.

- [ ] **Step 3: Commit the test-only RED**

```bash
git add tests/test_story_testimony_action_resolution.py
git commit -m "test: add testimony action resolver RED"
```

Record the exact RED SHA.

- [ ] **Step 4: Open the draft V3 PR and observe exact-head CI RED**

Open a draft PR from `work/narrative-evolution-matrix-v3` to `proof/narrative-dynamics-v0` with title:

```text
RED: add Narrative Evolution Matrix V3
```

The body must state that the current head intentionally contains only Task 1 RED and that `master` must not move.

Wait for the `proof` workflow associated with the exact RED head. Expected result:

- dependency/conformance/Lean build/theorem gates before Python remain green;
- Python fails because `resolve_testimony_action` is missing;
- no unrelated existing test failure is accepted as RED evidence.

- [ ] **Step 5: Implement the minimal shared resolver**

Append to `narrative_dynamics/story/replay_v2.py`:

```python
def resolve_testimony_action(story: NarrativeScenarioV2) -> str:
    """Resolve the unique decision action supported by the actor's latest evidence."""

    if not isinstance(story, NarrativeScenarioV2):
        raise TypeError("testimony action resolution requires a validated NarrativeScenarioV2")
    decision = story.decision
    location = latest_epistemic_location(
        testimony_state(story, decision.actor, at_time=decision.time),
        decision.object,
    )
    matching = tuple(
        action.id
        for action in decision.actions
        if action.location == location.location
    )
    if len(matching) != 1:
        raise ValueError(
            "testimony-aware target location must match exactly one decision action"
        )
    return matching[0]
```

Do not add this helper to `narrative_dynamics.story.__all__`.

- [ ] **Step 6: Refactor the adapter to reuse only the matching rule**

In `narrative_dynamics/adapters/story_testimony_search.py`, import `resolve_testimony_action` from `replay_v2`. Keep the existing `testimony_state` / `latest_epistemic_location` call because the adapter still needs the exact provenance basis. Replace only the local action-match block with:

```python
selected = resolve_testimony_action(story)
```

Do not change model name, parameter validation, scores, policy, event order, event kinds, basis keys, outcome keys, or RNG behavior.

- [ ] **Step 7: Run focused and existing testimony tests**

Run:

```bash
python3 -m unittest \
  tests.test_story_testimony_action_resolution \
  tests.test_story_testimony_models \
  tests.test_story_testimony_runtime -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit Task 1 GREEN**

```bash
git add \
  narrative_dynamics/story/replay_v2.py \
  narrative_dynamics/adapters/story_testimony_search.py
git commit -m "refactor: share testimony action resolution"
```

- [ ] **Step 9: Require fresh exact-head full CI GREEN**

On the exact Task 1 GREEN SHA, require the full `proof` workflow to succeed, including full Python discovery and both StoryState/Testimony Lean gates. Record run number, run ID, exact feature `head_sha`, and Python test count from logs before starting Task 2.

---

### Task 2: Baseline Temporal Evolution Matrix

**Files:**
- Create: `tests/test_story_evolution_v2.py`
- Create: `narrative_dynamics/story/evolution_v2.py`

**Interfaces:**
- Consumes: `NarrativeScenarioV2`, `objective_state`, `subjective_state`, `testimony_state`, `resolve_testimony_action`.
- Produces public module symbols (not package exports yet): `EvolutionSnapshotV2`, `EvolutionTrajectoryV2`, `EvolutionInterventionV2`, `EvolutionCounterfactualV2`, `EvolutionAnalysisV2`, `analyze_testimony_evolution`.
- Produces internal records: `_EvolutionAgentStateV2`, `_EvolutionDivergenceV2`.
- At the end of this task, `analyze_testimony_evolution` returns a correct baseline and an empty `counterfactuals` tuple; Tasks 3 and 4 fill the approved counterfactual surface.

- [ ] **Step 1: Add baseline-only RED tests to `tests/test_story_evolution_v2.py`**

Start the file with:

```python
from __future__ import annotations

import unittest

from narrative_dynamics.story.evolution_v2 import analyze_testimony_evolution
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
        self.assertEqual(
            (baseline.snapshots[2].agents["bob"].evidence_kind,
             baseline.snapshots[2].agents["bob"].supporting_id,
             baseline.snapshots[2].agents["bob"].source_agent),
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
        self.assertEqual(tuple(item.testimony_location for item in bob),
                         ("drawer", "drawer", "drawer", "drawer"))
        self.assertEqual(bob[1].evidence_kind, "direct_perception")
        self.assertEqual(bob[1].supporting_id, "e1")
        self.assertEqual(bob[2].evidence_kind, "testimony")
        self.assertEqual(bob[2].supporting_id, "r1")
        self.assertEqual(bob[2].source_agent, "alice")
        self.assertEqual(baseline.selected_action, "search_drawer")

    def test_analysis_requires_validated_v2_scenario(self):
        with self.assertRaisesRegex(TypeError, "validated NarrativeScenarioV2"):
            analyze_testimony_evolution(object())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run focused test and observe local RED if possible**

```bash
python3 -m unittest tests.test_story_evolution_v2 -v
```

Expected RED: missing `narrative_dynamics.story.evolution_v2`.

- [ ] **Step 3: Commit Task 2 test-only RED and observe exact-head CI failure**

```bash
git add tests/test_story_evolution_v2.py
git commit -m "test: add evolution baseline RED"
```

Require the PR `proof` workflow on this exact RED SHA to fail only because the new evolution module is absent; existing suites/gates must remain clean before the Python failure.

- [ ] **Step 4: Implement immutable baseline records and serialization**

Create `narrative_dynamics/story/evolution_v2.py`. Use frozen dataclasses and a mapping proxy for snapshot agent maps. The core record shapes must be exactly:

```python
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from narrative_dynamics.story.replay import objective_state, subjective_state
from narrative_dynamics.story.replay_v2 import resolve_testimony_action, testimony_state
from narrative_dynamics.story.scenario_v2 import NarrativeScenarioV2


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

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline": self.baseline.to_dict(),
            "counterfactuals": [item.to_dict() for item in self.counterfactuals],
            "mechanism_uniqueness_claimed": self.mechanism_uniqueness_claimed,
        }
```

Do not make `_EvolutionAgentStateV2` or `_EvolutionDivergenceV2` package exports.

- [ ] **Step 5: Implement deterministic baseline helpers**

Implement these private helpers in the same file:

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
```

Implement trigger resolution so an event ID, report ID, or decision ID at the snapshot time becomes `trigger_ids`. `trigger_kind` is `relocation`, `report`, `decision`, or `mixed` when more than one trigger class occurs.

Implement `_build_trajectory(story)` with these invariants:

```python
selected = resolve_testimony_action(story)
target = story.decision.object
tracked = _tracked_agents(story)

snapshots = []
for logical_time in _snapshot_times(story):
    objective = objective_state(story.events, at_time=logical_time).get(target)
    agents = MappingProxyType({
        agent: _agent_state(story, agent, target, logical_time)
        for agent in tracked
    })
    snapshots.append(
        EvolutionSnapshotV2(
            logical_time=logical_time,
            trigger_kind=...,
            trigger_ids=...,
            objective_location=None if objective is None else objective.location,
            agents=agents,
            selected_action=(selected if logical_time == story.decision.time else None),
        )
    )

return EvolutionTrajectoryV2(
    target_object=target,
    tracked_agents=tracked,
    snapshots=tuple(snapshots),
    selected_action=selected,
)
```

- [ ] **Step 6: Implement the baseline-only primary entry point**

```python
def analyze_testimony_evolution(story: NarrativeScenarioV2) -> EvolutionAnalysisV2:
    if not isinstance(story, NarrativeScenarioV2):
        raise TypeError("evolution analysis requires a validated NarrativeScenarioV2")
    return EvolutionAnalysisV2(
        baseline=_build_trajectory(story),
        counterfactuals=(),
    )
```

Do not export this from `narrative_dynamics.story.__init__` yet; Task 5 owns the public-package boundary change.

- [ ] **Step 7: Run focused baseline tests**

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

Require full `proof` success on the Task 2 GREEN SHA. Record exact `head_sha`, workflow run number/ID, full Python count, Lean build success, StoryState success, and Testimony success.

---

### Task 3: Valid Minimal Counterfactuals and First Divergence

**Files:**
- Modify: `tests/test_story_evolution_v2.py`
- Modify: `narrative_dynamics/story/evolution_v2.py`

**Interfaces:**
- Consumes: Task 2 `_build_trajectory`, immutable V3 records, typed V2 scenario fields.
- Produces: automatic valid `remove_reception` and `change_report_content` counterfactuals, canonical first-divergence evidence, deterministic counterfactual ordering.

- [ ] **Step 1: Add test-only RED for reception/content interventions and divergence**

Extend `tests/test_story_evolution_v2.py` with a second class:

```python
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
        truthful = self.by_kind(
            analyze_testimony_evolution(self.truthful), "remove_reception"
        )
        self.assertEqual(len(truthful), 1)
        item = truthful[0]
        self.assertEqual(item.status, "valid")
        self.assertIsNotNone(item.trajectory)
        self.assertEqual(item.trajectory.selected_action, "search_drawer")
        self.assertEqual(item.first_divergence.logical_time, 3)
        self.assertTrue(item.first_divergence.action_changed)
        self.assertIn(
            "agents.bob.evidence_kind", item.first_divergence.changed_fields
        )
        self.assertIn(
            "agents.bob.supporting_id", item.first_divergence.changed_fields
        )
        self.assertEqual(
            item.trajectory.snapshots[2].agents["bob"].evidence_kind,
            "direct_perception",
        )
        self.assertEqual(
            item.trajectory.snapshots[2].agents["bob"].supporting_id,
            "e1",
        )

        stale = self.by_kind(
            analyze_testimony_evolution(self.stale), "remove_reception"
        )[0]
        self.assertEqual(stale.status, "valid")
        self.assertEqual(stale.trajectory.selected_action, "search_drawer")
        self.assertEqual(stale.first_divergence.logical_time, 3)
        self.assertFalse(stale.first_divergence.action_changed)
        self.assertIn("agents.bob.evidence_kind", stale.first_divergence.changed_fields)

    def test_change_report_content_diverges_exactly_at_report_time_and_flips_action(self):
        truthful = self.by_kind(
            analyze_testimony_evolution(self.truthful), "change_report_content"
        )
        self.assertEqual(len(truthful), 1)
        item = truthful[0]
        self.assertEqual(item.intervention.subject_id, "r1")
        self.assertEqual(item.intervention.from_value, "box")
        self.assertEqual(item.intervention.to_value, "drawer")
        self.assertEqual(item.status, "valid")
        self.assertEqual(item.first_divergence.logical_time, 3)
        self.assertTrue(item.first_divergence.action_changed)
        self.assertEqual(item.trajectory.selected_action, "search_drawer")
        self.assertIn(
            "agents.bob.testimony_location", item.first_divergence.changed_fields
        )

        stale = self.by_kind(
            analyze_testimony_evolution(self.stale), "change_report_content"
        )[0]
        self.assertEqual(stale.intervention.from_value, "drawer")
        self.assertEqual(stale.intervention.to_value, "box")
        self.assertEqual(stale.first_divergence.logical_time, 3)
        self.assertEqual(stale.trajectory.selected_action, "search_box")
        self.assertTrue(stale.first_divergence.action_changed)

    def test_counterfactual_order_and_changed_paths_are_deterministic(self):
        analysis = analyze_testimony_evolution(self.truthful)
        keys = tuple(
            (
                item.intervention.kind,
                -1 if item.intervention.logical_time is None else item.intervention.logical_time,
                item.intervention.subject_id,
                "" if item.intervention.to_value is None else item.intervention.to_value,
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

At this Task 3 RED stage, assert only `remove_reception` and `change_report_content` kinds when filtering; Task 4 will add the two observation-removal intervention kinds.

- [ ] **Step 2: Run focused test and observe RED**

```bash
python3 -m unittest tests.test_story_evolution_v2 -v
```

Expected RED: baseline tests pass, but counterfactual tuples are empty so the new counterfactual assertions fail.

- [ ] **Step 3: Commit Task 3 test-only RED and observe exact-head CI RED**

```bash
git add tests/test_story_evolution_v2.py
git commit -m "test: add valid evolution counterfactual RED"
```

Require the exact RED head's `proof` workflow to fail only in the newly added counterfactual tests.

- [ ] **Step 4: Implement canonical trajectory comparison**

Add a canonical snapshot flattener and first-divergence comparator. The compared paths are exactly:

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
```

Then compare aligned snapshot times in ascending order. If the time tuples differ, raise an implementation-level `RuntimeError` because the approved interventions do not alter event/report/decision times. At the first differing time:

```python
changed = tuple(sorted(
    key for key in baseline_fields
    if baseline_fields[key] != counterfactual_fields[key]
))
return _EvolutionDivergenceV2(
    logical_time=logical_time,
    changed_fields=changed,
    action_changed=(baseline.selected_action != counterfactual.selected_action),
)
```

Return `None` only if all compared canonical fields are identical.

- [ ] **Step 5: Implement valid candidate evaluation**

Add a helper that receives an already constructed valid candidate scenario and its `EvolutionInterventionV2`, builds the trajectory, compares it with the baseline, and returns:

```python
EvolutionCounterfactualV2(
    intervention=intervention,
    status="valid",
    trajectory=trajectory,
    first_divergence=_first_divergence(baseline, trajectory),
    rejection_stage=None,
    rejection_reason=None,
    rejection_logical_time=None,
)
```

Do not label a valid trajectory as causal or uniquely identified.

- [ ] **Step 6: Generate `remove_reception` candidates**

For each target report received by the decision actor, remove only that `(report.id, decision.actor)` reception. Use:

```python
EvolutionInterventionV2(
    kind="remove_reception",
    subject_id=report.id,
    agent=story.decision.actor,
    from_value="received",
    to_value=None,
    logical_time=report.logical_time,
)
```

Construct a new `NarrativeScenarioV2` explicitly with the baseline `entities/events/observations/reports/decision` and the filtered `receptions`. Do not mutate the baseline object.

- [ ] **Step 7: Generate `change_report_content` candidates**

For each report about the target, iterate decision action locations in decision order, excluding the current report location. Replace only that report's `location`; retain speaker, support event, report ID/time/kind, receptions, observations, events, and decision. The intervention record is:

```python
EvolutionInterventionV2(
    kind="change_report_content",
    subject_id=report.id,
    agent=report.speaker,
    from_value=report.location,
    to_value=alternate_location,
    logical_time=report.logical_time,
)
```

Do not compare the alternate report content with objective truth.

- [ ] **Step 8: Sort valid counterfactuals deterministically and wire the primary entry point**

Sort all current counterfactual records by:

```python
(
    item.intervention.kind,
    -1 if item.intervention.logical_time is None else item.intervention.logical_time,
    item.intervention.subject_id,
    "" if item.intervention.to_value is None else item.intervention.to_value,
    "" if item.intervention.agent is None else item.intervention.agent,
)
```

Update `analyze_testimony_evolution` so it computes the baseline once and returns the sorted valid counterfactual tuple.

- [ ] **Step 9: Run focused counterfactual tests**

```bash
python3 -m unittest tests.test_story_evolution_v2 -v
```

Expected: baseline and Task 3 valid-counterfactual tests pass.

- [ ] **Step 10: Commit Task 3 GREEN**

```bash
git add narrative_dynamics/story/evolution_v2.py
git commit -m "feat: compare valid evolution counterfactuals"
```

- [ ] **Step 11: Require fresh exact-head full CI GREEN**

Require full `proof` success on the exact Task 3 GREEN SHA before Task 4.

---

### Task 4: Validation-Boundary Counterfactuals and Mechanism-Safety Witness

**Files:**
- Modify: `tests/test_story_evolution_v2.py`
- Modify: `narrative_dynamics/story/evolution_v2.py`

**Interfaces:**
- Consumes: Task 3 candidate evaluation/divergence pipeline.
- Produces: `remove_direct_observation`, `remove_support_observation`, rejected counterfactual records, action-resolution rejection support, explicit same-action/different-provenance mechanism-safety evidence.

- [ ] **Step 1: Add RED tests for validation-boundary interventions**

Append tests equivalent to:

```python
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

    def test_remove_decision_actor_only_observation_is_rejected_by_v2_validation(self):
        item = self.one(
            analyze_testimony_evolution(self.truthful), "remove_direct_observation"
        )
        self.assertEqual(item.intervention.subject_id, "e1")
        self.assertEqual(item.intervention.agent, "bob")
        self.assertEqual(item.intervention.logical_time, 1)
        self.assertEqual(item.status, "rejected")
        self.assertIsNone(item.trajectory)
        self.assertIsNone(item.first_divergence)
        self.assertEqual(item.rejection_stage, "scenario_validation")
        self.assertEqual(item.rejection_logical_time, 1)
        self.assertIn(
            "decision actor must have observed a relocation of the target object",
            item.rejection_reason,
        )

    def test_remove_speaker_support_observation_is_rejected_by_provenance_validation(self):
        item = self.one(
            analyze_testimony_evolution(self.truthful), "remove_support_observation"
        )
        self.assertEqual(item.intervention.subject_id, "r1")
        self.assertEqual(item.intervention.agent, "alice")
        self.assertEqual(item.intervention.from_value, "e2")
        self.assertEqual(item.intervention.logical_time, 2)
        self.assertEqual(item.status, "rejected")
        self.assertIsNone(item.trajectory)
        self.assertIsNone(item.first_divergence)
        self.assertEqual(item.rejection_stage, "scenario_validation")
        self.assertEqual(item.rejection_logical_time, 2)
        self.assertIn(
            "report speaker must have directly observed the support_event",
            item.rejection_reason,
        )

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
                "remove_reception",
                "change_report_content",
                "remove_direct_observation",
                "remove_support_observation",
            },
        )
```

- [ ] **Step 2: Run focused tests and observe RED**

```bash
python3 -m unittest tests.test_story_evolution_v2 -v
```

Expected RED: missing observation-removal counterfactual kinds. Existing baseline and Task 3 tests remain green.

- [ ] **Step 3: Commit Task 4 test-only RED and observe exact-head CI RED**

```bash
git add tests/test_story_evolution_v2.py
git commit -m "test: add evolution validation-boundary RED"
```

Require the exact RED head `proof` run to fail only the new Task 4 assertions.

- [ ] **Step 4: Generalize candidate evaluation to preserve expected rejections as data**

Use a helper that takes the intervention and a zero-argument candidate constructor. Catch only domain-boundary `ValueError` from scenario construction and action resolution; allow `TypeError`, `RuntimeError`, and other programming errors to propagate.

The required shape is:

```python
def _evaluate_candidate(
    baseline: EvolutionTrajectoryV2,
    intervention: EvolutionInterventionV2,
    build_story,
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

Use this same path for Task 3 valid interventions too, so validation is never bypassed.

- [ ] **Step 5: Generate decision-actor direct-observation removals**

Build an event lookup. For every observation where:

```python
observation.agent == story.decision.actor
and event_by_id[observation.event].object == story.decision.object
```

generate exactly one intervention:

```python
EvolutionInterventionV2(
    kind="remove_direct_observation",
    subject_id=observation.event,
    agent=observation.agent,
    from_value="observed",
    to_value=None,
    logical_time=event_by_id[observation.event].logical_time,
)
```

The candidate removes only that exact `(event, agent)` observation and reconstructs `NarrativeScenarioV2` normally. In the committed fixtures, removing Bob/e1 must be rejected by the existing decision-actor observation requirement.

- [ ] **Step 6: Generate speaker support-observation removals**

For every target report, generate one intervention keyed by the report ID so multiple reports sharing support can still be distinguished in analysis:

```python
EvolutionInterventionV2(
    kind="remove_support_observation",
    subject_id=report.id,
    agent=report.speaker,
    from_value=report.support_event,
    to_value=None,
    logical_time=event_by_id[report.support_event].logical_time,
)
```

Remove only the exact `(report.support_event, report.speaker)` observation and reconstruct the scenario through ordinary validation. For the committed fixtures this must become `status="rejected"`, `rejection_stage="scenario_validation"` with the existing speaker-support error.

- [ ] **Step 7: Keep full counterfactual ordering stable**

Collect all four intervention families and sort using the Task 3 stable key. Do not special-case rejected records in ordering.

- [ ] **Step 8: Run the complete evolution test file**

```bash
python3 -m unittest tests.test_story_evolution_v2 -v
```

Expected: all baseline, valid-counterfactual, validation-boundary, ordering, and mechanism-safety tests pass.

- [ ] **Step 9: Commit Task 4 GREEN**

```bash
git add narrative_dynamics/story/evolution_v2.py
git commit -m "feat: expose evolution validation boundaries"
```

- [ ] **Step 10: Require fresh exact-head full CI GREEN**

Require full `proof` success on exact Task 4 GREEN SHA before Task 5.

---

### Task 5: Public V3 Story API, Serialization Lock, and Full Regression Surface

**Files:**
- Modify: `tests/test_story_testimony_runtime.py`
- Modify: `tests/test_story_evolution_v2.py`
- Modify: `narrative_dynamics/story/__init__.py`

**Interfaces:**
- Consumes: completed `evolution_v2` public module symbols.
- Produces exact `narrative_dynamics.story` export expansion with six V3 names and no package-root leakage.

- [ ] **Step 1: Add exact V3 export RED to the runtime test**

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

Rename the exact export test to reflect V1+V2+V3 and require:

```python
expected = _V1_EXPORTS | _V2_EXPORTS | _V3_EXPORTS
self.assertEqual(set(getattr(story, "__all__", ())), expected)
for name in expected:
    self.assertTrue(hasattr(story, name), name)
```

Extend root isolation:

```python
for name in _V2_EXPORTS | _V3_EXPORTS:
    self.assertFalse(hasattr(narrative_dynamics, name), name)
```

Also lock the internal helper/records out of the public story surface:

```python
for name in (
    "resolve_testimony_action",
    "_EvolutionAgentStateV2",
    "_EvolutionDivergenceV2",
):
    self.assertNotIn(name, getattr(story, "__all__", ()))
```

- [ ] **Step 2: Add deterministic JSON-structural serialization assertions**

In `tests/test_story_evolution_v2.py`, add:

```python
import json


def test_analysis_to_dict_is_deterministic_json_serializable(self):
    analysis = analyze_testimony_evolution(self.truthful)
    first = analysis.to_dict()
    second = analyze_testimony_evolution(self.truthful).to_dict()
    self.assertEqual(first, second)
    encoded = json.dumps(first, sort_keys=True, separators=(",", ":"))
    self.assertEqual(json.loads(encoded), first)
    self.assertFalse(first["mechanism_uniqueness_claimed"])
```

Place this in the baseline test class or another focused serialization class with the same `setUp` fixture.

- [ ] **Step 3: Run focused tests and observe export RED**

```bash
python3 -m unittest \
  tests.test_story_evolution_v2 \
  tests.test_story_testimony_runtime -v
```

Expected RED: serialization already passes from earlier implementation, but the exact package export assertion fails because the six V3 names are not yet in `story.__all__`.

- [ ] **Step 4: Commit Task 5 test-only RED and observe exact-head CI RED**

```bash
git add tests/test_story_evolution_v2.py tests/test_story_testimony_runtime.py
git commit -m "test: add Narrative Evolution V3 public API RED"
```

Require the exact RED head `proof` run to fail only the new export-surface assertion.

- [ ] **Step 5: Export exactly the six approved V3 names**

In `narrative_dynamics/story/__init__.py`, import:

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

- [ ] **Step 6: Run focused V3 and story runtime/model regression tests**

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

- [ ] **Step 7: Run full Python discovery before committing GREEN when local execution is available**

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

On the exact Task 5 GREEN SHA, require the complete `proof` workflow to succeed. Record:

- exact feature `head_sha`;
- workflow run number and run ID;
- full Lean build success;
- full Python unittest count and `OK`;
- StoryState theorem gate success;
- Testimony theorem gate success.

Do not describe this remote CI as independent standalone `compileall` evidence.

---

### Task 6: Final Evidence, Review, and Research-Branch Integration

**Files:**
- No production change expected.
- PR title/body metadata may be updated.
- Any defect found during review requires a new focused test-only RED before a production fix.

**Interfaces:**
- Consumes: exact final V3 feature head and the pre-V3 research SHA `29a15a49e3191e3d6fde995064642630a4f8ea2a`.
- Produces: reviewed/verified V3 research head and a non-forced fast-forward of `proof/narrative-dynamics-v0` only.

- [ ] **Step 1: Re-read the spec and build an explicit requirement checklist**

Check every spec section against the implementation. At minimum confirm:

- baseline snapshots at relocation/report/decision times;
- decision actor + relevant speaker tracking only;
- objective/direct/testimony/provenance states are distinct;
- no objective fallback;
- all four single-variable intervention kinds only;
- report-content changes do not consult objective truth;
- valid first divergence is canonical-state-only;
- rejected scenarios have no trajectory/divergence;
- scenario/action rejection stages are distinct;
- stale same-action/different-provenance witness exists;
- mechanism uniqueness remains false;
- deterministic serialization and ordering;
- exact six-name V3 public story surface;
- no package-root leakage;
- shared action resolver does not alter adapter output;
- no runtime/registry/metrics/prison/Lean/V1 semantic changes.

If any requirement is missing, do not integrate. Add a focused RED test before fixing it.

- [ ] **Step 2: Review the complete diff against the exact pre-V3 research SHA**

Compare:

```text
29a15a49e3191e3d6fde995064642630a4f8ea2a
...
<exact final work/narrative-evolution-matrix-v3 head>
```

Expected changed production/test/document files are only:

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

Any additional production file is a review blocker until explicitly justified against the approved spec.

- [ ] **Step 3: Inspect scientific/claim boundaries in code, tests, and PR text**

Search changed text for unsupported claims/labels such as:

```text
unique_cause
true_mechanism
identified_mechanism
human Theory of Mind
lying intent
deception intent
empirical human validity
population representative
causal effect
```

The fixture provenance may legitimately contain existing negative flags such as `population_representative=false`; the review concern is positive unsupported claims.

- [ ] **Step 4: Obtain fresh exact-head final verification**

Require the latest `proof` workflow associated with the exact final feature SHA to be `completed/success`. Inspect job steps/logs rather than relying only on PR badges. Confirm full Python discovery and both story theorem gates.

If the final head changed after the last GREEN workflow for any reason, trigger/await a fresh run for the new exact head before claiming completion.

- [ ] **Step 5: Independently attempt compileall without overclaiming**

If a usable local checkout exists, run:

```bash
python3 -m compileall -q narrative_dynamics
```

If it exits 0, record it as independent evidence. If the environment cannot obtain/run the checkout, record the gap explicitly; do not infer compileall success from GitHub Actions.

- [ ] **Step 6: Update the draft PR to final V3 evidence**

Change title to:

```text
feat: add Narrative Evolution Matrix V3
```

Replace the RED-only body with a final summary that includes:

- temporal baseline matrix purpose;
- four minimal intervention families;
- valid first-divergence vs validation-rejection distinction;
- stale same-action/different-provenance non-identification witness;
- exact public API boundary;
- exact final feature SHA;
- final `proof` run number/ID and Python/Lean evidence;
- standalone compileall evidence only if independently obtained;
- strongest allowed claim: canonical temporal evolution plus minimal single-variable counterfactual comparison, not unique hidden-mechanism identification or empirical human causal validity;
- target remains `proof/narrative-dynamics-v0`; `master` must not move.

- [ ] **Step 7: Lock branch topology immediately before integration**

Freshly verify:

1. `proof/narrative-dynamics-v0` is still exactly `29a15a49e3191e3d6fde995064642630a4f8ea2a`.
2. `work/narrative-evolution-matrix-v3` is the exact reviewed/verified final head.
3. Feature vs research is `ahead > 0`, `behind = 0`, merge base exactly `29a15a49...`.
4. Capture the current exact `master` SHA as `MASTER_BEFORE`.

If research moved or feature is behind/diverged, stop. Do not force-update or silently rebase during final integration.

- [ ] **Step 8: Fast-forward only the research branch non-forced**

Move `proof/narrative-dynamics-v0` to the exact final feature SHA with force disabled.

The equivalent Git operation is:

```bash
git push origin \
  <exact-final-feature-sha>:refs/heads/proof/narrative-dynamics-v0
```

with fast-forward semantics only; when using the GitHub ref API, set `force=false`.

Do not update `master`.

- [ ] **Step 9: Verify post-integration identity and master preservation**

Freshly verify:

- `proof/narrative-dynamics-v0` and `work/narrative-evolution-matrix-v3` are `identical` at the exact final SHA;
- `master` is still exactly `MASTER_BEFORE`;
- no feature branch deletion occurred.

GitHub may automatically mark the PR closed/merged when base and head refs become identical. Record the actual PR state; do not misdescribe that automatic state change as a merge into `master`.

- [ ] **Step 10: Final completion report**

Report only evidence actually observed in this task. Include final research SHA, exact final workflow/run evidence, Python count, Lean gates, branch identity, `master` unchanged, PR final state, and the standalone compileall status/gap.

The final research claim must remain no stronger than:

> Narrative Dynamics can reconstruct a canonical temporal evolution of objective, direct-perception, and testimony-aware decision state for a validated testimony scenario, and can compare that trajectory with minimal single-variable counterfactuals to identify the earliest canonical state divergence or validation boundary.

Do not claim that final behavior uniquely reveals the cognitive mechanism.
