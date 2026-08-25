from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from types import MappingProxyType
from collections.abc import Mapping

from narrative_dynamics.narrative.decision import (
    DecisionChoice,
    DecisionModelSpec,
    run_decision_model,
)
from narrative_dynamics.narrative.domain import DomainSpec, validate_narrative
from narrative_dynamics.narrative.ir import (
    Decision,
    GenericNarrative,
    StateCellRef,
    TypedValue,
)
from narrative_dynamics.narrative.replay import (
    EpistemicCellView,
    direct_state,
    epistemic_state,
    objective_state,
)


def _cell_key(cell: StateCellRef) -> tuple[str, str, str]:
    return (
        cell.subject.entity_type,
        cell.subject.entity_id,
        cell.state_variable,
    )


def _cell_value_rows(
    values: Mapping[StateCellRef, TypedValue | None],
) -> list[dict[str, object]]:
    return [
        {
            "cell": cell.to_dict(),
            "value": None if value is None else value.to_dict(),
        }
        for cell, value in sorted(values.items(), key=lambda item: _cell_key(item[0]))
    ]


def _cell_view_rows(
    values: Mapping[StateCellRef, EpistemicCellView],
) -> list[dict[str, object]]:
    return [
        {"cell": cell.to_dict(), "view": view.to_dict()}
        for cell, view in sorted(values.items(), key=lambda item: _cell_key(item[0]))
    ]


@dataclass(frozen=True)
class TriggerRef:
    kind: str
    ref_id: str

    def __post_init__(self) -> None:
        if self.kind not in {"event", "claim", "reception", "decision"}:
            raise ValueError("analysis trigger kind is not supported")
        if not isinstance(self.ref_id, str) or not self.ref_id:
            raise ValueError("analysis trigger ref id must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "ref_id": self.ref_id}


@dataclass(frozen=True)
class AnalysisScope:
    decision_id: str
    tracked_state_cells: tuple[StateCellRef, ...]
    tracked_agents: tuple[str, ...]
    snapshot_times: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.decision_id, str) or not self.decision_id:
            raise ValueError("analysis decision id must be non-empty")
        cells = tuple(self.tracked_state_cells)
        if not cells or any(not isinstance(item, StateCellRef) for item in cells):
            raise TypeError("analysis tracked state cells must be non-empty StateCellRef values")
        if len(set(cells)) != len(cells):
            raise ValueError("analysis tracked state cells must be unique")
        agents = tuple(self.tracked_agents)
        if not agents or any(not isinstance(item, str) or not item for item in agents):
            raise ValueError("analysis tracked agents must be non-empty strings")
        if len(set(agents)) != len(agents):
            raise ValueError("analysis tracked agents must be unique")
        times = tuple(self.snapshot_times)
        if not times or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in times
        ):
            raise ValueError("analysis snapshot times must be non-negative integers")
        if tuple(sorted(set(times))) != times:
            raise ValueError("analysis snapshot times must be unique and increasing")
        object.__setattr__(self, "tracked_state_cells", cells)
        object.__setattr__(self, "tracked_agents", agents)
        object.__setattr__(self, "snapshot_times", times)

    @property
    def tracked_cells(self) -> tuple[StateCellRef, ...]:
        return self.tracked_state_cells

    def to_dict(self) -> dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "tracked_state_cells": [item.to_dict() for item in self.tracked_state_cells],
            "tracked_agents": list(self.tracked_agents),
            "snapshot_times": list(self.snapshot_times),
        }


@dataclass(frozen=True)
class AgentEvolutionView:
    agent_id: str
    direct_cells: Mapping[StateCellRef, EpistemicCellView]
    epistemic_cells: Mapping[StateCellRef, EpistemicCellView]

    def __post_init__(self) -> None:
        if not isinstance(self.agent_id, str) or not self.agent_id:
            raise ValueError("analysis agent id must be non-empty")
        direct = dict(self.direct_cells)
        epistemic = dict(self.epistemic_cells)
        for values in (direct, epistemic):
            if any(
                not isinstance(cell, StateCellRef)
                or not isinstance(view, EpistemicCellView)
                or view.cell != cell
                for cell, view in values.items()
            ):
                raise TypeError("analysis agent cells must map StateCellRef to matching views")
        object.__setattr__(self, "direct_cells", MappingProxyType(direct))
        object.__setattr__(self, "epistemic_cells", MappingProxyType(epistemic))

    def to_dict(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "direct_cells": _cell_view_rows(self.direct_cells),
            "epistemic_cells": _cell_view_rows(self.epistemic_cells),
        }


@dataclass(frozen=True)
class EvolutionSnapshot:
    logical_time: int
    triggers: tuple[TriggerRef, ...]
    objective_cells: Mapping[StateCellRef, TypedValue | None]
    agent_views: Mapping[str, AgentEvolutionView]
    decision_result: DecisionChoice | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.logical_time, int)
            or isinstance(self.logical_time, bool)
            or self.logical_time < 0
        ):
            raise ValueError("analysis snapshot logical time must be non-negative")
        triggers = tuple(self.triggers)
        if any(not isinstance(item, TriggerRef) for item in triggers):
            raise TypeError("analysis snapshot triggers must be TriggerRef values")
        objective = dict(self.objective_cells)
        if any(
            not isinstance(cell, StateCellRef)
            or (value is not None and not isinstance(value, TypedValue))
            for cell, value in objective.items()
        ):
            raise TypeError("analysis objective cells have invalid entries")
        views = dict(self.agent_views)
        if any(
            not isinstance(agent, str)
            or not isinstance(view, AgentEvolutionView)
            or view.agent_id != agent
            for agent, view in views.items()
        ):
            raise TypeError("analysis agent views must map ids to matching views")
        if self.decision_result is not None and not isinstance(
            self.decision_result, DecisionChoice
        ):
            raise TypeError("analysis decision result must be DecisionChoice or None")
        object.__setattr__(self, "triggers", triggers)
        object.__setattr__(self, "objective_cells", MappingProxyType(objective))
        object.__setattr__(self, "agent_views", MappingProxyType(views))

    def to_dict(self) -> dict[str, object]:
        return {
            "logical_time": self.logical_time,
            "triggers": [item.to_dict() for item in self.triggers],
            "objective_cells": _cell_value_rows(self.objective_cells),
            "agent_views": [
                self.agent_views[agent_id].to_dict()
                for agent_id in sorted(self.agent_views)
            ],
            "decision_result": (
                None if self.decision_result is None else self.decision_result.to_dict()
            ),
        }


@dataclass(frozen=True)
class NarrativeTrajectory:
    scope: AnalysisScope
    model_id: str
    model_hash: str
    snapshots: tuple[EvolutionSnapshot, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.scope, AnalysisScope):
            raise TypeError("narrative trajectory requires AnalysisScope")
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("narrative trajectory model id must be non-empty")
        if not isinstance(self.model_hash, str) or not self.model_hash.startswith("sha256:"):
            raise ValueError("narrative trajectory model hash must be a content hash")
        snapshots = tuple(self.snapshots)
        if any(not isinstance(item, EvolutionSnapshot) for item in snapshots):
            raise TypeError("narrative trajectory snapshots must be EvolutionSnapshot values")
        if tuple(item.logical_time for item in snapshots) != self.scope.snapshot_times:
            raise ValueError("narrative trajectory snapshots must match frozen scope times")
        object.__setattr__(self, "snapshots", snapshots)

    @property
    def decision_result(self) -> DecisionChoice:
        results = tuple(
            item.decision_result for item in self.snapshots if item.decision_result is not None
        )
        if len(results) != 1:
            raise ValueError("narrative trajectory must contain exactly one decision result")
        return results[0]

    @property
    def selected_action(self) -> str:
        return self.decision_result.selected_action

    def to_dict(self) -> dict[str, object]:
        return {
            "scope": self.scope.to_dict(),
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "snapshots": [item.to_dict() for item in self.snapshots],
        }


@dataclass(frozen=True)
class MechanismPairwiseComparison:
    left_model_id: str
    right_model_id: str
    action_equal: bool
    basis_equal: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "left_model_id": self.left_model_id,
            "right_model_id": self.right_model_id,
            "action_equal": self.action_equal,
            "basis_equal": self.basis_equal,
        }


@dataclass(frozen=True)
class MechanismComparison:
    decision_id: str
    trajectories: tuple[NarrativeTrajectory, ...]
    pairwise: tuple[MechanismPairwiseComparison, ...]
    mechanism_uniqueness_claimed: bool = False

    def __post_init__(self) -> None:
        if self.mechanism_uniqueness_claimed:
            raise ValueError("generic analysis never claims mechanism uniqueness")
        trajectories = tuple(self.trajectories)
        pairwise = tuple(self.pairwise)
        if any(not isinstance(item, NarrativeTrajectory) for item in trajectories):
            raise TypeError("mechanism trajectories must be NarrativeTrajectory values")
        if any(not isinstance(item, MechanismPairwiseComparison) for item in pairwise):
            raise TypeError("mechanism pairwise rows must be comparison values")
        object.__setattr__(self, "trajectories", trajectories)
        object.__setattr__(self, "pairwise", pairwise)

    def to_dict(self) -> dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "trajectories": [item.to_dict() for item in self.trajectories],
            "pairwise": [item.to_dict() for item in self.pairwise],
            "mechanism_uniqueness_claimed": self.mechanism_uniqueness_claimed,
        }


@dataclass(frozen=True)
class NarrativeAnalysis:
    baseline: NarrativeTrajectory
    mechanism_comparison: MechanismComparison

    @property
    def baseline_trajectory(self) -> NarrativeTrajectory:
        return self.baseline

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline": self.baseline.to_dict(),
            "mechanism_comparison": self.mechanism_comparison.to_dict(),
        }



def _one_decision(story: GenericNarrative, decision_id: str) -> Decision:
    matches = tuple(item for item in story.decisions if item.id == decision_id)
    if len(matches) != 1:
        raise ValueError("analysis decision id must identify exactly one decision")
    return matches[0]


def _unknown_view(cell: StateCellRef) -> EpistemicCellView:
    return EpistemicCellView(
        cell=cell,
        status="unknown",
        resolved_value=None,
        constraints=(),
        evidence_kind=None,
        supporting_id=None,
        source_agent=None,
        evidence_logical_time=None,
        evidence_refs=(),
    )


def _relevant_event_times(
    story: GenericNarrative,
    domain: DomainSpec,
    cells: tuple[StateCellRef, ...],
    decision: Decision,
) -> set[int]:
    times: set[int] = set()
    for event in story.events:
        if event.logical_time > decision.logical_time:
            continue
        before = (
            {}
            if event.logical_time == 0
            else objective_state(story, domain, at_time=event.logical_time - 1)
        )
        after = objective_state(story, domain, at_time=event.logical_time)
        if any(before.get(cell) != after.get(cell) for cell in cells):
            times.add(event.logical_time)
    return times


def derive_analysis_scope(
    story: GenericNarrative,
    domain: DomainSpec,
    decision_id: str,
) -> AnalysisScope:
    validate_narrative(story, domain)
    decision = _one_decision(story, decision_id)
    cells = tuple(decision.context_cells)
    agents = [decision.actor_id]
    times = _relevant_event_times(story, domain, cells, decision)
    received = {
        (reception.claim_id, reception.recipient_id)
        for reception in story.receptions
    }
    for claim in sorted(story.claims, key=lambda item: item.logical_time):
        if claim.logical_time > decision.logical_time:
            continue
        cell = StateCellRef(
            claim.proposition.subject,
            claim.proposition.state_variable,
        )
        if cell not in cells or (claim.id, decision.actor_id) not in received:
            continue
        times.add(claim.logical_time)
        if claim.speaker_id not in agents:
            agents.append(claim.speaker_id)
    times.add(decision.logical_time)
    return AnalysisScope(
        decision_id=decision.id,
        tracked_state_cells=cells,
        tracked_agents=tuple(agents),
        snapshot_times=tuple(sorted(times)),
    )


def _triggers_at(
    story: GenericNarrative,
    scope: AnalysisScope,
    logical_time: int,
) -> tuple[TriggerRef, ...]:
    relevant_claim_ids = {
        claim.id
        for claim in story.claims
        if claim.logical_time == logical_time
        and StateCellRef(
            claim.proposition.subject,
            claim.proposition.state_variable,
        ) in scope.tracked_state_cells
    }
    rows: list[TriggerRef] = []
    rows.extend(
        TriggerRef("event", item.id)
        for item in story.events
        if item.logical_time == logical_time
    )
    rows.extend(
        TriggerRef("claim", item.id)
        for item in story.claims
        if item.id in relevant_claim_ids
    )
    rows.extend(
        TriggerRef("reception", item.id)
        for item in story.receptions
        if item.claim_id in relevant_claim_ids
    )
    rows.extend(
        TriggerRef("decision", item.id)
        for item in story.decisions
        if item.id == scope.decision_id and item.logical_time == logical_time
    )
    return tuple(rows)


def build_trajectory(
    story: GenericNarrative,
    domain: DomainSpec,
    scope: AnalysisScope,
    model: DecisionModelSpec,
) -> NarrativeTrajectory:
    validate_narrative(story, domain)
    if not isinstance(scope, AnalysisScope):
        raise TypeError("trajectory construction requires AnalysisScope")
    if not isinstance(model, DecisionModelSpec):
        raise TypeError("trajectory construction requires DecisionModelSpec")
    decision = _one_decision(story, scope.decision_id)
    if decision.logical_time not in scope.snapshot_times:
        raise ValueError("frozen analysis scope must include decision time")

    snapshots: list[EvolutionSnapshot] = []
    for logical_time in scope.snapshot_times:
        objective = objective_state(story, domain, at_time=logical_time)
        objective_cells = {
            cell: objective.get(cell) for cell in scope.tracked_state_cells
        }
        agent_views: dict[str, AgentEvolutionView] = {}
        for agent_id in scope.tracked_agents:
            direct = direct_state(story, domain, agent_id, at_time=logical_time)
            epistemic = epistemic_state(story, domain, agent_id, at_time=logical_time)
            direct_cells = {
                cell: direct.cells.get(cell, _unknown_view(cell))
                for cell in scope.tracked_state_cells
            }
            epistemic_cells = {
                cell: epistemic.cells.get(cell, _unknown_view(cell))
                for cell in scope.tracked_state_cells
            }
            agent_views[agent_id] = AgentEvolutionView(
                agent_id,
                direct_cells,
                epistemic_cells,
            )
        result = (
            run_decision_model(story, domain, decision.id, model)
            if logical_time == decision.logical_time
            else None
        )
        snapshots.append(
            EvolutionSnapshot(
                logical_time=logical_time,
                triggers=_triggers_at(story, scope, logical_time),
                objective_cells=objective_cells,
                agent_views=agent_views,
                decision_result=result,
            )
        )
    return NarrativeTrajectory(
        scope=scope,
        model_id=model.model_id,
        model_hash=model.content_hash,
        snapshots=tuple(snapshots),
    )


def _basis(choice: DecisionChoice) -> tuple[object, ...]:
    return (
        choice.evidence_refs,
        tuple(
            (
                cell.subject.entity_type,
                cell.subject.entity_id,
                cell.state_variable,
            )
            for cell in choice.inspected_cells
        ),
    )


def _compare_trajectories(
    trajectories: tuple[NarrativeTrajectory, ...],
) -> tuple[MechanismPairwiseComparison, ...]:
    rows: list[MechanismPairwiseComparison] = []
    for left, right in combinations(trajectories, 2):
        left_result = left.decision_result
        right_result = right.decision_result
        rows.append(
            MechanismPairwiseComparison(
                left_model_id=left.model_id,
                right_model_id=right.model_id,
                action_equal=left_result.selected_action == right_result.selected_action,
                basis_equal=_basis(left_result) == _basis(right_result),
            )
        )
    return tuple(rows)


def analyze_narrative(
    story: GenericNarrative,
    domain: DomainSpec,
    decision_id: str,
    model: DecisionModelSpec,
    *,
    comparison_models: tuple[DecisionModelSpec, ...] = (),
) -> NarrativeAnalysis:
    scope = derive_analysis_scope(story, domain, decision_id)
    models = (model,) + tuple(comparison_models)
    trajectories = tuple(
        build_trajectory(story, domain, scope, item) for item in models
    )
    comparison = MechanismComparison(
        decision_id=scope.decision_id,
        trajectories=trajectories,
        pairwise=_compare_trajectories(trajectories),
        mechanism_uniqueness_claimed=False,
    )
    return NarrativeAnalysis(
        baseline=trajectories[0],
        mechanism_comparison=comparison,
    )
