"""Transport-neutral World Studio application service."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, TypeAlias

from narrative_dynamics.abm.scenario_coordinator import ScenarioCoordinator
from narrative_dynamics.abm.scenario_coordinator_contracts import (
    ScenarioCommandCapability,
    ScenarioCommandKind,
    ScenarioCommandRequest,
    ScenarioCommandResult,
    ScenarioForkRequest,
    ScenarioForkResult,
    ScenarioRunView,
)
from narrative_dynamics.abm.scenario_package_contracts import CompiledSituatedScenario
from narrative_dynamics.abm.simulation_output_contracts import (
    SimulationAgentDecisionPayload,
    SimulationAudienceCapability,
    SimulationBlenderDeltaPayload,
    SimulationCommandResultPayload,
    SimulationDiagnosticPayload,
    SimulationMemoryUpdatePayload,
    SimulationNarrativeScenePayload,
    SimulationNetworkMetricsPayload,
    SimulationObjectiveEventPayload,
    SimulationOutputAudience,
    SimulationOutputRecord,
    SimulationOutputView,
    SimulationPrivatePerceptPayload,
    SimulationSocialUpdatePayload,
    SimulationStateDeltaPayload,
    SimulationStoryProgressPayload,
)
from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.studio.capabilities import StudioCapability
from narrative_dynamics.studio.contracts import (
    DraftOperationKind,
    JsonValue,
    ScenarioDiagnosticReport,
    ScenarioDraftOperation,
    ScenarioDraftOperationResult,
    ScenarioDraftSnapshot,
    ScenarioProjectExport,
    thaw_json,
)
from narrative_dynamics.studio.project_store import (
    ScenarioProjectCapacityError,
    ScenarioProjectConflictError,
    ScenarioProjectNotFoundError,
    ScenarioProjectStorageError,
    ScenarioProjectValidationError,
)
from narrative_dynamics.studio.run_registry import (
    ScenarioRunRegistry,
    ScenarioRunReservationConflictError,
)
from narrative_dynamics.studio.workspace import ScenarioProjectWorkspace


JsonObject: TypeAlias = dict[str, JsonValue]


class StudioError(RuntimeError):
    """Base for stable service failures safe for protocol normalization."""

    rpc_code: int
    error_code: str

    def __init__(self, message: str, *, data: JsonObject | None = None) -> None:
        self.data = {} if data is None else data
        super().__init__(message)


class StudioAuthorizationError(StudioError):
    rpc_code = -32010
    error_code = "unauthorized"

    def __init__(self) -> None:
        super().__init__("studio authority is insufficient")


class StudioStaleStateError(StudioError):
    rpc_code = -32011
    error_code = "stale_state"

    def __init__(self) -> None:
        super().__init__("studio state is stale")


class StudioValidationError(StudioError):
    rpc_code = -32012
    error_code = "validation_failed"

    def __init__(self, report: ScenarioDiagnosticReport | None = None) -> None:
        data: JsonObject = {}
        if report is not None:
            data = {
                "diagnostic_report_hash": report.content_hash,
                "diagnostic_codes": tuple(item.code for item in report.diagnostics),
                "diagnostic_pointers": tuple(item.pointer for item in report.diagnostics),
            }
        super().__init__("studio validation failed", data=data)


class StudioRunLifecycleError(StudioError):
    rpc_code = -32013
    error_code = "invalid_run_lifecycle"

    def __init__(self) -> None:
        super().__init__("studio run lifecycle is invalid")


class StudioNotFoundError(StudioError):
    rpc_code = -32014
    error_code = "not_found"

    def __init__(self) -> None:
        super().__init__("studio resource is unavailable")


class StudioConflictError(StudioError):
    rpc_code = -32015
    error_code = "identity_conflict"

    def __init__(self) -> None:
        super().__init__("studio identity conflicts")


class StudioHistoryGapError(StudioError):
    rpc_code = -32016
    error_code = "history_gap"

    def __init__(self) -> None:
        super().__init__("studio history is unavailable")


class StudioCapacityError(StudioError):
    rpc_code = -32017
    error_code = "capacity_limit"

    def __init__(self) -> None:
        super().__init__("studio capacity is exhausted")


class StudioMethodNotFoundError(LookupError):
    pass


class StudioInvalidParamsError(ValueError):
    pass


class ScenarioCoordinatorFactory(Protocol):
    def create(
        self,
        scenario: CompiledSituatedScenario,
        *,
        project_id: str,
        run_id: str,
        stream_id: str,
    ) -> ScenarioCoordinator: ...

    def fork(
        self,
        coordinator: ScenarioCoordinator,
        request: ScenarioForkRequest,
        capability: ScenarioCommandCapability,
    ) -> tuple[ScenarioCoordinator, ScenarioForkResult]: ...

    def abort_create(
        self,
        *,
        project_id: str,
        run_id: str,
        stream_id: str,
    ) -> None: ...

    def abort_fork(
        self,
        coordinator: ScenarioCoordinator,
        request: ScenarioForkRequest,
    ) -> None: ...


def _params(
    value: object,
    *,
    required: tuple[str, ...] = (),
    optional: tuple[str, ...] = (),
) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise StudioInvalidParamsError("studio params must be a JSON object")
    keys = set(value)
    if not set(required).issubset(keys) or not keys.issubset(set(required) | set(optional)):
        raise StudioInvalidParamsError("studio params do not match the method schema")
    return value


def _text(value: object, *, label: str, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.encode("utf-8")) > maximum:
        raise StudioInvalidParamsError(f"{label} must be bounded non-empty text")
    return value


def _integer(value: object, *, label: str, positive: bool = True) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or (positive and value <= 0):
        raise StudioInvalidParamsError(f"{label} must be a positive integer")
    return value


def _optional_text(value: object, *, label: str) -> str | None:
    return None if value is None else _text(value, label=label)


def _document(document: object) -> JsonObject:
    return {
        "role": document.role,
        "logical_id": document.logical_id,
        "value": thaw_json(document.value),
        "content_hash": document.content_hash,
    }


def _diagnostic_report(report: ScenarioDiagnosticReport) -> JsonObject:
    return {
        "schema": "narrative-dynamics.scenario-diagnostic-report/v1",
        "project_id": report.project_id,
        "revision": report.revision,
        "diagnostics": [
            {
                "severity": item.severity.value,
                "code": item.code,
                "document_role": item.document_role,
                "logical_id": item.logical_id,
                "pointer": item.pointer,
                "related_ids": item.related_ids,
                "message_key": item.message_key,
            }
            for item in report.diagnostics
        ],
        "content_hash": report.content_hash,
    }


def _snapshot(snapshot: ScenarioDraftSnapshot) -> JsonObject:
    return {
        "schema": "narrative-dynamics.scenario-draft-snapshot/v1",
        "project_id": snapshot.project_id,
        "revision": snapshot.revision,
        "scenario_id": snapshot.scenario_id,
        "version": snapshot.version,
        "documents": tuple(_document(item) for item in snapshot.documents),
        "document_semantic_hash": snapshot.document_semantic_hash,
        "layout": thaw_json(snapshot.layout),
        "layout_hash": snapshot.layout_hash,
        "diagnostic_report_hash": snapshot.diagnostic_report_hash,
        "compiled_scenario_hash": snapshot.compiled_scenario_hash,
        "content_hash": snapshot.content_hash,
    }


def _operation_result(result: ScenarioDraftOperationResult) -> JsonObject:
    return {
        "operation_hash": result.operation_hash,
        "prior_revision": result.prior_revision,
        "next_snapshot": _snapshot(result.next_snapshot),
        "diagnostic_report": _diagnostic_report(result.diagnostic_report),
    }


def _project_export(result: ScenarioProjectExport) -> JsonObject:
    return {
        "project_id": result.project_id,
        "revision": result.revision,
        "snapshot_hash": result.snapshot_hash,
        "target_name": result.target_name,
        "package_hash": result.package_hash,
        "compiled_scenario_hash": result.compiled_scenario_hash,
        "content_hash": result.content_hash,
    }


def _run_view(view: ScenarioRunView) -> JsonObject:
    return {
        "schema": view.schema,
        "run_id": view.run_id,
        "stream_id": view.stream_id,
        "scenario_hash": view.scenario_hash,
        "coordinator_epoch": view.coordinator_epoch,
        "status": view.status.value,
        "round_index": view.round_index,
        "state_hash": view.state_hash,
        "next_sequence": view.next_sequence,
        "output_batch_hashes": view.output_batch_hashes,
        "checkpoint_hashes": view.checkpoint_hashes,
        "parent_checkpoint_hash": view.parent_checkpoint_hash,
        "content_hash": view.content_hash,
    }


def _command_result(result: ScenarioCommandResult) -> JsonObject:
    return {
        "schema": result.schema,
        "command_id": result.command_id,
        "idempotency_key": result.idempotency_key,
        "request_hash": result.request_hash,
        "capability_hash": result.capability_hash,
        "run_id": result.run_id,
        "scenario_hash": result.scenario_hash,
        "coordinator_epoch": result.coordinator_epoch,
        "kind": result.kind.value,
        "accepted": result.accepted,
        "reason": result.reason.value,
        "prior_status": result.prior_status.value,
        "next_status": result.next_status.value,
        "prior_state_hash": result.prior_state_hash,
        "next_state_hash": result.next_state_hash,
        "round_index": result.round_index,
        "output_batch_hash": result.output_batch_hash,
        "checkpoint_hash": result.checkpoint_hash,
        "content_hash": result.content_hash,
    }


def _fork_result(result: ScenarioForkResult) -> JsonObject:
    return {
        "schema": result.schema,
        "fork_id": result.fork_id,
        "idempotency_key": result.idempotency_key,
        "request_hash": result.request_hash,
        "capability_hash": result.capability_hash,
        "source_run_id": result.source_run_id,
        "child_run_id": result.child_run_id,
        "child_stream_id": result.child_stream_id,
        "scenario_hash": result.scenario_hash,
        "child_epoch": result.child_epoch,
        "checkpoint_hash": result.checkpoint_hash,
        "child_state_hash": result.child_state_hash,
        "content_hash": result.content_hash,
    }


def _metrics(metrics: object) -> JsonObject:
    return {
        "snapshot_hash": metrics.snapshot_hash,
        "round_index": metrics.round_index,
        "population_size": metrics.population_size,
        "occupied_place_count": metrics.occupied_place_count,
        "adopted_count": metrics.adopted_count,
        "adoption_rate": metrics.adoption_rate,
        "tracked_belief_mean": metrics.tracked_belief_mean,
        "tracked_belief_variance": metrics.tracked_belief_variance,
        "active_relationship_edge_count": metrics.active_relationship_edge_count,
        "mean_relationship_trust": metrics.mean_relationship_trust,
        "direct_interaction_pair_count": metrics.direct_interaction_pair_count,
        "latest_tell_event_count": metrics.latest_tell_event_count,
        "transmission_count": metrics.transmission_count,
        "reached_observer_count": metrics.reached_observer_count,
        "exact_transmission_count": metrics.exact_transmission_count,
        "detected_transmission_count": metrics.detected_transmission_count,
        "identified_transmission_count": metrics.identified_transmission_count,
        "active_claim_count": metrics.active_claim_count,
        "confirmed_claim_count": metrics.confirmed_claim_count,
        "contradicted_claim_count": metrics.contradicted_claim_count,
        "superseded_claim_count": metrics.superseded_claim_count,
        "forgotten_claim_count": metrics.forgotten_claim_count,
    }


def _public_state(view: object) -> JsonObject:
    return {
        "schema": view.schema,
        "run_id": view.run_id,
        "scenario_hash": view.scenario_hash,
        "round_index": view.round_index,
        "state_hash": view.state_hash,
        "agent_places": view.agent_places,
        "passage_states": view.passage_states,
        "object_placements": view.object_placements,
        "metrics": _metrics(view.metrics),
        "content_hash": view.content_hash,
    }


def _belief(belief: object) -> JsonObject:
    return {"probabilities": {key: belief.probabilities[key] for key in sorted(belief.probabilities)}}


def _mind(mind: object) -> JsonObject:
    return {
        "agent_id": mind.agent_id,
        "belief": _belief(mind.belief),
        "own_place_id": mind.own_place_id,
        "processed_observation_ids": mind.processed_observation_ids,
        "observed_event_ids": mind.observed_event_ids,
        "selected_action_ids": mind.selected_action_ids,
        "decision_count": mind.decision_count,
        "recalled_memory_ids": mind.recalled_memory_ids,
        "observation_floor_round": mind.observation_floor_round,
    }


def _relationship(item: object) -> JsonObject:
    return {
        "observer_agent_id": item.observer_agent_id,
        "source_agent_id": item.source_agent_id,
        "trust": item.trust,
        "affinity": item.affinity,
        "confirmation_count": item.confirmation_count,
        "contradiction_count": item.contradiction_count,
    }


def _claim(item: object) -> JsonObject:
    return {
        "claim_id": item.claim_id,
        "observer_agent_id": item.observer_agent_id,
        "source_agent_id": item.source_agent_id,
        "topic_id": item.topic_id,
        "symbol_id": item.symbol_id,
        "event_ids": item.event_ids,
        "memory_ids": item.memory_ids,
        "first_round": item.first_round,
        "last_round": item.last_round,
        "support_count": item.support_count,
        "status": item.status.value,
    }


def _agent_state(view: object) -> JsonObject:
    return {
        "schema": view.schema,
        "run_id": view.run_id,
        "scenario_hash": view.scenario_hash,
        "round_index": view.round_index,
        "state_hash": view.state_hash,
        "agent_id": view.agent_id,
        "place_id": view.place_id,
        "mind": _mind(view.mind),
        "relationships": tuple(_relationship(item) for item in view.relationships),
        "claims": tuple(_claim(item) for item in view.claims),
        "content_hash": view.content_hash,
    }


def _network_state(snapshot: object) -> JsonObject:
    return {
        "model_id": snapshot.model_id,
        "model_hash": snapshot.model_hash,
        "round_index": snapshot.round_index,
        "story_hash": snapshot.story_hash,
        "cognitive_state_hash": snapshot.cognitive_state_hash,
        "social_state_hash": snapshot.social_state_hash,
        "nodes": tuple(
            {
                "agent_id": item.agent_id,
                "role_id": item.role_id,
                "place_id": item.place_id,
                "tracked_belief_probability": item.tracked_belief_probability,
                "active_claim_count": item.active_claim_count,
            }
            for item in snapshot.nodes
        ),
        "relationship_edges": tuple(
            {
                "source_agent_id": item.source_agent_id,
                "observer_agent_id": item.observer_agent_id,
                "trust": item.trust,
                "affinity": item.affinity,
                "confirmation_count": item.confirmation_count,
                "contradiction_count": item.contradiction_count,
                "active": item.active,
            }
            for item in snapshot.relationship_edges
        ),
        "access_edges": tuple(
            {
                "source_agent_id": item.source_agent_id,
                "observer_agent_id": item.observer_agent_id,
                "visual_cost": item.visual_cost,
                "auditory_loss": item.auditory_loss,
                "direct_interaction": item.direct_interaction,
            }
            for item in snapshot.access_edges
        ),
        "transmissions": tuple(
            {
                "event_id": item.event_id,
                "source_agent_id": item.source_agent_id,
                "observer_agent_id": item.observer_agent_id,
                "fidelity": item.fidelity.value,
                "channels": tuple(channel.value for channel in item.channels),
            }
            for item in snapshot.transmissions
        ),
        "latest_tell_event_count": snapshot.latest_tell_event_count,
        "content_hash": snapshot.content_hash,
    }


def _percept(percept: object) -> JsonObject:
    return {
        "percept_id": percept.percept_id,
        "round_index": percept.round_index,
        "agent_id": percept.agent_id,
        "source_event_id": percept.source_event_id,
        "source_event_hash": percept.source_event_hash,
        "channels": tuple(item.value for item in percept.channels),
        "fidelity": percept.fidelity.value,
        "actor_agent_id": percept.actor_agent_id,
        "kind": None if percept.kind is None else percept.kind.value,
        "place_id": percept.place_id,
        "outcome": percept.outcome,
        "details": tuple({"name": item.name, "value": item.value} for item in percept.details),
    }


def _intent(intent: object) -> JsonObject:
    return {
        "action_id": intent.action_id,
        "agent_id": intent.agent_id,
        "kind": intent.kind.value,
        "target_id": intent.target_id,
        "message": intent.message,
        "source_event_ids": intent.source_event_ids,
    }


def _decision(decision: object) -> JsonObject:
    return {
        "agent_id": decision.agent_id,
        "round_index": decision.round_index,
        "prior_belief": _belief(decision.prior_belief),
        "posterior_belief": _belief(decision.posterior_belief),
        "admitted_observation_ids": decision.admitted_observation_ids,
        "admitted_symbol_ids": decision.admitted_symbol_ids,
        "feasible_action_ids": decision.feasible_action_ids,
        "action_values": dict(decision.action_values),
        "action_policy": dict(decision.action_policy),
        "selected_action_id": decision.selected_action_id,
        "selected_goal_contributions": dict(decision.selected_goal_contributions),
        "intent": _intent(decision.intent),
        "recalled_memory_ids": decision.recalled_memory_ids,
        "recalled_symbol_ids": decision.recalled_symbol_ids,
    }


def _payload(payload: object) -> JsonObject:
    if type(payload) is SimulationStateDeltaPayload:
        return {
            "prior_snapshot_hash": payload.prior_snapshot_hash,
            "next_snapshot_hash": payload.next_snapshot_hash,
            "changed_agent_ids": payload.changed_agent_ids,
            "changed_passage_ids": payload.changed_passage_ids,
            "changed_object_ids": payload.changed_object_ids,
        }
    if type(payload) is SimulationObjectiveEventPayload:
        return {
            "event_id": payload.event_id,
            "event_hash": payload.event_hash,
            "action_id": payload.action_id,
            "action_kind": payload.action_kind,
            "actor_agent_id": payload.actor_agent_id,
            "place_id": payload.place_id,
            "target_id": payload.target_id,
            "success": payload.success,
            "cause_event_ids": payload.cause_event_ids,
        }
    if type(payload) is SimulationPrivatePerceptPayload:
        return {"percept": _percept(payload.percept)}
    if type(payload) is SimulationAgentDecisionPayload:
        return {"decision": _decision(payload.decision)}
    if type(payload) is SimulationMemoryUpdatePayload:
        return {
            "agent_id": payload.agent_id,
            "prior_mind_hash": payload.prior_mind_hash,
            "next_mind_hash": payload.next_mind_hash,
            "recalled_memory_ids": payload.recalled_memory_ids,
            "admitted_memory_ids": payload.admitted_memory_ids,
        }
    if type(payload) is SimulationSocialUpdatePayload:
        return {
            "observer_agent_id": payload.observer_agent_id,
            "prior_claim_hashes": payload.prior_claim_hashes,
            "next_claim_hashes": payload.next_claim_hashes,
            "prior_relationship_hashes": payload.prior_relationship_hashes,
            "next_relationship_hashes": payload.next_relationship_hashes,
            "admitted_evidence_ids": payload.admitted_evidence_ids,
        }
    if type(payload) is SimulationNetworkMetricsPayload:
        return {"metrics": _metrics(payload.metrics)}
    if type(payload) is SimulationStoryProgressPayload:
        return {
            "active_scene_id": payload.active_scene_id,
            "completed_scene_ids": payload.completed_scene_ids,
            "status": payload.status,
        }
    if type(payload) is SimulationNarrativeScenePayload:
        return {
            "scene_id": payload.scene_id,
            "projection_hash": payload.projection_hash,
            "realization_hash": payload.realization_hash,
        }
    if type(payload) is SimulationBlenderDeltaPayload:
        return {
            "agent_places": payload.agent_places,
            "passage_states": payload.passage_states,
            "object_placements": payload.object_placements,
        }
    if type(payload) is SimulationCommandResultPayload:
        return {
            "command_id": payload.command_id,
            "accepted": payload.accepted,
            "reason_code": payload.reason_code,
        }
    if type(payload) is SimulationDiagnosticPayload:
        return {"code": payload.code, "message": payload.message}
    raise TypeError("unsupported trusted simulation output payload")


def _output_record(record: SimulationOutputRecord) -> JsonObject:
    return {
        "schema": record.schema,
        "stream_id": record.stream_id,
        "scenario_hash": record.scenario_hash,
        "sequence": record.sequence,
        "round_index": record.round_index,
        "state_hash": record.state_hash,
        "kind": record.kind.value,
        "audience": record.audience.value,
        "owner_agent_id": record.owner_agent_id,
        "source_artifact_hashes": record.source_artifact_hashes,
        "payload": _payload(record.payload),
        "payload_hash": record.payload.content_hash,
        "content_hash": record.content_hash,
    }


def _output_view(view: SimulationOutputView) -> JsonObject:
    return {
        "schema": view.schema,
        "stream_id": view.stream_id,
        "scenario_hash": view.scenario_hash,
        "prior_state_hash": view.prior_state_hash,
        "next_state_hash": view.next_state_hash,
        "round_result_hash": view.round_result_hash,
        "first_sequence": view.first_sequence,
        "last_sequence": view.last_sequence,
        "records": tuple(_output_record(item) for item in view.records),
        "source_batch_hash": view.source_batch_hash,
        "checkpoint": view.checkpoint,
        "content_hash": view.content_hash,
    }


class WorldStudioService:
    """Dispatch the closed V22 application surface without transport concerns."""

    def __init__(
        self,
        workspace: ScenarioProjectWorkspace,
        run_registry: ScenarioRunRegistry,
        coordinator_factory: ScenarioCoordinatorFactory,
        *,
        import_sources: Mapping[str, str | Path] | None = None,
    ) -> None:
        if not isinstance(workspace, ScenarioProjectWorkspace):
            raise TypeError("World Studio service requires ScenarioProjectWorkspace")
        self._workspace = workspace
        self._runs = run_registry
        self._factory = coordinator_factory
        self._import_sources = dict(import_sources or {})
        self._methods = {
            "project.create": self._project_create,
            "project.import": self._project_import,
            "project.snapshot": self._project_snapshot,
            "project.apply": self._project_apply,
            "project.undo": self._project_undo,
            "project.redo": self._project_redo,
            "project.export": self._project_export,
            "scenario.validate": self._scenario_validate,
            "scenario.compile": self._scenario_compile,
            "run.create": self._run_create,
            "run.command": self._run_command,
            "run.fork": self._run_fork,
            "run.view": self._run_view,
            "run.list": self._run_list,
            "state.public": self._state_public,
            "state.agent": self._state_agent,
            "state.network": self._state_network,
            "output.get": self._output_get,
            "command.get": self._command_get,
        }

    @property
    def methods(self) -> tuple[str, ...]:
        return tuple(sorted(self._methods))

    def is_state_changing(self, method: str) -> bool:
        return method in {
            "project.create",
            "project.import",
            "project.apply",
            "project.undo",
            "project.redo",
            "project.export",
            "run.create",
            "run.command",
            "run.fork",
        }

    def invoke(
        self,
        method: str,
        params: JsonObject,
        capability: StudioCapability,
    ) -> JsonValue:
        if not isinstance(method, str):
            raise StudioMethodNotFoundError("studio method is unavailable")
        if not isinstance(capability, StudioCapability):
            raise TypeError("World Studio invocation requires StudioCapability")
        try:
            handler = self._methods[method]
        except KeyError:
            raise StudioMethodNotFoundError("studio method is unavailable") from None
        try:
            return handler(params, capability)
        except StudioError:
            raise
        except ScenarioProjectConflictError as error:
            if "stale" in str(error):
                raise StudioStaleStateError() from None
            raise StudioConflictError() from None
        except ScenarioProjectNotFoundError:
            raise StudioNotFoundError() from None
        except ScenarioProjectValidationError as error:
            raise StudioValidationError(error.diagnostic_report) from None
        except ScenarioProjectCapacityError:
            raise StudioCapacityError() from None
        except ScenarioProjectStorageError:
            raise RuntimeError("studio storage failure") from None
        except PermissionError:
            raise StudioAuthorizationError() from None
        except KeyError:
            raise StudioNotFoundError() from None
        except StudioInvalidParamsError:
            raise

    @staticmethod
    def _require_project(
        capability: StudioCapability, project_id: str, permission: str
    ) -> None:
        if permission not in capability.permissions or project_id not in capability.project_ids:
            raise StudioAuthorizationError()

    @staticmethod
    def _require_run(capability: StudioCapability, run_id: str, permission: str) -> None:
        if permission not in capability.permissions or run_id not in capability.run_ids:
            raise StudioAuthorizationError()

    @staticmethod
    def _require_agent(capability: StudioCapability, agent_id: str) -> None:
        if agent_id not in capability.agent_ids:
            raise StudioAuthorizationError()

    def _project_create(self, raw: object, capability: StudioCapability) -> JsonObject:
        values = _params(raw, required=("project_id",))
        project_id = _text(values["project_id"], label="project ID")
        self._require_project(capability, project_id, "project.create")
        return _snapshot(self._workspace.create_project(project_id))

    def _project_import(self, raw: object, capability: StudioCapability) -> JsonObject:
        values = _params(
            raw,
            required=("project_id", "source_id", "expected_revision", "expected_snapshot_hash"),
        )
        project_id = _text(values["project_id"], label="project ID")
        self._require_project(capability, project_id, "project.write")
        source_id = _text(values["source_id"], label="source ID")
        try:
            source = self._import_sources[source_id]
        except KeyError:
            raise StudioNotFoundError() from None
        result = self._workspace.import_package(
            project_id,
            source,
            expected_revision=_integer(values["expected_revision"], label="expected revision"),
            expected_snapshot_hash=_text(values["expected_snapshot_hash"], label="expected snapshot hash"),
        )
        return _snapshot(result)

    def _project_snapshot(self, raw: object, capability: StudioCapability) -> JsonObject:
        values = _params(raw, required=("project_id",))
        project_id = _text(values["project_id"], label="project ID")
        self._require_project(capability, project_id, "project.read")
        return _snapshot(self._workspace.snapshot(project_id))

    def _project_apply(self, raw: object, capability: StudioCapability) -> JsonObject:
        required = (
            "operation_id", "idempotency_key", "project_id", "expected_revision",
            "expected_snapshot_hash", "document_role", "logical_id", "kind",
        )
        values = _params(raw, required=required, optional=("pointer", "value", "from_pointer"))
        project_id = _text(values["project_id"], label="project ID")
        self._require_project(capability, project_id, "project.write")
        try:
            operation = ScenarioDraftOperation(
                _text(values["operation_id"], label="operation ID"),
                _text(values["idempotency_key"], label="idempotency key"),
                project_id,
                _integer(values["expected_revision"], label="expected revision"),
                _text(values["expected_snapshot_hash"], label="expected snapshot hash"),
                _text(values["document_role"], label="document role"),
                _optional_text(values["logical_id"], label="logical ID"),
                DraftOperationKind(_text(values["kind"], label="operation kind")),
                pointer=values.get("pointer", ""),
                value=values.get("value"),
                from_pointer=values.get("from_pointer"),
            )
        except (TypeError, ValueError):
            raise StudioInvalidParamsError("project operation is invalid") from None
        return _operation_result(self._workspace.apply(operation))

    def _history(self, raw: object, capability: StudioCapability, *, redo: bool) -> JsonObject:
        values = _params(raw, required=("project_id", "expected_revision", "expected_snapshot_hash"))
        project_id = _text(values["project_id"], label="project ID")
        self._require_project(capability, project_id, "project.write")
        function = self._workspace.redo if redo else self._workspace.undo
        return _snapshot(
            function(
                project_id,
                _integer(values["expected_revision"], label="expected revision"),
                _text(values["expected_snapshot_hash"], label="expected snapshot hash"),
            )
        )

    def _project_undo(self, raw: object, capability: StudioCapability) -> JsonObject:
        return self._history(raw, capability, redo=False)

    def _project_redo(self, raw: object, capability: StudioCapability) -> JsonObject:
        return self._history(raw, capability, redo=True)

    def _project_export(self, raw: object, capability: StudioCapability) -> JsonObject:
        values = _params(raw, required=("project_id", "target_name"))
        project_id = _text(values["project_id"], label="project ID")
        self._require_project(capability, project_id, "project.write")
        return _project_export(
            self._workspace.export(project_id, _text(values["target_name"], label="target name"))
        )

    def _scenario_validate(self, raw: object, capability: StudioCapability) -> JsonObject:
        values = _params(raw, required=("project_id",))
        project_id = _text(values["project_id"], label="project ID")
        self._require_project(capability, project_id, "project.read")
        return _diagnostic_report(self._workspace.validate(project_id))

    def _scenario_compile(self, raw: object, capability: StudioCapability) -> JsonObject:
        values = _params(
            raw,
            required=("project_id", "expected_revision", "expected_snapshot_hash"),
        )
        project_id = _text(values["project_id"], label="project ID")
        self._require_project(capability, project_id, "scenario.compile")
        snapshot, scenario = self._workspace.compile_exact(
            project_id,
            _integer(values["expected_revision"], label="expected revision"),
            _text(values["expected_snapshot_hash"], label="expected snapshot hash"),
        )
        return {
            "project_id": project_id,
            "revision": snapshot.revision,
            "snapshot_hash": snapshot.content_hash,
            "scenario_id": scenario.scenario_id,
            "version": scenario.version,
            "package_hash": scenario.package_hash,
            "scenario_hash": scenario.content_hash,
        }

    def _run_create(self, raw: object, capability: StudioCapability) -> JsonObject:
        values = _params(
            raw,
            required=(
                "project_id", "run_id", "stream_id", "expected_revision", "expected_snapshot_hash"
            ),
        )
        project_id = _text(values["project_id"], label="project ID")
        run_id = _text(values["run_id"], label="run ID")
        stream_id = _text(values["stream_id"], label="stream ID")
        self._require_project(capability, project_id, "run.create")
        self._require_run(capability, run_id, "run.create")
        expected_revision = _integer(values["expected_revision"], label="expected revision")
        expected_hash = _text(values["expected_snapshot_hash"], label="expected snapshot hash")
        request_hash = stable_content_hash(
            {
                "operation": "run.create",
                "capability_hash": capability.content_hash,
                "project_id": project_id,
                "run_id": run_id,
                "stream_id": stream_id,
                "expected_revision": expected_revision,
                "expected_snapshot_hash": expected_hash,
            }
        )
        try:
            reservation = self._runs.reserve(run_id, request_hash)
        except ScenarioRunReservationConflictError:
            raise StudioConflictError() from None
        if not reservation.acquired:
            if reservation.coordinator is None:
                raise RuntimeError("scenario run reservation lost its coordinator")
            return _run_view(reservation.coordinator.run_view())

        factory_started = False
        try:
            _, scenario = self._workspace.compile_exact(
                project_id,
                expected_revision,
                expected_hash,
            )
            factory_started = True
            coordinator = self._factory.create(
                scenario,
                project_id=project_id,
                run_id=run_id,
                stream_id=stream_id,
            )
            self._runs.commit(reservation, coordinator)
        except Exception:
            try:
                if factory_started:
                    self._factory.abort_create(
                        project_id=project_id,
                        run_id=run_id,
                        stream_id=stream_id,
                    )
            finally:
                self._runs.abort(reservation)
            raise
        return _run_view(coordinator.run_view())

    def _coordinator(self, capability: StudioCapability, run_id: str, permission: str) -> ScenarioCoordinator:
        self._require_run(capability, run_id, permission)
        return self._runs.resolve(run_id)

    @staticmethod
    def _command_capability(
        capability: StudioCapability,
        run_id: str,
        *,
        command: bool = False,
        fork: bool = False,
        audit: bool = False,
    ) -> ScenarioCommandCapability:
        kinds = tuple(sorted(ScenarioCommandKind, key=lambda item: item.value)) if command else ()
        return ScenarioCommandCapability(
            capability.authority_id,
            run_id,
            kinds,
            can_fork=fork,
            can_read_all_audit=audit,
        )

    def _run_command(self, raw: object, capability: StudioCapability) -> JsonObject:
        values = _params(
            raw,
            required=(
                "command_id", "idempotency_key", "run_id", "scenario_hash",
                "coordinator_epoch", "expected_state_hash", "kind",
            ),
            optional=("requested_checkpoint_id",),
        )
        run_id = _text(values["run_id"], label="run ID")
        coordinator = self._coordinator(capability, run_id, "run.command")
        try:
            request = ScenarioCommandRequest(
                _text(values["command_id"], label="command ID"),
                _text(values["idempotency_key"], label="idempotency key"),
                run_id,
                _text(values["scenario_hash"], label="scenario hash"),
                _integer(values["coordinator_epoch"], label="coordinator epoch"),
                _text(values["expected_state_hash"], label="expected state hash"),
                capability.authority_id,
                ScenarioCommandKind(_text(values["kind"], label="command kind")),
                _optional_text(values.get("requested_checkpoint_id"), label="checkpoint ID"),
            )
        except (TypeError, ValueError):
            raise StudioInvalidParamsError("scenario command is invalid") from None
        try:
            result = coordinator.submit_command(
                request,
                self._command_capability(capability, run_id, command=True),
            )
        except ValueError as error:
            detail = str(error)
            if "was reused" in detail:
                raise StudioConflictError() from None
            if "history limit" in detail or "maximum output records" in detail:
                raise StudioCapacityError() from None
            if "checkpoint id" in detail:
                raise StudioRunLifecycleError() from None
            raise
        except RuntimeError as error:
            if str(error) == "scenario checkpoint store is not configured":
                raise StudioRunLifecycleError() from None
            raise
        return _command_result(result)

    def _run_fork(self, raw: object, capability: StudioCapability) -> JsonObject:
        values = _params(
            raw,
            required=(
                "fork_id", "idempotency_key", "source_run_id", "scenario_hash",
                "source_epoch", "checkpoint_hash", "child_run_id", "child_stream_id",
            ),
        )
        source_run_id = _text(values["source_run_id"], label="source run ID")
        child_run_id = _text(values["child_run_id"], label="child run ID")
        self._require_run(capability, source_run_id, "run.fork")
        self._require_run(capability, child_run_id, "run.fork")
        coordinator = self._runs.resolve(source_run_id)
        try:
            request = ScenarioForkRequest(
                _text(values["fork_id"], label="fork ID"),
                _text(values["idempotency_key"], label="idempotency key"),
                source_run_id,
                _text(values["scenario_hash"], label="scenario hash"),
                _integer(values["source_epoch"], label="source epoch"),
                _text(values["checkpoint_hash"], label="checkpoint hash"),
                child_run_id,
                _text(values["child_stream_id"], label="child stream ID"),
            )
        except (TypeError, ValueError):
            raise StudioInvalidParamsError("scenario fork is invalid") from None
        command_capability = self._command_capability(
            capability,
            source_run_id,
            fork=True,
        )
        reservation_hash = stable_content_hash(
            {
                "operation": "run.fork",
                "request_hash": request.content_hash,
                "capability_hash": command_capability.content_hash,
            }
        )
        try:
            reservation = self._runs.reserve(child_run_id, reservation_hash)
        except ScenarioRunReservationConflictError:
            raise StudioConflictError() from None
        if not reservation.acquired:
            if not isinstance(reservation.outcome, ScenarioForkResult):
                raise RuntimeError("scenario fork reservation lost its result")
            return _fork_result(reservation.outcome)

        factory_started = False
        try:
            factory_started = True
            try:
                child, result = self._factory.fork(
                    coordinator,
                    request,
                    command_capability,
                )
            except ValueError as error:
                detail = str(error)
                if "was reused" in detail or "already exists" in detail:
                    raise StudioConflictError() from None
                if "history limit" in detail:
                    raise StudioCapacityError() from None
                if any(
                    marker in detail
                    for marker in (
                        "source run does not match",
                        "scenario does not match",
                        "source epoch does not match",
                        "child run must differ",
                        "child stream must differ",
                    )
                ):
                    raise StudioRunLifecycleError() from None
                raise
            except RuntimeError as error:
                if str(error) == "scenario checkpoint store is not configured":
                    raise StudioRunLifecycleError() from None
                raise
            self._runs.commit(reservation, child, outcome=result)
        except Exception:
            try:
                if factory_started:
                    self._factory.abort_fork(coordinator, request)
            finally:
                self._runs.abort(reservation)
            raise
        return _fork_result(result)

    def _run_view(self, raw: object, capability: StudioCapability) -> JsonObject:
        values = _params(raw, required=("run_id",))
        run_id = _text(values["run_id"], label="run ID")
        return _run_view(self._coordinator(capability, run_id, "run.read").run_view())

    def _run_list(self, raw: object, capability: StudioCapability) -> tuple[JsonObject, ...]:
        _params(raw)
        if "run.read" not in capability.permissions:
            raise StudioAuthorizationError()
        return tuple(_run_view(item) for item in self._runs.list_for(capability))

    def _state_public(self, raw: object, capability: StudioCapability) -> JsonObject:
        values = _params(raw, required=("run_id",))
        run_id = _text(values["run_id"], label="run ID")
        return _public_state(
            self._coordinator(capability, run_id, "state.public").public_state_view()
        )

    def _state_agent(self, raw: object, capability: StudioCapability) -> JsonObject:
        values = _params(raw, required=("run_id", "agent_id"))
        run_id = _text(values["run_id"], label="run ID")
        agent_id = _text(values["agent_id"], label="agent ID")
        self._require_run(capability, run_id, "state.agent")
        self._require_agent(capability, agent_id)
        coordinator = self._runs.resolve(run_id)
        audience = SimulationAudienceCapability(SimulationOutputAudience.AGENT, agent_id)
        return _agent_state(coordinator.agent_state_view(agent_id, audience))

    def _state_network(self, raw: object, capability: StudioCapability) -> JsonObject:
        values = _params(raw, required=("run_id",))
        run_id = _text(values["run_id"], label="run ID")
        coordinator = self._coordinator(capability, run_id, "state.network")
        audience = SimulationAudienceCapability(SimulationOutputAudience.ANALYST)
        return _network_state(coordinator.network_state(audience))

    def _output_get(self, raw: object, capability: StudioCapability) -> JsonObject:
        values = _params(raw, required=("run_id", "batch_hash"), optional=("agent_id",))
        run_id = _text(values["run_id"], label="run ID")
        batch_hash = _text(values["batch_hash"], label="batch hash")
        agent_id = _optional_text(values.get("agent_id"), label="agent ID")
        self._require_run(capability, run_id, "output.read")
        if agent_id is not None:
            self._require_agent(capability, agent_id)
            audience = SimulationAudienceCapability(SimulationOutputAudience.AGENT, agent_id)
        elif "state.network" in capability.permissions:
            audience = SimulationAudienceCapability(SimulationOutputAudience.ANALYST)
        else:
            audience = SimulationAudienceCapability(SimulationOutputAudience.PUBLIC)
        coordinator = self._runs.resolve(run_id)
        return _output_view(
            coordinator.output_view(
                batch_hash,
                audience,
            )
        )

    def _command_get(self, raw: object, capability: StudioCapability) -> JsonObject:
        values = _params(raw, required=("run_id", "command_id"))
        run_id = _text(values["run_id"], label="run ID")
        coordinator = self._coordinator(capability, run_id, "command.read")
        command_capability = self._command_capability(
            capability,
            run_id,
            audit="command.audit" in capability.permissions,
        )
        return _command_result(
            coordinator.command_result(
                _text(values["command_id"], label="command ID"),
                command_capability,
            )
        )


__all__ = (
    "JsonObject",
    "ScenarioCoordinatorFactory",
    "StudioAuthorizationError",
    "StudioCapacityError",
    "StudioConflictError",
    "StudioError",
    "StudioHistoryGapError",
    "StudioInvalidParamsError",
    "StudioMethodNotFoundError",
    "StudioNotFoundError",
    "StudioRunLifecycleError",
    "StudioStaleStateError",
    "StudioValidationError",
    "WorldStudioService",
)
