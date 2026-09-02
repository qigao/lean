export type JsonScalar = null | boolean | number | string;
export type JsonValue = JsonScalar | JsonValue[] | JsonObject;
export type JsonObject = { [key: string]: JsonValue };

export type DraftOperationKind =
  | "replace_document"
  | "set_value"
  | "insert_value"
  | "remove_value"
  | "move_value"
  | "set_layout";

export interface DraftDocument {
  role: string;
  logical_id: string | null;
  value: JsonObject;
  content_hash: string;
}

export interface ScenarioDiagnostic {
  severity: "error" | "warning";
  code: string;
  document_role: string;
  logical_id: string | null;
  pointer: string;
  related_ids: string[];
  message_key: string;
}

export interface DiagnosticReport {
  schema: "narrative-dynamics.scenario-diagnostic-report/v1";
  project_id: string;
  revision: number;
  diagnostics: ScenarioDiagnostic[];
  content_hash: string;
}

export interface ProjectSnapshot {
  schema: "narrative-dynamics.scenario-draft-snapshot/v1";
  project_id: string;
  revision: number;
  scenario_id: string | null;
  version: string | null;
  documents: DraftDocument[];
  document_semantic_hash: string;
  layout: JsonObject;
  layout_hash: string;
  diagnostic_report_hash: string;
  compiled_scenario_hash: string | null;
  content_hash: string;
}

export interface ProjectApplyParams extends JsonObject {
  operation_id: string;
  idempotency_key: string;
  project_id: string;
  expected_revision: number;
  expected_snapshot_hash: string;
  document_role: string;
  logical_id: string | null;
  kind: DraftOperationKind;
  pointer?: string;
  value?: JsonValue;
  from_pointer?: string;
}

export interface ProjectApplyResult {
  operation_hash: string;
  prior_revision: number;
  next_snapshot: ProjectSnapshot;
  diagnostic_report: DiagnosticReport;
}

export type OperationIntent =
  | {
      document_role: string;
      logical_id: string | null;
      kind: "replace_document" | "set_value" | "insert_value" | "set_layout";
      pointer?: string;
      value: JsonValue;
    }
  | {
      document_role: string;
      logical_id: string | null;
      kind: "remove_value";
      pointer: string;
    }
  | {
      document_role: string;
      logical_id: string | null;
      kind: "move_value";
      pointer: string;
      from_pointer: string;
    };

export interface DiagnosticTarget {
  documentRole: string;
  logicalId: string | null;
  pointer: string;
}

export type ScenarioRunStatus = "created" | "running" | "paused" | "stopped" | "completed";
export type ScenarioCommandKind = "start" | "pause" | "resume" | "step" | "stop" | "checkpoint";
export type ScenarioCommandReason =
  | "accepted"
  | "unauthorized"
  | "run_mismatch"
  | "scenario_mismatch"
  | "epoch_mismatch"
  | "stale_state"
  | "invalid_status"
  | "maximum_rounds";

export interface RunAuthority {
  authority_id: string;
  project_ids: string[];
  run_ids: string[];
  agent_ids: string[];
  permissions: string[];
}

export interface StudioSession {
  schema: "narrative-dynamics.studio-session/v1";
  authority: RunAuthority;
}

export interface ScenarioRunView {
  schema: "narrative-dynamics.scenario-run-view/v1";
  run_id: string;
  stream_id: string;
  scenario_hash: string;
  coordinator_epoch: number;
  status: ScenarioRunStatus;
  round_index: number;
  state_hash: string;
  next_sequence: number;
  output_batch_hashes: string[];
  checkpoint_hashes: string[];
  parent_checkpoint_hash: string | null;
  content_hash: string;
}

export interface ScenarioCommandResult {
  schema: "narrative-dynamics.scenario-command-result/v1";
  command_id: string;
  idempotency_key: string;
  request_hash: string;
  capability_hash: string;
  run_id: string;
  scenario_hash: string;
  coordinator_epoch: number;
  kind: ScenarioCommandKind;
  accepted: boolean;
  reason: ScenarioCommandReason;
  prior_status: ScenarioRunStatus;
  next_status: ScenarioRunStatus;
  prior_state_hash: string;
  next_state_hash: string;
  round_index: number;
  output_batch_hash: string | null;
  checkpoint_hash: string | null;
  content_hash: string;
}

export interface ScenarioForkResult {
  schema: "narrative-dynamics.scenario-fork-result/v1";
  fork_id: string;
  idempotency_key: string;
  request_hash: string;
  capability_hash: string;
  source_run_id: string;
  child_run_id: string;
  child_stream_id: string;
  scenario_hash: string;
  child_epoch: number;
  checkpoint_hash: string;
  child_state_hash: string;
  content_hash: string;
}

export interface ScenarioCompileResult {
  project_id: string;
  revision: number;
  snapshot_hash: string;
  scenario_id: string;
  version: string;
  package_hash: string;
  scenario_hash: string;
}

export interface SituatedNetworkMetrics extends JsonObject {
  snapshot_hash: string;
  round_index: number;
  population_size: number;
  occupied_place_count: number;
  adopted_count: number;
  adoption_rate: number;
  tracked_belief_mean: number;
  tracked_belief_variance: number;
  active_relationship_edge_count: number;
  mean_relationship_trust: number;
  direct_interaction_pair_count: number;
  latest_tell_event_count: number;
  transmission_count: number;
  reached_observer_count: number;
  exact_transmission_count: number;
  detected_transmission_count: number;
  identified_transmission_count: number;
  active_claim_count: number;
  confirmed_claim_count: number;
  contradicted_claim_count: number;
  superseded_claim_count: number;
  forgotten_claim_count: number;
}

export interface ScenarioPublicStateView {
  schema: "narrative-dynamics.scenario-public-state-view/v1";
  run_id: string;
  scenario_hash: string;
  round_index: number;
  state_hash: string;
  agent_places: [string, string][];
  passage_states: [string, boolean][];
  object_placements: [string, string | null, string | null][];
  metrics: SituatedNetworkMetrics;
  content_hash: string;
}

export interface ScenarioAgentStateView {
  schema: "narrative-dynamics.scenario-agent-state-view/v1";
  run_id: string;
  scenario_hash: string;
  round_index: number;
  state_hash: string;
  agent_id: string;
  place_id: string;
  mind: JsonObject;
  relationships: JsonObject[];
  claims: JsonObject[];
  content_hash: string;
}

export interface ScenarioNetworkStateView {
  schema: "narrative-dynamics.scenario-network-state-view/v1";
  run_id: string;
  scenario_hash: string;
  state_hash: string;
  model_id: string;
  model_hash: string;
  round_index: number;
  story_hash: string;
  cognitive_state_hash: string;
  social_state_hash: string;
  nodes: JsonObject[];
  relationship_edges: JsonObject[];
  access_edges: JsonObject[];
  transmissions: JsonObject[];
  latest_tell_event_count: number;
  content_hash: string;
}

export type SimulationOutputAudience = "public" | "objective" | "agent" | "analyst" | "internal";
export type SimulationOutputKind =
  | "state.delta"
  | "event.objective"
  | "percept.private"
  | "agent.decision"
  | "memory.update"
  | "social.update"
  | "network.metrics"
  | "story.progress"
  | "narrative.scene"
  | "blender.delta"
  | "command.result"
  | "diagnostic";

export interface SimulationStateDeltaPayload extends JsonObject {
  prior_snapshot_hash: string;
  next_snapshot_hash: string;
  changed_agent_ids: string[];
  changed_passage_ids: string[];
  changed_object_ids: string[];
}

export interface SimulationObjectiveEventPayload extends JsonObject {
  event_id: string;
  event_hash: string;
  action_id: string;
  action_kind: string;
  actor_agent_id: string;
  place_id: string;
  target_id: string | null;
  success: boolean;
  cause_event_ids: string[];
}

export interface SimulationPrivatePerceptPayload extends JsonObject { percept: JsonObject; }
export interface SimulationAgentDecisionPayload extends JsonObject { decision: JsonObject; }
export interface SimulationMemoryUpdatePayload extends JsonObject {
  agent_id: string;
  prior_mind_hash: string;
  next_mind_hash: string;
  recalled_memory_ids: string[];
  admitted_memory_ids: string[];
}
export interface SimulationSocialUpdatePayload extends JsonObject {
  observer_agent_id: string;
  prior_claim_hashes: string[];
  next_claim_hashes: string[];
  prior_relationship_hashes: string[];
  next_relationship_hashes: string[];
  admitted_evidence_ids: string[];
}
export interface SimulationNetworkMetricsPayload extends JsonObject { metrics: SituatedNetworkMetrics; }
export interface SimulationStoryProgressPayload extends JsonObject {
  active_scene_id: string | null;
  completed_scene_ids: string[];
  status: string;
}
export interface SimulationNarrativeScenePayload extends JsonObject {
  scene_id: string;
  projection_hash: string;
  realization_hash: string;
}
export interface SimulationBlenderDeltaPayload extends JsonObject {
  agent_places: [string, string][];
  passage_states: [string, boolean][];
  object_placements: [string, string | null, string | null][];
}
export interface SimulationCommandResultPayload extends JsonObject {
  command_id: string;
  accepted: boolean;
  reason_code: string;
}
export interface SimulationDiagnosticPayload extends JsonObject { code: string; message: string; }

export interface SimulationOutputPayloadByKind {
  "state.delta": SimulationStateDeltaPayload;
  "event.objective": SimulationObjectiveEventPayload;
  "percept.private": SimulationPrivatePerceptPayload;
  "agent.decision": SimulationAgentDecisionPayload;
  "memory.update": SimulationMemoryUpdatePayload;
  "social.update": SimulationSocialUpdatePayload;
  "network.metrics": SimulationNetworkMetricsPayload;
  "story.progress": SimulationStoryProgressPayload;
  "narrative.scene": SimulationNarrativeScenePayload;
  "blender.delta": SimulationBlenderDeltaPayload;
  "command.result": SimulationCommandResultPayload;
  diagnostic: SimulationDiagnosticPayload;
}

type SimulationOutputRecordFor<K extends SimulationOutputKind> = {
  schema: "narrative-dynamics.simulation-output-record/v1";
  stream_id: string;
  scenario_hash: string;
  sequence: number;
  round_index: number;
  state_hash: string;
  kind: K;
  audience: SimulationOutputAudience;
  owner_agent_id: string | null;
  source_artifact_hashes: string[];
  payload: SimulationOutputPayloadByKind[K];
  payload_hash: string;
  content_hash: string;
};

export type SimulationOutputRecord = {
  [K in SimulationOutputKind]: SimulationOutputRecordFor<K>
}[SimulationOutputKind];

export interface SimulationOutputView {
  schema: "narrative-dynamics.simulation-output-view/v1";
  stream_id: string;
  scenario_hash: string;
  prior_state_hash: string;
  next_state_hash: string;
  round_result_hash: string;
  first_sequence: number;
  last_sequence: number;
  records: SimulationOutputRecord[];
  source_batch_hash: string;
  checkpoint: boolean;
  content_hash: string;
}

export interface StreamBinding {
  subscription_id: string;
  run_id: string;
  stream_id: string;
  scenario_hash: string;
  kinds: SimulationOutputKind[];
  audience: "public" | "agent" | "analyst";
  owner_agent_id: string | null;
}

export type TimelineMarkerKind =
  | "gap"
  | "recovery-started"
  | "recovery-completed"
  | "recovery-failed"
  | "resume";

export interface TimelineMarker extends JsonObject {
  kind: TimelineMarkerKind;
  after_sequence: number;
  before_sequence: number;
  label: string;
}
