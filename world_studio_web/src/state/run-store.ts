import {
  JsonRpcError,
  JsonRpcProtocolError,
  type RpcCaller,
} from "../rpc/client";
import type {
  JsonObject,
  RunAuthority,
  ScenarioAgentStateView,
  ScenarioCommandKind,
  ScenarioCommandResult,
  ScenarioForkResult,
  ScenarioNetworkStateView,
  ScenarioPublicStateView,
  ScenarioRunStatus,
  ScenarioRunView,
  SimulationOutputView,
  TimelineMarker,
} from "../schema/studio-types";

const HASH = /^sha256:[0-9a-f]{64}$/;
const STATUSES = new Set<ScenarioRunStatus>(["created", "running", "paused", "stopped", "completed"]);
const COMMANDS = new Set<ScenarioCommandKind>(["start", "pause", "resume", "step", "stop", "checkpoint"]);

export interface RunIdentitySource {
  commandId(): string;
  idempotencyKey(): string;
  forkId(): string;
  forkIdempotencyKey(): string;
}

export interface RunStoreLimits {
  maximumOutputBatches: number;
  maximumOutputRecords: number;
  maximumOutputBytes: number;
  maximumAnnouncements: number;
  maximumTimelineMarkers: number;
}

export interface RunAudienceSelection {
  audience: "public" | "agent" | "network";
  owner_agent_id: string | null;
}

export interface ScopedRunState {
  audience: RunAudienceSelection["audience"];
  owner_agent_id: string | null;
  value: ScenarioPublicStateView | ScenarioAgentStateView | ScenarioNetworkStateView | JsonObject;
}

type PendingMutation = {
  signature: string;
  method: "run.command" | "run.fork";
  params: JsonObject;
};

function defaultIdentity(prefix: string): string {
  return `${prefix}-${crypto.randomUUID()}`;
}

function freeze<T>(value: T): T {
  if (value && typeof value === "object" && !Object.isFrozen(value)) {
    Object.freeze(value);
    for (const child of Object.values(value as Record<string, unknown>)) freeze(child);
  }
  return value;
}

function record(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function text(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}

function hash(value: unknown): value is string {
  return typeof value === "string" && HASH.test(value);
}

function nonnegative(value: unknown): value is number {
  return Number.isInteger(value) && (value as number) >= 0;
}

function positive(value: unknown): value is number {
  return Number.isInteger(value) && (value as number) > 0;
}

function hashes(value: unknown): value is string[] {
  return Array.isArray(value) && value.every(hash) && new Set(value).size === value.length;
}

function exactKeys(value: Record<string, unknown>, keys: readonly string[]): boolean {
  const actual = Object.keys(value);
  return actual.length === keys.length && actual.every((key) => keys.includes(key));
}

function sameStrings(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function sameRunView(left: ScenarioRunView, right: ScenarioRunView): boolean {
  return left.schema === right.schema && left.run_id === right.run_id &&
    left.stream_id === right.stream_id && left.scenario_hash === right.scenario_hash &&
    left.coordinator_epoch === right.coordinator_epoch && left.status === right.status &&
    left.round_index === right.round_index && left.state_hash === right.state_hash &&
    left.next_sequence === right.next_sequence &&
    sameStrings(left.output_batch_hashes, right.output_batch_hashes) &&
    sameStrings(left.checkpoint_hashes, right.checkpoint_hashes) &&
    left.parent_checkpoint_hash === right.parent_checkpoint_hash &&
    left.content_hash === right.content_hash;
}

export function assertScenarioRunView(value: unknown): asserts value is ScenarioRunView {
  if (!record(value) || !exactKeys(value, [
    "schema", "run_id", "stream_id", "scenario_hash", "coordinator_epoch", "status",
    "round_index", "state_hash", "next_sequence", "output_batch_hashes",
    "checkpoint_hashes", "parent_checkpoint_hash", "content_hash",
  ]) || value.schema !== "narrative-dynamics.scenario-run-view/v1" ||
      !text(value.run_id) || !text(value.stream_id) || !hash(value.scenario_hash) ||
      !positive(value.coordinator_epoch) || !STATUSES.has(value.status as ScenarioRunStatus) ||
      !nonnegative(value.round_index) || !hash(value.state_hash) || !positive(value.next_sequence) ||
      !hashes(value.output_batch_hashes) || !hashes(value.checkpoint_hashes) ||
      !(value.parent_checkpoint_hash === null || hash(value.parent_checkpoint_hash)) ||
      !hash(value.content_hash)) {
    throw new JsonRpcProtocolError("Scenario run view is malformed");
  }
}

function assertCommandResult(value: unknown): asserts value is ScenarioCommandResult {
  if (!record(value) || !exactKeys(value, [
    "schema", "command_id", "idempotency_key", "request_hash", "capability_hash",
    "run_id", "scenario_hash", "coordinator_epoch", "kind", "accepted", "reason",
    "prior_status", "next_status", "prior_state_hash", "next_state_hash", "round_index",
    "output_batch_hash", "checkpoint_hash", "content_hash",
  ]) || value.schema !== "narrative-dynamics.scenario-command-result/v1" ||
      !text(value.command_id) || !text(value.idempotency_key) || !hash(value.request_hash) ||
      !hash(value.capability_hash) || !text(value.run_id) || !hash(value.scenario_hash) ||
      !positive(value.coordinator_epoch) || !COMMANDS.has(value.kind as ScenarioCommandKind) ||
      typeof value.accepted !== "boolean" || !text(value.reason) ||
      !STATUSES.has(value.prior_status as ScenarioRunStatus) ||
      !STATUSES.has(value.next_status as ScenarioRunStatus) ||
      !hash(value.prior_state_hash) || !hash(value.next_state_hash) || !nonnegative(value.round_index) ||
      !(value.output_batch_hash === null || hash(value.output_batch_hash)) ||
      !(value.checkpoint_hash === null || hash(value.checkpoint_hash)) || !hash(value.content_hash) ||
      value.accepted !== (value.reason === "accepted")) {
    throw new JsonRpcProtocolError("Scenario command result is malformed");
  }
}

function assertForkResult(value: unknown): asserts value is ScenarioForkResult {
  if (!record(value) || !exactKeys(value, [
    "schema", "fork_id", "idempotency_key", "request_hash", "capability_hash",
    "source_run_id", "child_run_id", "child_stream_id", "scenario_hash", "child_epoch",
    "checkpoint_hash", "child_state_hash", "content_hash",
  ]) || value.schema !== "narrative-dynamics.scenario-fork-result/v1" ||
      !text(value.fork_id) || !text(value.idempotency_key) || !hash(value.request_hash) ||
      !hash(value.capability_hash) || !text(value.source_run_id) || !text(value.child_run_id) ||
      !text(value.child_stream_id) || !hash(value.scenario_hash) || !positive(value.child_epoch) ||
      !hash(value.checkpoint_hash) || !hash(value.child_state_hash) || !hash(value.content_hash)) {
    throw new JsonRpcProtocolError("Scenario fork result is malformed");
  }
}

function asParams(value: Record<string, unknown>): JsonObject {
  return value as unknown as JsonObject;
}

export class RunStore extends EventTarget {
  private readonly authority: RunAuthority;
  private readonly ids: RunIdentitySource;
  private readonly limits: RunStoreLimits;
  private readonly views = new Map<string, ScenarioRunView>();
  private readonly busyRuns = new Set<string>();
  private readonly retries = new Map<string, PendingMutation>();
  private readonly scoped = new Map<string, ScopedRunState>();
  private readonly audienceGenerations = new Map<string, number>();
  private readonly pendingAudiences = new Map<string, number>();
  private readonly retained = new Map<string, SimulationOutputView[]>();
  private readonly notices: string[] = [];
  private readonly markers = new Map<string, TimelineMarker[]>();

  constructor(
    private readonly rpc: RpcCaller,
    authority: RunAuthority,
    ids?: RunIdentitySource,
    limits: Partial<RunStoreLimits> = {},
  ) {
    super();
    this.authority = freeze(structuredClone(authority));
    this.ids = ids ?? {
      commandId: () => defaultIdentity("command"),
      idempotencyKey: () => defaultIdentity("command-key"),
      forkId: () => defaultIdentity("fork"),
      forkIdempotencyKey: () => defaultIdentity("fork-key"),
    };
    this.limits = {
      maximumOutputBatches: limits.maximumOutputBatches ?? 128,
      maximumOutputRecords: limits.maximumOutputRecords ?? 4_000,
      maximumOutputBytes: limits.maximumOutputBytes ?? 4 * 1024 * 1024,
      maximumAnnouncements: limits.maximumAnnouncements ?? 8,
      maximumTimelineMarkers: limits.maximumTimelineMarkers ?? 64,
    };
    for (const [name, value] of Object.entries(this.limits)) {
      if (!Number.isInteger(value) || value <= 0) throw new TypeError(`${name} must be a positive integer`);
    }
  }

  get authorizedAgentIds(): string[] { return [...this.authority.agent_ids]; }
  get permissions(): string[] { return [...this.authority.permissions]; }
  get announcements(): string[] { return [...this.notices]; }
  timelineMarkers(runId: string): TimelineMarker[] {
    return structuredClone(this.markers.get(runId) ?? []);
  }

  private changed(): void { this.dispatchEvent(new Event("change")); }

  announce(message: string): void {
    const bounded = message.trim().slice(0, 240);
    if (!bounded) return;
    this.notices.push(bounded);
    while (this.notices.length > this.limits.maximumAnnouncements) this.notices.shift();
    this.changed();
  }

  markTimeline(runId: string, marker: TimelineMarker): void {
    if (!this.views.has(runId)) throw new Error("Load the run before marking its timeline");
    if (!record(marker) || ![
      "gap", "recovery-started", "recovery-completed", "recovery-failed", "resume",
    ].includes(marker.kind) || !nonnegative(marker.after_sequence) ||
        !nonnegative(marker.before_sequence) || marker.before_sequence < marker.after_sequence ||
        !text(marker.label)) {
      throw new TypeError("Timeline marker is malformed");
    }
    const accepted = freeze(structuredClone({
      ...marker,
      label: marker.label.trim().slice(0, 240),
    }));
    const retained = this.markers.get(runId) ?? [];
    retained.push(accepted);
    while (retained.length > this.limits.maximumTimelineMarkers) retained.shift();
    this.markers.set(runId, retained);
    this.announce(accepted.label);
  }

  run(runId: string): ScenarioRunView | null { return this.views.get(runId) ?? null; }
  runs(): ScenarioRunView[] { return [...this.views.values()]; }
  isBusy(runId: string): boolean { return this.busyRuns.has(runId); }
  isAudienceBusy(runId: string): boolean { return (this.pendingAudiences.get(runId) ?? 0) > 0; }
  scopedState(runId: string): ScopedRunState | null { return this.scoped.get(runId) ?? null; }
  outputs(runId: string): SimulationOutputView[] { return [...(this.retained.get(runId) ?? [])]; }

  acceptRun(value: unknown): ScenarioRunView {
    assertScenarioRunView(value);
    if (!this.authority.run_ids.includes(value.run_id)) throw new JsonRpcProtocolError("Run is outside the authoritative capability");
    const accepted = freeze(structuredClone(value));
    this.views.set(accepted.run_id, accepted);
    this.changed();
    return accepted;
  }

  async create(
    project: { project_id: string; revision: number; content_hash: string },
    runId: string,
    streamId: string,
  ): Promise<ScenarioRunView> {
    if (!this.authority.permissions.includes("run.create") ||
        !this.authority.project_ids.includes(project.project_id) ||
        !this.authority.run_ids.includes(runId)) throw new Error("Run creation is not authorized");
    return this.withGates([runId], async () => {
      const result = await this.rpc.call<ScenarioRunView>("run.create", {
        project_id: project.project_id,
        run_id: runId,
        stream_id: streamId,
        expected_revision: project.revision,
        expected_snapshot_hash: project.content_hash,
      }, {
        stateChanging: true,
        attempts: 2,
        requestId: `run-create:${project.project_id}:${project.revision}:${runId}`,
      });
      if (result.run_id !== runId || result.stream_id !== streamId) {
        throw new JsonRpcProtocolError("Created run does not match the request");
      }
      this.announce(`Run ${runId} was created.`);
      return this.acceptRun(result);
    });
  }

  private async withGates<T>(runIds: string[], action: () => Promise<T>): Promise<T> {
    const unique = [...new Set(runIds)];
    if (unique.some((runId) => this.busyRuns.has(runId))) throw new Error("A state-changing request is already in flight for this run");
    unique.forEach((runId) => this.busyRuns.add(runId));
    this.changed();
    try { return await action(); }
    finally {
      unique.forEach((runId) => this.busyRuns.delete(runId));
      this.changed();
    }
  }

  private commandMutation(view: ScenarioRunView, kind: ScenarioCommandKind, checkpointId?: string): PendingMutation {
    const signature = JSON.stringify({ kind, checkpointId: checkpointId ?? null, state: view.state_hash });
    const previous = this.retries.get(view.run_id);
    if (previous) {
      if (previous.signature !== signature || previous.method !== "run.command") {
        throw new Error("A different logical retry is pending for this run");
      }
      return previous;
    }
    const params: Record<string, unknown> = {
      command_id: this.ids.commandId(),
      idempotency_key: this.ids.idempotencyKey(),
      run_id: view.run_id,
      scenario_hash: view.scenario_hash,
      coordinator_epoch: view.coordinator_epoch,
      expected_state_hash: view.state_hash,
      kind,
    };
    if (kind === "checkpoint") {
      if (!checkpointId) throw new Error("Checkpoint commands require an ID");
      params.requested_checkpoint_id = checkpointId;
    } else if (checkpointId) {
      throw new Error("Only checkpoint commands accept a checkpoint ID");
    }
    return { signature, method: "run.command", params: asParams(params) };
  }

  async command(runId: string, kind: ScenarioCommandKind, checkpointId?: string): Promise<ScenarioCommandResult> {
    const view = this.views.get(runId);
    if (!view) throw new Error("Load the run before issuing a command");
    if (!this.authority.permissions.includes("run.command")) throw new Error("Run commands are not authorized");
    return this.withGates([runId], async () => {
      const mutation = this.commandMutation(view, kind, checkpointId);
      this.retries.set(runId, mutation);
      let outcomeReceived = false;
      try {
        const result = await this.rpc.call<ScenarioCommandResult>(mutation.method, mutation.params, {
          stateChanging: true,
          attempts: 2,
          requestId: mutation.params.command_id as string,
        });
        outcomeReceived = true;
        assertCommandResult(result);
        if (result.command_id !== mutation.params.command_id ||
            result.idempotency_key !== mutation.params.idempotency_key ||
            result.run_id !== runId || result.scenario_hash !== view.scenario_hash ||
            result.coordinator_epoch !== view.coordinator_epoch || result.kind !== kind ||
            result.prior_status !== view.status || result.prior_state_hash !== view.state_hash) {
          throw new JsonRpcProtocolError("Scenario command result does not bind the accepted run");
        }
        if (!result.accepted) {
          if (result.next_status !== view.status || result.next_state_hash !== view.state_hash ||
              result.round_index !== view.round_index || result.output_batch_hash !== null ||
              result.checkpoint_hash !== null) {
            throw new JsonRpcProtocolError("Rejected command result attempts an authoritative transition");
          }
          this.retries.delete(runId);
          this.announce(`${kind} was rejected: ${result.reason}.`);
          return freeze(structuredClone(result));
        }
        const refreshed = await this.rpc.call<ScenarioRunView>("run.view", { run_id: runId });
        assertScenarioRunView(refreshed);
        const expectedOutputHashes = result.output_batch_hash === null
          ? view.output_batch_hashes : [...view.output_batch_hashes, result.output_batch_hash];
        const expectedCheckpointHashes = result.checkpoint_hash === null
          ? view.checkpoint_hashes : [...view.checkpoint_hashes, result.checkpoint_hash];
        if (refreshed.status !== result.next_status || refreshed.state_hash !== result.next_state_hash ||
            refreshed.round_index !== result.round_index || refreshed.run_id !== runId ||
            refreshed.stream_id !== view.stream_id || refreshed.scenario_hash !== view.scenario_hash ||
            refreshed.coordinator_epoch !== view.coordinator_epoch ||
            refreshed.parent_checkpoint_hash !== view.parent_checkpoint_hash ||
            !sameStrings(refreshed.output_batch_hashes, expectedOutputHashes) ||
            !sameStrings(refreshed.checkpoint_hashes, expectedCheckpointHashes)) {
          throw new JsonRpcProtocolError("Refreshed run does not match the accepted command result");
        }
        this.acceptRun(refreshed);
        this.retries.delete(runId);
        this.announce(`${kind} was accepted for ${runId}.`);
        return freeze(structuredClone(result));
      } catch (error) {
        if (!outcomeReceived && error instanceof JsonRpcError) this.retries.delete(runId);
        this.announce(`${kind} was not accepted for ${runId}.`);
        throw error;
      }
    });
  }

  async fork(
    sourceRunId: string,
    checkpointHash: string,
    childRunId: string,
    childStreamId: string,
  ): Promise<ScenarioForkResult> {
    const source = this.views.get(sourceRunId);
    if (!source) throw new Error("Load the source run before forking");
    if (!this.authority.permissions.includes("run.fork") || !this.authority.run_ids.includes(childRunId)) {
      throw new Error("Run fork is not authorized");
    }
    return this.withGates([sourceRunId, childRunId], async () => {
      const signature = JSON.stringify({ sourceRunId, checkpointHash, childRunId, childStreamId, state: source.state_hash });
      let mutation = this.retries.get(sourceRunId);
      if (mutation && (mutation.method !== "run.fork" || mutation.signature !== signature)) {
        throw new Error("A different logical retry is pending for this run");
      }
      if (!mutation) {
        mutation = {
          signature,
          method: "run.fork",
          params: {
            fork_id: this.ids.forkId(),
            idempotency_key: this.ids.forkIdempotencyKey(),
            source_run_id: sourceRunId,
            scenario_hash: source.scenario_hash,
            source_epoch: source.coordinator_epoch,
            checkpoint_hash: checkpointHash,
            child_run_id: childRunId,
            child_stream_id: childStreamId,
          },
        };
      }
      this.retries.set(sourceRunId, mutation);
      let outcomeReceived = false;
      try {
        const result = await this.rpc.call<ScenarioForkResult>(mutation.method, mutation.params, {
          stateChanging: true,
          attempts: 2,
          requestId: mutation.params.fork_id as string,
        });
        outcomeReceived = true;
        assertForkResult(result);
        if (result.fork_id !== mutation.params.fork_id ||
            result.idempotency_key !== mutation.params.idempotency_key ||
            result.source_run_id !== sourceRunId || result.child_run_id !== childRunId ||
            result.child_stream_id !== childStreamId || result.scenario_hash !== source.scenario_hash ||
            result.checkpoint_hash !== checkpointHash) {
          throw new JsonRpcProtocolError("Fork result does not match the requested lineage");
        }
        const [parent, child] = await Promise.all([
          this.rpc.call<ScenarioRunView>("run.view", { run_id: sourceRunId }),
          this.rpc.call<ScenarioRunView>("run.view", { run_id: childRunId }),
        ]);
        assertScenarioRunView(parent);
        assertScenarioRunView(child);
        if (!sameRunView(parent, source) ||
            child.run_id !== result.child_run_id || child.stream_id !== result.child_stream_id ||
            child.scenario_hash !== result.scenario_hash ||
            child.coordinator_epoch !== result.child_epoch ||
            child.parent_checkpoint_hash !== result.checkpoint_hash ||
            child.state_hash !== result.child_state_hash) {
          throw new JsonRpcProtocolError("Fork refreshed views do not match the requested lineage");
        }
        this.acceptRun(parent);
        this.acceptRun(child);
        this.retries.delete(sourceRunId);
        this.announce(`Fork ${childRunId} was created from ${sourceRunId}.`);
        return freeze(structuredClone(result));
      } catch (error) {
        if (!outcomeReceived && error instanceof JsonRpcError) this.retries.delete(sourceRunId);
        this.announce(`Fork from ${sourceRunId} was not accepted.`);
        throw error;
      }
    });
  }

  acceptScopedState(
    runId: string,
    audience: RunAudienceSelection["audience"],
    ownerAgentId: string | null,
    value: JsonObject,
  ): void {
    this.scoped.set(runId, freeze({ audience, owner_agent_id: ownerAgentId, value: structuredClone(value) }));
    this.changed();
  }

  clearAudienceData(runId: string): void {
    this.scoped.delete(runId);
    this.retained.delete(runId);
    this.changed();
  }

  async selectAudience(runId: string, selection: RunAudienceSelection): Promise<ScopedRunState> {
    const run = this.views.get(runId);
    if (!run) throw new Error("Load the run before selecting an audience");
    if (selection.audience === "agent" &&
        (selection.owner_agent_id === null || !this.authority.agent_ids.includes(selection.owner_agent_id))) {
      throw new Error("Agent owner is not authorized");
    }
    if (selection.audience !== "agent" && selection.owner_agent_id !== null) {
      throw new Error("Only Agent audience may name an owner");
    }
    const permission = selection.audience === "public" ? "state.public"
      : selection.audience === "agent" ? "state.agent" : "state.network";
    if (!this.authority.permissions.includes(permission)) throw new Error("State audience is not authorized");
    const generation = (this.audienceGenerations.get(runId) ?? 0) + 1;
    this.audienceGenerations.set(runId, generation);
    this.pendingAudiences.set(runId, (this.pendingAudiences.get(runId) ?? 0) + 1);
    this.clearAudienceData(runId);
    const method = selection.audience === "public" ? "state.public"
      : selection.audience === "agent" ? "state.agent" : "state.network";
    const params = selection.audience === "agent"
      ? { run_id: runId, agent_id: selection.owner_agent_id! }
      : { run_id: runId };
    try {
      const value = await this.rpc.call<JsonObject>(method, params);
      const expectedSchema = selection.audience === "public"
        ? "narrative-dynamics.scenario-public-state-view/v1"
        : selection.audience === "agent"
          ? "narrative-dynamics.scenario-agent-state-view/v1"
          : "narrative-dynamics.scenario-network-state-view/v1";
      if (!record(value) || !hash(value.content_hash) || !nonnegative(value.round_index) ||
          value.round_index !== run.round_index || value.schema !== expectedSchema ||
          value.run_id !== run.run_id || value.scenario_hash !== run.scenario_hash ||
          value.state_hash !== run.state_hash) {
        throw new JsonRpcProtocolError("Scoped state does not bind the accepted run state");
      }
      if (selection.audience === "agent" && value.agent_id !== selection.owner_agent_id) {
        throw new JsonRpcProtocolError("Agent state does not bind the selected owner");
      }
      if (this.audienceGenerations.get(runId) !== generation) {
        throw new JsonRpcProtocolError("Scoped state response was superseded by a newer audience");
      }
      this.acceptScopedState(runId, selection.audience, selection.owner_agent_id, value);
      return this.scoped.get(runId)!;
    } finally {
      const pending = (this.pendingAudiences.get(runId) ?? 1) - 1;
      if (pending === 0) this.pendingAudiences.delete(runId);
      else this.pendingAudiences.set(runId, pending);
      this.changed();
    }
  }

  acceptOutput(runId: string, output: SimulationOutputView): void {
    const run = this.views.get(runId);
    if (!run || output.stream_id !== run.stream_id || output.scenario_hash !== run.scenario_hash) {
      throw new JsonRpcProtocolError("Output does not match the accepted run");
    }
    const batches = this.retained.get(runId) ?? [];
    batches.push(freeze(structuredClone(output)));
    const byteSize = () => new TextEncoder().encode(JSON.stringify(batches)).length;
    const recordCount = () => batches.reduce((total, batch) => total + batch.records.length, 0);
    while (batches.length && (batches.length > this.limits.maximumOutputBatches ||
      recordCount() > this.limits.maximumOutputRecords || byteSize() > this.limits.maximumOutputBytes)) batches.shift();
    this.retained.set(runId, batches);
    this.changed();
  }
}
