import { assertJsonValue, JsonRpcError, JsonRpcProtocolError } from "./client";
import type {
  JsonObject,
  JsonValue,
  SimulationOutputAudience,
  SimulationOutputKind,
  SimulationOutputRecord,
  SimulationOutputView,
  StreamBinding,
  TimelineMarker,
} from "../schema/studio-types";

const PROTOCOL = "nd-jsonrpc-v1";
const HASH = /^sha256:[0-9a-f]{64}$/;
const OUTPUT_KINDS = new Set<SimulationOutputKind>([
  "state.delta", "event.objective", "percept.private", "agent.decision",
  "memory.update", "social.update", "network.metrics", "story.progress",
  "narrative.scene", "blender.delta", "command.result", "diagnostic",
]);
const AUDIENCES = new Set<SimulationOutputAudience>(["public", "objective", "agent", "analyst", "internal"]);

export interface WebSocketLike {
  readonly protocol: string;
  readonly readyState: number;
  onopen: ((event: Event) => void) | null;
  onmessage: ((event: MessageEvent) => void) | null;
  onclose: ((event: CloseEvent) => void) | null;
  onerror: ((event: Event) => void) | null;
  send(data: string): void;
  close(code?: number, reason?: string): void;
}

export type StreamConnectionStatus = "disconnected" | "connecting" | "connected" | "recovering" | "error";

export interface StreamCursor {
  sequence: number;
  batch_hash: string;
}

export interface StreamGapRecoveryRequest extends JsonObject {
  run_id: string;
  audience: StreamBinding["audience"];
  owner_agent_id: string | null;
  snapshot_token: string;
}

export interface StreamClientOptions {
  socketFactory?: (url: string, protocols: string[]) => WebSocketLike;
  onOutput?: (output: SimulationOutputView) => void | Promise<void>;
  onProtocolError?: (error: Error) => void;
  onStatus?: (status: StreamConnectionStatus, message: string) => void;
  onMarker?: (runId: string, marker: TimelineMarker) => void;
  recoverGap?: (request: StreamGapRecoveryRequest) => void | Promise<void>;
  maximumRetainedIdentities?: number;
}

type PendingControl = {
  resolve: (value: JsonValue) => void;
  reject: (error: Error) => void;
};

interface DeliveryGeneration {
  generation: number;
  binding: StreamBinding;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function assertNoDuplicateObjectKeys(raw: string): void {
  let index = 0;
  const whitespace = () => {
    while (index < raw.length && /\s/.test(raw[index]!)) index += 1;
  };
  const jsonString = (): string => {
    const start = index;
    index += 1;
    while (index < raw.length) {
      if (raw[index] === "\\") {
        index += 2;
      } else if (raw[index] === '"') {
        index += 1;
        return JSON.parse(raw.slice(start, index)) as string;
      } else {
        index += 1;
      }
    }
    return "";
  };
  const value = (): void => {
    whitespace();
    if (raw[index] === "{") {
      index += 1;
      whitespace();
      const keys = new Set<string>();
      while (raw[index] !== "}" && index < raw.length) {
        const key = jsonString();
        if (keys.has(key)) throw new StreamProtocolError(`Stream JSON contains duplicate object key ${key}`);
        keys.add(key);
        whitespace();
        index += 1;
        value();
        whitespace();
        if (raw[index] === ",") { index += 1; whitespace(); }
      }
      index += 1;
      return;
    }
    if (raw[index] === "[") {
      index += 1;
      whitespace();
      while (raw[index] !== "]" && index < raw.length) {
        value();
        whitespace();
        if (raw[index] === ",") { index += 1; whitespace(); }
      }
      index += 1;
      return;
    }
    if (raw[index] === '"') {
      jsonString();
      return;
    }
    while (index < raw.length && !/[\s,}\]]/.test(raw[index]!)) index += 1;
  };
  value();
}

function hasExactKeys(value: Record<string, unknown>, required: string[], optional: string[] = []): boolean {
  const keys = Object.keys(value);
  return required.every((key) => keys.includes(key)) &&
    keys.every((key) => required.includes(key) || optional.includes(key));
}

function isHash(value: unknown): value is string {
  return typeof value === "string" && HASH.test(value);
}

function isPositiveInteger(value: unknown): value is number {
  return Number.isInteger(value) && (value as number) > 0;
}

function isNonnegativeInteger(value: unknown): value is number {
  return Number.isInteger(value) && (value as number) >= 0;
}

function assertOutputRecord(value: unknown, output: Record<string, unknown>): asserts value is SimulationOutputRecord {
  if (!isRecord(value) || !hasExactKeys(value, [
    "schema", "stream_id", "scenario_hash", "sequence", "round_index", "state_hash",
    "kind", "audience", "owner_agent_id", "source_artifact_hashes", "payload",
    "payload_hash", "content_hash",
  ]) || value.schema !== "narrative-dynamics.simulation-output-record/v1" ||
      value.stream_id !== output.stream_id || value.scenario_hash !== output.scenario_hash ||
      !isPositiveInteger(value.sequence) || !isPositiveInteger(value.round_index) ||
      !isHash(value.state_hash) || value.state_hash !== output.next_state_hash ||
      !OUTPUT_KINDS.has(value.kind as SimulationOutputKind) ||
      !AUDIENCES.has(value.audience as SimulationOutputAudience) || value.audience === "internal" ||
      !(value.owner_agent_id === null || typeof value.owner_agent_id === "string" && value.owner_agent_id.length > 0) ||
      !Array.isArray(value.source_artifact_hashes) || !value.source_artifact_hashes.every(isHash) ||
      !isRecord(value.payload) || !isHash(value.payload_hash) || !isHash(value.content_hash)) {
    throw new StreamProtocolError("Stream output record is malformed");
  }
  assertJsonValue(value.payload);
}

export function assertSimulationOutputView(value: unknown): asserts value is SimulationOutputView {
  if (!isRecord(value) || !hasExactKeys(value, [
    "schema", "stream_id", "scenario_hash", "prior_state_hash", "next_state_hash",
    "round_result_hash", "first_sequence", "last_sequence", "records",
    "source_batch_hash", "checkpoint", "content_hash",
  ]) || value.schema !== "narrative-dynamics.simulation-output-view/v1" ||
      typeof value.stream_id !== "string" || !value.stream_id || !isHash(value.scenario_hash) ||
      !isHash(value.prior_state_hash) || !isHash(value.next_state_hash) || !isHash(value.round_result_hash) ||
      !isPositiveInteger(value.first_sequence) || !isPositiveInteger(value.last_sequence) ||
      value.last_sequence < value.first_sequence || !Array.isArray(value.records) ||
      !isHash(value.source_batch_hash) || typeof value.checkpoint !== "boolean" || !isHash(value.content_hash)) {
    throw new StreamProtocolError("Stream output view is malformed");
  }
  let priorSequence = 0;
  for (const item of value.records) {
    assertOutputRecord(item, value);
    if (item.sequence < value.first_sequence || item.sequence > value.last_sequence || item.sequence <= priorSequence) {
      throw new StreamProtocolError("Filtered output records do not preserve source order");
    }
    priorSequence = item.sequence;
  }
}

function assertBinding(binding: StreamBinding): void {
  if (!binding.subscription_id || !binding.run_id || !binding.stream_id || !isHash(binding.scenario_hash) ||
      !Array.isArray(binding.kinds) || binding.kinds.some((kind) => !OUTPUT_KINDS.has(kind)) ||
      new Set(binding.kinds).size !== binding.kinds.length ||
      !["public", "agent", "analyst"].includes(binding.audience) ||
      (binding.audience === "agent") !== (typeof binding.owner_agent_id === "string" && !!binding.owner_agent_id)) {
    throw new StreamProtocolError("Stream binding is malformed");
  }
}

function subscriptionParams(binding: StreamBinding): JsonObject {
  const params: JsonObject = {
    subscription_id: binding.subscription_id,
    run_id: binding.run_id,
    stream_id: binding.stream_id,
    kinds: [...binding.kinds],
    audience: binding.audience,
  };
  if (binding.owner_agent_id !== null) params.owner_agent_id = binding.owner_agent_id;
  return params;
}

function outputAllowed(record: SimulationOutputRecord, binding: StreamBinding): boolean {
  if (!binding.kinds.includes(record.kind)) return false;
  if (binding.audience === "public") return record.audience === "public" && record.owner_agent_id === null;
  if (binding.audience === "agent") {
    return record.audience === "public" && record.owner_agent_id === null ||
      record.audience === "agent" && record.owner_agent_id === binding.owner_agent_id;
  }
  return record.owner_agent_id === null && ["public", "objective", "analyst"].includes(record.audience);
}

export class StreamProtocolError extends JsonRpcProtocolError {
  constructor(message: string) {
    super(message);
    this.name = "StreamProtocolError";
  }
}

export class StreamClient extends EventTarget {
  private readonly socketFactory: (url: string, protocols: string[]) => WebSocketLike;
  private readonly onOutput: (output: SimulationOutputView) => void | Promise<void>;
  private readonly onProtocolError: (error: Error) => void;
  private readonly onStatus: (status: StreamConnectionStatus, message: string) => void;
  private readonly onMarker: (runId: string, marker: TimelineMarker) => void;
  private readonly recoverGap: (request: StreamGapRecoveryRequest) => void | Promise<void>;
  private readonly maximumRetainedIdentities: number;
  private socket: WebSocketLike | null = null;
  private activeBinding: StreamBinding | null = null;
  private streamScopeBinding: StreamBinding | null = null;
  private pending = new Map<string, PendingControl>();
  private requestNumber = 0;
  private state: StreamConnectionStatus = "disconnected";
  private committedCursor: StreamCursor | null = null;
  private confirmedAcknowledgementCursor: StreamCursor | null = null;
  private readonly identities = new Map<string, string>();
  private deliveryChain: Promise<void> = Promise.resolve();
  private audienceGeneration = 0;
  private audienceSwitchTail: Promise<void> = Promise.resolve();
  private readonly retiredSubscriptionIds = new Set<string>();
  private readonly unsubscribedSubscriptionIds = new Set<string>();

  constructor(private readonly endpoint: string, options: StreamClientOptions = {}) {
    super();
    this.socketFactory = options.socketFactory ?? ((url, protocols) => new WebSocket(url, protocols));
    this.onOutput = options.onOutput ?? (() => undefined);
    this.onProtocolError = options.onProtocolError ?? (() => undefined);
    this.onStatus = options.onStatus ?? (() => undefined);
    this.onMarker = options.onMarker ?? (() => undefined);
    this.recoverGap = options.recoverGap ?? (() => {
      throw new StreamProtocolError("No scoped recovery handler is configured");
    });
    this.maximumRetainedIdentities = options.maximumRetainedIdentities ?? 128;
    if (!Number.isInteger(this.maximumRetainedIdentities) || this.maximumRetainedIdentities <= 0) {
      throw new TypeError("maximumRetainedIdentities must be a positive integer");
    }
  }

  get status(): StreamConnectionStatus { return this.state; }
  get binding(): StreamBinding | null { return this.activeBinding ? structuredClone(this.activeBinding) : null; }
  get cursor(): StreamCursor | null { return this.committedCursor ? { ...this.committedCursor } : null; }
  get retainedIdentityCount(): number { return this.identities.size; }

  private updateStatus(status: StreamConnectionStatus, message: string): void {
    this.state = status;
    this.onStatus(status, message.slice(0, 240));
    this.dispatchEvent(new Event("change"));
  }

  private fail(error: unknown): void {
    const normalized = error instanceof Error ? error : new StreamProtocolError("Unknown stream protocol failure");
    this.onProtocolError(normalized);
    this.updateStatus("error", normalized.message);
  }

  private mark(binding: StreamBinding, marker: TimelineMarker): void {
    try { this.onMarker(binding.run_id, freezeMarker(marker)); }
    catch { /* A view callback cannot change stream protocol authority. */ }
  }

  private retireSubscription(subscriptionId: string): void {
    this.retiredSubscriptionIds.add(subscriptionId);
    while (this.retiredSubscriptionIds.size > this.maximumRetainedIdentities) {
      const oldest = this.retiredSubscriptionIds.values().next().value as string | undefined;
      if (oldest === undefined) break;
      this.retiredSubscriptionIds.delete(oldest);
    }
  }

  private async unsubscribeRetiring(subscriptionId: string): Promise<void> {
    if (this.unsubscribedSubscriptionIds.has(subscriptionId)) return;
    let result: JsonValue;
    try {
      result = await this.sendControl("stream.unsubscribe", { subscription_id: subscriptionId });
    } catch (error) {
      this.close();
      throw error;
    }
    if (!isRecord(result) || !hasExactKeys(result, ["subscription_id", "unsubscribed"]) ||
        result.subscription_id !== subscriptionId || result.unsubscribed !== true) {
      this.close();
      throw new StreamProtocolError("Audience switch unsubscribe response is malformed");
    }
    this.unsubscribedSubscriptionIds.add(subscriptionId);
    while (this.unsubscribedSubscriptionIds.size > this.maximumRetainedIdentities) {
      const oldest = this.unsubscribedSubscriptionIds.values().next().value as string | undefined;
      if (oldest === undefined) break;
      this.unsubscribedSubscriptionIds.delete(oldest);
    }
  }

  private switchedBinding(binding: StreamBinding, generation: number): StreamBinding {
    let suffix = generation;
    let subscriptionId = `${binding.subscription_id}-generation-${suffix}`;
    while (this.retiredSubscriptionIds.has(subscriptionId) ||
           subscriptionId === this.activeBinding?.subscription_id) {
      suffix += 1;
      subscriptionId = `${binding.subscription_id}-generation-${suffix}`;
    }
    return { ...structuredClone(binding), subscription_id: subscriptionId };
  }

  async connect(binding: StreamBinding): Promise<void> {
    assertBinding(binding);
    if (this.socket && this.socket.readyState < 2) throw new Error("Stream connection is already active");
    const prior = this.activeBinding;
    if (prior && JSON.stringify(prior) !== JSON.stringify(binding)) {
      this.identities.clear();
      this.committedCursor = null;
      this.confirmedAcknowledgementCursor = null;
    }
    this.activeBinding = structuredClone(binding);
    this.streamScopeBinding = structuredClone(binding);
    this.updateStatus("connecting", "Connecting to the run output stream.");
    const socket = this.socketFactory(this.endpoint, [PROTOCOL]);
    this.socket = socket;
    socket.onmessage = (event) => this.receive(event.data);
    socket.onerror = () => this.fail(new StreamProtocolError("Stream transport failed"));
    socket.onclose = () => {
      if (this.socket === socket) {
        this.rejectPending(new StreamProtocolError("Stream connection closed"));
        this.updateStatus("disconnected", "Run output stream disconnected.");
      }
    };
    await new Promise<void>((resolve, reject) => {
      socket.onopen = () => {
        if (socket.protocol !== PROTOCOL) {
          const error = new StreamProtocolError(`Server did not negotiate ${PROTOCOL}`);
          socket.close(1002, "subprotocol required");
          reject(error);
          return;
        }
        void this.subscribe(binding).then(() => {
          this.updateStatus("connected", "Run output stream connected.");
          resolve();
        }, reject);
      };
    });
  }

  private rejectPending(error: Error): void {
    const pending = [...this.pending.values()];
    this.pending.clear();
    for (const item of pending) item.reject(error);
  }

  private sendControl(method: string, params: JsonObject): Promise<JsonValue> {
    if (!this.socket || this.socket.readyState !== 1) return Promise.reject(new StreamProtocolError("Stream is not connected"));
    const id = `stream-control-${++this.requestNumber}`;
    const request = { jsonrpc: "2.0", id, method, params };
    return new Promise<JsonValue>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      try { this.socket!.send(JSON.stringify(request)); }
      catch (error) {
        this.pending.delete(id);
        reject(error instanceof Error ? error : new StreamProtocolError("Stream send failed"));
      }
    });
  }

  private receive(raw: unknown): void {
    if (typeof raw !== "string") {
      this.fail(new StreamProtocolError("Stream frames must be UTF-8 JSON text"));
      return;
    }
    let value: unknown;
    try {
      value = JSON.parse(raw);
      assertNoDuplicateObjectKeys(raw);
    }
    catch (error) {
      this.fail(error instanceof StreamProtocolError
        ? error : new StreamProtocolError("Stream frame is not valid JSON"));
      return;
    }
    if (!isRecord(value) || value.jsonrpc !== "2.0") {
      this.fail(new StreamProtocolError("Stream frame is not JSON-RPC 2.0"));
      return;
    }
    if (Object.prototype.hasOwnProperty.call(value, "id")) {
      this.receiveResponse(value);
      return;
    }
    try {
      if (!hasExactKeys(value, ["jsonrpc", "method", "params"]) || value.method !== "stream.output" ||
          !isRecord(value.params) || !hasExactKeys(value.params, ["subscription_id", "output"])) {
        throw new StreamProtocolError("Stream notification envelope is malformed");
      }
      const params = value.params;
      const delivery: DeliveryGeneration | null = this.activeBinding === null ? null : {
        generation: this.audienceGeneration,
        binding: this.activeBinding,
      };
      this.deliveryChain = this.deliveryChain
        .then(() => this.commitOutput(params.subscription_id, params.output, delivery))
        .catch((error) => this.fail(error));
    } catch (error) { this.fail(error); }
  }

  private receiveResponse(value: Record<string, unknown>): void {
    const id = value.id;
    if (typeof id !== "string") {
      this.fail(new StreamProtocolError("Stream response ID is malformed"));
      return;
    }
    const pending = this.pending.get(id);
    if (!pending) {
      this.fail(new StreamProtocolError("Stream response ID is not pending"));
      return;
    }
    this.pending.delete(id);
    const hasResult = Object.prototype.hasOwnProperty.call(value, "result");
    const hasError = Object.prototype.hasOwnProperty.call(value, "error");
    if (hasResult === hasError) {
      pending.reject(new StreamProtocolError("Stream response must contain one outcome"));
      return;
    }
    if (hasResult) {
      if (!hasExactKeys(value, ["jsonrpc", "id", "result"])) {
        pending.reject(new StreamProtocolError("Stream response result envelope is malformed"));
        return;
      }
      try {
        assertJsonValue(value.result);
        pending.resolve(value.result);
      } catch (error) { pending.reject(error as Error); }
      return;
    }
    if (!hasExactKeys(value, ["jsonrpc", "id", "error"]) || !isRecord(value.error) ||
        !hasExactKeys(value.error, ["code", "message"], ["data"]) ||
        !Number.isInteger(value.error.code) || typeof value.error.message !== "string") {
      pending.reject(new StreamProtocolError("Stream error envelope is malformed"));
      return;
    }
    try {
      if (Object.prototype.hasOwnProperty.call(value.error, "data")) assertJsonValue(value.error.data);
      pending.reject(new JsonRpcError(
        value.error.code as number,
        value.error.message,
        value.error.data as JsonValue | undefined,
      ));
    } catch (error) { pending.reject(error as Error); }
  }

  private async subscribe(binding: StreamBinding): Promise<void> {
    const result = await this.sendControl("stream.subscribe", subscriptionParams(binding));
    const required = ["subscription_id", "run_id", "stream_id", "kinds", "audience"];
    const optional = binding.audience === "agent" ? ["owner_agent_id"] : [];
    if (!isRecord(result) || !hasExactKeys(result, required, optional) ||
        result.subscription_id !== binding.subscription_id || result.run_id !== binding.run_id ||
        result.stream_id !== binding.stream_id || !Array.isArray(result.kinds) ||
        JSON.stringify(result.kinds) !== JSON.stringify(binding.kinds) ||
        result.audience !== binding.audience ||
        (binding.audience === "agent" && result.owner_agent_id !== binding.owner_agent_id)) {
      throw new StreamProtocolError("Subscription response does not match the requested binding");
    }
  }

  private async commitOutput(
    subscriptionId: unknown,
    rawOutput: unknown,
    delivery: DeliveryGeneration | null,
  ): Promise<void> {
    if (delivery === null || delivery.generation !== this.audienceGeneration) return;
    const binding = this.activeBinding;
    if (!binding || binding.subscription_id !== delivery.binding.subscription_id) return;
    if (subscriptionId !== binding.subscription_id) {
      if (typeof subscriptionId === "string" && this.retiredSubscriptionIds.has(subscriptionId)) return;
      throw new StreamProtocolError("Output subscription binding is invalid");
    }
    assertSimulationOutputView(rawOutput);
    if (rawOutput.stream_id !== binding.stream_id || rawOutput.scenario_hash !== binding.scenario_hash ||
        rawOutput.records.some((item) => !outputAllowed(item, binding))) {
      throw new StreamProtocolError("Output stream, scenario, kind, or audience binding is invalid");
    }
    const bounds = `${rawOutput.first_sequence}:${rawOutput.last_sequence}`;
    const priorHash = this.identities.get(bounds);
    if (priorHash !== undefined) {
      if (priorHash === rawOutput.source_batch_hash) {
        if (this.committedCursor?.sequence === rawOutput.last_sequence &&
            this.committedCursor.batch_hash === rawOutput.source_batch_hash &&
            (!this.confirmedAcknowledgementCursor ||
             this.confirmedAcknowledgementCursor.sequence < rawOutput.last_sequence)) {
          await this.acknowledge(binding, this.committedCursor);
        }
        return;
      }
      throw new StreamProtocolError("Output sequence bounds conflict with a different batch hash");
    }
    if (this.committedCursor && rawOutput.first_sequence !== this.committedCursor.sequence + 1) {
      if (rawOutput.first_sequence > this.committedCursor.sequence + 1) {
        this.mark(binding, {
          kind: "gap",
          after_sequence: this.committedCursor.sequence,
          before_sequence: rawOutput.first_sequence,
          label: `Output gap detected before source sequence ${rawOutput.first_sequence}.`,
        });
      }
      throw new StreamProtocolError(rawOutput.last_sequence <= this.committedCursor.sequence
        ? "Output sequence regressed" : "Output sequence gap requires recovery");
    }
    await this.onOutput(freezeOutput(rawOutput));
    this.identities.set(bounds, rawOutput.source_batch_hash);
    while (this.identities.size > this.maximumRetainedIdentities) {
      const oldest = this.identities.keys().next().value as string | undefined;
      if (oldest === undefined) break;
      this.identities.delete(oldest);
    }
    this.committedCursor = { sequence: rawOutput.last_sequence, batch_hash: rawOutput.source_batch_hash };
    await this.acknowledge(binding, this.committedCursor);
  }

  private async acknowledge(binding: StreamBinding, cursor: StreamCursor): Promise<void> {
    const result = await this.sendControl("stream.acknowledge", {
      subscription_id: binding.subscription_id,
      stream_id: binding.stream_id,
      last_sequence: cursor.sequence,
      last_batch_hash: cursor.batch_hash,
    });
    if (!isRecord(result) || !hasExactKeys(result, ["subscription_id", "acknowledged"]) ||
        result.subscription_id !== binding.subscription_id || result.acknowledged !== true) {
      throw new StreamProtocolError("Stream acknowledgement response is malformed");
    }
    if (this.confirmedAcknowledgementCursor && cursor.sequence <= this.confirmedAcknowledgementCursor.sequence) {
      throw new StreamProtocolError("Stream acknowledgement did not advance monotonically");
    }
    this.confirmedAcknowledgementCursor = { ...cursor };
  }

  async reconnect(): Promise<void> {
    const binding = this.activeBinding;
    if (!binding) throw new Error("No released subscription is available to reconnect");
    const cursor = this.committedCursor ? { ...this.committedCursor } : null;
    await this.connect(binding);
    if (!cursor) return;
    this.updateStatus("recovering", "Resuming run output from the last committed batch.");
    try {
      const result = await this.sendControl("stream.resume", {
        subscription_id: binding.subscription_id,
        stream_id: binding.stream_id,
        last_sequence: cursor.sequence,
        last_batch_hash: cursor.batch_hash,
      });
      if (!isRecord(result) || result.subscription_id !== binding.subscription_id ||
          result.resumed_from_sequence !== cursor.sequence || !isNonnegativeInteger(result.replayed_count) ||
          !isPositiveInteger(result.current_sequence) || !isHash(result.current_batch_hash)) {
        throw new StreamProtocolError("Stream resume response is malformed");
      }
      this.mark(binding, {
        kind: "resume",
        after_sequence: cursor.sequence,
        before_sequence: result.current_sequence,
        label: `Stream resumed from source sequence ${cursor.sequence}.`,
      });
      this.updateStatus("connected", "Run output stream resumed.");
    } catch (error) {
      if (!(error instanceof JsonRpcError) || error.code !== -32016) throw error;
      await this.recoverHistoryGap(error, binding);
    }
  }

  private async recoverHistoryGap(error: JsonRpcError, binding: StreamBinding): Promise<void> {
    const data = error.data;
    if (!isRecord(data) || !hasExactKeys(data, ["current_sequence", "current_batch_hash", "snapshot_token"]) ||
        !isPositiveInteger(data.current_sequence) || !isHash(data.current_batch_hash) || !isHash(data.snapshot_token)) {
      throw new StreamProtocolError("History-gap recovery data is malformed");
    }
    const after = this.committedCursor?.sequence ?? 0;
    const boundary = { after_sequence: after, before_sequence: data.current_sequence };
    this.mark(binding, {
      kind: "gap", ...boundary,
      label: `Retained stream history has a gap at source sequence ${after}.`,
    });
    this.mark(binding, {
      kind: "recovery-started", ...boundary,
      label: "Capability-scoped state recovery begun.",
    });
    try {
      await this.recoverGap({
        run_id: binding.run_id,
        audience: binding.audience,
        owner_agent_id: binding.owner_agent_id,
        snapshot_token: data.snapshot_token,
      });
      this.identities.clear();
      this.committedCursor = { sequence: data.current_sequence, batch_hash: data.current_batch_hash };
      this.confirmedAcknowledgementCursor = { ...this.committedCursor };
      const unsubscribed = await this.sendControl("stream.unsubscribe", { subscription_id: binding.subscription_id });
      if (!isRecord(unsubscribed) || unsubscribed.subscription_id !== binding.subscription_id || unsubscribed.unsubscribed !== true) {
        throw new StreamProtocolError("Gap recovery unsubscribe response is malformed");
      }
      await this.subscribe(binding);
      this.mark(binding, {
        kind: "recovery-completed", ...boundary,
        label: `Capability-scoped state recovery completed at source sequence ${data.current_sequence}.`,
      });
      this.updateStatus("connected", "Run output recovered from a capability-scoped state snapshot.");
    } catch (recoveryError) {
      this.mark(binding, {
        kind: "recovery-failed", ...boundary,
        label: "Capability-scoped state recovery failed.",
      });
      throw recoveryError;
    }
  }

  switchAudience(binding: StreamBinding, clearPrivateData: () => void): Promise<void> {
    assertBinding(binding);
    const prior = this.streamScopeBinding;
    if (!prior || !this.socket || this.socket.readyState !== 1) throw new Error("Stream is not connected");
    if (binding.run_id !== prior.run_id || binding.stream_id !== prior.stream_id ||
        binding.scenario_hash !== prior.scenario_hash) throw new StreamProtocolError("Audience switch must retain the run binding");
    const retiring = this.activeBinding;
    const generation = ++this.audienceGeneration;
    if (retiring !== null) this.retireSubscription(retiring.subscription_id);
    this.activeBinding = null;
    this.identities.clear();
    this.committedCursor = null;
    this.confirmedAcknowledgementCursor = null;
    clearPrivateData();
    const nextBinding = this.switchedBinding(binding, generation);
    const operation = this.audienceSwitchTail.catch(() => undefined).then(async () => {
      if (retiring !== null) {
        await this.unsubscribeRetiring(retiring.subscription_id);
      }
      if (generation !== this.audienceGeneration) {
        throw new StreamProtocolError("Audience switch was superseded by a newer selection");
      }
      await this.subscribe(nextBinding);
      if (generation !== this.audienceGeneration) {
        this.retireSubscription(nextBinding.subscription_id);
        await this.unsubscribeRetiring(nextBinding.subscription_id);
        throw new StreamProtocolError("Audience switch was superseded by a newer selection");
      }
      this.activeBinding = structuredClone(nextBinding);
      this.streamScopeBinding = structuredClone(nextBinding);
      this.updateStatus("connected", `Run output audience changed to ${nextBinding.audience}.`);
    });
    this.audienceSwitchTail = operation.then(() => undefined, () => undefined);
    return operation;
  }

  close(): void {
    const socket = this.socket;
    this.socket = null;
    this.rejectPending(new StreamProtocolError("Stream client closed"));
    socket?.close(1000, "client closed");
    this.updateStatus("disconnected", "Run output stream closed.");
  }
}

function freezeOutput(output: SimulationOutputView): SimulationOutputView {
  const copy = structuredClone(output);
  const visit = (value: unknown): void => {
    if (value && typeof value === "object" && !Object.isFrozen(value)) {
      Object.freeze(value);
      for (const child of Object.values(value as Record<string, unknown>)) visit(child);
    }
  };
  visit(copy);
  return copy;
}

function freezeMarker(marker: TimelineMarker): TimelineMarker {
  return Object.freeze(structuredClone(marker));
}
