import {
  assertJsonValue,
  JsonRpcError,
  JsonRpcProtocolError,
  type RpcCaller,
} from "../rpc/client";
import type {
  DiagnosticReport,
  DraftDocument,
  JsonObject,
  JsonValue,
  OperationIntent,
  ProjectApplyParams,
  ProjectApplyResult,
  ProjectSnapshot,
  ScenarioDiagnostic,
} from "../schema/studio-types";

export type ProjectStoreStatus = "empty" | "loading" | "ready" | "applying" | "conflict" | "error";

export interface ProjectConflict {
  kind: "stale_state";
  attempted_revision: number;
  current_revision: number;
}

export interface ProjectIdSource {
  operationId: () => string;
  idempotencyKey: () => string;
}

function freezeJson<T>(value: T): T {
  if (value && typeof value === "object" && !Object.isFrozen(value)) {
    Object.freeze(value);
    for (const child of Object.values(value as Record<string, unknown>)) freezeJson(child);
  }
  return value;
}

function defaultIdentity(prefix: string): string {
  return `${prefix}-${crypto.randomUUID()}`;
}

const HASH = /^sha256:[0-9a-f]{64}$/;

function isRecord(value: unknown): value is Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function hasExactKeys(value: Record<string, unknown>, expected: readonly string[]): boolean {
  const keys = Object.keys(value);
  return keys.length === expected.length && expected.every((key) => keys.includes(key));
}

function isText(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}

function isHash(value: unknown): value is string {
  return typeof value === "string" && HASH.test(value);
}

function isPositiveInteger(value: unknown): value is number {
  return Number.isInteger(value) && (value as number) > 0;
}

function assertDocument(value: unknown): asserts value is DraftDocument {
  if (!isRecord(value) || !hasExactKeys(value, ["role", "logical_id", "value", "content_hash"]) ||
      !isText(value.role) || !(value.logical_id === null || isText(value.logical_id)) ||
      !isRecord(value.value) || !isHash(value.content_hash)) {
    throw new JsonRpcProtocolError("Project draft document is malformed");
  }
  assertJsonValue(value.value);
}

function assertDiagnostic(value: unknown): asserts value is ScenarioDiagnostic {
  if (!isRecord(value) || !hasExactKeys(value, [
    "severity", "code", "document_role", "logical_id", "pointer", "related_ids", "message_key",
  ]) || !["error", "warning"].includes(value.severity as string) ||
      !isText(value.code) || !isText(value.document_role) ||
      !(value.logical_id === null || isText(value.logical_id)) || typeof value.pointer !== "string" ||
      !Array.isArray(value.related_ids) || !value.related_ids.every(isText) || !isText(value.message_key)) {
    throw new JsonRpcProtocolError("Project diagnostic is malformed");
  }
}

function assertReport(value: unknown): asserts value is DiagnosticReport {
  if (!isRecord(value) || !hasExactKeys(value, [
    "schema", "project_id", "revision", "diagnostics", "content_hash",
  ]) || value.schema !== "narrative-dynamics.scenario-diagnostic-report/v1" ||
      !isText(value.project_id) || !isPositiveInteger(value.revision) ||
      !Array.isArray(value.diagnostics) || !isHash(value.content_hash)) {
    throw new JsonRpcProtocolError("Project diagnostic report is malformed");
  }
  value.diagnostics.forEach(assertDiagnostic);
}

function assertSnapshot(value: unknown): asserts value is ProjectSnapshot {
  if (!isRecord(value) || !hasExactKeys(value, [
    "schema", "project_id", "revision", "scenario_id", "version", "documents",
    "document_semantic_hash", "layout", "layout_hash", "diagnostic_report_hash",
    "compiled_scenario_hash", "content_hash",
  ]) || value.schema !== "narrative-dynamics.scenario-draft-snapshot/v1" ||
      !isText(value.project_id) || !isPositiveInteger(value.revision) ||
      !(value.scenario_id === null || isText(value.scenario_id)) ||
      !(value.version === null || isText(value.version)) ||
      (value.scenario_id === null) !== (value.version === null) ||
      !Array.isArray(value.documents) || !isHash(value.document_semantic_hash) ||
      !isRecord(value.layout) || !isHash(value.layout_hash) || !isHash(value.diagnostic_report_hash) ||
      !(value.compiled_scenario_hash === null || isHash(value.compiled_scenario_hash)) ||
      !isHash(value.content_hash)) {
    throw new JsonRpcProtocolError("Project snapshot is malformed");
  }
  value.documents.forEach(assertDocument);
  const identities = value.documents.map((document) => `${document.role}\u0000${document.logical_id ?? ""}`);
  if (new Set(identities).size !== identities.length) {
    throw new JsonRpcProtocolError("Project snapshot document identities are duplicated");
  }
  assertJsonValue(value.layout);
}

function assertApplyResult(value: unknown): asserts value is ProjectApplyResult {
  if (!isRecord(value) || !hasExactKeys(value, [
    "operation_hash", "prior_revision", "next_snapshot", "diagnostic_report",
  ]) || !isHash(value.operation_hash) || !isPositiveInteger(value.prior_revision)) {
    throw new JsonRpcProtocolError("Project apply result is malformed");
  }
  assertSnapshot(value.next_snapshot);
  assertReport(value.diagnostic_report);
}

function operationParams(snapshot: ProjectSnapshot, intent: OperationIntent, ids: ProjectIdSource): ProjectApplyParams {
  const params: ProjectApplyParams = {
    operation_id: ids.operationId(),
    idempotency_key: ids.idempotencyKey(),
    project_id: snapshot.project_id,
    expected_revision: snapshot.revision,
    expected_snapshot_hash: snapshot.content_hash,
    document_role: intent.document_role,
    logical_id: intent.logical_id,
    kind: intent.kind,
  };
  if ("pointer" in intent && intent.pointer !== undefined) params.pointer = intent.pointer;
  if ("value" in intent) params.value = intent.value;
  if ("from_pointer" in intent) params.from_pointer = intent.from_pointer;
  return params;
}

export class ProjectStore extends EventTarget {
  private acceptedSnapshot: ProjectSnapshot | null = null;
  private acceptedReport: DiagnosticReport | null = null;
  private state: ProjectStoreStatus = "empty";
  private conflictState: ProjectConflict | null = null;
  private readonly ids: ProjectIdSource;

  constructor(private readonly rpc: RpcCaller, ids?: ProjectIdSource) {
    super();
    this.ids = ids ?? {
      operationId: () => defaultIdentity("operation"),
      idempotencyKey: () => defaultIdentity("operation-key"),
    };
  }

  get snapshot(): ProjectSnapshot | null {
    return this.acceptedSnapshot;
  }

  get diagnosticReport(): DiagnosticReport | null {
    return this.acceptedReport;
  }

  get status(): ProjectStoreStatus {
    return this.state;
  }

  get conflict(): ProjectConflict | null {
    return this.conflictState;
  }

  private announce(): void {
    this.dispatchEvent(new Event("change"));
  }

  accept(snapshot: ProjectSnapshot, report: DiagnosticReport | null = null): void {
    assertSnapshot(snapshot);
    if (report !== null) {
      assertReport(report);
      if (report.project_id !== snapshot.project_id || report.revision !== snapshot.revision ||
          report.content_hash !== snapshot.diagnostic_report_hash) {
        throw new JsonRpcProtocolError("Project diagnostic report does not match the snapshot");
      }
    }
    this.acceptedSnapshot = freezeJson(snapshot);
    this.acceptedReport = report === null ? null : freezeJson(report);
    this.conflictState = null;
    this.state = "ready";
    this.announce();
  }

  async load(projectId: string): Promise<ProjectSnapshot> {
    this.state = "loading";
    this.announce();
    try {
      const snapshot = await this.rpc.call<ProjectSnapshot>("project.snapshot", { project_id: projectId });
      this.accept(snapshot);
      return snapshot;
    } catch (error) {
      this.state = "error";
      this.announce();
      throw error;
    }
  }

  async apply(intent: OperationIntent): Promise<ProjectApplyResult> {
    const prior = this.acceptedSnapshot;
    if (prior === null) throw new Error("Load a project before applying an operation");
    if (this.state === "applying") throw new Error("A project operation is already in flight");
    const params = operationParams(prior, intent, this.ids);
    this.state = "applying";
    this.conflictState = null;
    this.announce();
    try {
      const result = await this.rpc.call<ProjectApplyResult>("project.apply", params as JsonObject, {
        stateChanging: true,
        attempts: 2,
      });
      assertApplyResult(result);
      if (result.prior_revision !== prior.revision ||
          result.next_snapshot.project_id !== prior.project_id ||
          result.next_snapshot.revision !== prior.revision + 1 ||
          result.diagnostic_report.project_id !== prior.project_id ||
          result.diagnostic_report.revision !== result.next_snapshot.revision ||
          result.diagnostic_report.content_hash !== result.next_snapshot.diagnostic_report_hash) {
        throw new JsonRpcProtocolError("Project apply result did not match the requested next revision");
      }
      this.acceptedSnapshot = freezeJson(result.next_snapshot);
      this.acceptedReport = freezeJson(result.diagnostic_report);
      this.state = "ready";
      this.announce();
      return result;
    } catch (error) {
      if (error instanceof JsonRpcError && error.code === -32011) {
        this.acceptedReport = null;
        try {
          const reloaded = await this.rpc.call<ProjectSnapshot>("project.snapshot", { project_id: prior.project_id });
          assertSnapshot(reloaded);
          this.acceptedSnapshot = freezeJson(reloaded);
          this.conflictState = {
            kind: "stale_state",
            attempted_revision: prior.revision,
            current_revision: reloaded.revision,
          };
          this.state = "conflict";
        } catch (reloadError) {
          this.conflictState = null;
          this.state = "error";
          throw reloadError;
        } finally {
          if (this.state === "applying") this.state = "error";
          this.announce();
        }
        throw error;
      }
      this.state = "error";
      this.announce();
      throw error;
    }
  }
}

export function projectOperationIntent(
  documentRole: string,
  logicalId: string | null,
  kind: OperationIntent["kind"],
  pointer: string,
  value?: JsonValue,
): OperationIntent {
  if (kind === "remove_value") return { document_role: documentRole, logical_id: logicalId, kind, pointer };
  if (kind === "move_value") throw new Error("Move operations require an explicit source pointer");
  return { document_role: documentRole, logical_id: logicalId, kind, pointer, value: value ?? null };
}
