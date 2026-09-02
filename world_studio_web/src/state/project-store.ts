import {
  JsonRpcError,
  JsonRpcProtocolError,
  type RpcCaller,
} from "../rpc/client";
import type {
  DiagnosticReport,
  JsonObject,
  JsonValue,
  OperationIntent,
  ProjectApplyParams,
  ProjectApplyResult,
  ProjectSnapshot,
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

function isSnapshot(value: unknown): value is ProjectSnapshot {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const snapshot = value as Partial<ProjectSnapshot>;
  return snapshot.schema === "narrative-dynamics.scenario-draft-snapshot/v1" &&
    typeof snapshot.project_id === "string" &&
    Number.isInteger(snapshot.revision) &&
    (snapshot.revision ?? 0) > 0 &&
    typeof snapshot.content_hash === "string" &&
    Array.isArray(snapshot.documents) &&
    !!snapshot.layout && typeof snapshot.layout === "object" && !Array.isArray(snapshot.layout);
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
    if (!isSnapshot(snapshot)) throw new JsonRpcProtocolError("Project snapshot is malformed");
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
          if (!isSnapshot(reloaded)) throw new JsonRpcProtocolError("Reloaded project snapshot is malformed");
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
