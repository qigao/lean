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
