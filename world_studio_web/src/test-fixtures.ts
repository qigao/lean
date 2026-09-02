import type {
  DiagnosticReport,
  JsonObject,
  ProjectSnapshot,
} from "./schema/studio-types";

const HASH_A = `sha256:${"a".repeat(64)}`;
const HASH_B = `sha256:${"b".repeat(64)}`;
const HASH_C = `sha256:${"c".repeat(64)}`;
const HASH_D = `sha256:${"d".repeat(64)}`;

export function snapshotFixture(revision = 2, overrides: Partial<ProjectSnapshot> = {}): ProjectSnapshot {
  return {
    schema: "narrative-dynamics.scenario-draft-snapshot/v1",
    project_id: "law-firm",
    revision,
    scenario_id: "law-firm-case",
    version: "1",
    documents: [
      {
        role: "physical.world",
        logical_id: null,
        value: {
          model_id: "law-firm-world",
          places: [
            { place_id: "lobby", label: "Public lobby" },
            { place_id: "meeting", label: "Meeting room" },
            { place_id: "archive", label: "Restricted archive" },
          ],
          passages: [
            { passage_id: "lobby-meeting", source_place_id: "lobby", target_place_id: "meeting", initially_open: true },
            { passage_id: "meeting-archive", source_place_id: "meeting", target_place_id: "archive", initially_open: false },
          ],
          objects: [
            { object_id: "case-file", kind: "document", initial_place_id: "archive", portable: true },
          ],
          version: "1",
        },
        content_hash: HASH_A,
      },
    ],
    document_semantic_hash: HASH_A,
    layout: {},
    layout_hash: HASH_B,
    diagnostic_report_hash: HASH_C,
    compiled_scenario_hash: HASH_D,
    content_hash: revision === 2 ? HASH_A : HASH_B,
    ...overrides,
  };
}

export function reportFixture(revision = 3, diagnostics: DiagnosticReport["diagnostics"] = []): DiagnosticReport {
  return {
    schema: "narrative-dynamics.scenario-diagnostic-report/v1",
    project_id: "law-firm",
    revision,
    diagnostics,
    content_hash: HASH_C,
  };
}

export function documentValue(snapshot: ProjectSnapshot, role: string): JsonObject {
  const value = snapshot.documents.find((document) => document.role === role)?.value;
  if (!value || Array.isArray(value) || typeof value !== "object") throw new Error(`Missing fixture document: ${role}`);
  return value;
}
