import "../editors/graph-editor";
import "../editors/json-editor";
import "../editors/map-editor";
import { projectGraph, type GraphEditorElement, type GraphMode } from "../editors/graph-editor";
import type { JsonEditorElement } from "../editors/json-editor";
import type { MapEditorElement } from "../editors/map-editor";
import type {
  DiagnosticReport,
  DraftDocument,
  JsonValue,
  OperationIntent,
  ProjectApplyResult,
  ProjectSnapshot,
  ScenarioDiagnostic,
} from "../schema/studio-types";
import "./diagnostic-list";
import type { DiagnosticListElement } from "./diagnostic-list";
import "./project-tree";
import type { ProjectTreeElement } from "./project-tree";
import "./property-inspector";
import type { InspectorSelection, PropertyInspectorElement } from "./property-inspector";

const MODES = [
  ["physical", "Physical"],
  ["social", "Social"],
  ["story", "Story"],
  ["resources", "Resources"],
  ["map", "Map"],
  ["json", "Raw JSON"],
] as const;

type StudioMode = (typeof MODES)[number][0];

export interface StudioStoreLike {
  apply(intent: OperationIntent): Promise<ProjectApplyResult | undefined>;
}

function pointerTokens(pointer: string): string[] {
  if (!pointer) return [];
  return pointer.slice(1).split("/").map((token) => token.replaceAll("~1", "/").replaceAll("~0", "~"));
}

function valueAtPointer(root: JsonValue, pointer: string): JsonValue {
  let current = root;
  for (const token of pointerTokens(pointer)) {
    if (Array.isArray(current)) {
      current = current[Number(token)] ?? null;
    } else if (current !== null && typeof current === "object") {
      current = current[token] ?? null;
    } else {
      return null;
    }
  }
  return current;
}

export class StudioShellElement extends HTMLElement {
  projectId = "Untitled project";
  store: StudioStoreLike | null = null;
  private projectSnapshot: ProjectSnapshot | null = null;
  private diagnosticReport: DiagnosticReport | null = null;
  private activeMode: StudioMode = "physical";
  private selectedDocument: DraftDocument | null = null;
  private inspectorSelection: InspectorSelection | null = null;
  private eventsBound = false;

  set snapshot(value: ProjectSnapshot | null) {
    this.projectSnapshot = value;
    if (value) {
      this.projectId = value.project_id;
      const stillPresent = value.documents.find((document) =>
        document.role === this.selectedDocument?.role && document.logical_id === this.selectedDocument.logical_id);
      this.selectedDocument = stillPresent ?? value.documents[0] ?? null;
    }
    if (this.isConnected) this.render();
  }

  get snapshot(): ProjectSnapshot | null { return this.projectSnapshot; }

  set diagnostics(value: DiagnosticReport | null) {
    this.diagnosticReport = value;
    if (this.isConnected) {
      const list = this.querySelector("diagnostic-list") as DiagnosticListElement | null;
      if (list) list.report = value;
    }
  }

  get diagnostics(): DiagnosticReport | null { return this.diagnosticReport; }

  connectedCallback(): void {
    if (!this.eventsBound) this.bindShellEvents();
    this.render();
  }

  private connectionMessage(message: string): void {
    const status = this.querySelector<HTMLElement>(".connection-status");
    if (status) status.textContent = message;
  }

  private async applyOperation(intent: OperationIntent): Promise<void> {
    if (!this.store) {
      this.connectionMessage("No project store is connected; the authoritative snapshot was not changed.");
      return;
    }
    this.connectionMessage("Applying project operation.");
    try {
      const result = await this.store.apply(intent);
      if (result) {
        this.projectSnapshot = result.next_snapshot;
        this.diagnosticReport = result.diagnostic_report;
        this.selectedDocument = result.next_snapshot.documents.find((document) =>
          document.role === this.selectedDocument?.role && document.logical_id === this.selectedDocument.logical_id
        ) ?? result.next_snapshot.documents[0] ?? null;
        this.render();
      }
      this.connectionMessage("Project revision accepted by the server.");
    } catch (error) {
      const conflict = error instanceof Error && error.name === "JsonRpcError";
      this.connectionMessage(conflict
        ? "Project conflict detected. The latest server revision is now authoritative."
        : "Project operation failed. The last accepted snapshot remains authoritative.");
    }
  }

  private bindShellEvents(): void {
    this.eventsBound = true;
    this.addEventListener("studio-operation", (event) => {
      const intent = (event as CustomEvent<OperationIntent>).detail;
      if (intent) void this.applyOperation(intent);
    });
    this.addEventListener("studio-operations", (event) => {
      const intents = (event as CustomEvent<OperationIntent[]>).detail;
      if (!Array.isArray(intents)) return;
      void (async () => {
        for (const intent of intents) await this.applyOperation(intent);
      })();
    });
    this.addEventListener("studio-document-select", (event) => {
      const detail = (event as CustomEvent<{ role: string; logicalId: string | null }>).detail;
      const document = this.projectSnapshot?.documents.find((item) =>
        item.role === detail.role && item.logical_id === detail.logicalId);
      if (!document) return;
      this.selectedDocument = document;
      this.inspectorSelection = {
        label: document.logical_id ? `${document.role}: ${document.logical_id}` : document.role,
        documentRole: document.role,
        logicalId: document.logical_id,
        pointer: "",
        value: document.value,
      };
      this.activateMode("json", false);
      (this.querySelector("property-inspector") as PropertyInspectorElement | null)?.focusValue();
    });
    this.addEventListener("studio-diagnostic-select", (event) => {
      this.revealDiagnostic((event as CustomEvent<ScenarioDiagnostic>).detail);
    });
    this.addEventListener("studio-select", (event) => {
      const detail = (event as CustomEvent<{ mode: GraphMode; id: string }>).detail;
      if (!this.projectSnapshot) return;
      const cell = [
        ...projectGraph(this.projectSnapshot, detail.mode).nodes,
        ...projectGraph(this.projectSnapshot, detail.mode).edges,
      ].find(({ id }) => id === detail.id);
      if (!cell) return;
      const document = this.projectSnapshot.documents.find((item) =>
        item.role === cell.documentRole && item.logical_id === cell.logicalId);
      this.inspectorSelection = {
        label: cell.label,
        documentRole: cell.documentRole,
        logicalId: cell.logicalId,
        pointer: cell.pointer,
        value: document ? valueAtPointer(document.value, cell.pointer) : null,
      };
      const inspector = this.querySelector("property-inspector") as PropertyInspectorElement | null;
      if (inspector) inspector.selection = this.inspectorSelection;
    });
  }

  private revealDiagnostic(diagnostic: ScenarioDiagnostic): void {
    const document = this.projectSnapshot?.documents.find((item) =>
      item.role === diagnostic.document_role && item.logical_id === diagnostic.logical_id) ?? null;
    if (document) this.selectedDocument = document;
    this.inspectorSelection = {
      label: diagnostic.message_key.replaceAll("_", " "),
      documentRole: diagnostic.document_role,
      logicalId: diagnostic.logical_id,
      pointer: diagnostic.pointer,
      value: document ? valueAtPointer(document.value, diagnostic.pointer) : null,
    };
    const targetMode: StudioMode = this.activeMode === "json"
      ? "json"
      : diagnostic.document_role === "physical.map"
        ? "map"
        : diagnostic.document_role.startsWith("social.")
          ? "social"
          : diagnostic.document_role.startsWith("story.")
            ? "story"
            : diagnostic.document_role === "knowledge.catalog" || diagnostic.document_role === "asset.catalog"
              ? "resources"
              : "physical";
    this.activateMode(targetMode, false);
    const inspector = this.querySelector("property-inspector") as PropertyInspectorElement | null;
    if (inspector) inspector.selection = this.inspectorSelection;
    if (targetMode === "json") {
      (this.querySelector("json-editor") as JsonEditorElement | null)?.focusPointer(diagnostic.pointer);
    } else if (targetMode === "map") {
      (this.querySelector("map-editor") as MapEditorElement | null)?.focusPointer(diagnostic.pointer.replace(/\/(x|y)$/, ""));
    } else {
      inspector?.focusValue();
    }
  }

  private activateMode(mode: StudioMode, focus = false): void {
    this.activeMode = mode;
    this.querySelectorAll<HTMLButtonElement>("[role=tab]").forEach((tab) => {
      const selected = tab.dataset.mode === mode;
      tab.setAttribute("aria-selected", String(selected));
      tab.tabIndex = selected ? 0 : -1;
    });
    this.querySelectorAll<HTMLElement>("[role=tabpanel]").forEach((panel) => {
      panel.hidden = panel.id !== `studio-panel-${mode}`;
    });
    this.renderActiveEditor();
    if (focus) this.querySelector<HTMLElement>(`#studio-tab-${mode}`)?.focus();
  }

  private onTabKeyDown(event: KeyboardEvent, index: number): void {
    let nextIndex: number | undefined;
    if (event.key === "ArrowRight") nextIndex = (index + 1) % MODES.length;
    if (event.key === "ArrowLeft") nextIndex = (index - 1 + MODES.length) % MODES.length;
    if (event.key === "Home") nextIndex = 0;
    if (event.key === "End") nextIndex = MODES.length - 1;
    if (event.key === "Enter" || event.key === " ") nextIndex = index;
    if (nextIndex === undefined) return;
    event.preventDefault();
    const next = MODES[nextIndex];
    if (next) this.activateMode(next[0], true);
  }

  private bindTabs(): void {
    this.querySelectorAll<HTMLButtonElement>("[role=tab]").forEach((tab, index) => {
      tab.addEventListener("click", () => this.activateMode(tab.dataset.mode as StudioMode));
      tab.addEventListener("keydown", (event) => this.onTabKeyDown(event, index));
    });
  }

  private renderActiveEditor(): void {
    this.querySelectorAll<HTMLElement>("[role=tabpanel]").forEach((panel) => panel.replaceChildren());
    const panel = this.querySelector<HTMLElement>(`#studio-panel-${this.activeMode}`);
    if (!panel) return;
    if (this.activeMode === "map") {
      const editor = document.createElement("map-editor") as MapEditorElement;
      editor.snapshot = this.projectSnapshot;
      editor.active = true;
      panel.append(editor);
      return;
    }
    if (this.activeMode === "json") {
      const editor = document.createElement("json-editor") as JsonEditorElement;
      editor.document = this.selectedDocument;
      editor.diagnostics = this.diagnosticReport;
      editor.active = true;
      panel.append(editor);
      return;
    }
    const editor = document.createElement("graph-editor") as GraphEditorElement;
    editor.dataset.mode = this.activeMode;
    editor.snapshot = this.projectSnapshot;
    editor.mode = this.activeMode;
    editor.active = true;
    panel.append(editor);
  }

  render(): void {
    const tabs = MODES.map(([mode, label]) => {
      const selected = mode === this.activeMode;
      return `<button id="studio-tab-${mode}" type="button" role="tab" data-mode="${mode}" aria-selected="${selected}" aria-controls="studio-panel-${mode}" tabindex="${selected ? "0" : "-1"}">${label}</button>`;
    }).join("");
    const panels = MODES.map(([mode]) => `<div id="studio-panel-${mode}" role="tabpanel" tabindex="0" aria-labelledby="studio-tab-${mode}"${mode === this.activeMode ? "" : " hidden"}></div>`).join("");
    this.innerHTML = `
      <a class="skip-link" href="#studio-main">Skip to content</a>
      <header class="studio-header">
        <div><p class="eyebrow">World Studio</p><h1>Scenario authoring</h1></div>
        <div class="run-toolbar" role="toolbar" aria-label="Run controls">
          <span class="toolbar-label">Run console</span>
          <button type="button" disabled>Start</button><button type="button" disabled>Step</button>
          <span class="status-cue">Available in the live console</span>
        </div>
        <p class="connection-status" role="status" aria-live="polite">Editor shell ready.</p>
      </header>
      <project-tree></project-tree>
      <main id="studio-main" tabindex="-1">
        <section class="panel primary-editor" role="region" aria-label="Primary editor">
          <h2>Scenario editor</h2>
          <div class="mode-tabs" role="tablist" aria-label="Editor modes">${tabs}</div>
          ${panels}
        </section>
        <property-inspector></property-inspector>
        <diagnostic-list></diagnostic-list>
        <section class="panel timeline" role="region" aria-label="Timeline">
          <details open><summary><h2>Timeline</h2></summary><p>Run events will appear here when the live console is connected.</p></details>
        </section>
      </main>
    `;
    const tree = this.querySelector("project-tree") as ProjectTreeElement | null;
    if (tree) { tree.projectId = this.projectId; tree.snapshot = this.projectSnapshot; }
    const inspector = this.querySelector("property-inspector") as PropertyInspectorElement | null;
    if (inspector) inspector.selection = this.inspectorSelection;
    const diagnostics = this.querySelector("diagnostic-list") as DiagnosticListElement | null;
    if (diagnostics) diagnostics.report = this.diagnosticReport;
    this.bindTabs();
    this.renderActiveEditor();
  }
}

if (!customElements.get("studio-shell")) customElements.define("studio-shell", StudioShellElement);
