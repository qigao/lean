import "../editors/graph-editor";
import "../editors/json-editor";
import "../editors/map-editor";
import { projectGraph, type GraphEditorElement, type GraphMode } from "../editors/graph-editor";
import type { JsonEditorElement } from "../editors/json-editor";
import type { MapEditorElement } from "../editors/map-editor";
import { JsonRpcError } from "../rpc/client";
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
import type { RunStore } from "../state/run-store";
import "./fork-comparison";
import type { ForkComparisonElement } from "./fork-comparison";
import "./output-timeline";
import type { OutputTimelineElement } from "./output-timeline";
import "./run-toolbar";
import type { RunToolbarElement } from "./run-toolbar";
import "./state-inspector";
import type { StateInspectorElement } from "./state-inspector";
import "./studio-workflow";
import type { StudioWorkflowElement } from "./studio-workflow";
import type { StreamClient } from "../rpc/stream-client";

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
  private authoritativeRuns: RunStore | null = null;
  private selectedAuthoritativeRunId = "";
  private outputStream: StreamClient | null = null;
  private eventsBound = false;
  private readonly onRunChange = () => this.syncRunConsole();

  set runStore(value: RunStore | null) {
    this.authoritativeRuns?.removeEventListener("change", this.onRunChange);
    this.authoritativeRuns = value;
    value?.addEventListener("change", this.onRunChange);
    if (this.isConnected) this.syncRunConsole();
  }

  get runStore(): RunStore | null { return this.authoritativeRuns; }

  set selectedRunId(value: string) {
    this.selectedAuthoritativeRunId = value;
    if (this.isConnected) this.syncRunConsole();
  }

  get selectedRunId(): string { return this.selectedAuthoritativeRunId; }

  set streamClient(value: StreamClient | null) {
    this.outputStream = value;
    if (this.isConnected) this.syncRunConsole();
  }
  get streamClient(): StreamClient | null { return this.outputStream; }

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
    this.authoritativeRuns?.addEventListener("change", this.onRunChange);
    if (!this.eventsBound) this.bindShellEvents();
    this.render();
  }

  disconnectedCallback(): void {
    this.authoritativeRuns?.removeEventListener("change", this.onRunChange);
  }

  private syncRunConsole(): void {
    const runId = this.selectedAuthoritativeRunId || this.authoritativeRuns?.runs()[0]?.run_id || "";
    const toolbar = this.querySelector("run-toolbar") as RunToolbarElement | null;
    if (toolbar) {
      if (toolbar.store !== this.authoritativeRuns) toolbar.store = this.authoritativeRuns;
      if (toolbar.streamClient !== this.outputStream) toolbar.streamClient = this.outputStream;
      toolbar.runId = runId;
    }
    const inspector = this.querySelector("state-inspector") as StateInspectorElement | null;
    if (inspector) {
      if (inspector.store !== this.authoritativeRuns) inspector.store = this.authoritativeRuns;
      inspector.streamClient = this.outputStream;
      inspector.runId = runId;
    }
    const timeline = this.querySelector("output-timeline") as OutputTimelineElement | null;
    if (timeline) {
      timeline.batches = this.authoritativeRuns?.outputs(runId) ?? [];
      timeline.markers = this.authoritativeRuns?.timelineMarkers(runId) ?? [];
    }
    const placeholder = this.querySelector<HTMLElement>(".timeline-placeholder");
    if (placeholder) placeholder.hidden = !!this.authoritativeRuns;

    const selected = this.authoritativeRuns?.run(runId) ?? null;
    const parent = selected?.parent_checkpoint_hash
      ? this.authoritativeRuns?.runs().find((candidate) => candidate.checkpoint_hashes.includes(selected.parent_checkpoint_hash!)) ?? selected
      : selected;
    const child = parent ? this.authoritativeRuns?.runs().find((candidate) =>
      candidate.run_id !== parent.run_id && candidate.parent_checkpoint_hash !== null &&
      parent.checkpoint_hashes.includes(candidate.parent_checkpoint_hash)) ?? null : null;
    const comparison = this.querySelector("fork-comparison") as ForkComparisonElement | null;
    if (comparison) {
      comparison.parent = parent;
      comparison.child = child;
      comparison.parentMetrics = this.networkMetrics(runId);
      comparison.childMetrics = child ? this.networkMetrics(child.run_id) : null;
    }
    const announcementList = this.querySelector<HTMLOListElement>(".run-announcements");
    if (announcementList) {
      announcementList.replaceChildren();
      for (const message of this.authoritativeRuns?.announcements ?? []) {
        const item = document.createElement("li");
        item.textContent = message;
        announcementList.append(item);
      }
    }
    const workflow = this.querySelector("studio-workflow") as StudioWorkflowElement | null;
    if (workflow) {
      if (workflow.runStore !== this.authoritativeRuns) workflow.runStore = this.authoritativeRuns;
      workflow.streamClient = this.outputStream;
    }
  }

  private networkMetrics(runId: string) {
    const records = (this.authoritativeRuns?.outputs(runId) ?? []).flatMap((batch) => batch.records);
    const record = records.reverse().find((candidate) => candidate.kind === "network.metrics");
    return record?.kind === "network.metrics" ? record.payload.metrics : null;
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
      if (error instanceof JsonRpcError && error.code === -32011) {
        this.connectionMessage("Project conflict detected. The latest server revision is now authoritative.");
      } else if (error instanceof JsonRpcError) {
        this.connectionMessage(`Project operation was rejected by the server (RPC ${error.code}). The last accepted snapshot remains authoritative.`);
      } else {
        this.connectionMessage("Project operation failed. The last accepted snapshot remains authoritative.");
      }
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
    this.addEventListener("studio-map-select", (event) => {
      const detail = (event as CustomEvent<{ id: string; pointer: string }>).detail;
      const document = this.projectSnapshot?.documents.find((item) =>
        item.role === "physical.map" && item.logical_id === null);
      if (!document || !detail?.pointer) return;
      this.inspectorSelection = {
        label: detail.id,
        documentRole: document.role,
        logicalId: document.logical_id,
        pointer: detail.pointer,
        value: valueAtPointer(document.value, detail.pointer),
      };
      const inspector = this.querySelector("property-inspector") as PropertyInspectorElement | null;
      if (inspector) inspector.selection = this.inspectorSelection;
    });
    this.addEventListener("studio-run-select", (event) => {
      const runId = (event as CustomEvent<{ runId: string }>).detail?.runId;
      if (runId && this.authoritativeRuns?.run(runId)) this.selectedRunId = runId;
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
    if (!this.querySelector(".connection-status")) {
      this.innerHTML = `
        <a class="skip-link" href="#studio-main">Skip to content</a>
        <header class="studio-header">
          <div><p class="eyebrow">World Studio</p><h1>Scenario authoring</h1></div>
          <run-toolbar></run-toolbar>
          <p class="connection-status" role="status" aria-live="polite">Editor shell ready.</p>
          <ol class="run-announcements visually-hidden" role="log" aria-live="polite" aria-relevant="additions text"></ol>
        </header>
        <project-tree></project-tree>
        <main id="studio-main" tabindex="-1">
          <studio-workflow></studio-workflow>
          <section class="panel primary-editor" role="region" aria-label="Primary editor">
            <h2>Scenario editor</h2>
            <div class="mode-tabs" role="tablist" aria-label="Editor modes">${tabs}</div>
            ${panels}
          </section>
          <property-inspector></property-inspector>
          <diagnostic-list></diagnostic-list>
          <state-inspector></state-inspector>
          <section class="panel timeline" role="region" aria-label="Timeline">
            <details open><summary><h2>Timeline</h2></summary><p class="timeline-placeholder">Run events will appear here when the live console is connected.</p><output-timeline></output-timeline></details>
          </section>
          <section class="panel fork-panel" role="region" aria-label="Fork comparison"><h2>Fork comparison</h2><fork-comparison></fork-comparison></section>
        </main>
      `;
      this.bindTabs();
    }
    const tree = this.querySelector("project-tree") as ProjectTreeElement | null;
    if (tree) { tree.projectId = this.projectId; tree.snapshot = this.projectSnapshot; }
    const inspector = this.querySelector("property-inspector") as PropertyInspectorElement | null;
    if (inspector) inspector.selection = this.inspectorSelection;
    const diagnostics = this.querySelector("diagnostic-list") as DiagnosticListElement | null;
    if (diagnostics) diagnostics.report = this.diagnosticReport;
    this.renderActiveEditor();
    this.syncRunConsole();
  }
}

if (!customElements.get("studio-shell")) customElements.define("studio-shell", StudioShellElement);
