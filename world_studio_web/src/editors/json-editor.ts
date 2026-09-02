import type {
  DiagnosticReport,
  DraftDocument,
  JsonObject,
  OperationIntent,
} from "../schema/studio-types";

export interface JsonDiagnosticMarker {
  severity: "error" | "warning";
  code: string;
  message: string;
  pointer: string;
}

export interface JsonEditorAdapter {
  dispose: () => void;
  focusPointer: (pointer: string) => void;
}

export type JsonEditorLoader = (
  container: HTMLElement,
  text: string,
  markers: JsonDiagnosticMarker[],
  onChange: (text: string) => void,
) => Promise<JsonEditorAdapter>;

export function replaceDocumentIntent(document: DraftDocument, value: JsonObject): OperationIntent {
  return {
    document_role: document.role,
    logical_id: document.logical_id,
    kind: "replace_document",
    pointer: "",
    value,
  };
}

export function markersForDocument(report: DiagnosticReport | null, document: DraftDocument): JsonDiagnosticMarker[] {
  if (!report) return [];
  return report.diagnostics
    .filter((diagnostic) => diagnostic.document_role === document.role && diagnostic.logical_id === document.logical_id)
    .map((diagnostic) => ({
      severity: diagnostic.severity,
      code: diagnostic.code,
      message: diagnostic.message_key,
      pointer: diagnostic.pointer,
    }));
}

function prettyJson(value: JsonObject): string {
  return JSON.stringify(value, null, 2);
}

function pointerToken(pointer: string): string {
  const token = pointer.split("/").at(-1) ?? "";
  return token.replaceAll("~1", "/").replaceAll("~0", "~");
}

function tokenRange(source: string, pointer: string): [number, number] {
  const token = JSON.stringify(pointerToken(pointer));
  const start = Math.max(0, source.indexOf(token));
  return [start, Math.min(source.length, start + Math.max(1, token.length))];
}

function lineAndColumn(source: string, offset: number): { line: number; column: number } {
  const before = source.slice(0, offset).split("\n");
  return { line: before.length, column: (before.at(-1)?.length ?? 0) + 1 };
}

async function loadMonacoEditor(
  container: HTMLElement,
  source: string,
  markers: JsonDiagnosticMarker[],
  onChange: (text: string) => void,
): Promise<JsonEditorAdapter> {
  const [monaco] = await Promise.all([
    import("monaco-editor/editor/editor.api.js"),
    import("monaco-editor/language/json/monaco.contribution.js"),
  ]);
  const model = monaco.editor.createModel(source, "json");
  const editor = monaco.editor.create(container, {
    model,
    automaticLayout: true,
    minimap: { enabled: false },
    scrollBeyondLastLine: false,
  });
  monaco.editor.setModelMarkers(model, "world-studio-server", markers.map((marker) => {
    const [startOffset, endOffset] = tokenRange(source, marker.pointer);
    const start = lineAndColumn(source, startOffset);
    const end = lineAndColumn(source, endOffset);
    return {
      severity: marker.severity === "error" ? monaco.MarkerSeverity.Error : monaco.MarkerSeverity.Warning,
      message: marker.message,
      code: marker.code,
      startLineNumber: start.line,
      startColumn: start.column,
      endLineNumber: end.line,
      endColumn: end.column,
    };
  }));
  const change = model.onDidChangeContent(() => onChange(model.getValue()));
  return {
    dispose: () => {
      change.dispose();
      editor.dispose();
      model.dispose();
    },
    focusPointer: (pointer) => {
      const [startOffset, endOffset] = tokenRange(model.getValue(), pointer);
      const start = model.getPositionAt(startOffset);
      const end = model.getPositionAt(endOffset);
      editor.setSelection({
        startLineNumber: start.lineNumber,
        startColumn: start.column,
        endLineNumber: end.lineNumber,
        endColumn: end.column,
      });
      editor.revealPositionInCenter(start);
      editor.focus();
    },
  };
}

export class JsonEditorElement extends HTMLElement {
  private selectedDocument: DraftDocument | null = null;
  private report: DiagnosticReport | null = null;
  private enabled = false;
  private adapter: JsonEditorAdapter | undefined;
  editorLoader: JsonEditorLoader = loadMonacoEditor;

  set document(value: DraftDocument | null) {
    this.selectedDocument = value;
    if (this.isConnected) this.render();
  }

  get document(): DraftDocument | null {
    return this.selectedDocument;
  }

  set diagnostics(value: DiagnosticReport | null) {
    this.report = value;
    if (this.isConnected) this.render();
  }

  get diagnostics(): DiagnosticReport | null {
    return this.report;
  }

  set active(value: boolean) {
    this.enabled = value;
    if (this.isConnected && value) void this.initializeEditor();
  }

  get active(): boolean {
    return this.enabled;
  }

  connectedCallback(): void {
    this.render();
  }

  disconnectedCallback(): void {
    this.adapter?.dispose();
  }

  focusPointer(pointer: string): void {
    if (this.adapter) {
      this.adapter.focusPointer(pointer);
      return;
    }
    const textarea = this.querySelector<HTMLTextAreaElement>("#document-json");
    if (!textarea) return;
    const [start, end] = tokenRange(textarea.value, pointer);
    textarea.dataset.pointer = pointer;
    textarea.focus();
    textarea.setSelectionRange(start, end);
  }

  private async initializeEditor(): Promise<void> {
    const container = this.querySelector<HTMLElement>(".monaco-host");
    const textarea = this.querySelector<HTMLTextAreaElement>("#document-json");
    const status = this.querySelector<HTMLElement>(".editor-status");
    if (!container || !textarea || !status || !this.selectedDocument || !this.enabled || this.adapter) return;
    if (this.editorLoader === loadMonacoEditor && navigator.userAgent.includes("jsdom")) {
      status.textContent = "Enhanced JSON editor unavailable. The text fallback remains available.";
      return;
    }
    status.textContent = "Loading enhanced JSON editor.";
    try {
      this.adapter = await this.editorLoader(
        container,
        textarea.value,
        markersForDocument(this.report, this.selectedDocument),
        (text) => { textarea.value = text; },
      );
      status.textContent = "Enhanced JSON editor ready. The text fallback remains available.";
    } catch {
      status.textContent = "Enhanced JSON editor unavailable. The text fallback remains available.";
    }
  }

  private bind(): void {
    this.querySelector<HTMLFormElement>("form")?.addEventListener("submit", (event) => {
      event.preventDefault();
      const textarea = this.querySelector<HTMLTextAreaElement>("#document-json");
      const error = this.querySelector<HTMLElement>("#json-error");
      if (!textarea || !this.selectedDocument) return;
      try {
        const parsed: unknown = JSON.parse(textarea.value);
        if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("root");
        textarea.removeAttribute("aria-invalid");
        textarea.removeAttribute("aria-describedby");
        if (error) error.textContent = "";
        this.dispatchEvent(new CustomEvent<OperationIntent>("studio-operation", {
          bubbles: true,
          composed: true,
          detail: replaceDocumentIntent(this.selectedDocument, parsed as JsonObject),
        }));
      } catch {
        textarea.setAttribute("aria-invalid", "true");
        textarea.setAttribute("aria-describedby", "json-error");
        if (error) error.textContent = "Enter a valid JSON object.";
        textarea.focus();
      }
    });
  }

  render(): void {
    this.adapter?.dispose();
    this.adapter = undefined;
    const source = this.selectedDocument ? prettyJson(this.selectedDocument.value) : "{}";
    this.innerHTML = `
      <section class="json-editor" role="region" aria-label="Raw JSON editor">
        <div class="monaco-host" aria-label="Enhanced JSON editor"></div>
        <p class="editor-status" role="status" aria-live="polite">Text JSON editor ready.</p>
        <form aria-label="Edit raw JSON" novalidate>
          <label for="document-json">Document JSON</label>
          <textarea id="document-json" name="document_json" rows="20" spellcheck="false">${source.replaceAll("&", "&amp;").replaceAll("<", "&lt;")}</textarea>
          <p id="json-error" class="field-error"></p>
          <button type="submit">Apply JSON replacement</button>
        </form>
      </section>
    `;
    this.bind();
    if (this.enabled) void this.initializeEditor();
  }
}

if (!customElements.get("json-editor")) {
  customElements.define("json-editor", JsonEditorElement);
}
