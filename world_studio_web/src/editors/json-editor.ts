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

interface JsonRangeNode {
  type: "object" | "array" | "scalar";
  start: number;
  end: number;
  members?: Array<{ key: string; value: JsonRangeNode }>;
  items?: JsonRangeNode[];
}

class RangeJsonParser {
  private offset = 0;

  constructor(private readonly source: string) {}

  parse(): JsonRangeNode {
    this.skipWhitespace();
    const node = this.parseValue();
    this.skipWhitespace();
    if (this.offset !== this.source.length) throw new SyntaxError("Unexpected trailing JSON content");
    return node;
  }

  private skipWhitespace(): void {
    while (/\s/.test(this.source[this.offset] ?? "")) this.offset += 1;
  }

  private parseValue(): JsonRangeNode {
    this.skipWhitespace();
    const character = this.source[this.offset];
    if (character === "{") return this.parseObject();
    if (character === "[") return this.parseArray();
    if (character === '"') {
      const string = this.parseString();
      return { type: "scalar", start: string.start, end: string.end };
    }
    const start = this.offset;
    const tail = this.source.slice(this.offset);
    const literal = tail.match(/^(?:true|false|null)/)?.[0] ??
      tail.match(/^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/)?.[0];
    if (!literal) throw new SyntaxError("Expected a JSON value");
    this.offset += literal.length;
    return { type: "scalar", start, end: this.offset };
  }

  private parseString(): { start: number; end: number; value: string } {
    const start = this.offset;
    this.offset += 1;
    while (this.offset < this.source.length) {
      const character = this.source[this.offset];
      if (character === "\\") {
        this.offset += 2;
        continue;
      }
      this.offset += 1;
      if (character === '"') {
        const raw = this.source.slice(start, this.offset);
        const value: unknown = JSON.parse(raw);
        if (typeof value !== "string") throw new SyntaxError("Expected a JSON string");
        return { start, end: this.offset, value };
      }
    }
    throw new SyntaxError("Unterminated JSON string");
  }

  private parseObject(): JsonRangeNode {
    const start = this.offset;
    const members: Array<{ key: string; value: JsonRangeNode }> = [];
    this.offset += 1;
    this.skipWhitespace();
    if (this.source[this.offset] === "}") {
      this.offset += 1;
      return { type: "object", start, end: this.offset, members };
    }
    while (this.offset < this.source.length) {
      const key = this.parseString();
      this.skipWhitespace();
      if (this.source[this.offset] !== ":") throw new SyntaxError("Expected an object member separator");
      this.offset += 1;
      const value = this.parseValue();
      members.push({ key: key.value, value });
      this.skipWhitespace();
      if (this.source[this.offset] === "}") {
        this.offset += 1;
        return { type: "object", start, end: this.offset, members };
      }
      if (this.source[this.offset] !== ",") throw new SyntaxError("Expected another object member");
      this.offset += 1;
      this.skipWhitespace();
    }
    throw new SyntaxError("Unterminated JSON object");
  }

  private parseArray(): JsonRangeNode {
    const start = this.offset;
    const items: JsonRangeNode[] = [];
    this.offset += 1;
    this.skipWhitespace();
    if (this.source[this.offset] === "]") {
      this.offset += 1;
      return { type: "array", start, end: this.offset, items };
    }
    while (this.offset < this.source.length) {
      items.push(this.parseValue());
      this.skipWhitespace();
      if (this.source[this.offset] === "]") {
        this.offset += 1;
        return { type: "array", start, end: this.offset, items };
      }
      if (this.source[this.offset] !== ",") throw new SyntaxError("Expected another array item");
      this.offset += 1;
    }
    throw new SyntaxError("Unterminated JSON array");
  }
}

function pointerTokens(pointer: string): string[] | null {
  if (pointer === "") return [];
  if (!pointer.startsWith("/")) return null;
  const tokens: string[] = [];
  for (const encoded of pointer.slice(1).split("/")) {
    let token = "";
    for (let index = 0; index < encoded.length; index += 1) {
      const character = encoded[index];
      if (character !== "~") {
        token += character;
        continue;
      }
      const escape = encoded[index + 1];
      if (escape === "0") token += "~";
      else if (escape === "1") token += "/";
      else return null;
      index += 1;
    }
    tokens.push(token);
  }
  return tokens;
}

function tokenRange(source: string, pointer: string): [number, number] {
  try {
    let node = new RangeJsonParser(source).parse();
    const tokens = pointerTokens(pointer);
    if (tokens === null) throw new SyntaxError("Invalid RFC 6901 pointer");
    for (const token of tokens) {
      if (node.type === "object") {
        const member = node.members?.filter(({ key }) => key === token).at(-1);
        if (!member) throw new SyntaxError("JSON pointer member was not found");
        node = member.value;
      } else if (node.type === "array") {
        if (!/^(?:0|[1-9]\d*)$/.test(token)) throw new SyntaxError("Invalid JSON pointer array index");
        const item = node.items?.[Number(token)];
        if (!item) throw new SyntaxError("JSON pointer array item was not found");
        node = item;
      } else {
        throw new SyntaxError("JSON pointer traversed through a scalar");
      }
    }
    return [node.start, node.end];
  } catch {
    return [0, Math.min(source.length, 1)];
  }
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
        <div class="monaco-host" role="region" aria-label="Enhanced JSON editor"></div>
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
