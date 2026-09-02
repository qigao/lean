import type { JsonValue, OperationIntent } from "../schema/studio-types";

export interface InspectorSelection {
  label: string;
  documentRole: string;
  logicalId: string | null;
  pointer: string;
  value: JsonValue;
}

function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character] ?? character);
}

export class PropertyInspectorElement extends HTMLElement {
  private selected: InspectorSelection | null = null;

  set selection(value: InspectorSelection | null) {
    this.selected = value;
    if (this.isConnected) this.render();
  }

  get selection(): InspectorSelection | null { return this.selected; }

  connectedCallback(): void { this.render(); }

  focusValue(): void { this.querySelector<HTMLElement>("#inspector-value")?.focus(); }

  private bind(): void {
    this.querySelector<HTMLFormElement>("form")?.addEventListener("submit", (event) => {
      event.preventDefault();
      const textarea = this.querySelector<HTMLTextAreaElement>("#inspector-value");
      const error = this.querySelector<HTMLElement>("#inspector-error");
      if (!textarea || !this.selected) return;
      try {
        const value = JSON.parse(textarea.value) as JsonValue;
        textarea.removeAttribute("aria-invalid");
        textarea.removeAttribute("aria-describedby");
        if (error) error.textContent = "";
        const intent: OperationIntent = {
          document_role: this.selected.documentRole,
          logical_id: this.selected.logicalId,
          kind: "set_value",
          pointer: this.selected.pointer,
          value,
        };
        this.dispatchEvent(new CustomEvent<OperationIntent>("studio-operation", {
          bubbles: true,
          composed: true,
          detail: intent,
        }));
      } catch {
        textarea.setAttribute("aria-invalid", "true");
        textarea.setAttribute("aria-describedby", "inspector-error");
        if (error) error.textContent = "Enter a valid JSON value.";
        textarea.focus();
      }
    });
  }

  render(): void {
    const body = this.selected ? `
      <p class="selection-label">${escapeHtml(this.selected.label)}</p>
      <dl><dt>Document</dt><dd>${escapeHtml(this.selected.documentRole)}</dd><dt>Pointer</dt><dd><code>${escapeHtml(this.selected.pointer || "/")}</code></dd></dl>
      <form aria-label="Edit selected property" novalidate>
        <label for="inspector-value">JSON value</label>
        <textarea id="inspector-value" name="value" rows="5">${escapeHtml(JSON.stringify(this.selected.value, null, 2))}</textarea>
        <p id="inspector-error" class="field-error"></p>
        <button type="submit">Apply property</button>
      </form>
    ` : "<p>Select an item to inspect its properties.</p>";
    this.innerHTML = `<section class="panel inspector" role="region" aria-label="Property inspector"><details open><summary><h2>Inspector</h2></summary>${body}</details></section>`;
    this.bind();
  }
}

if (!customElements.get("property-inspector")) customElements.define("property-inspector", PropertyInspectorElement);
