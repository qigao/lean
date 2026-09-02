import type { DiagnosticReport } from "../schema/studio-types";

function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character] ?? character);
}

function message(value: string): string { return value.replaceAll("_", " "); }

export class DiagnosticListElement extends HTMLElement {
  private diagnosticReport: DiagnosticReport | null = null;

  set report(value: DiagnosticReport | null) {
    this.diagnosticReport = value;
    if (this.isConnected) this.render();
  }

  get report(): DiagnosticReport | null { return this.diagnosticReport; }

  connectedCallback(): void { this.render(); }

  private bind(): void {
    this.querySelectorAll<HTMLButtonElement>("[data-diagnostic-index]").forEach((button) => {
      button.addEventListener("click", () => {
        const diagnostic = this.diagnosticReport?.diagnostics[Number(button.dataset.diagnosticIndex)];
        if (diagnostic) {
          this.dispatchEvent(new CustomEvent("studio-diagnostic-select", {
            bubbles: true,
            composed: true,
            detail: diagnostic,
          }));
        }
      });
    });
  }

  render(): void {
    const diagnostics = this.diagnosticReport?.diagnostics ?? [];
    const errors = diagnostics.filter(({ severity }) => severity === "error").length;
    const warnings = diagnostics.length - errors;
    const count = `${diagnostics.length} diagnostic${diagnostics.length === 1 ? "" : "s"}: ${errors} error${errors === 1 ? "" : "s"}, ${warnings} warning${warnings === 1 ? "" : "s"}.`;
    const items = diagnostics.map((diagnostic, index) => {
      const severity = diagnostic.severity === "error" ? "Error" : "Warning";
      const label = `${severity}: ${message(diagnostic.message_key)}`;
      return `<li><button type="button" data-diagnostic-index="${index}" aria-label="${escapeHtml(label)}"><span class="severity-cue">${severity}</span><span>${escapeHtml(message(diagnostic.message_key))}</span><code>${escapeHtml(diagnostic.pointer || "/")}</code></button></li>`;
    }).join("");
    if (!this.querySelector(".diagnostic-status")) {
      this.innerHTML = `
        <section class="panel diagnostics" role="region" aria-label="Diagnostics">
          <h2>Diagnostics</h2>
          <p class="diagnostic-status" role="status" aria-live="polite"></p>
          <div class="diagnostic-items"></div>
        </section>
      `;
    }
    const status = this.querySelector<HTMLElement>(".diagnostic-status");
    const content = this.querySelector<HTMLElement>(".diagnostic-items");
    if (status) status.textContent = count;
    if (content) content.innerHTML = items ? `<ul>${items}</ul>` : "<p>No diagnostics.</p>";
    this.bind();
  }
}

if (!customElements.get("diagnostic-list")) customElements.define("diagnostic-list", DiagnosticListElement);
