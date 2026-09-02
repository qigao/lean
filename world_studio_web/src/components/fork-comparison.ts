import type { JsonObject, ScenarioRunView } from "../schema/studio-types";

export class ForkComparisonElement extends HTMLElement {
  private parentRun: ScenarioRunView | null = null;
  private childRun: ScenarioRunView | null = null;
  private parentMetricValues: JsonObject | null = null;
  private childMetricValues: JsonObject | null = null;
  private initialized = false;

  set parent(value: ScenarioRunView | null) { this.parentRun = value; if (this.isConnected) this.renderValues(); }
  get parent(): ScenarioRunView | null { return this.parentRun; }
  set child(value: ScenarioRunView | null) { this.childRun = value; if (this.isConnected) this.renderValues(); }
  get child(): ScenarioRunView | null { return this.childRun; }
  set parentMetrics(value: JsonObject | null) { this.parentMetricValues = value; if (this.isConnected) this.renderValues(); }
  get parentMetrics(): JsonObject | null { return this.parentMetricValues; }
  set childMetrics(value: JsonObject | null) { this.childMetricValues = value; if (this.isConnected) this.renderValues(); }
  get childMetrics(): JsonObject | null { return this.childMetricValues; }

  connectedCallback(): void {
    if (!this.initialized) {
      this.initialized = true;
      this.innerHTML = `
        <section class="comparison-side" role="region" aria-label="Parent run"><h3>Parent</h3><dl></dl></section>
        <section class="comparison-side" role="region" aria-label="Child run"><h3>Child</h3><dl></dl></section>
      `;
    }
    this.renderValues();
  }

  private renderSide(label: "Parent" | "Child", run: ScenarioRunView | null, metrics: JsonObject | null): void {
    const section = this.querySelector<HTMLElement>(`[aria-label="${label} run"]`)!;
    const list = section.querySelector("dl")!;
    list.replaceChildren();
    const rows: [string, string][] = run ? [
      ["Run ID", run.run_id], ["Parent checkpoint", run.parent_checkpoint_hash ?? "None"],
      ["Status", run.status], ["Round", String(run.round_index)], ["Epoch", String(run.coordinator_epoch)],
      ["State hash", run.state_hash],
    ] : [["Run", "Not selected"]];
    if (metrics) {
      for (const key of Object.keys(metrics).sort()) rows.push([key.replaceAll("_", " "), String(metrics[key])]);
    }
    for (const [name, value] of rows) {
      const term = document.createElement("dt"); term.textContent = name;
      const detail = document.createElement("dd"); detail.textContent = value;
      list.append(term, detail);
    }
  }

  private renderValues(): void {
    if (!this.initialized) return;
    this.renderSide("Parent", this.parentRun, this.parentMetricValues);
    this.renderSide("Child", this.childRun, this.childMetricValues);
  }
}

if (!customElements.get("fork-comparison")) customElements.define("fork-comparison", ForkComparisonElement);
