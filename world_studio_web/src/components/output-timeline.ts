import type { SimulationOutputView } from "../schema/studio-types";

export interface TimelineMarker {
  kind: "gap" | "recovery";
  after_sequence: number;
  before_sequence: number;
  label: string;
}

export class OutputTimelineElement extends HTMLElement {
  private sourceBatches: SimulationOutputView[] = [];
  private sourceMarkers: TimelineMarker[] = [];
  private batchLimit = 128;
  private initialized = false;

  set maximumBatches(value: number) {
    if (!Number.isInteger(value) || value <= 0) throw new TypeError("Timeline batch limit must be positive");
    this.batchLimit = value;
    if (this.isConnected) this.renderItems();
  }
  get maximumBatches(): number { return this.batchLimit; }
  set batches(value: SimulationOutputView[]) { this.sourceBatches = structuredClone(value); if (this.isConnected) this.renderItems(); }
  get batches(): SimulationOutputView[] { return structuredClone(this.sourceBatches); }
  set markers(value: TimelineMarker[]) { this.sourceMarkers = structuredClone(value); if (this.isConnected) this.renderItems(); }
  get markers(): TimelineMarker[] { return structuredClone(this.sourceMarkers); }

  connectedCallback(): void {
    if (!this.initialized) {
      this.initialized = true;
      this.innerHTML = `
        <ol class="output-events" aria-label="Run output records"></ol>
        <p class="timeline-status" role="status" aria-live="polite">No output retained.</p>
      `;
    }
    this.renderItems();
  }

  private renderItems(): void {
    if (!this.initialized) return;
    const ordered = [...this.sourceBatches]
      .sort((left, right) => left.first_sequence - right.first_sequence)
      .slice(-this.batchLimit);
    const firstRetained = ordered[0]?.first_sequence ?? Number.POSITIVE_INFINITY;
    const lastRetained = ordered.at(-1)?.last_sequence ?? 0;
    const markers = this.sourceMarkers.filter((marker) =>
      marker.after_sequence >= firstRetained - 1 && marker.before_sequence <= lastRetained);
    const events: ({ type: "batch"; value: SimulationOutputView } | { type: "marker"; value: TimelineMarker })[] = [];
    for (const batch of ordered) {
      for (const marker of markers.filter((item) => item.after_sequence < batch.first_sequence &&
        !events.some((event) => event.type === "marker" && event.value === item))) {
        events.push({ type: "marker", value: marker });
      }
      events.push({ type: "batch", value: batch });
    }
    const list = this.querySelector<HTMLOListElement>("ol")!;
    list.replaceChildren();
    for (const event of events) {
      const item = document.createElement("li");
      if (event.type === "marker") {
        item.className = `timeline-marker ${event.value.kind}`;
        item.textContent = `Gap ${event.value.after_sequence + 1}–${event.value.before_sequence - 1}. ${event.value.label}`;
      } else {
        const batch = event.value;
        const bounds = batch.first_sequence === batch.last_sequence
          ? `Sequence ${batch.first_sequence}` : `Sequences ${batch.first_sequence}–${batch.last_sequence}`;
        const kinds = batch.records.length
          ? batch.records.map((record) => `${record.kind} (${record.audience}${
            record.owner_agent_id === null ? "" : `, owner ${record.owner_agent_id}`
          })`).join(", ")
          : "No records visible for this audience";
        item.textContent = `${bounds}. ${kinds}. Batch ${batch.source_batch_hash}.`;
      }
      list.append(item);
    }
    this.querySelector<HTMLElement>(".timeline-status")!.textContent = ordered.length
      ? `${ordered.length} retained batches. Latest source sequence ${lastRetained}.`
      : "No output retained.";
  }
}

if (!customElements.get("output-timeline")) customElements.define("output-timeline", OutputTimelineElement);
