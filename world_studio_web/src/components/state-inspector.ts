import type { RunStore } from "../state/run-store";
import type { StreamClient } from "../rpc/stream-client";
import type { SimulationOutputKind } from "../schema/studio-types";

const OUTPUT_KINDS: SimulationOutputKind[] = [
  "state.delta", "event.objective", "percept.private", "agent.decision",
  "memory.update", "social.update", "network.metrics", "story.progress",
  "narrative.scene", "blender.delta", "command.result", "diagnostic",
];

export class StateInspectorElement extends HTMLElement {
  private runStore: RunStore | null = null;
  private selectedRunId = "";
  private initialized = false;
  streamClient: StreamClient | null = null;
  private readonly onStoreChange = () => this.update();

  set store(value: RunStore | null) {
    this.runStore?.removeEventListener("change", this.onStoreChange);
    this.runStore = value;
    value?.addEventListener("change", this.onStoreChange);
    if (this.isConnected) this.update();
  }
  get store(): RunStore | null { return this.runStore; }
  set runId(value: string) { this.selectedRunId = value; if (this.isConnected) this.update(); }
  get runId(): string { return this.selectedRunId; }

  connectedCallback(): void {
    this.runStore?.addEventListener("change", this.onStoreChange);
    this.ensureDom();
    this.update();
  }
  disconnectedCallback(): void { this.runStore?.removeEventListener("change", this.onStoreChange); }

  private ensureDom(): void {
    if (this.initialized) return;
    this.initialized = true;
    this.innerHTML = `
      <section class="panel run-state" role="region" aria-label="Run state inspector" aria-busy="false">
        <h2>Run state</h2>
        <dl class="run-facts"></dl>
        <label for="run-audience">State audience</label>
        <select id="run-audience" name="run-audience"></select>
        <p class="state-status" role="status" aria-live="polite">Choose an audience to load scoped state.</p>
        <pre class="scoped-state" tabindex="0"></pre>
      </section>
    `;
    this.querySelector<HTMLSelectElement>("select")!.addEventListener("change", (event) => {
      void this.select((event.currentTarget as HTMLSelectElement).value);
    });
  }

  private update(): void {
    this.ensureDom();
    const run = this.runStore?.run(this.selectedRunId) ?? null;
    const facts = this.querySelector<HTMLElement>(".run-facts")!;
    facts.replaceChildren();
    const rows: [string, string][] = run ? [
      ["Run ID", run.run_id], ["Status", run.status], ["Round", String(run.round_index)],
      ["Epoch", String(run.coordinator_epoch)], ["State hash", run.state_hash],
      ["Next sequence", String(run.next_sequence)],
    ] : [["Run", "No run selected"]];
    for (const [label, value] of rows) {
      const term = document.createElement("dt"); term.textContent = label;
      const detail = document.createElement("dd"); detail.textContent = value;
      facts.append(term, detail);
    }
    const select = this.querySelector<HTMLSelectElement>("select")!;
    const prior = select.value;
    select.replaceChildren();
    const choices: [string, string][] = [];
    if (this.runStore?.permissions.includes("state.public")) choices.push(["public", "Public"]);
    if (this.runStore?.permissions.includes("state.agent")) {
      for (const agentId of this.runStore.authorizedAgentIds) choices.push([`agent:${agentId}`, `Agent: ${agentId}`]);
    }
    if (this.runStore?.permissions.includes("state.network")) choices.push(["network", "Network metrics"]);
    for (const [value, label] of choices) {
      const option = document.createElement("option"); option.value = value; option.textContent = label;
      select.append(option);
    }
    if (choices.some(([value]) => value === prior)) select.value = prior;
    select.disabled = !run || choices.length === 0;
    const scoped = this.runStore?.scopedState(this.selectedRunId);
    this.querySelector<HTMLElement>(".scoped-state")!.textContent = scoped
      ? JSON.stringify(scoped.value, null, 2)
      : "No scoped state loaded.";
  }

  private async select(value: string): Promise<void> {
    if (!this.runStore || !this.selectedRunId) return;
    const section = this.querySelector<HTMLElement>(".run-state")!;
    const status = this.querySelector<HTMLElement>(".state-status")!;
    const [prefix, owner] = value.split(":", 2);
    const audience = prefix === "agent" ? "agent" : prefix === "network" ? "network" : "public";
    section.setAttribute("aria-busy", "true");
    status.textContent = `Loading ${audience} state.`;
    try {
      const run = this.runStore.run(this.selectedRunId);
      if (this.streamClient && run) {
        await this.streamClient.switchAudience({
          subscription_id: `subscription-${run.run_id}-${audience}${owner ? `-${owner}` : ""}`,
          run_id: run.run_id,
          stream_id: run.stream_id,
          scenario_hash: run.scenario_hash,
          kinds: [...OUTPUT_KINDS],
          audience,
          owner_agent_id: audience === "agent" ? owner ?? null : null,
        }, () => this.runStore?.clearAudienceData(this.selectedRunId));
      }
      await this.runStore.selectAudience(this.selectedRunId, {
        audience,
        owner_agent_id: audience === "agent" ? owner ?? null : null,
      });
      status.textContent = `${audience} state loaded.`;
    } catch {
      status.textContent = "Scoped state could not be loaded.";
    } finally {
      section.setAttribute("aria-busy", "false");
      this.update();
    }
  }
}

if (!customElements.get("state-inspector")) customElements.define("state-inspector", StateInspectorElement);
