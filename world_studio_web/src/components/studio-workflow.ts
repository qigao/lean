import { JsonRpcError, JsonRpcProtocolError, type RpcCaller } from "../rpc/client";
import type { StreamClient } from "../rpc/stream-client";
import type {
  ProjectSnapshot,
  ScenarioCompileResult,
  SimulationOutputKind,
  StreamBinding,
} from "../schema/studio-types";
import type { ProjectStore } from "../state/project-store";
import type { RunStore } from "../state/run-store";

const OUTPUT_KINDS: SimulationOutputKind[] = [
  "state.delta", "event.objective", "percept.private", "agent.decision",
  "memory.update", "social.update", "network.metrics", "story.progress",
  "narrative.scene", "blender.delta", "command.result", "diagnostic",
];

export class StudioWorkflowElement extends HTMLElement {
  rpc: RpcCaller | null = null;
  projectStore: ProjectStore | null = null;
  streamClient: StreamClient | null = null;
  private runs: RunStore | null = null;
  private initialized = false;
  private selectedRun = "";
  private readonly onRunChange = () => this.updateRuns();

  set runStore(value: RunStore | null) {
    this.runs?.removeEventListener("change", this.onRunChange);
    this.runs = value;
    value?.addEventListener("change", this.onRunChange);
    if (this.isConnected) this.updateRuns();
  }
  get runStore(): RunStore | null { return this.runs; }

  connectedCallback(): void {
    this.runs?.addEventListener("change", this.onRunChange);
    this.ensureDom();
    this.updateRuns();
  }

  disconnectedCallback(): void { this.runs?.removeEventListener("change", this.onRunChange); }

  private ensureDom(): void {
    if (this.initialized) return;
    this.initialized = true;
    this.innerHTML = `
      <section class="panel workflow" role="region" aria-label="Studio workflow">
        <h2>Project and run</h2>
        <div class="workflow-grid">
          <label>Project ID<input name="project-id" value="law-firm" autocomplete="off"></label>
          <label>Configured source ID<input name="source-id" value="law-firm-fixture" autocomplete="off"></label>
          <div class="workflow-actions">
            <button type="button" data-action="import">Import project</button>
            <button type="button" data-action="open">Open project</button>
            <button type="button" data-action="compile">Compile revision</button>
          </div>
          <label>Run ID<input name="run-id" value="run-parent" autocomplete="off"></label>
          <label>Stream ID<input name="stream-id" value="stream-parent" autocomplete="off"></label>
          <button type="button" data-action="create-run">Create run</button>
          <label>Selected run<select name="selected-run"></select></label>
        </div>
        <p class="workflow-status" role="status" aria-live="polite">Ready for an explicit project action.</p>
      </section>
    `;
    this.querySelector("[data-action=import]")!.addEventListener("click", () => void this.importProject());
    this.querySelector("[data-action=open]")!.addEventListener("click", () => void this.openProject());
    this.querySelector("[data-action=compile]")!.addEventListener("click", () => void this.compile());
    this.querySelector("[data-action=create-run]")!.addEventListener("click", () => void this.createRun());
    this.querySelector<HTMLSelectElement>("[name=selected-run]")!.addEventListener("change", (event) => {
      this.selectRun((event.currentTarget as HTMLSelectElement).value);
    });
  }

  private value(name: string): string {
    return this.querySelector<HTMLInputElement>(`[name=${name}]`)?.value.trim() ?? "";
  }

  private status(message: string): void {
    this.querySelector<HTMLElement>(".workflow-status")!.textContent = message.slice(0, 240);
  }

  private busy(value: boolean): void {
    this.querySelector<HTMLElement>(".workflow")!.setAttribute("aria-busy", String(value));
    this.querySelectorAll<HTMLButtonElement>("button").forEach((button) => { button.disabled = value; });
  }

  private requireProjectValues(): { projectId: string; sourceId: string } {
    const projectId = this.value("project-id");
    const sourceId = this.value("source-id");
    if (!projectId || !sourceId) throw new Error("Project and configured source IDs are required");
    return { projectId, sourceId };
  }

  private accept(snapshot: ProjectSnapshot): void {
    if (!this.projectStore) throw new Error("Project store is not connected");
    this.projectStore.accept(snapshot);
  }

  private async importProject(): Promise<void> {
    if (!this.rpc) return;
    this.busy(true);
    this.status("Creating and importing the configured project source.");
    try {
      const { projectId, sourceId } = this.requireProjectValues();
      let created: ProjectSnapshot;
      try {
        created = await this.rpc.call<ProjectSnapshot>("project.create", { project_id: projectId }, {
          stateChanging: true, attempts: 1, requestId: `project-create:${projectId}`,
        });
        this.accept(created);
      } catch (createError) {
        if (createError instanceof JsonRpcProtocolError ||
            createError instanceof JsonRpcError && createError.code !== -32015) {
          throw createError;
        }
        created = await this.rpc.call<ProjectSnapshot>("project.snapshot", { project_id: projectId });
        this.accept(created);
      }
      let imported: ProjectSnapshot;
      try {
        imported = await this.rpc.call<ProjectSnapshot>("project.import", {
          project_id: projectId,
          source_id: sourceId,
          expected_revision: created.revision,
          expected_snapshot_hash: created.content_hash,
        }, {
          stateChanging: true, attempts: 1, requestId: `project-import:${projectId}:${created.revision}`,
        });
        this.accept(imported);
      } catch (importError) {
        if (importError instanceof JsonRpcError || importError instanceof JsonRpcProtocolError) {
          throw importError;
        }
        const reconciled = await this.rpc.call<ProjectSnapshot>(
          "project.snapshot", { project_id: projectId },
        );
        this.accept(reconciled);
        this.status(
          `Project ${projectId} import outcome is ambiguous at authoritative revision ` +
          `${reconciled.revision}. Review it or retry import.`,
        );
        return;
      }
      this.accept(imported);
      this.status(`Project ${projectId} imported at revision ${imported.revision}.`);
    } catch {
      this.status("Project import failed. No browser path was sent and the last accepted revision remains authoritative.");
    } finally { this.busy(false); }
  }

  private async openProject(): Promise<void> {
    if (!this.projectStore) return;
    this.busy(true);
    try {
      const projectId = this.value("project-id");
      const snapshot = await this.projectStore.load(projectId);
      this.status(`Project ${projectId} reopened at saved revision ${snapshot.revision}.`);
    } catch {
      this.status("Saved project could not be opened.");
    } finally { this.busy(false); }
  }

  private async compile(): Promise<void> {
    const snapshot = this.projectStore?.snapshot;
    if (!this.rpc || !snapshot) { this.status("Import or open a project before compiling."); return; }
    this.busy(true);
    try {
      const result = await this.rpc.call<ScenarioCompileResult>("scenario.compile", {
        project_id: snapshot.project_id,
        expected_revision: snapshot.revision,
        expected_snapshot_hash: snapshot.content_hash,
      });
      if (result.project_id !== snapshot.project_id || result.revision !== snapshot.revision ||
          result.snapshot_hash !== snapshot.content_hash) throw new Error("Compile result mismatch");
      this.status(`Revision ${result.revision} compiled as ${result.scenario_hash}.`);
    } catch { this.status("Revision compile failed. Repair diagnostics before retrying."); }
    finally { this.busy(false); }
  }

  private binding(runId: string): StreamBinding {
    const run = this.runs?.run(runId);
    if (!run) throw new Error("Created run is unavailable");
    return {
      subscription_id: `subscription-${runId}-public`,
      run_id: run.run_id,
      stream_id: run.stream_id,
      scenario_hash: run.scenario_hash,
      kinds: [...OUTPUT_KINDS],
      audience: "public",
      owner_agent_id: null,
    };
  }

  private async createRun(): Promise<void> {
    const snapshot = this.projectStore?.snapshot;
    if (!this.runs || !snapshot) { this.status("Compile an accepted project before creating a run."); return; }
    this.busy(true);
    try {
      const runId = this.value("run-id");
      const streamId = this.value("stream-id");
      await this.runs.create(snapshot, runId, streamId);
      this.selectRun(runId, false);
      if (this.streamClient) await this.streamClient.connect(this.binding(runId));
      this.status(`Run ${runId} created and its public output stream connected.`);
    } catch { this.status("Run creation or stream connection failed."); }
    finally { this.busy(false); }
  }

  private selectRun(runId: string, reconnect = true): void {
    this.selectedRun = runId;
    const select = this.querySelector<HTMLSelectElement>("[name=selected-run]");
    if (select) select.value = runId;
    this.dispatchEvent(new CustomEvent("studio-run-select", {
      bubbles: true, composed: true, detail: { runId },
    }));
    if (reconnect) void this.changeRunStream(runId);
  }

  private async changeRunStream(runId: string): Promise<void> {
    if (!this.streamClient || !this.runs?.run(runId) || this.streamClient.binding?.run_id === runId) return;
    const priorRunId = this.streamClient.binding?.run_id;
    this.streamClient.close();
    if (priorRunId) this.runs.clearAudienceData(priorRunId);
    try {
      await this.streamClient.connect(this.binding(runId));
      this.runs.announce(`Public output stream connected for ${runId}.`);
    } catch {
      this.runs.announce(`Output stream could not connect for ${runId}.`);
    }
  }

  private updateRuns(): void {
    this.ensureDom();
    const select = this.querySelector<HTMLSelectElement>("[name=selected-run]")!;
    const runs = this.runs?.runs() ?? [];
    const prior = this.selectedRun || select.value;
    select.replaceChildren();
    for (const run of runs) {
      const option = document.createElement("option");
      option.value = run.run_id;
      option.textContent = `${run.run_id} (${run.status})`;
      select.append(option);
    }
    select.disabled = runs.length === 0;
    const selected = runs.some((run) => run.run_id === prior) ? prior : runs[0]?.run_id ?? "";
    if (selected) { select.value = selected; this.selectedRun = selected; }
  }
}

if (!customElements.get("studio-workflow")) customElements.define("studio-workflow", StudioWorkflowElement);
