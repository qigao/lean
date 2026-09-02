import type { ScenarioCommandKind, ScenarioRunStatus } from "../schema/studio-types";
import type { RunStore } from "../state/run-store";
import type { StreamClient } from "../rpc/stream-client";

const COMMANDS: { kind: ScenarioCommandKind; label: string; statuses: ScenarioRunStatus[] }[] = [
  { kind: "start", label: "Start run", statuses: ["created"] },
  { kind: "pause", label: "Pause run", statuses: ["running"] },
  { kind: "resume", label: "Resume run", statuses: ["paused"] },
  { kind: "step", label: "Step run", statuses: ["running", "paused"] },
  { kind: "stop", label: "Stop run", statuses: ["created", "running", "paused"] },
];

export class RunToolbarElement extends HTMLElement {
  private runStore: RunStore | null = null;
  private selectedRunId = "";
  private initialized = false;
  private stream: StreamClient | null = null;
  private localMessage = "No run selected.";
  private readonly onStoreChange = () => this.update();

  set store(value: RunStore | null) {
    this.runStore?.removeEventListener("change", this.onStoreChange);
    this.runStore = value;
    value?.addEventListener("change", this.onStoreChange);
    if (this.isConnected) this.update();
  }
  get store(): RunStore | null { return this.runStore; }

  set streamClient(value: StreamClient | null) {
    this.stream?.removeEventListener("change", this.onStoreChange);
    this.stream = value;
    value?.addEventListener("change", this.onStoreChange);
    if (this.isConnected) this.update();
  }
  get streamClient(): StreamClient | null { return this.stream; }

  set runId(value: string) { this.selectedRunId = value; if (this.isConnected) this.update(); }
  get runId(): string { return this.selectedRunId; }

  connectedCallback(): void {
    this.runStore?.addEventListener("change", this.onStoreChange);
    this.stream?.addEventListener("change", this.onStoreChange);
    this.ensureDom();
    this.update();
  }
  disconnectedCallback(): void {
    this.runStore?.removeEventListener("change", this.onStoreChange);
    this.stream?.removeEventListener("change", this.onStoreChange);
  }

  private ensureDom(): void {
    if (this.initialized) return;
    this.initialized = true;
    this.setAttribute("role", "toolbar");
    this.setAttribute("aria-label", "Run controls");
    this.innerHTML = `
      <span class="toolbar-label">Run console</span>
      <span class="lifecycle-actions"></span>
      <span class="stream-actions"><button type="button" data-stream-action="disconnect">Disconnect stream</button><button type="button" data-stream-action="reconnect">Reconnect stream</button></span>
      <details class="run-advanced">
        <summary>Checkpoint and fork</summary>
        <div class="run-advanced-fields">
          <label>Checkpoint ID<input name="checkpoint-id" autocomplete="off" spellcheck="false"></label>
          <button class="checkpoint-action" type="button" aria-label="Create checkpoint">Checkpoint</button>
          <label>Child run ID<input name="child-run-id" autocomplete="off" spellcheck="false"></label>
          <label>Child stream ID<input name="child-stream-id" autocomplete="off" spellcheck="false"></label>
          <button class="fork-action" type="button" aria-label="Fork run">Fork</button>
        </div>
      </details>
      <span class="run-command-status" role="status" aria-live="polite"></span>
      <span class="stream-status">Output stream not connected.</span>
    `;
    const actions = this.querySelector(".lifecycle-actions")!;
    for (const command of COMMANDS) {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.command = command.kind;
      button.setAttribute("aria-label", command.label);
      button.textContent = command.label.replace(" run", "");
      button.addEventListener("click", () => void this.execute(command.kind));
      actions.append(button);
    }
    this.querySelector<HTMLButtonElement>(".checkpoint-action")!
      .addEventListener("click", () => void this.checkpoint());
    this.querySelector<HTMLButtonElement>(".fork-action")!
      .addEventListener("click", () => void this.fork());
    this.querySelector<HTMLButtonElement>("[data-stream-action=disconnect]")!
      .addEventListener("click", () => this.disconnectStream());
    this.querySelector<HTMLButtonElement>("[data-stream-action=reconnect]")!
      .addEventListener("click", () => void this.reconnectStream());
  }

  private update(): void {
    this.ensureDom();
    const run = this.runStore?.run(this.selectedRunId) ?? null;
    const busy = !!this.runStore?.isBusy(this.selectedRunId);
    const canCommand = !!this.runStore?.permissions.includes("run.command");
    this.setAttribute("aria-busy", String(busy));
    for (const command of COMMANDS) {
      const button = this.querySelector<HTMLButtonElement>(`[data-command="${command.kind}"]`)!;
      button.disabled = busy || !run || !canCommand || !command.statuses.includes(run.status);
    }
    const canCheckpoint = canCommand && !!run && run.status !== "stopped";
    const checkpoint = this.querySelector<HTMLButtonElement>(".checkpoint-action")!;
    const checkpointInput = this.querySelector<HTMLInputElement>("[name=checkpoint-id]")!;
    checkpoint.hidden = !canCheckpoint;
    checkpointInput.closest("label")!.toggleAttribute("hidden", !canCheckpoint);
    checkpoint.disabled = busy;
    checkpointInput.disabled = busy;

    const canFork = !!run && this.runStore?.permissions.includes("run.fork");
    const fork = this.querySelector<HTMLButtonElement>(".fork-action")!;
    fork.hidden = !canFork;
    fork.disabled = busy;
    for (const name of ["child-run-id", "child-stream-id"]) {
      const input = this.querySelector<HTMLInputElement>(`[name=${name}]`)!;
      input.closest("label")!.toggleAttribute("hidden", !canFork);
      input.disabled = busy;
    }
    const advanced = this.querySelector<HTMLDetailsElement>(".run-advanced")!;
    advanced.hidden = !canCheckpoint && !canFork;
    advanced.setAttribute("aria-hidden", String(!canCheckpoint && !canFork));
    const status = this.querySelector<HTMLElement>(".run-command-status")!;
    status.textContent = busy
      ? `${this.busyLabel()} in progress. Canonical run state has not advanced.`
      : run ? `${run.status}. Round ${run.round_index}.` : this.localMessage;
    const connected = this.stream?.status === "connected" || this.stream?.status === "recovering";
    this.querySelector<HTMLButtonElement>("[data-stream-action=disconnect]")!.disabled = busy || !connected;
    this.querySelector<HTMLButtonElement>("[data-stream-action=reconnect]")!.disabled = busy ||
      this.stream?.status !== "disconnected" || this.stream.binding === null;
    this.querySelector<HTMLElement>(".stream-status")!.textContent = this.stream
      ? `Output stream ${this.stream.status}.` : "Output stream not connected.";
  }

  private busyLabel(): string {
    const active = this.querySelector<HTMLButtonElement>("[data-active=true]");
    if (!active) return "Run command";
    const command = active.dataset.command ?? "command";
    return command.charAt(0).toUpperCase() + command.slice(1);
  }

  private async execute(kind: ScenarioCommandKind, checkpointId?: string): Promise<void> {
    if (!this.runStore || !this.selectedRunId) return;
    const button = this.querySelector<HTMLButtonElement>(`[data-command="${kind}"]`);
    button?.setAttribute("data-active", "true");
    this.update();
    try {
      const result = await this.runStore.command(this.selectedRunId, kind, checkpointId);
      this.localMessage = result.accepted ? `${kind} accepted.` : `${kind} rejected: ${result.reason}.`;
    } catch {
      this.localMessage = `${kind} failed. The last accepted run remains authoritative.`;
    } finally {
      button?.removeAttribute("data-active");
      this.update();
    }
  }

  private async checkpoint(): Promise<void> {
    const input = this.querySelector<HTMLInputElement>("[name=checkpoint-id]")!;
    const checkpointId = input.value.trim();
    if (!checkpointId) {
      this.localMessage = "Enter a checkpoint ID.";
      input.setAttribute("aria-invalid", "true");
      input.focus();
      this.update();
      return;
    }
    input.removeAttribute("aria-invalid");
    await this.execute("checkpoint", checkpointId);
  }

  private async fork(): Promise<void> {
    if (!this.runStore || !this.selectedRunId) return;
    const run = this.runStore.run(this.selectedRunId);
    const checkpoint = run?.checkpoint_hashes.at(-1);
    const child = this.querySelector<HTMLInputElement>("[name=child-run-id]")!;
    const stream = this.querySelector<HTMLInputElement>("[name=child-stream-id]")!;
    if (!checkpoint || !child.value.trim() || !stream.value.trim()) {
      this.localMessage = "A retained checkpoint, child run ID, and child stream ID are required.";
      this.update();
      return;
    }
    try {
      await this.runStore.fork(this.selectedRunId, checkpoint, child.value.trim(), stream.value.trim());
      this.localMessage = `Fork ${child.value.trim()} accepted.`;
    } catch {
      this.localMessage = "Fork failed. The accepted run views remain authoritative.";
    }
    this.update();
  }

  private disconnectStream(): void {
    this.stream?.close();
    this.runStore?.announce("Run output stream was disconnected. Accepted run state is unchanged.");
  }

  private async reconnectStream(): Promise<void> {
    if (!this.stream) return;
    try {
      await this.stream.reconnect();
      this.runStore?.announce("Run output stream reconnected and resumed from its committed cursor.");
    } catch {
      this.runStore?.announce("Run output stream recovery failed.");
    }
  }
}

if (!customElements.get("run-toolbar")) customElements.define("run-toolbar", RunToolbarElement);
