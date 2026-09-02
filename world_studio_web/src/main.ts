import { JsonRpcClient } from "./rpc/client";
import { ProjectStore } from "./state/project-store";
import { RunStore } from "./state/run-store";
import { StreamClient } from "./rpc/stream-client";
import type { RunAuthority, StudioSession } from "./schema/studio-types";
import "./components/studio-shell";
import type { StudioShellElement } from "./components/studio-shell";
import type { StudioWorkflowElement } from "./components/studio-workflow";
import { establishBrowserSession } from "./session-bootstrap";
import "./styles.css";

const shell = document.querySelector<StudioShellElement>("#studio-app");

function authority(value: unknown): RunAuthority {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Session authority is malformed");
  const item = value as Partial<RunAuthority>;
  if (typeof item.authority_id !== "string" || !item.authority_id ||
      !Array.isArray(item.project_ids) || !item.project_ids.every((entry) => typeof entry === "string") ||
      !Array.isArray(item.run_ids) || !item.run_ids.every((entry) => typeof entry === "string") ||
      !Array.isArray(item.agent_ids) || !item.agent_ids.every((entry) => typeof entry === "string") ||
      !Array.isArray(item.permissions) || !item.permissions.every((entry) => typeof entry === "string")) {
    throw new Error("Session authority is malformed");
  }
  return item as RunAuthority;
}

async function bootstrap(shell: StudioShellElement): Promise<void> {
  const rpc = new JsonRpcClient("/rpc");
  const store = new ProjectStore(rpc);
  shell.store = store;
  store.addEventListener("change", () => {
    if (store.snapshot && shell.snapshot !== store.snapshot) shell.snapshot = store.snapshot;
    if (shell.diagnostics !== store.diagnosticReport) shell.diagnostics = store.diagnosticReport;
  });
  const session: StudioSession = await establishBrowserSession(shell);
  const runStore = new RunStore(rpc, authority(session.authority));
  const streamUrl = new URL("/v1/stream", window.location.href);
  streamUrl.protocol = streamUrl.protocol === "https:" ? "wss:" : "ws:";
  const stream = new StreamClient(streamUrl.href, {
    onOutput: async (output) => {
      const run = runStore.runs().find((candidate) => candidate.stream_id === output.stream_id);
      if (!run) throw new Error("Output arrived before its run authority");
      runStore.acceptOutput(run.run_id, output);
    },
    onStatus: (status, message) => runStore.announce(`${status}: ${message}`),
    onMarker: (runId, marker) => runStore.markTimeline(runId, marker),
    recoverGap: async (request) => runStore.selectAudience(request.run_id, {
      audience: request.audience === "analyst" ? "network" : request.audience,
      owner_agent_id: request.owner_agent_id,
    }).then(() => undefined),
  });
  shell.runStore = runStore;
  shell.streamClient = stream;
  const workflow = shell.querySelector("studio-workflow") as StudioWorkflowElement | null;
  if (workflow) {
    workflow.rpc = rpc;
    workflow.projectStore = store;
    workflow.runStore = runStore;
    workflow.streamClient = stream;
  }
  const projectId = new URL(window.location.href).searchParams.get("project");
  if (projectId) {
    await store.load(projectId).catch(() => {
      const status = shell.querySelector<HTMLElement>(".connection-status");
      if (status) status.textContent = "The project could not be loaded. Check the connection and project access.";
    });
  }
}

if (shell) {
  void bootstrap(shell).catch(() => {
    const status = shell.querySelector<HTMLElement>(".connection-status");
    if (status) status.textContent = "World Studio could not establish an authenticated session.";
  });
}
