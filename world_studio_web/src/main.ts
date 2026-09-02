import { JsonRpcClient } from "./rpc/client";
import { ProjectStore } from "./state/project-store";
import "./components/studio-shell";
import type { StudioShellElement } from "./components/studio-shell";
import "./styles.css";

const shell = document.querySelector<StudioShellElement>("#studio-app");

if (shell) {
  const store = new ProjectStore(new JsonRpcClient("/rpc"));
  shell.store = store;
  store.addEventListener("change", () => {
    if (store.snapshot && shell.snapshot !== store.snapshot) shell.snapshot = store.snapshot;
    if (shell.diagnostics !== store.diagnosticReport) shell.diagnostics = store.diagnosticReport;
  });
  const projectId = new URL(window.location.href).searchParams.get("project");
  if (projectId) {
    void store.load(projectId).catch(() => {
      const status = shell.querySelector<HTMLElement>(".connection-status");
      if (status) status.textContent = "The project could not be loaded. Check the connection and project access.";
    });
  }
}
