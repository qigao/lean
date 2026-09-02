import { expect, test, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { mkdtempSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const webRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repositoryRoot = resolve(webRoot, "..");
const fixture = resolve(repositoryRoot, "examples", "law_firm_scenario");
const temporaryRoot = mkdtempSync(resolve(tmpdir(), "world-studio-e2e-"));
const workspaceRoot = resolve(temporaryRoot, "workspace");
const exportRoot = resolve(temporaryRoot, "export");
mkdirSync(workspaceRoot);
mkdirSync(exportRoot);

let server: ChildProcessWithoutNullStreams | null = null;
let serverOutput = "";

function launcherArguments(): string[] {
  return [
    "-m", "tools.run_world_studio",
    "--workspace-root", workspaceRoot,
    "--import-root", resolve(repositoryRoot, "examples"),
    "--import-source", `law-firm-fixture=${fixture}`,
    "--export-root", exportRoot,
    "--static-root", resolve(webRoot, "dist"),
    "--bind-host", "127.0.0.1",
    "--bind-port", "8766",
    "--origin", "http://127.0.0.1:8766",
    "--development-trust-all",
    "--authority-id", "e2e-operator",
    "--project-id", "law-firm",
    "--run-id", "run-parent",
    "--run-id", "run-child",
    "--agent-id", "alice",
  ];
}

async function waitForServer(): Promise<void> {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (server?.exitCode !== null) throw new Error(`World Studio exited early.\n${serverOutput}`);
    try {
      const response = await fetch("http://127.0.0.1:8766/health");
      if (response.ok) return;
    } catch {
      // Startup is expected to race this bounded readiness probe.
    }
    await new Promise((resolveDelay) => setTimeout(resolveDelay, 100));
  }
  throw new Error(`World Studio did not become ready.\n${serverOutput}`);
}

async function startServer(): Promise<void> {
  serverOutput = "";
  server = spawn(process.env.PYTHON ?? "python", launcherArguments(), {
    cwd: repositoryRoot,
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
    stdio: "pipe",
  });
  const retain = (chunk: Buffer) => {
    serverOutput = (serverOutput + chunk.toString("utf8")).slice(-20_000);
  };
  server.stdout.on("data", retain);
  server.stderr.on("data", retain);
  await waitForServer();
}

async function stopServer(): Promise<void> {
  const process = server;
  server = null;
  if (!process || process.exitCode !== null) return;
  const exited = new Promise<void>((resolveExit) => process.once("exit", () => resolveExit()));
  process.kill("SIGTERM");
  await Promise.race([exited, new Promise<void>((resolveDelay) => setTimeout(resolveDelay, 5_000))]);
  if (process.exitCode === null) {
    process.kill("SIGKILL");
    await exited;
  }
}

async function applyInspectorJson(page: Page, value: unknown): Promise<void> {
  await page.getByLabel("JSON value").fill(JSON.stringify(value, null, 2));
  await page.getByRole("button", { name: "Apply property" }).click();
  await expect(page.locator(".connection-status")).toHaveText("Project revision accepted by the server.");
}

test.beforeAll(async () => { await startServer(); });
test.afterAll(async () => {
  await stopServer();
  rmSync(temporaryRoot, { recursive: true, force: true });
});

test("law-firm authoring, privacy, recovery, fork, and persistence use real authorities", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator(".connection-status")).toHaveText("Editor shell ready.");

  // 1. Import the configured examples/law_firm_scenario source by server-owned ID.
  await page.getByRole("button", { name: "Import project" }).click();
  await expect(page.locator(".workflow-status")).toContainText("imported at revision 2");
  await expect(page.getByRole("navigation", { name: "Project documents" })).toContainText("law-firm");

  // 2. Edit one place label and one relationship through visual selection + property view.
  await page.getByRole("button", { name: "Select place Public lobby" }).click();
  await applyInspectorJson(page, { label: "Client reception", place_id: "lobby" });
  await expect(page.getByRole("button", { name: "Select place Client reception" })).toBeVisible();
  await page.getByRole("tab", { name: "Social" }).click();
  await page.getByRole("button", { name: "Select relationship supervises" }).first().click();
  await applyInspectorJson(page, {
    relationship_type: "supervises", source_agent_id: "alice", strength: 0.85, target_agent_id: "bob",
  });

  // 3. Introduce a dangling scene dependency, observe diagnostics, and repair it.
  await page.getByRole("tab", { name: "Story" }).click();
  await page.getByRole("button", { name: "Select dependency precedes" }).click();
  await applyInspectorJson(page, { predecessor_scene_id: "discover", successor_scene_id: "missing-scene" });
  const diagnostics = page.getByRole("region", { name: "Diagnostics" });
  await expect(diagnostics.getByRole("status")).not.toContainText("0 diagnostics");
  await diagnostics.getByRole("button").first().click();
  await expect(page.getByRole("region", { name: "Property inspector" })).toContainText("/dependencies/0/successor_scene_id");
  await applyInspectorJson(page, "confront");
  await expect(diagnostics.getByRole("status")).toContainText("0 diagnostics");

  // 4. Compile the exact accepted project revision.
  await page.getByRole("button", { name: "Compile revision" }).click();
  await expect(page.locator(".workflow-status")).toContainText("compiled as sha256:");

  // Accessibility is checked against the real populated product surface.
  const accessibility = await new AxeBuilder({ page }).analyze();
  expect(accessibility.violations.filter((item) => item.impact === "critical" || item.impact === "serious")).toEqual([]);

  // 5. Create, start, and step a real run.
  await page.getByRole("button", { name: "Create run" }).click();
  await expect(page.locator(".workflow-status")).toContainText("public output stream connected");
  await page.getByRole("button", { name: "Start run" }).click();
  await expect(page.getByRole("toolbar", { name: "Run controls" }).getByRole("status")).toContainText("running");
  await page.getByRole("button", { name: "Step run" }).click();
  const publicTimelineItem = page.getByRole("list", { name: "Run output records" }).getByRole("listitem").last();
  await expect(publicTimelineItem).toContainText("Sequence");
  await expect(publicTimelineItem).toContainText("(public)");

  // 6. Verify public output, then owner-Agent selection with no unauthorized owner exposed.
  const audience = page.getByLabel("State audience");
  await expect(audience.locator("option")).toHaveText(["Public", "Agent: alice", "Network metrics"]);
  await audience.selectOption("agent:alice");
  await expect(page.locator(".state-status")).toContainText("agent state loaded");
  await expect(page.locator(".scoped-state")).toContainText('"agent_id": "alice"');
  await expect(page.locator(".scoped-state")).not.toContainText('"agent_id": "bob"');
  await page.getByRole("button", { name: "Step run" }).click();
  const timeline = page.getByRole("list", { name: "Run output records" });
  await expect(timeline.getByRole("listitem").last()).toContainText("Sequence");
  await expect(timeline).not.toContainText("owner bob");

  // 7. Force a real WSS disconnect, recreate the identical subscription, and resume.
  await page.getByRole("button", { name: "Disconnect stream" }).click();
  await expect(page.locator(".stream-status")).toContainText("disconnected");
  await page.getByRole("button", { name: "Reconnect stream" }).click();
  await expect(page.locator(".stream-status")).toContainText("connected");
  await expect(page.getByRole("log")).toContainText("resumed from its acknowledged cursor");

  // 8. Checkpoint and fork through the real checkpoint store/coordinator.
  await page.getByText("Checkpoint and fork").click();
  await page.getByLabel("Checkpoint ID").fill("fork-base");
  await page.getByRole("button", { name: "Create checkpoint" }).click();
  await expect(page.getByRole("log")).toContainText("checkpoint was accepted");
  await page.getByLabel("Child run ID").fill("run-child");
  await page.getByLabel("Child stream ID").fill("stream-child");
  await page.getByRole("button", { name: "Fork run" }).click();
  await expect(page.getByRole("region", { name: "Child run" })).toContainText("run-child");

  // 9. Step parent and child independently and display distinct authoritative hashes.
  await page.getByRole("button", { name: "Step run" }).click();
  await page.getByRole("button", { name: "Step run" }).click();
  await page.getByLabel("Selected run").selectOption("run-child");
  await expect(page.locator(".stream-status")).toContainText("connected");
  await page.getByRole("button", { name: "Step run" }).click();
  const parentText = await page.getByRole("region", { name: "Parent run" }).textContent();
  const childText = await page.getByRole("region", { name: "Child run" }).textContent();
  const parentHash = parentText?.match(/sha256:[0-9a-f]{64}/)?.[0];
  const childHash = childText?.match(/sha256:[0-9a-f]{64}/)?.[0];
  expect(parentHash).toBeTruthy();
  expect(childHash).toBeTruthy();
  expect(childHash).not.toBe(parentHash);

  // 10. Restart the actual service process and reopen the persisted SQLite project revision.
  const savedRevision = await page.locator(".workflow-status").textContent();
  await stopServer();
  await startServer();
  await page.reload();
  await page.getByRole("button", { name: "Open project" }).click();
  await expect(page.locator(".workflow-status")).toContainText("reopened at saved revision");
  await expect(page.getByRole("button", { name: "Select place Client reception" })).toBeVisible();
  expect(savedRevision).toBeTruthy();
});
