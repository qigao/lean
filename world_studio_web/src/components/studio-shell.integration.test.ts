import { fireEvent, getByLabelText, getByRole } from "@testing-library/dom";
import { afterEach, describe, expect, it } from "vitest";
import { JsonRpcError } from "../rpc/client";
import type { OperationIntent } from "../schema/studio-types";
import { reportFixture, snapshotFixture } from "../test-fixtures";
import type { JsonEditorElement } from "../editors/json-editor";
import type { MapEditorElement } from "../editors/map-editor";
import type { StudioShellElement } from "./studio-shell";
import { RunStore } from "../state/run-store";
import type { JsonObject, RunAuthority, ScenarioRunView } from "../schema/studio-types";
import "./studio-shell";

afterEach(() => document.body.replaceChildren());

describe("studio shell integration", () => {
  it("wires the authoritative run store into the toolbar, inspector, and bounded timeline without replacing live nodes", () => {
    const rpc = { call: async <T,>(_method: string, _params: JsonObject) => ({}) as T };
    const authority: RunAuthority = {
      authority_id: "operator", project_ids: ["law-firm"], run_ids: ["run-parent"],
      agent_ids: ["alice"],
      permissions: ["run.command", "run.read", "state.public", "output.read"],
    };
    const runStore = new RunStore(rpc, authority);
    const run: ScenarioRunView = {
      schema: "narrative-dynamics.scenario-run-view/v1", run_id: "run-parent", stream_id: "stream-parent",
      scenario_hash: `sha256:${"a".repeat(64)}`, coordinator_epoch: 1, status: "running", round_index: 2,
      state_hash: `sha256:${"b".repeat(64)}`, next_sequence: 3, output_batch_hashes: [], checkpoint_hashes: [],
      parent_checkpoint_hash: null, content_hash: `sha256:${"c".repeat(64)}`,
    };
    runStore.acceptRun(run);
    const shell = document.createElement("studio-shell") as StudioShellElement;
    shell.runStore = runStore;
    shell.selectedRunId = "run-parent";
    document.body.append(shell);

    const toolbar = getByRole(shell, "toolbar", { name: "Run controls" });
    const status = getByRole(toolbar, "status");
    expect((getByRole(toolbar, "button", { name: "Pause run" }) as HTMLButtonElement).disabled).toBe(false);
    expect(getByRole(shell, "region", { name: "Run state inspector" }).textContent).toContain(run.state_hash);
    expect(getByRole(shell, "list", { name: "Run output records" })).toBeTruthy();

    runStore.acceptRun({ ...run, round_index: 3, content_hash: `sha256:${"d".repeat(64)}` });
    expect(getByRole(toolbar, "status")).toBe(status);
    expect(status.textContent).toContain("Round 3");
  });

  it("loads only the selected heavy editor while preserving semantic fallbacks", () => {
    const shell = document.createElement("studio-shell") as StudioShellElement;
    shell.snapshot = snapshotFixture();
    document.body.append(shell);
    expect(shell.querySelector("graph-editor")?.getAttribute("data-mode")).toBe("physical");
    expect(shell.querySelector("map-editor")).toBeNull();
    expect(shell.querySelector("json-editor")).toBeNull();

    fireEvent.click(getByRole(shell, "tab", { name: "Map" }));
    expect((shell.querySelector("map-editor") as MapEditorElement | null)?.active).toBe(true);
    expect(shell.querySelector("graph-editor")).toBeNull();

    fireEvent.click(getByRole(shell, "tab", { name: "Raw JSON" }));
    const json = shell.querySelector("json-editor") as JsonEditorElement | null;
    expect(json?.active).toBe(true);
    expect(getByLabelText(json!, "Document JSON")).toBeTruthy();
  });

  it("forwards exact semantic operations to the project store", async () => {
    const applied: OperationIntent[] = [];
    const shell = document.createElement("studio-shell") as StudioShellElement;
    shell.snapshot = snapshotFixture();
    shell.store = {
      apply: async (intent: OperationIntent) => {
        applied.push(intent);
        return undefined;
      },
    };
    document.body.append(shell);
    const operation: OperationIntent = {
      document_role: "physical.world",
      logical_id: null,
      kind: "set_value",
      pointer: "/places/0/label",
      value: "Reception",
    };
    shell.querySelector("graph-editor")?.dispatchEvent(new CustomEvent("studio-operation", {
      bubbles: true,
      detail: operation,
    }));
    await Promise.resolve();
    expect(applied).toEqual([operation]);
  });

  it("announces a stale conflict only for JSON-RPC code -32011", async () => {
    const shell = document.createElement("studio-shell") as StudioShellElement;
    shell.snapshot = snapshotFixture();
    let rejection: Error = new JsonRpcError(-32011, "Stale state");
    shell.store = { apply: async () => { throw rejection; } };
    document.body.append(shell);
    const editor = shell.querySelector("graph-editor")!;
    const operation: OperationIntent = {
      document_role: "physical.world",
      logical_id: null,
      kind: "remove_value",
      pointer: "/places/0",
    };

    editor.dispatchEvent(new CustomEvent("studio-operation", { bubbles: true, detail: operation }));
    await Promise.resolve();
    await Promise.resolve();
    expect(shell.querySelector(".connection-status")?.textContent).toContain("conflict detected");

    rejection = new JsonRpcError(-32010, "Application failure with untrusted detail");
    editor.dispatchEvent(new CustomEvent("studio-operation", { bubbles: true, detail: operation }));
    await Promise.resolve();
    await Promise.resolve();
    const message = shell.querySelector(".connection-status")?.textContent ?? "";
    expect(message).toContain("RPC -32010");
    expect(message).not.toContain("conflict");
    expect(message).not.toContain("untrusted detail");
  });

  it("keeps one connected polite status node while rendering and announcing updates", async () => {
    const shell = document.createElement("studio-shell") as StudioShellElement;
    shell.snapshot = snapshotFixture(2);
    shell.store = { apply: async () => undefined };
    document.body.append(shell);
    const status = shell.querySelector(".connection-status");

    shell.snapshot = snapshotFixture(3);
    expect(shell.querySelector(".connection-status")).toBe(status);

    shell.querySelector("graph-editor")?.dispatchEvent(new CustomEvent("studio-operation", {
      bubbles: true,
      detail: {
        document_role: "physical.world",
        logical_id: null,
        kind: "remove_value",
        pointer: "/places/0",
      } satisfies OperationIntent,
    }));
    await Promise.resolve();
    await Promise.resolve();
    expect(shell.querySelector(".connection-status")).toBe(status);
    expect(status?.textContent).toBe("Project revision accepted by the server.");
  });

  it("routes a selected diagnostic into the active raw JSON pointer and inspector", () => {
    const report = reportFixture(2, [{
      severity: "error",
      code: "unknown_place",
      document_role: "physical.world",
      logical_id: null,
      pointer: "/passages/0/source_place_id",
      related_ids: ["missing"],
      message_key: "passage_source_unknown",
    }]);
    const shell = document.createElement("studio-shell") as StudioShellElement;
    shell.snapshot = snapshotFixture();
    shell.diagnostics = report;
    document.body.append(shell);
    fireEvent.click(getByRole(shell, "tab", { name: "Raw JSON" }));
    fireEvent.click(getByRole(shell, "button", { name: "Error: passage source unknown" }));

    const textarea = getByLabelText(shell, "Document JSON") as HTMLTextAreaElement;
    expect(document.activeElement).toBe(textarea);
    expect(textarea.dataset.pointer).toBe("/passages/0/source_place_id");
    expect(getByRole(shell, "region", { name: "Property inspector" }).textContent).toContain("/passages/0/source_place_id");
  });

  it("routes semantic or Pixi map selection into the property inspector", () => {
    const base = snapshotFixture();
    const shell = document.createElement("studio-shell") as StudioShellElement;
    shell.snapshot = {
      ...base,
      documents: [...base.documents, {
        role: "physical.map",
        logical_id: null,
        content_hash: base.content_hash,
        value: {
          orientation: "orthogonal",
          tilewidth: 10,
          tileheight: 10,
          layers: [{
            type: "objectgroup",
            objects: [{ id: 1, name: "lobby", class: "place", x: 4, y: 8, width: 20, height: 20 }],
          }],
        },
      }],
    };
    document.body.append(shell);
    fireEvent.click(getByRole(shell, "tab", { name: "Map" }));
    fireEvent.click(getByRole(shell, "button", { name: "Public lobby at 4, 8" }));

    const inspector = getByRole(shell, "region", { name: "Property inspector" });
    expect(inspector.textContent).toContain("lobby");
    expect(inspector.textContent).toContain("/layers/0/objects/0");
  });

  it("keeps the skip link first and has no positive tabindex or unlabeled form controls", () => {
    const shell = document.createElement("studio-shell") as StudioShellElement;
    shell.snapshot = snapshotFixture();
    document.body.append(shell);
    expect(shell.querySelector("a,button,input,select,textarea")?.classList.contains("skip-link")).toBe(true);
    expect(shell.querySelector("[tabindex='1'],[tabindex='2'],[tabindex='3']")).toBeNull();
    for (const control of shell.querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>("input,select,textarea")) {
      expect(control.labels?.length || control.getAttribute("aria-label")).toBeTruthy();
    }
    expect(getByRole(shell, "region", { name: "Property inspector" }).querySelector("details[open] > summary")).toBeTruthy();
    expect(getByRole(shell, "region", { name: "Timeline" }).querySelector("details[open] > summary")).toBeTruthy();
  });
});
