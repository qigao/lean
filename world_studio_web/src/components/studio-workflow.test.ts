import { fireEvent, getByLabelText, getByRole } from "@testing-library/dom";
import { afterEach, describe, expect, it } from "vitest";
import type { JsonObject, ProjectSnapshot } from "../schema/studio-types";
import type { RpcCaller } from "../rpc/client";
import { ProjectStore } from "../state/project-store";
import type { StudioWorkflowElement } from "./studio-workflow";
import "./studio-workflow";

const HASH = (value: string) => `sha256:${value.repeat(64)}`;

function snapshot(revision: number): ProjectSnapshot {
  return {
    schema: "narrative-dynamics.scenario-draft-snapshot/v1", project_id: "law-firm", revision,
    scenario_id: revision > 1 ? "law-firm-case" : null, version: revision > 1 ? "1" : null,
    documents: [], document_semantic_hash: HASH("1"), layout: {}, layout_hash: HASH("2"),
    diagnostic_report_hash: HASH("3"), compiled_scenario_hash: revision > 1 ? HASH("4") : null,
    content_hash: HASH(String(revision)),
  };
}

class Rpc implements RpcCaller {
  calls: { method: string; params: JsonObject }[] = [];
  async call<T>(method: string, params: JsonObject): Promise<T> {
    this.calls.push({ method, params });
    if (method === "project.create") return snapshot(1) as T;
    if (method === "project.import") return snapshot(2) as T;
    return {
      project_id: "law-firm", revision: 2, snapshot_hash: HASH("2"),
      scenario_id: "law-firm-case", version: "1", package_hash: HASH("5"), scenario_hash: HASH("4"),
    } as T;
  }
}

afterEach(() => document.body.replaceChildren());

describe("studio workflow", () => {
  it("imports by configured source identity, compiles the exact accepted revision, and keeps one status node", async () => {
    const rpc = new Rpc();
    const projectStore = new ProjectStore(rpc);
    const workflow = document.createElement("studio-workflow") as StudioWorkflowElement;
    workflow.rpc = rpc;
    workflow.projectStore = projectStore;
    document.body.append(workflow);
    const status = getByRole(workflow, "status");
    (getByLabelText(workflow, "Project ID") as HTMLInputElement).value = "law-firm";
    (getByLabelText(workflow, "Configured source ID") as HTMLInputElement).value = "law-firm-fixture";

    fireEvent.click(getByRole(workflow, "button", { name: "Import project" }));
    for (let index = 0; index < 8; index += 1) await Promise.resolve();
    expect(rpc.calls.slice(0, 2)).toEqual([
      { method: "project.create", params: { project_id: "law-firm" } },
      { method: "project.import", params: {
        project_id: "law-firm", source_id: "law-firm-fixture",
        expected_revision: 1, expected_snapshot_hash: HASH("1"),
      } },
    ]);
    expect(JSON.stringify(rpc.calls)).not.toContain("examples/");

    fireEvent.click(getByRole(workflow, "button", { name: "Compile revision" }));
    for (let index = 0; index < 5; index += 1) await Promise.resolve();
    expect(rpc.calls.at(-1)).toEqual({ method: "scenario.compile", params: {
      project_id: "law-firm", expected_revision: 2, expected_snapshot_hash: HASH("2"),
    } });
    expect(getByRole(workflow, "status")).toBe(status);
    expect(status.textContent).toContain("compiled");
  });
});
