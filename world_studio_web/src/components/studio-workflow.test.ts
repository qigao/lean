import { fireEvent, getByLabelText, getByRole } from "@testing-library/dom";
import { afterEach, describe, expect, it } from "vitest";
import type { JsonObject, ProjectSnapshot } from "../schema/studio-types";
import {
  JsonRpcError,
  JsonRpcProtocolError,
  type JsonRpcCallOptions,
  type RpcCaller,
} from "../rpc/client";
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
  calls: { method: string; params: JsonObject; options: JsonRpcCallOptions | undefined }[] = [];
  async call<T>(method: string, params: JsonObject, options?: JsonRpcCallOptions): Promise<T> {
    this.calls.push({ method, params, options });
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
    expect(rpc.calls.slice(0, 2).map(({ method, params }) => ({ method, params }))).toEqual([
      { method: "project.create", params: { project_id: "law-firm" } },
      { method: "project.import", params: {
        project_id: "law-firm", source_id: "law-firm-fixture",
        expected_revision: 1, expected_snapshot_hash: HASH("1"),
      } },
    ]);
    expect(JSON.stringify(rpc.calls)).not.toContain("examples/");

    fireEvent.click(getByRole(workflow, "button", { name: "Compile revision" }));
    for (let index = 0; index < 5; index += 1) await Promise.resolve();
    expect(rpc.calls.at(-1)).toMatchObject({ method: "scenario.compile", params: {
      project_id: "law-firm", expected_revision: 2, expected_snapshot_hash: HASH("2"),
    } });
    expect(getByRole(workflow, "status")).toBe(status);
    expect(status.textContent).toContain("compiled");
    expect(rpc.calls[0]!.options?.attempts).toBe(1);
    expect(rpc.calls[1]!.options?.attempts).toBe(1);
  });

  it.each(["create", "import"] as const)(
    "reconciles an ambiguous %s response through the authoritative snapshot",
    async (ambiguousMethod) => {
      class AmbiguousRpc implements RpcCaller {
        calls: string[] = [];
        private failed = false;

        async call<T>(method: string, _params: JsonObject, options?: JsonRpcCallOptions): Promise<T> {
          this.calls.push(`${method}:${options?.attempts ?? 1}`);
          if (method === "project.create") {
            if (ambiguousMethod === "create" && !this.failed) {
              this.failed = true;
              throw new Error("response lost after commit");
            }
            return snapshot(1) as T;
          }
          if (method === "project.import") {
            if (ambiguousMethod === "import" && !this.failed) {
              this.failed = true;
              throw new Error("response lost after commit");
            }
            return snapshot(2) as T;
          }
          if (method === "project.snapshot") {
            return snapshot(ambiguousMethod === "create" ? 1 : 2) as T;
          }
          throw new Error("unexpected method");
        }
      }

      const rpc = new AmbiguousRpc();
      const projectStore = new ProjectStore(rpc);
      const workflow = document.createElement("studio-workflow") as StudioWorkflowElement;
      workflow.rpc = rpc;
      workflow.projectStore = projectStore;
      document.body.append(workflow);

      fireEvent.click(getByRole(workflow, "button", { name: "Import project" }));
      for (let index = 0; index < 16; index += 1) await Promise.resolve();

      expect(projectStore.snapshot?.revision).toBe(2);
      expect(rpc.calls).toEqual(ambiguousMethod === "create"
        ? ["project.create:1", "project.snapshot:1", "project.import:1"]
        : ["project.create:1", "project.import:1", "project.snapshot:1"]);
      expect(getByRole(workflow, "status").textContent).toContain(
        ambiguousMethod === "create" ? "imported" : "ambiguous",
      );
    },
  );

  it("imports an existing empty project after create reports a conflict", async () => {
    class ExistingRpc implements RpcCaller {
      calls: string[] = [];
      async call<T>(method: string, _params: JsonObject, options?: JsonRpcCallOptions): Promise<T> {
        this.calls.push(`${method}:${options?.attempts ?? 1}`);
        if (method === "project.create") throw new JsonRpcError(-32015, "Conflict");
        if (method === "project.snapshot") return snapshot(1) as T;
        if (method === "project.import") return snapshot(2) as T;
        throw new Error("unexpected method");
      }
    }
    const rpc = new ExistingRpc();
    const projectStore = new ProjectStore(rpc);
    const workflow = document.createElement("studio-workflow") as StudioWorkflowElement;
    workflow.rpc = rpc;
    workflow.projectStore = projectStore;
    document.body.append(workflow);

    fireEvent.click(getByRole(workflow, "button", { name: "Import project" }));
    for (let index = 0; index < 16; index += 1) await Promise.resolve();

    expect(projectStore.snapshot?.revision).toBe(2);
    expect(rpc.calls).toEqual(["project.create:1", "project.snapshot:1", "project.import:1"]);
  });

  it.each([
    ["authorization", () => new JsonRpcError(-32012, "Not authorized")],
    ["validation", () => new JsonRpcError(-32602, "Invalid params")],
    ["protocol", () => new JsonRpcProtocolError("Malformed create response")],
  ])("does not reinterpret an explicit create %s failure as an existing project", async (_label, failure) => {
    class ExplicitCreateFailureRpc implements RpcCaller {
      calls: string[] = [];
      async call<T>(method: string): Promise<T> {
        this.calls.push(method);
        throw failure();
      }
    }
    const rpc = new ExplicitCreateFailureRpc();
    const projectStore = new ProjectStore(rpc);
    const workflow = document.createElement("studio-workflow") as StudioWorkflowElement;
    workflow.rpc = rpc;
    workflow.projectStore = projectStore;
    document.body.append(workflow);

    fireEvent.click(getByRole(workflow, "button", { name: "Import project" }));
    for (let index = 0; index < 12; index += 1) await Promise.resolve();

    expect(rpc.calls).toEqual(["project.create"]);
    expect(projectStore.snapshot).toBeNull();
    expect(getByRole(workflow, "status").textContent).toContain("failed");
  });

  it.each([
    ["conflict", () => new JsonRpcError(-32015, "Import conflict")],
    ["validation", () => new JsonRpcError(-32602, "Invalid import")],
    ["protocol", () => new JsonRpcProtocolError("Malformed import response")],
  ])("does not reinterpret an explicit import %s failure as unrelated revision progress", async (_label, failure) => {
    class ExplicitImportFailureRpc implements RpcCaller {
      calls: string[] = [];
      async call<T>(method: string): Promise<T> {
        this.calls.push(method);
        if (method === "project.create") return snapshot(1) as T;
        if (method === "project.import") throw failure();
        if (method === "project.snapshot") return snapshot(9) as T;
        throw new Error("unexpected method");
      }
    }
    const rpc = new ExplicitImportFailureRpc();
    const projectStore = new ProjectStore(rpc);
    const workflow = document.createElement("studio-workflow") as StudioWorkflowElement;
    workflow.rpc = rpc;
    workflow.projectStore = projectStore;
    document.body.append(workflow);

    fireEvent.click(getByRole(workflow, "button", { name: "Import project" }));
    for (let index = 0; index < 12; index += 1) await Promise.resolve();

    expect(rpc.calls).toEqual(["project.create", "project.import"]);
    expect(projectStore.snapshot?.revision).toBe(1);
    expect(getByRole(workflow, "status").textContent).toContain("failed");
  });

  it("reports a lost import response as ambiguous and retries from the accepted snapshot", async () => {
    class RetriableAmbiguousRpc implements RpcCaller {
      calls: string[] = [];
      private importAttempts = 0;
      async call<T>(method: string, _params: JsonObject, options?: JsonRpcCallOptions): Promise<T> {
        this.calls.push(`${method}:${options?.attempts ?? 1}`);
        if (method === "project.create") {
          if (this.importAttempts > 0) throw new JsonRpcError(-32015, "Already exists");
          return snapshot(1) as T;
        }
        if (method === "project.import") {
          this.importAttempts += 1;
          if (this.importAttempts === 1) throw new Error("response lost after commit");
          return snapshot(3) as T;
        }
        if (method === "project.snapshot") return snapshot(2) as T;
        throw new Error("unexpected method");
      }
    }
    const rpc = new RetriableAmbiguousRpc();
    const projectStore = new ProjectStore(rpc);
    const workflow = document.createElement("studio-workflow") as StudioWorkflowElement;
    workflow.rpc = rpc;
    workflow.projectStore = projectStore;
    document.body.append(workflow);
    const importButton = getByRole(workflow, "button", { name: "Import project" });

    fireEvent.click(importButton);
    for (let index = 0; index < 16; index += 1) await Promise.resolve();
    expect(projectStore.snapshot?.revision).toBe(2);
    expect(getByRole(workflow, "status").textContent).toContain("ambiguous");
    expect(getByRole(workflow, "status").textContent).not.toContain(" imported ");

    fireEvent.click(importButton);
    for (let index = 0; index < 20; index += 1) await Promise.resolve();
    expect(projectStore.snapshot?.revision).toBe(3);
    expect(getByRole(workflow, "status").textContent).toContain("imported");
    expect(rpc.calls).toEqual([
      "project.create:1", "project.import:1", "project.snapshot:1",
      "project.create:1", "project.snapshot:1", "project.import:1",
    ]);
  });
});
