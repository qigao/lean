import { describe, expect, it } from "vitest";
import { JsonRpcError, JsonRpcProtocolError } from "../rpc/client";
import type { JsonObject, ProjectApplyResult, ProjectSnapshot } from "../schema/studio-types";
import { reportFixture, snapshotFixture } from "../test-fixtures";
import { ProjectStore } from "./project-store";

class Deferred<T> {
  readonly promise: Promise<T>;
  readonly resolve: (value: T) => void;
  readonly reject: (reason: unknown) => void;

  constructor() {
    let resolvePromise!: (value: T) => void;
    let rejectPromise!: (reason: unknown) => void;
    this.promise = new Promise<T>((resolve, reject) => {
      resolvePromise = resolve;
      rejectPromise = reject;
    });
    this.resolve = resolvePromise;
    this.reject = rejectPromise;
  }
}

class RpcStub {
  calls: Array<{ method: string; params: JsonObject }> = [];
  handler: (method: string, params: JsonObject) => Promise<unknown> = async () => ({});

  async call<T>(method: string, params: JsonObject): Promise<T> {
    this.calls.push({ method, params });
    return await this.handler(method, params) as T;
  }
}

function resultFor(prior: ProjectSnapshot, next: ProjectSnapshot): ProjectApplyResult {
  return {
    operation_hash: `sha256:${"e".repeat(64)}`,
    prior_revision: prior.revision,
    next_snapshot: next,
    diagnostic_report: reportFixture(next.revision),
  };
}

describe("ProjectStore", () => {
  it("retains the immutable accepted snapshot until an exact next revision returns", async () => {
    const rpc = new RpcStub();
    const current = snapshotFixture(2);
    const next = snapshotFixture(3);
    const deferred = new Deferred<ProjectApplyResult>();
    rpc.handler = async () => deferred.promise;
    const store = new ProjectStore(rpc, {
      operationId: () => "operation-1",
      idempotencyKey: () => "operation-key-1",
    });
    store.accept(current);

    const pending = store.apply({
      document_role: "physical.world",
      logical_id: null,
      kind: "set_value",
      pointer: "/places/0/label",
      value: "Reception",
    });

    expect(store.snapshot).toBe(current);
    expect(store.status).toBe("applying");
    expect(Object.isFrozen(store.snapshot)).toBe(true);
    expect(rpc.calls[0]).toEqual({
      method: "project.apply",
      params: {
        operation_id: "operation-1",
        idempotency_key: "operation-key-1",
        project_id: "law-firm",
        expected_revision: 2,
        expected_snapshot_hash: current.content_hash,
        document_role: "physical.world",
        logical_id: null,
        kind: "set_value",
        pointer: "/places/0/label",
        value: "Reception",
      },
    });

    deferred.resolve(resultFor(current, next));
    await pending;
    expect(store.snapshot).toEqual(next);
    expect(store.status).toBe("ready");
  });

  it("reloads exactly once on stale state and exposes a conflict without merging", async () => {
    const rpc = new RpcStub();
    const current = snapshotFixture(2);
    const remote = snapshotFixture(5);
    rpc.handler = async (method) => {
      if (method === "project.apply") throw new JsonRpcError(-32011, "Stale state", { code: "stale_state" });
      return remote;
    };
    const store = new ProjectStore(rpc, {
      operationId: () => "operation-2",
      idempotencyKey: () => "operation-key-2",
    });
    store.accept(current);

    await expect(store.apply({
      document_role: "physical.world",
      logical_id: null,
      kind: "remove_value",
      pointer: "/places/0",
    })).rejects.toBeInstanceOf(JsonRpcError);

    expect(rpc.calls.map(({ method }) => method)).toEqual(["project.apply", "project.snapshot"]);
    expect(rpc.calls[1]?.params).toEqual({ project_id: "law-firm" });
    expect(store.snapshot).toEqual(remote);
    expect(store.conflict).toEqual({ kind: "stale_state", attempted_revision: 2, current_revision: 5 });
  });

  it("rejects a non-matching apply result without replacing authority", async () => {
    const rpc = new RpcStub();
    const current = snapshotFixture(2);
    rpc.handler = async () => resultFor(current, snapshotFixture(4));
    const store = new ProjectStore(rpc, {
      operationId: () => "operation-3",
      idempotencyKey: () => "operation-key-3",
    });
    store.accept(current);

    await expect(store.apply({
      document_role: "physical.world",
      logical_id: null,
      kind: "set_value",
      pointer: "/places/0/label",
      value: "Reception",
    })).rejects.toBeInstanceOf(JsonRpcProtocolError);
    expect(store.snapshot).toBe(current);
  });

  it("normalizes a malformed method result without losing the accepted snapshot", async () => {
    const rpc = new RpcStub();
    const current = snapshotFixture(2);
    rpc.handler = async () => ({});
    const store = new ProjectStore(rpc, {
      operationId: () => "operation-4",
      idempotencyKey: () => "operation-key-4",
    });
    store.accept(current);

    await expect(store.apply({
      document_role: "physical.world",
      logical_id: null,
      kind: "remove_value",
      pointer: "/places/0",
    })).rejects.toBeInstanceOf(JsonRpcProtocolError);
    expect(store.snapshot).toBe(current);
  });
});
