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
  it.each([
    ["serialized manifest-missing", {
      severity: "error" as const,
      code: "manifest_missing",
      document_role: "package",
      logical_id: null,
      pointer: "",
      related_ids: [],
      message_key: "manifest_missing",
    }],
    ["package compiler diagnostic with independent logical ID", {
      severity: "error" as const,
      code: "package_invalid",
      document_role: "package",
      logical_id: "manifest-v1",
      pointer: "/documents/~0metadata/~1source",
      related_ids: ["agent-a", "agent-b"],
      message_key: "package_invalid",
    }],
  ])("accepts a real-shape %s diagnostic", (_label, diagnostic) => {
    const store = new ProjectStore(new RpcStub());
    const accepted = snapshotFixture(2);
    const report = reportFixture(2, [diagnostic]);

    expect(() => store.accept(accepted, report)).not.toThrow();
    expect(store.snapshot).toBe(accepted);
    expect(store.diagnosticReport).toBe(report);
    expect(store.status).toBe("ready");
  });

  it.each([
    ["snapshot extra key", { ...snapshotFixture(2), unexpected: true }],
    ["snapshot missing key", (() => {
      const value = { ...snapshotFixture(2) } as Record<string, unknown>;
      delete value.layout_hash;
      return value;
    })()],
    ["document hash", {
      ...snapshotFixture(2),
      documents: [{ ...snapshotFixture(2).documents[0]!, content_hash: "bad" }],
    }],
    ["non-finite layout", { ...snapshotFixture(2), layout: { x: Number.POSITIVE_INFINITY } }],
    ["unknown document role", {
      ...snapshotFixture(2),
      documents: [{ ...snapshotFixture(2).documents[0]!, role: "physical.unknown" }],
    }],
    ["singleton document logical ID", {
      ...snapshotFixture(2),
      documents: [{ ...snapshotFixture(2).documents[0]!, logical_id: "unexpected" }],
    }],
    ["agent document without logical ID", {
      ...snapshotFixture(2),
      documents: [{ ...snapshotFixture(2).documents[0]!, role: "agent", logical_id: null }],
    }],
  ])("rejects malformed %s without replacing accepted authority", (_label, malformed) => {
    const store = new ProjectStore(new RpcStub());
    const accepted = snapshotFixture(3);
    store.accept(accepted, reportFixture(3));

    expect(() => store.accept(malformed as unknown as ProjectSnapshot)).toThrow(JsonRpcProtocolError);
    expect(store.snapshot).toBe(accepted);
    expect(store.diagnosticReport).toEqual(reportFixture(3));
  });

  it.each([
    ["report extra key", { ...reportFixture(2), unexpected: true }],
    ["diagnostic severity", reportFixture(2, [{
      severity: "fatal" as "error", code: "bad", document_role: "physical.world",
      logical_id: null, pointer: "", related_ids: [], message_key: "bad",
    }])],
    ["report hash", { ...reportFixture(2), content_hash: "bad" }],
    ["diagnostic message key syntax", reportFixture(2, [{
      severity: "warning", code: "warning_code", document_role: "physical.world",
      logical_id: null, pointer: "", related_ids: [], message_key: "Bad-Key",
    }])],
    ["diagnostic duplicate related IDs", reportFixture(2, [{
      severity: "warning", code: "warning_code", document_role: "physical.world",
      logical_id: null, pointer: "", related_ids: ["alpha", "alpha"], message_key: "warning_key",
    }])],
    ["diagnostic noncanonical related ID order", reportFixture(2, [{
      severity: "warning", code: "warning_code", document_role: "physical.world",
      logical_id: null, pointer: "", related_ids: ["beta", "alpha"], message_key: "warning_key",
    }])],
    ["diagnostic relative pointer", reportFixture(2, [{
      severity: "warning", code: "warning_code", document_role: "physical.world",
      logical_id: null, pointer: "documents/0", related_ids: [], message_key: "warning_key",
    }])],
    ["diagnostic invalid pointer escape", reportFixture(2, [{
      severity: "warning", code: "warning_code", document_role: "physical.world",
      logical_id: null, pointer: "/documents/~2", related_ids: [], message_key: "warning_key",
    }])],
    ["diagnostic dangling pointer escape", reportFixture(2, [{
      severity: "warning", code: "warning_code", document_role: "physical.world",
      logical_id: null, pointer: "/documents/~", related_ids: [], message_key: "warning_key",
    }])],
    ["noncanonical diagnostic report order", reportFixture(2, [{
      severity: "error", code: "world_warning", document_role: "physical.world",
      logical_id: null, pointer: "", related_ids: [], message_key: "world_warning",
    }, {
      severity: "warning", code: "agent_warning", document_role: "agent",
      logical_id: "alice", pointer: "", related_ids: [], message_key: "agent_warning",
    }])],
  ])("rejects malformed %s without replacing accepted authority", (_label, malformed) => {
    const store = new ProjectStore(new RpcStub());
    const accepted = snapshotFixture(2);
    store.accept(accepted);

    expect(() => store.accept(accepted, malformed as ReturnType<typeof reportFixture>))
      .toThrow(JsonRpcProtocolError);
    expect(store.snapshot).toBe(accepted);
    expect(store.diagnosticReport).toBeNull();
  });

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

  it("accepts a valid layout apply whose authoritative report is manifest-missing", async () => {
    const rpc = new RpcStub();
    const current = snapshotFixture(2);
    const next = snapshotFixture(3, { layout: { package: { x: 12, y: 24 } } });
    const manifestMissing = reportFixture(3, [{
      severity: "error", code: "manifest_missing", document_role: "package",
      logical_id: null, pointer: "", related_ids: [], message_key: "manifest_missing",
    }]);
    rpc.handler = async () => ({
      ...resultFor(current, next),
      diagnostic_report: manifestMissing,
    });
    const store = new ProjectStore(rpc);
    store.accept(current);

    await expect(store.apply({
      document_role: "physical.world",
      logical_id: null,
      kind: "set_layout",
      value: { package: { x: 12, y: 24 } },
    })).resolves.toMatchObject({ next_snapshot: next });
    expect(store.snapshot).toBe(next);
    expect(store.diagnosticReport).toBe(manifestMissing);
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
    store.accept(current, reportFixture(2));

    await expect(store.apply({
      document_role: "physical.world",
      logical_id: null,
      kind: "remove_value",
      pointer: "/places/0",
    })).rejects.toBeInstanceOf(JsonRpcError);

    expect(rpc.calls.map(({ method }) => method)).toEqual(["project.apply", "project.snapshot"]);
    expect(rpc.calls[1]?.params).toEqual({ project_id: "law-firm" });
    expect(store.snapshot).toEqual(remote);
    expect(store.diagnosticReport).toBeNull();
    expect(store.status).toBe("conflict");
    expect(store.conflict).toEqual({ kind: "stale_state", attempted_revision: 2, current_revision: 5 });
  });

  it("ends in an error state and clears the stale report when stale recovery reload fails", async () => {
    const rpc = new RpcStub();
    const current = snapshotFixture(2);
    const reloadFailure = new Error("snapshot service unavailable");
    rpc.handler = async (method) => {
      if (method === "project.apply") throw new JsonRpcError(-32011, "Stale state", { code: "stale_state" });
      throw reloadFailure;
    };
    const store = new ProjectStore(rpc, {
      operationId: () => "operation-reload-failure",
      idempotencyKey: () => "operation-key-reload-failure",
    });
    store.accept(current, reportFixture(2));

    await expect(store.apply({
      document_role: "physical.world",
      logical_id: null,
      kind: "remove_value",
      pointer: "/places/0",
    })).rejects.toBe(reloadFailure);

    expect(store.status).toBe("error");
    expect(store.snapshot).toBe(current);
    expect(store.diagnosticReport).toBeNull();
    expect(store.conflict).toBeNull();
    expect(rpc.calls.map(({ method }) => method)).toEqual(["project.apply", "project.snapshot"]);
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

  it.each([
    ["wrapper extra key", (current: ProjectSnapshot) => ({
      ...resultFor(current, snapshotFixture(3)), unexpected: true,
    })],
    ["operation hash", (current: ProjectSnapshot) => ({
      ...resultFor(current, snapshotFixture(3)), operation_hash: "bad",
    })],
    ["nested snapshot", (current: ProjectSnapshot) => ({
      ...resultFor(current, snapshotFixture(3)),
      next_snapshot: { ...snapshotFixture(3), documents: [{
        ...snapshotFixture(3).documents[0]!, value: [] as unknown as JsonObject,
      }] },
    })],
    ["nested report", (current: ProjectSnapshot) => ({
      ...resultFor(current, snapshotFixture(3)),
      diagnostic_report: { ...reportFixture(3), diagnostics: [{
        severity: "warning", code: "warn", document_role: "physical.world",
        logical_id: null, pointer: "", related_ids: ["ok", 7 as unknown as string],
        message_key: "warn",
      }] },
    })],
  ])("rejects malformed apply %s before changing authority", async (_label, malformed) => {
    const rpc = new RpcStub();
    const current = snapshotFixture(2);
    const report = reportFixture(2);
    rpc.handler = async () => malformed(current);
    const store = new ProjectStore(rpc);
    store.accept(current, report);

    await expect(store.apply({
      document_role: "physical.world", logical_id: null, kind: "remove_value", pointer: "/places/0",
    })).rejects.toBeInstanceOf(JsonRpcProtocolError);
    expect(store.snapshot).toBe(current);
    expect(store.diagnosticReport).toBe(report);
    expect(store.status).toBe("error");
  });
});
