import { afterEach, describe, expect, it } from "vitest";
import { JsonRpcError, type JsonRpcCallOptions, type RpcCaller } from "../rpc/client";
import type {
  JsonObject,
  RunAuthority,
  ScenarioCommandResult,
  ScenarioRunView,
} from "../schema/studio-types";
import { RunStore } from "./run-store";

const HASH = (character: string) => `sha256:${character.repeat(64)}`;

function runView(overrides: Partial<ScenarioRunView> = {}): ScenarioRunView {
  return {
    schema: "narrative-dynamics.scenario-run-view/v1",
    run_id: "run-1",
    stream_id: "stream-1",
    scenario_hash: HASH("a"),
    coordinator_epoch: 1,
    status: "created",
    round_index: 0,
    state_hash: HASH("b"),
    next_sequence: 1,
    output_batch_hashes: [],
    checkpoint_hashes: [],
    parent_checkpoint_hash: null,
    content_hash: HASH("c"),
    ...overrides,
  };
}

function commandResult(overrides: Partial<ScenarioCommandResult> = {}): ScenarioCommandResult {
  return {
    schema: "narrative-dynamics.scenario-command-result/v1",
    command_id: "command-1",
    idempotency_key: "command-key-1",
    request_hash: HASH("d"),
    capability_hash: HASH("e"),
    run_id: "run-1",
    scenario_hash: HASH("a"),
    coordinator_epoch: 1,
    kind: "start",
    accepted: true,
    reason: "accepted",
    prior_status: "created",
    next_status: "running",
    prior_state_hash: HASH("b"),
    next_state_hash: HASH("b"),
    round_index: 0,
    output_batch_hash: null,
    checkpoint_hash: null,
    content_hash: HASH("f"),
    ...overrides,
  };
}

const AUTHORITY: RunAuthority = {
  authority_id: "operator",
  project_ids: ["law-firm"],
  run_ids: ["run-1", "run-2", "run-child"],
  agent_ids: ["alice"],
  permissions: [
    "run.create", "run.command", "run.fork", "run.read", "state.public",
    "state.agent", "state.network", "output.read",
  ],
};

interface RecordedCall {
  method: string;
  params: JsonObject;
  options: JsonRpcCallOptions | undefined;
}

class TransportRpc implements RpcCaller {
  readonly calls: RecordedCall[] = [];
  handler: (method: string, params: JsonObject) => unknown | Promise<unknown> = () => {
    throw new Error("unconfigured transport");
  };

  async call<T>(method: string, params: JsonObject, options?: JsonRpcCallOptions): Promise<T> {
    this.calls.push({ method, params: structuredClone(params), options });
    return await this.handler(method, params) as T;
  }
}

function identities() {
  let value = 0;
  return {
    commandId: () => `command-${++value}`,
    idempotencyKey: () => `command-key-${value}`,
    forkId: () => `fork-${++value}`,
    forkIdempotencyKey: () => `fork-key-${value}`,
  };
}

afterEach(() => document.body.replaceChildren());

describe("RunStore authoritative transitions", () => {
  it("creates from an exact project revision and accepts only the returned run view", async () => {
    const rpc = new TransportRpc();
    const created = runView();
    rpc.handler = () => created;
    const store = new RunStore(rpc, AUTHORITY, identities());

    await store.create({
      project_id: "law-firm",
      revision: 7,
      content_hash: HASH("9"),
    }, "run-1", "stream-1");

    expect(rpc.calls).toEqual([{
      method: "run.create",
      params: {
        project_id: "law-firm",
        run_id: "run-1",
        stream_id: "stream-1",
        expected_revision: 7,
        expected_snapshot_hash: HASH("9"),
      },
      options: { stateChanging: true, attempts: 2, requestId: "run-create:law-firm:7:run-1" },
    }]);
    expect(store.run("run-1")).toEqual(created);
  });

  it("does not optimistically advance and rejects a second mutation for the same run", async () => {
    const rpc = new TransportRpc();
    const created = runView();
    let finish!: (value: ScenarioCommandResult) => void;
    const pending = new Promise<ScenarioCommandResult>((resolve) => { finish = resolve; });
    rpc.handler = (method) => method === "run.command" ? pending : created;
    const store = new RunStore(rpc, AUTHORITY, identities());
    store.acceptRun(created);

    const first = store.command("run-1", "start");
    expect(store.run("run-1")).toEqual(created);
    expect(store.isBusy("run-1")).toBe(true);
    await expect(store.command("run-1", "step")).rejects.toThrow("already in flight");

    const accepted = commandResult();
    const running = runView({ status: "running", content_hash: HASH("1") });
    rpc.handler = (method) => method === "run.view" ? running : pending;
    finish(accepted);
    await expect(first).resolves.toEqual(accepted);
    expect(store.run("run-1")).toEqual(running);
    expect(store.isBusy("run-1")).toBe(false);
  });

  it("preserves the exact logical command identity and params for a caller retry", async () => {
    const rpc = new TransportRpc();
    const created = runView();
    let fail = true;
    rpc.handler = (method) => {
      if (method === "run.command" && fail) throw new TypeError("network disconnected");
      if (method === "run.command") return commandResult();
      return runView({ status: "running", content_hash: HASH("1") });
    };
    const store = new RunStore(rpc, AUTHORITY, identities());
    store.acceptRun(created);

    await expect(store.command("run-1", "start")).rejects.toThrow("network disconnected");
    const first = rpc.calls.at(-1)!;
    expect(store.run("run-1")).toEqual(created);

    fail = false;
    await store.command("run-1", "start");
    const second = [...rpc.calls].reverse().find((call: RecordedCall) => call.method === "run.command")!;
    expect(second.params).toEqual(first.params);
    expect(second.params.command_id).toBe("command-1");
    expect(second.params.idempotency_key).toBe("command-key-1");
    expect(first.options?.requestId).toBe("command-1");
    expect(second.options?.requestId).toBe("command-1");
  });

  it("preserves command identity when authoritative refresh fails after acceptance", async () => {
    const rpc = new TransportRpc();
    const created = runView();
    let refreshFails = true;
    rpc.handler = (method) => {
      if (method === "run.command") return commandResult();
      if (refreshFails) throw new TypeError("run view transport disconnected");
      return runView({ status: "running", content_hash: HASH("1") });
    };
    const store = new RunStore(rpc, AUTHORITY, identities());
    store.acceptRun(created);

    await expect(store.command("run-1", "start")).rejects.toThrow("run view transport disconnected");
    const first = rpc.calls.find((call) => call.method === "run.command")!;
    refreshFails = false;
    await store.command("run-1", "start");
    const commands = rpc.calls.filter((call) => call.method === "run.command");

    expect(commands).toHaveLength(2);
    expect(commands[1]?.params).toEqual(first.params);
    expect(commands[1]?.options?.requestId).toBe(first.options?.requestId);
  });

  it("leaves the last accepted run intact when the server rejects a command", async () => {
    const rpc = new TransportRpc();
    const created = runView();
    rpc.handler = () => { throw new JsonRpcError(-32013, "Invalid run lifecycle"); };
    const store = new RunStore(rpc, AUTHORITY, identities());
    store.acceptRun(created);

    await expect(store.command("run-1", "pause")).rejects.toMatchObject({ code: -32013 });
    expect(store.run("run-1")).toEqual(created);
    expect(store.isBusy("run-1")).toBe(false);
  });

  it("uses checkpoint and fork contracts and serializes both source and child run gates", async () => {
    const rpc = new TransportRpc();
    const parent = runView({ status: "paused", checkpoint_hashes: [HASH("7")] });
    const child = runView({
      run_id: "run-child",
      stream_id: "stream-child",
      status: "paused",
      parent_checkpoint_hash: HASH("7"),
      state_hash: HASH("8"),
      content_hash: HASH("6"),
    });
    rpc.handler = (method, params) => {
      if (method === "run.fork") return {
        schema: "narrative-dynamics.scenario-fork-result/v1",
        fork_id: "fork-1",
        idempotency_key: "fork-key-1",
        request_hash: HASH("1"),
        capability_hash: HASH("2"),
        source_run_id: "run-1",
        child_run_id: "run-child",
        child_stream_id: "stream-child",
        scenario_hash: HASH("a"),
        child_epoch: 2,
        checkpoint_hash: HASH("7"),
        child_state_hash: HASH("8"),
        content_hash: HASH("3"),
      };
      if (method === "run.view") return params.run_id === "run-child" ? child : parent;
      throw new Error("unexpected call");
    };
    const store = new RunStore(rpc, AUTHORITY, identities());
    store.acceptRun(parent);

    const result = await store.fork("run-1", HASH("7"), "run-child", "stream-child");
    expect(result.child_state_hash).toBe(HASH("8"));
    expect(store.run("run-1")).toEqual(parent);
    expect(store.run("run-child")).toEqual(child);
    expect(rpc.calls[0]?.params).toMatchObject({
      source_run_id: "run-1",
      source_epoch: 1,
      checkpoint_hash: HASH("7"),
      child_run_id: "run-child",
      child_stream_id: "stream-child",
    });
  });

  it("preserves fork identity when child-view refresh fails after acceptance", async () => {
    const rpc = new TransportRpc();
    const parent = runView({ status: "paused", checkpoint_hashes: [HASH("7")] });
    const child = runView({
      run_id: "run-child", stream_id: "stream-child", status: "paused",
      parent_checkpoint_hash: HASH("7"), state_hash: HASH("8"), content_hash: HASH("6"),
    });
    let childRefreshFails = true;
    rpc.handler = (method, params) => {
      if (method === "run.fork") return {
        schema: "narrative-dynamics.scenario-fork-result/v1",
        fork_id: "fork-1", idempotency_key: "fork-key-1",
        request_hash: HASH("1"), capability_hash: HASH("2"),
        source_run_id: "run-1", child_run_id: "run-child", child_stream_id: "stream-child",
        scenario_hash: HASH("a"), child_epoch: 2, checkpoint_hash: HASH("7"),
        child_state_hash: HASH("8"), content_hash: HASH("3"),
      };
      if (params.run_id === "run-child" && childRefreshFails) {
        throw new TypeError("child view transport disconnected");
      }
      return params.run_id === "run-child" ? child : parent;
    };
    const store = new RunStore(rpc, AUTHORITY, identities());
    store.acceptRun(parent);

    await expect(store.fork("run-1", HASH("7"), "run-child", "stream-child"))
      .rejects.toThrow("child view transport disconnected");
    const first = rpc.calls.find((call) => call.method === "run.fork")!;
    childRefreshFails = false;
    await store.fork("run-1", HASH("7"), "run-child", "stream-child");
    const forks = rpc.calls.filter((call) => call.method === "run.fork");

    expect(forks).toHaveLength(2);
    expect(forks[1]?.params).toEqual(first.params);
    expect(forks[1]?.options?.requestId).toBe(first.options?.requestId);
  });

  it("clears private state and retained output before requesting another audience", async () => {
    const rpc = new TransportRpc();
    const store = new RunStore(rpc, AUTHORITY, identities());
    store.acceptRun(runView({ status: "running" }));
    store.acceptScopedState("run-1", "agent", "alice", { agent_id: "alice", secret: "private" });
    store.acceptOutput("run-1", {
      schema: "narrative-dynamics.simulation-output-view/v1",
      stream_id: "stream-1",
      scenario_hash: HASH("a"),
      prior_state_hash: HASH("b"),
      next_state_hash: HASH("b"),
      round_result_hash: HASH("4"),
      first_sequence: 1,
      last_sequence: 1,
      records: [],
      source_batch_hash: HASH("5"),
      checkpoint: false,
      content_hash: HASH("6"),
    });
    let observedCleared = false;
    rpc.handler = () => {
      observedCleared = store.scopedState("run-1") === null && store.outputs("run-1").length === 0;
      return {
        schema: "narrative-dynamics.scenario-public-state-view/v1",
        run_id: "run-1", scenario_hash: HASH("a"), round_index: 0,
        state_hash: HASH("b"), content_hash: HASH("9"),
      };
    };

    await store.selectAudience("run-1", { audience: "public", owner_agent_id: null });
    expect(observedCleared).toBe(true);
    expect(rpc.calls[0]).toMatchObject({ method: "state.public", params: { run_id: "run-1" } });
  });

  it("never exposes or probes an owner absent from authoritative capability data", async () => {
    const rpc = new TransportRpc();
    const store = new RunStore(rpc, AUTHORITY, identities());
    store.acceptRun(runView());

    expect(store.authorizedAgentIds).toEqual(["alice"]);
    await expect(store.selectAudience("run-1", { audience: "agent", owner_agent_id: "bob" }))
      .rejects.toThrow("not authorized");
    expect(rpc.calls).toHaveLength(0);
  });
});
