import { fireEvent, getAllByRole, getByLabelText, getByRole, queryByRole } from "@testing-library/dom";
import { afterEach, describe, expect, it } from "vitest";
import type { JsonObject, RunAuthority, ScenarioRunView, SimulationOutputView } from "../schema/studio-types";
import type { RpcCaller } from "../rpc/client";
import { RunStore } from "../state/run-store";
import type { ForkComparisonElement } from "./fork-comparison";
import type { OutputTimelineElement } from "./output-timeline";
import type { RunToolbarElement } from "./run-toolbar";
import type { StateInspectorElement } from "./state-inspector";
import "./fork-comparison";
import "./output-timeline";
import "./run-toolbar";
import "./state-inspector";

const HASH = (character: string) => `sha256:${character.repeat(64)}`;

function view(overrides: Partial<ScenarioRunView> = {}): ScenarioRunView {
  return {
    schema: "narrative-dynamics.scenario-run-view/v1",
    run_id: "run-parent",
    stream_id: "stream-parent",
    scenario_hash: HASH("a"),
    coordinator_epoch: 3,
    status: "running",
    round_index: 4,
    state_hash: HASH("b"),
    next_sequence: 11,
    output_batch_hashes: [HASH("1")],
    checkpoint_hashes: [HASH("2")],
    parent_checkpoint_hash: null,
    content_hash: HASH("c"),
    ...overrides,
  };
}

function authority(overrides: Partial<RunAuthority> = {}): RunAuthority {
  return {
    authority_id: "operator",
    project_ids: ["law-firm"],
    run_ids: ["run-parent", "run-child"],
    agent_ids: ["alice"],
    permissions: ["run.command", "run.fork", "run.read", "state.public", "state.agent", "state.network", "output.read"],
    ...overrides,
  };
}

class Rpc implements RpcCaller {
  calls: { method: string; params: JsonObject }[] = [];
  handler: (method: string, params: JsonObject) => unknown | Promise<unknown> = () => ({});
  async call<T>(method: string, params: JsonObject): Promise<T> {
    this.calls.push({ method, params });
    return await this.handler(method, params) as T;
  }
}

function storeWith(run: ScenarioRunView, auth = authority(), rpc = new Rpc()): [RunStore, Rpc] {
  const store = new RunStore(rpc, auth, {
    commandId: () => "command-ui",
    idempotencyKey: () => "command-key-ui",
    forkId: () => "fork-ui",
    forkIdempotencyKey: () => "fork-key-ui",
  });
  store.acceptRun(run);
  return [store, rpc];
}

function output(
  first: number,
  last: number,
  hash: string,
  records: SimulationOutputView["records"] = [],
): SimulationOutputView {
  return {
    schema: "narrative-dynamics.simulation-output-view/v1",
    stream_id: "stream-parent",
    scenario_hash: HASH("a"),
    prior_state_hash: HASH("b"),
    next_state_hash: HASH("b"),
    round_result_hash: HASH("d"),
    first_sequence: first,
    last_sequence: last,
    records,
    source_batch_hash: HASH(hash),
    checkpoint: false,
    content_hash: HASH("e"),
  };
}

afterEach(() => document.body.replaceChildren());

describe("run toolbar", () => {
  it("enables lifecycle actions only from authoritative status and capability", () => {
    const [store] = storeWith(view({ status: "created" }));
    const toolbar = document.createElement("run-toolbar") as RunToolbarElement;
    toolbar.store = store;
    toolbar.runId = "run-parent";
    document.body.append(toolbar);

    expect((getByRole(toolbar, "button", { name: "Start run" }) as HTMLButtonElement).disabled).toBe(false);
    expect((getByRole(toolbar, "button", { name: "Pause run" }) as HTMLButtonElement).disabled).toBe(true);
    expect((getByRole(toolbar, "button", { name: "Resume run" }) as HTMLButtonElement).disabled).toBe(true);
    expect((getByRole(toolbar, "button", { name: "Step run" }) as HTMLButtonElement).disabled).toBe(true);
    expect((getByRole(toolbar, "button", { name: "Stop run" }) as HTMLButtonElement).disabled).toBe(false);
    expect(getByRole(toolbar, "button", { name: "Create checkpoint" })).toBeTruthy();
    expect(getByRole(toolbar, "button", { name: "Fork run" })).toBeTruthy();
    expect(toolbar.querySelector("[tabindex='1']")).toBeNull();
  });

  it("exposes textual and programmatic busy state while a step is in flight", async () => {
    const [store, rpc] = storeWith(view());
    let complete!: (value: unknown) => void;
    rpc.handler = (method) => method === "run.command"
      ? new Promise((resolve) => { complete = resolve; })
      : view({ round_index: 5, state_hash: HASH("f"), next_sequence: 12, content_hash: HASH("6") });
    const toolbar = document.createElement("run-toolbar") as RunToolbarElement;
    toolbar.store = store;
    toolbar.runId = "run-parent";
    document.body.append(toolbar);

    fireEvent.click(getByRole(toolbar, "button", { name: "Step run" }));
    expect(toolbar.getAttribute("aria-busy")).toBe("true");
    expect(getByRole(toolbar, "status").textContent).toContain("Step in progress");
    expect(getAllByRole(toolbar, "button").every((button) => (button as HTMLButtonElement).disabled)).toBe(true);

    complete({
      schema: "narrative-dynamics.scenario-command-result/v1",
      command_id: "command-ui", idempotency_key: "command-key-ui",
      request_hash: HASH("1"), capability_hash: HASH("2"), run_id: "run-parent",
      scenario_hash: HASH("a"), coordinator_epoch: 3, kind: "step", accepted: true,
      reason: "accepted", prior_status: "running", next_status: "running",
      prior_state_hash: HASH("b"), next_state_hash: HASH("f"), round_index: 5,
      output_batch_hash: HASH("3"), checkpoint_hash: null, content_hash: HASH("4"),
    });
    for (let index = 0; index < 8; index += 1) await Promise.resolve();
    expect(toolbar.getAttribute("aria-busy")).toBe("false");
    expect(getByRole(toolbar, "status").textContent).toContain("Round 5");
  });

  it("does not render checkpoint or fork actions without authoritative permissions", () => {
    const [store] = storeWith(view(), authority({ permissions: ["run.read"] }));
    const toolbar = document.createElement("run-toolbar") as RunToolbarElement;
    toolbar.store = store;
    toolbar.runId = "run-parent";
    document.body.append(toolbar);

    expect(queryByRole(toolbar, "button", { name: "Create checkpoint" })).toBeNull();
    expect(queryByRole(toolbar, "button", { name: "Fork run" })).toBeNull();
  });
});

describe("state inspector", () => {
  it("shows exact run state and only server-authorized audience choices", () => {
    const [store] = storeWith(view());
    const inspector = document.createElement("state-inspector") as StateInspectorElement;
    inspector.store = store;
    inspector.runId = "run-parent";
    document.body.append(inspector);

    expect(getByRole(inspector, "region", { name: "Run state inspector" }).textContent).toContain(HASH("b"));
    const audience = getByLabelText(inspector, "State audience") as HTMLSelectElement;
    expect([...audience.options].map((option) => option.value)).toEqual(["public", "agent:alice", "network"]);
    expect(inspector.textContent).not.toContain("bob");
  });

  it("clears the prior private rendering before loading public state", async () => {
    const [store, rpc] = storeWith(view());
    store.acceptScopedState("run-parent", "agent", "alice", { secret: "private-for-alice" });
    let wasCleared = false;
    rpc.handler = () => {
      wasCleared = !document.body.textContent?.includes("private-for-alice");
      return {
        schema: "narrative-dynamics.scenario-public-state-view/v1",
        run_id: "run-parent", scenario_hash: HASH("a"), round_index: 4,
        state_hash: HASH("b"), content_hash: HASH("3"),
      };
    };
    const inspector = document.createElement("state-inspector") as StateInspectorElement;
    inspector.store = store;
    inspector.runId = "run-parent";
    document.body.append(inspector);
    const audience = getByLabelText(inspector, "State audience") as HTMLSelectElement;

    fireEvent.change(audience, { target: { value: "public" } });
    for (let index = 0; index < 5; index += 1) await Promise.resolve();
    expect(wasCleared).toBe(true);
    expect(inspector.textContent).not.toContain("private-for-alice");
  });
});

describe("output timeline", () => {
  it("preserves source sequence, renders explicit gap/recovery markers, and bounds retention", () => {
    const timeline = document.createElement("output-timeline") as OutputTimelineElement;
    timeline.maximumBatches = 2;
    timeline.batches = [output(1, 1, "1"), output(2, 2, "2"), output(5, 5, "5")];
    timeline.markers = [{ kind: "gap", after_sequence: 2, before_sequence: 5, label: "Recovered from scoped snapshot" }];
    document.body.append(timeline);

    const items = getAllByRole(timeline, "listitem").map((item) => item.textContent ?? "");
    expect(items).toHaveLength(3);
    expect(items[0]).toContain("Sequence 2");
    expect(items[1]).toContain("Gap 3–4");
    expect(items[2]).toContain("Sequence 5");
    expect(items.join(" ")).not.toContain("Sequence 1");
    expect(getByRole(timeline, "status").textContent).toContain("2 retained batches");
  });

  it("renders the exact owner on private records so privacy remains inspectable", () => {
    const timeline = document.createElement("output-timeline") as OutputTimelineElement;
    timeline.batches = [output(1, 1, "1", [{
      schema: "narrative-dynamics.simulation-output-record/v1",
      stream_id: "stream-parent", scenario_hash: HASH("a"), sequence: 1, round_index: 4,
      state_hash: HASH("b"), kind: "diagnostic", audience: "agent", owner_agent_id: "alice",
      source_artifact_hashes: [], payload: { code: "visible", message: "Scoped diagnostic" },
      payload_hash: HASH("7"), content_hash: HASH("8"),
    }])];
    document.body.append(timeline);

    expect(getByRole(timeline, "listitem").textContent).toContain("diagnostic (agent, owner alice)");
  });
});

describe("fork comparison", () => {
  it("labels both sides and reports exact lineage and metrics without causal language", () => {
    const comparison = document.createElement("fork-comparison") as ForkComparisonElement;
    comparison.parent = view();
    comparison.child = view({
      run_id: "run-child", stream_id: "stream-child", coordinator_epoch: 4,
      status: "paused", round_index: 5, state_hash: HASH("f"),
      parent_checkpoint_hash: HASH("2"), content_hash: HASH("6"),
    });
    comparison.parentMetrics = { adoption_rate: 0.25 };
    comparison.childMetrics = { adoption_rate: 0.5 };
    document.body.append(comparison);

    expect(getByRole(comparison, "region", { name: "Parent run" }).textContent).toContain(HASH("b"));
    expect(getByRole(comparison, "region", { name: "Child run" }).textContent).toContain(HASH("f"));
    expect(comparison.textContent).toContain(HASH("2"));
    expect(comparison.textContent).toContain("0.25");
    expect(comparison.textContent).toContain("0.5");
    expect(comparison.textContent?.toLowerCase()).not.toContain("caused");
  });
});
