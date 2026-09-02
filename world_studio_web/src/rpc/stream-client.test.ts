import { describe, expect, it } from "vitest";
import type { JsonObject, SimulationOutputView, StreamBinding } from "../schema/studio-types";
import { StreamClient, StreamProtocolError, type WebSocketLike } from "./stream-client";

const HASH = (character: string) => `sha256:${character.repeat(64)}`;

class Socket implements WebSocketLike {
  protocol = "nd-jsonrpc-v1";
  readyState = 0;
  sent: string[] = [];
  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;

  send(data: string): void { this.sent.push(data); }
  close(): void { this.readyState = 3; this.onclose?.(new CloseEvent("close")); }
  open(): void { this.readyState = 1; this.onopen?.(new Event("open")); }
  receive(value: unknown): void {
    this.onmessage?.(new MessageEvent("message", { data: JSON.stringify(value) }));
  }
  receiveRaw(value: string): void {
    this.onmessage?.(new MessageEvent("message", { data: value }));
  }
  request(index: number): JsonObject { return JSON.parse(this.sent[index]!) as JsonObject; }
  respond(index: number, result: JsonObject): void {
    const request = this.request(index);
    this.receive({ jsonrpc: "2.0", id: request.id, result });
  }
}

function binding(overrides: Partial<StreamBinding> = {}): StreamBinding {
  return {
    subscription_id: "subscription-1",
    run_id: "run-1",
    stream_id: "stream-1",
    scenario_hash: HASH("a"),
    kinds: ["state.delta", "network.metrics"],
    audience: "public",
    owner_agent_id: null,
    ...overrides,
  };
}

function output(first: number, last: number, hashCharacter: string): SimulationOutputView {
  return {
    schema: "narrative-dynamics.simulation-output-view/v1",
    stream_id: "stream-1",
    scenario_hash: HASH("a"),
    prior_state_hash: HASH("b"),
    next_state_hash: HASH("c"),
    round_result_hash: HASH("d"),
    first_sequence: first,
    last_sequence: last,
    records: [{
      schema: "narrative-dynamics.simulation-output-record/v1",
      stream_id: "stream-1",
      scenario_hash: HASH("a"),
      sequence: first,
      round_index: 1,
      state_hash: HASH("c"),
      kind: "state.delta",
      audience: "public",
      owner_agent_id: null,
      source_artifact_hashes: [HASH("d")],
      payload: {
        prior_snapshot_hash: HASH("b"),
        next_snapshot_hash: HASH("c"),
        changed_agent_ids: [],
        changed_passage_ids: [],
        changed_object_ids: [],
      },
      payload_hash: HASH("e"),
      content_hash: HASH("f"),
    }],
    source_batch_hash: HASH(hashCharacter),
    checkpoint: false,
    content_hash: HASH("9"),
  };
}

async function flush(): Promise<void> {
  for (let index = 0; index < 12; index += 1) await Promise.resolve();
}

async function connectedClient(options: ConstructorParameters<typeof StreamClient>[1] = {}) {
  const sockets: Socket[] = [];
  const batches: SimulationOutputView[] = [];
  const client = new StreamClient("ws://example.test/v1/stream", {
    socketFactory: (_url, protocols) => {
      expect(protocols).toEqual(["nd-jsonrpc-v1"]);
      const socket = new Socket();
      sockets.push(socket);
      return socket;
    },
    onOutput: async (batch) => { batches.push(batch); },
    ...options,
  });
  const connecting = client.connect(binding());
  sockets[0]!.open();
  expect(sockets[0]!.request(0)).toMatchObject({
    jsonrpc: "2.0",
    method: "stream.subscribe",
    params: {
      subscription_id: "subscription-1",
      run_id: "run-1",
      stream_id: "stream-1",
      kinds: ["state.delta", "network.metrics"],
      audience: "public",
    },
  });
  sockets[0]!.respond(0, {
    subscription_id: "subscription-1", run_id: "run-1", stream_id: "stream-1",
    kinds: ["state.delta", "network.metrics"], audience: "public",
  });
  await connecting;
  return { client, sockets, batches };
}

describe("StreamClient strict JSON-RPC transport", () => {
  it("rejects duplicate JSON object keys before JSON-RPC dispatch", async () => {
    const failures: Error[] = [];
    const { sockets } = await connectedClient({ onProtocolError: (error) => failures.push(error) });
    sockets[0]!.receiveRaw('{"jsonrpc":"2.0","jsonrpc":"2.0","method":"stream.output","params":{}}');
    await flush();
    expect(failures).toHaveLength(1);
    expect(failures[0]?.message).toContain("duplicate");
  });

  it("bounds the retained logical batch identity ledger", async () => {
    const failures: Error[] = [];
    const { client, sockets, batches } = await connectedClient({
      maximumRetainedIdentities: 2,
      onProtocolError: (error) => failures.push(error),
    });
    for (let sequence = 1; sequence <= 3; sequence += 1) {
      sockets[0]!.receive({ jsonrpc: "2.0", method: "stream.output", params: {
        subscription_id: "subscription-1", output: output(sequence, sequence, String(sequence)),
      }});
      await flush();
      sockets[0]!.respond(sequence, { subscription_id: "subscription-1", acknowledged: true });
      await flush();
    }
    expect(batches).toHaveLength(3);
    expect(client.retainedIdentityCount).toBe(2);
    sockets[0]!.receive({ jsonrpc: "2.0", method: "stream.output", params: {
      subscription_id: "subscription-1", output: output(1, 1, "1"),
    }});
    await flush();
    expect(failures.at(-1)?.message).toContain("regressed");
  });

  it("requires the exact negotiated subprotocol", async () => {
    const socket = new Socket();
    socket.protocol = "";
    const client = new StreamClient("ws://example.test/v1/stream", {
      socketFactory: () => socket,
    });
    const connecting = client.connect(binding());
    socket.open();
    await expect(connecting).rejects.toThrow("nd-jsonrpc-v1");
    expect(socket.readyState).toBe(3);
  });

  it("validates binding, commits in source order, then acknowledges monotonically", async () => {
    const { client, sockets, batches } = await connectedClient();
    const first = output(1, 1, "1");
    sockets[0]!.receive({
      jsonrpc: "2.0", method: "stream.output",
      params: { subscription_id: "subscription-1", output: first },
    });
    await flush();

    expect(batches).toEqual([first]);
    expect(sockets[0]!.request(1)).toMatchObject({
      method: "stream.acknowledge",
      params: {
        subscription_id: "subscription-1",
        stream_id: "stream-1",
        last_sequence: 1,
        last_batch_hash: HASH("1"),
      },
    });
    sockets[0]!.respond(1, { subscription_id: "subscription-1", acknowledged: true });
    await flush();
    expect(client.cursor).toEqual({ sequence: 1, batch_hash: HASH("1") });
  });

  it("re-acknowledges a committed duplicate after an acknowledgement response is lost", async () => {
    const { client, sockets, batches } = await connectedClient();
    const first = output(1, 1, "1");
    sockets[0]!.receive({ jsonrpc: "2.0", method: "stream.output", params: {
      subscription_id: "subscription-1", output: first,
    }});
    await flush();
    expect(sockets[0]!.request(1)).toMatchObject({ method: "stream.acknowledge" });
    expect(client.cursor).toEqual({ sequence: 1, batch_hash: HASH("1") });

    sockets[0]!.close();
    await flush();
    const reconnecting = client.reconnect();
    sockets[1]!.open();
    sockets[1]!.respond(0, {
      subscription_id: "subscription-1", run_id: "run-1", stream_id: "stream-1",
      kinds: ["state.delta", "network.metrics"], audience: "public",
    });
    await flush();
    expect(sockets[1]!.request(1)).toMatchObject({ method: "stream.resume", params: {
      last_sequence: 1, last_batch_hash: HASH("1"),
    }});
    sockets[1]!.respond(1, {
      subscription_id: "subscription-1", resumed_from_sequence: 1,
      replayed_count: 0, current_sequence: 1, current_batch_hash: HASH("1"),
    });
    await reconnecting;

    sockets[1]!.receive({ jsonrpc: "2.0", method: "stream.output", params: {
      subscription_id: "subscription-1", output: first,
    }});
    await flush();
    expect(sockets[1]!.request(2)).toMatchObject({ method: "stream.acknowledge", params: {
      last_sequence: 1, last_batch_hash: HASH("1"),
    }});
    sockets[1]!.respond(2, { subscription_id: "subscription-1", acknowledged: true });
    await flush();

    const second = output(2, 2, "2");
    sockets[1]!.receive({ jsonrpc: "2.0", method: "stream.output", params: {
      subscription_id: "subscription-1", output: second,
    }});
    await flush();
    expect(sockets[1]!.request(3)).toMatchObject({ method: "stream.acknowledge", params: {
      last_sequence: 2, last_batch_hash: HASH("2"),
    }});
    sockets[1]!.respond(3, { subscription_id: "subscription-1", acknowledged: true });
    await flush();
    expect(batches).toEqual([first, second]);
    expect(client.cursor).toEqual({ sequence: 2, batch_hash: HASH("2") });
  });

  it("deduplicates an exact batch and rejects conflicts, regressions, and gaps", async () => {
    const failures: Error[] = [];
    const markers: JsonObject[] = [];
    const { sockets, batches } = await connectedClient({
      onProtocolError: (error) => failures.push(error),
      onMarker: (_runId, marker) => markers.push(marker),
    });
    const first = output(1, 1, "1");
    const notify = (batch: SimulationOutputView) => sockets[0]!.receive({
      jsonrpc: "2.0", method: "stream.output",
      params: { subscription_id: "subscription-1", output: batch },
    });
    notify(first);
    await flush();
    sockets[0]!.respond(1, { subscription_id: "subscription-1", acknowledged: true });
    await flush();
    notify(first);
    notify(output(1, 1, "2"));
    notify(output(3, 3, "3"));
    await flush();

    expect(batches).toEqual([first]);
    expect(failures).toHaveLength(2);
    expect(failures.every((failure) => failure instanceof StreamProtocolError)).toBe(true);
    expect(markers).toEqual([{
      kind: "gap", after_sequence: 1, before_sequence: 3,
      label: "Output gap detected before source sequence 3.",
    }]);
  });

  it("reconnects by recreating the identical released subscription before resuming", async () => {
    const markers: JsonObject[] = [];
    const { client, sockets } = await connectedClient({
      onMarker: (_runId, marker) => markers.push(marker),
    });
    const first = output(1, 1, "1");
    sockets[0]!.receive({ jsonrpc: "2.0", method: "stream.output", params: {
      subscription_id: "subscription-1", output: first,
    }});
    await flush();
    sockets[0]!.respond(1, { subscription_id: "subscription-1", acknowledged: true });
    await flush();

    sockets[0]!.close();
    const reconnecting = client.reconnect();
    sockets[1]!.open();
    expect(sockets[1]!.request(0)).toMatchObject({ method: "stream.subscribe", params: {
      subscription_id: "subscription-1", run_id: "run-1", stream_id: "stream-1",
    }});
    sockets[1]!.respond(0, {
      subscription_id: "subscription-1", run_id: "run-1", stream_id: "stream-1",
      kinds: ["state.delta", "network.metrics"], audience: "public",
    });
    await flush();
    expect(sockets[1]!.request(1)).toMatchObject({ method: "stream.resume", params: {
      subscription_id: "subscription-1", stream_id: "stream-1",
      last_sequence: 1, last_batch_hash: HASH("1"),
    }});
    sockets[1]!.respond(1, {
      subscription_id: "subscription-1", resumed_from_sequence: 1,
      replayed_count: 0, current_sequence: 1, current_batch_hash: HASH("1"),
    });
    await expect(reconnecting).resolves.toBeUndefined();
    expect(markers).toEqual([{
      kind: "resume", after_sequence: 1, before_sequence: 1,
      label: "Stream resumed from source sequence 1.",
    }]);
  });

  it("uses only a scoped gap token, resets the cursor, and resubscribes", async () => {
    const recovered: JsonObject[] = [];
    const markers: JsonObject[] = [];
    const { client, sockets } = await connectedClient({
      recoverGap: async (request) => { recovered.push(request); },
      onMarker: (_runId, marker) => markers.push(marker),
    });
    const first = output(1, 1, "1");
    sockets[0]!.receive({ jsonrpc: "2.0", method: "stream.output", params: {
      subscription_id: "subscription-1", output: first,
    }});
    await flush();
    sockets[0]!.respond(1, { subscription_id: "subscription-1", acknowledged: true });
    await flush();
    sockets[0]!.close();
    const reconnecting = client.reconnect();
    sockets[1]!.open();
    sockets[1]!.respond(0, {
      subscription_id: "subscription-1", run_id: "run-1", stream_id: "stream-1",
      kinds: ["state.delta", "network.metrics"], audience: "public",
    });
    await flush();
    const resume = sockets[1]!.request(1);
    sockets[1]!.receive({ jsonrpc: "2.0", id: resume.id, error: {
      code: -32016,
      message: "Subscription history gap",
      data: {
        current_sequence: 8,
        current_batch_hash: HASH("8"),
        snapshot_token: HASH("7"),
      },
    }});
    await flush();
    expect(recovered).toEqual([{
      run_id: "run-1",
      audience: "public",
      owner_agent_id: null,
      snapshot_token: HASH("7"),
    }]);
    expect(client.cursor).toEqual({ sequence: 8, batch_hash: HASH("8") });
    expect(sockets[1]!.request(2)).toMatchObject({ method: "stream.unsubscribe" });
    sockets[1]!.respond(2, { subscription_id: "subscription-1", unsubscribed: true });
    await flush();
    expect(sockets[1]!.request(3)).toMatchObject({ method: "stream.subscribe" });
    sockets[1]!.respond(3, {
      subscription_id: "subscription-1", run_id: "run-1", stream_id: "stream-1",
      kinds: ["state.delta", "network.metrics"], audience: "public",
    });
    await expect(reconnecting).resolves.toBeUndefined();
    expect(markers.map((marker) => marker.kind)).toEqual([
      "gap", "recovery-started", "recovery-completed",
    ]);
  });

  it("marks scoped recovery failure from the actual -32016 transition", async () => {
    const markers: JsonObject[] = [];
    const { client, sockets } = await connectedClient({
      recoverGap: async () => { throw new Error("scoped state failed"); },
      onMarker: (_runId, marker) => markers.push(marker),
    });
    const first = output(1, 1, "1");
    sockets[0]!.receive({ jsonrpc: "2.0", method: "stream.output", params: {
      subscription_id: "subscription-1", output: first,
    }});
    await flush();
    sockets[0]!.respond(1, { subscription_id: "subscription-1", acknowledged: true });
    await flush();
    sockets[0]!.close();
    const reconnecting = client.reconnect();
    sockets[1]!.open();
    sockets[1]!.respond(0, {
      subscription_id: "subscription-1", run_id: "run-1", stream_id: "stream-1",
      kinds: ["state.delta", "network.metrics"], audience: "public",
    });
    await flush();
    const resume = sockets[1]!.request(1);
    sockets[1]!.receive({ jsonrpc: "2.0", id: resume.id, error: {
      code: -32016, message: "Subscription history gap", data: {
        current_sequence: 8, current_batch_hash: HASH("8"), snapshot_token: HASH("7"),
      },
    }});

    await expect(reconnecting).rejects.toThrow("scoped state failed");
    expect(markers.map((marker) => marker.kind)).toEqual([
      "gap", "recovery-started", "recovery-failed",
    ]);
  });

  it("clears private data synchronously before requesting a new audience binding", async () => {
    const { client, sockets } = await connectedClient();
    let cleared = false;
    const switching = client.switchAudience(binding({
      subscription_id: "subscription-agent",
      audience: "agent",
      owner_agent_id: "alice",
    }), () => { cleared = true; });
    expect(cleared).toBe(true);
    await flush();
    expect(sockets[0]!.request(1)).toMatchObject({ method: "stream.unsubscribe" });
    sockets[0]!.respond(1, { subscription_id: "subscription-1", unsubscribed: true });
    await flush();
    expect(sockets[0]!.request(2)).toMatchObject({ method: "stream.subscribe", params: {
      subscription_id: "subscription-agent", audience: "agent", owner_agent_id: "alice",
    }});
    sockets[0]!.respond(2, {
      subscription_id: "subscription-agent", run_id: "run-1", stream_id: "stream-1",
      kinds: ["state.delta", "network.metrics"], audience: "agent", owner_agent_id: "alice",
    });
    await expect(switching).resolves.toBeUndefined();
  });

  it("never commits an older Agent switch after a newer public selection", async () => {
    const { client, sockets } = await connectedClient();
    let clears = 0;
    const older = client.switchAudience(binding({
      subscription_id: "subscription-agent", audience: "agent", owner_agent_id: "alice",
    }), () => { clears += 1; });
    await flush();
    expect(sockets[0]!.request(1)).toMatchObject({ method: "stream.unsubscribe" });

    const newer = client.switchAudience(binding({
      subscription_id: "subscription-public-new", audience: "public", owner_agent_id: null,
    }), () => { clears += 1; });
    expect(clears).toBe(2);
    sockets[0]!.respond(1, { subscription_id: "subscription-1", unsubscribed: true });
    await expect(older).rejects.toThrow("superseded");
    await flush();

    expect(sockets[0]!.sent).toHaveLength(3);
    expect(sockets[0]!.request(2)).toMatchObject({ method: "stream.subscribe", params: {
      subscription_id: "subscription-public-new", audience: "public",
    }});
    expect(sockets[0]!.sent.some((raw) => raw.includes("subscription-agent"))).toBe(false);
    sockets[0]!.respond(2, {
      subscription_id: "subscription-public-new", run_id: "run-1", stream_id: "stream-1",
      kinds: ["state.delta", "network.metrics"], audience: "public",
    });
    await expect(newer).resolves.toBeUndefined();
    expect(client.binding?.audience).toBe("public");
  });

  it("rejects output for another subscription, stream, run scenario, or malformed envelope", async () => {
    const failures: Error[] = [];
    const { sockets, batches } = await connectedClient({ onProtocolError: (error) => failures.push(error) });
    sockets[0]!.receive({ jsonrpc: "2.0", method: "stream.output", params: {
      subscription_id: "other", output: output(1, 1, "1"),
    }});
    sockets[0]!.receive({ jsonrpc: "2.0", method: "stream.output", params: {
      subscription_id: "subscription-1", output: { ...output(1, 1, "1"), stream_id: "other" },
    }});
    sockets[0]!.receive({ jsonrpc: "2.0", method: "stream.output", params: {
      subscription_id: "subscription-1", output: { ...output(1, 1, "1"), scenario_hash: HASH("0") },
    }});
    sockets[0]!.receive({ jsonrpc: "2.0", method: "stream.output", params: {}, extra: true });
    await flush();
    expect(batches).toHaveLength(0);
    expect(failures).toHaveLength(4);
  });
});
