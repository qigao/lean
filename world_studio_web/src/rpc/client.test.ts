import { describe, expect, it } from "vitest";
import { JsonRpcClient, JsonRpcError, JsonRpcProtocolError } from "./client";

describe("JsonRpcClient", () => {
  it("retries one logical mutation with the same request and idempotency identities", async () => {
    const bodies: string[] = [];
    let attempt = 0;
    const fetcher: typeof fetch = async (_input, init) => {
      bodies.push(String(init?.body));
      attempt += 1;
      if (attempt === 1) throw new TypeError("connection reset");
      return new Response(JSON.stringify({ jsonrpc: "2.0", id: "rpc-7", result: { accepted: true } }), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    };
    const client = new JsonRpcClient("/rpc", { fetcher, requestId: () => "rpc-7" });

    await expect(client.call("project.apply", { idempotency_key: "operation-key-1" }, {
      stateChanging: true,
      attempts: 2,
    })).resolves.toEqual({ accepted: true });

    expect(bodies).toEqual([
      '{"jsonrpc":"2.0","id":"rpc-7","method":"project.apply","params":{"idempotency_key":"operation-key-1"}}',
      '{"jsonrpc":"2.0","id":"rpc-7","method":"project.apply","params":{"idempotency_key":"operation-key-1"}}',
    ]);
  });

  it("rejects non-canonical request values before network I/O", async () => {
    let requests = 0;
    const client = new JsonRpcClient("/rpc", {
      fetcher: async () => {
        requests += 1;
        return new Response();
      },
      requestId: () => "rpc-1",
    });

    await expect(client.call("project.snapshot", { value: Number.NaN })).rejects.toBeInstanceOf(JsonRpcProtocolError);
    await expect(client.call("project.snapshot", { value: undefined } as never)).rejects.toBeInstanceOf(JsonRpcProtocolError);
    expect(requests).toBe(0);
  });

  it("normalizes application errors and rejects malformed response envelopes", async () => {
    const responses = [
      { jsonrpc: "2.0", id: "rpc-1", error: { code: -32011, message: "Stale state", data: { code: "stale_state" } } },
      { jsonrpc: "2.0", id: "wrong", result: {} },
    ];
    const client = new JsonRpcClient("/rpc", {
      fetcher: async () => new Response(JSON.stringify(responses.shift())),
      requestId: () => "rpc-1",
    });

    const stale = await client.call("project.apply", {}).catch((error: unknown) => error);
    expect(stale).toBeInstanceOf(JsonRpcError);
    expect((stale as JsonRpcError).code).toBe(-32011);
    await expect(client.call("project.snapshot", {})).rejects.toBeInstanceOf(JsonRpcProtocolError);
  });
});
