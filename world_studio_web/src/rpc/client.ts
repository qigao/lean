import type { JsonObject, JsonValue } from "../schema/studio-types";

export type JsonRpcId = number | string;

export interface JsonRpcCallOptions {
  stateChanging?: boolean;
  attempts?: number;
  requestId?: JsonRpcId;
}

export interface RpcCaller {
  call<T>(method: string, params: JsonObject, options?: JsonRpcCallOptions): Promise<T>;
}

export class JsonRpcProtocolError extends Error {
  constructor(message = "Malformed JSON-RPC envelope") {
    super(message);
    this.name = "JsonRpcProtocolError";
  }
}

export class JsonRpcError extends Error {
  readonly code: number;
  readonly data: JsonValue | undefined;

  constructor(code: number, message: string, data?: JsonValue) {
    super(message);
    this.name = "JsonRpcError";
    this.code = code;
    this.data = data;
  }
}

export interface JsonRpcClientDependencies {
  fetcher?: typeof fetch;
  requestId?: () => JsonRpcId;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

export function assertJsonValue(value: unknown, seen = new Set<object>()): asserts value is JsonValue {
  if (value === null || typeof value === "string" || typeof value === "boolean") return;
  if (typeof value === "number") {
    if (Number.isFinite(value)) return;
    throw new JsonRpcProtocolError("JSON values must contain only finite numbers");
  }
  if (typeof value !== "object") throw new JsonRpcProtocolError("Value is not JSON-compatible");
  if (seen.has(value)) throw new JsonRpcProtocolError("JSON values cannot be cyclic");
  seen.add(value);
  if (Array.isArray(value)) {
    for (const item of value) assertJsonValue(item, seen);
  } else {
    if (!isRecord(value)) throw new JsonRpcProtocolError("JSON objects must be plain records");
    for (const item of Object.values(value)) assertJsonValue(item, seen);
  }
  seen.delete(value);
}

function hasOnlyKeys(record: Record<string, unknown>, allowed: readonly string[]): boolean {
  const keys = Object.keys(record);
  return keys.every((key) => allowed.includes(key)) && new Set(keys).size === keys.length;
}

function decodeResponse(raw: unknown, requestId: JsonRpcId): JsonValue {
  if (!isRecord(raw) || raw.jsonrpc !== "2.0" || raw.id !== requestId) {
    throw new JsonRpcProtocolError();
  }
  const hasResult = Object.prototype.hasOwnProperty.call(raw, "result");
  const hasError = Object.prototype.hasOwnProperty.call(raw, "error");
  if (hasResult === hasError) throw new JsonRpcProtocolError();
  if (hasResult) {
    if (!hasOnlyKeys(raw, ["jsonrpc", "id", "result"])) throw new JsonRpcProtocolError();
    assertJsonValue(raw.result);
    return raw.result;
  }
  if (!hasOnlyKeys(raw, ["jsonrpc", "id", "error"]) || !isRecord(raw.error)) {
    throw new JsonRpcProtocolError();
  }
  if (!hasOnlyKeys(raw.error, ["code", "message", "data"]) ||
      !Number.isInteger(raw.error.code) || typeof raw.error.message !== "string") {
    throw new JsonRpcProtocolError();
  }
  if (Object.prototype.hasOwnProperty.call(raw.error, "data")) assertJsonValue(raw.error.data);
  throw new JsonRpcError(raw.error.code as number, raw.error.message, raw.error.data as JsonValue | undefined);
}

export class JsonRpcClient implements RpcCaller {
  private readonly fetcher: typeof fetch;
  private readonly requestId: () => JsonRpcId;

  constructor(private readonly endpoint = "/rpc", dependencies: JsonRpcClientDependencies = {}) {
    this.fetcher = dependencies.fetcher ?? fetch.bind(globalThis);
    this.requestId = dependencies.requestId ?? (() => crypto.randomUUID());
  }

  async call<T>(method: string, params: JsonObject, options: JsonRpcCallOptions = {}): Promise<T> {
    if (!method.trim()) throw new JsonRpcProtocolError("JSON-RPC method must be non-empty");
    assertJsonValue(params);
    const id = options.requestId ?? this.requestId();
    if ((typeof id !== "string" && typeof id !== "number") ||
        (typeof id === "number" && (!Number.isInteger(id) || !Number.isFinite(id)))) {
      throw new JsonRpcProtocolError("JSON-RPC request ID must be a string or integer");
    }
    const request = { jsonrpc: "2.0", id, method, params } as const;
    const body = JSON.stringify(request);
    const attempts = Math.max(1, Math.trunc(options.attempts ?? 1));
    let lastFailure: unknown;

    for (let attempt = 0; attempt < attempts; attempt += 1) {
      try {
        const response = await this.fetcher(this.endpoint, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body,
        });
        if (!response.ok) throw new Error(`RPC transport failed with HTTP ${response.status}`);
        let payload: unknown;
        try {
          payload = await response.json();
        } catch {
          throw new JsonRpcProtocolError("JSON-RPC response was not valid JSON");
        }
        return decodeResponse(payload, id) as T;
      } catch (error) {
        if (error instanceof JsonRpcError || error instanceof JsonRpcProtocolError) throw error;
        lastFailure = error;
      }
    }
    throw lastFailure;
  }
}
