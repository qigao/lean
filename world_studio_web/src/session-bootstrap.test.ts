import { fireEvent, getByLabelText, getByRole } from "@testing-library/dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { establishBrowserSession } from "./session-bootstrap";

function response(status: number, value: unknown = null): Response {
  const init: ResponseInit = { status };
  if (value !== null) init.headers = { "content-type": "application/json" };
  return new Response(value === null ? null : JSON.stringify(value), init);
}

async function flush(): Promise<void> {
  for (let index = 0; index < 8; index += 1) await Promise.resolve();
}

afterEach(() => document.body.replaceChildren());

describe("browser session bootstrap", () => {
  it("recovers from initial 401 with an accessible one-time bearer handoff", async () => {
    const shell = document.createElement("studio-shell");
    shell.hidden = true;
    document.body.append(shell);
    const fetcher = vi.fn()
      .mockResolvedValueOnce(response(401))
      .mockResolvedValueOnce(response(401))
      .mockResolvedValueOnce(response(200, {
        schema: "narrative-dynamics.studio-session/v1",
        authority: {
          authority_id: "operator", project_ids: [], run_ids: [], agent_ids: [], permissions: [],
        },
      }));

    const authenticating = establishBrowserSession(shell, fetcher);
    await flush();
    const form = getByRole(document.body, "form", { name: "Authenticate World Studio" });
    const token = getByLabelText(form, "Access token") as HTMLInputElement;
    expect(token.type).toBe("password");
    expect(document.activeElement).toBe(token);

    token.value = "do-not-reflect-this-token";
    fireEvent.submit(form);
    await flush();
    expect(getByRole(form, "status").textContent).toBe("Authentication failed. Check the access token.");
    expect(document.body.textContent).not.toContain("do-not-reflect-this-token");
    expect(token.value).toBe("");

    token.value = "valid-one-time-token";
    fireEvent.submit(form);
    const session = await authenticating;
    expect(session.schema).toBe("narrative-dynamics.studio-session/v1");
    expect(shell.hidden).toBe(false);
    expect(document.querySelector(".session-bootstrap")).toBeNull();
    expect(fetcher.mock.calls).toHaveLength(3);
    expect(fetcher.mock.calls[0]).toEqual(["/session", expect.objectContaining({ credentials: "same-origin" })]);
    expect(fetcher.mock.calls[1]?.[0]).toBe("/session");
    expect(fetcher.mock.calls[1]?.[1]).toMatchObject({
      method: "POST", credentials: "same-origin",
      headers: { accept: "application/json", authorization: "Bearer do-not-reflect-this-token" },
    });
    expect(fetcher.mock.calls[1]?.[1]).not.toHaveProperty("body");
    expect(fetcher.mock.calls[2]?.[1]).toMatchObject({
      headers: { authorization: "Bearer valid-one-time-token" },
    });
  });

  it("uses an existing same-origin cookie session without showing the token form", async () => {
    const shell = document.createElement("studio-shell");
    shell.hidden = true;
    document.body.append(shell);
    const fetcher = vi.fn().mockResolvedValue(response(200, {
      schema: "narrative-dynamics.studio-session/v1",
      authority: {
        authority_id: "operator", project_ids: [], run_ids: [], agent_ids: [], permissions: [],
      },
    }));

    await establishBrowserSession(shell, fetcher);

    expect(shell.hidden).toBe(false);
    expect(document.querySelector("form")).toBeNull();
    expect(fetcher).toHaveBeenCalledTimes(1);
  });
});
