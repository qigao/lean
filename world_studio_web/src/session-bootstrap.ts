import type { StudioSession } from "./schema/studio-types";

type Fetcher = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;

async function sessionFrom(response: Response): Promise<StudioSession> {
  const value = await response.json() as Partial<StudioSession>;
  if (value.schema !== "narrative-dynamics.studio-session/v1" ||
      value.authority === null || typeof value.authority !== "object") {
    throw new Error("Studio session is malformed");
  }
  return value as StudioSession;
}

function loginForm(): {
  container: HTMLElement;
  form: HTMLFormElement;
  token: HTMLInputElement;
  submit: HTMLButtonElement;
  status: HTMLElement;
} {
  const container = document.createElement("main");
  container.className = "session-bootstrap";
  const form = document.createElement("form");
  form.className = "panel session-form";
  form.setAttribute("aria-label", "Authenticate World Studio");
  const heading = document.createElement("h1");
  heading.textContent = "Open World Studio";
  const explanation = document.createElement("p");
  explanation.textContent = "Enter the access token supplied by the Studio operator.";
  const label = document.createElement("label");
  label.htmlFor = "studio-access-token";
  label.textContent = "Access token";
  const token = document.createElement("input");
  token.id = "studio-access-token";
  token.name = "access-token";
  token.type = "password";
  token.required = true;
  token.autocomplete = "off";
  token.spellcheck = false;
  const submit = document.createElement("button");
  submit.type = "submit";
  submit.textContent = "Authenticate";
  const status = document.createElement("p");
  status.className = "session-status";
  status.setAttribute("role", "status");
  status.setAttribute("aria-live", "polite");
  status.textContent = "Authentication is required.";
  form.append(heading, explanation, label, token, submit, status);
  container.append(form);
  return { container, form, token, submit, status };
}

export async function establishBrowserSession(
  shell: HTMLElement,
  fetcher: Fetcher = fetch,
): Promise<StudioSession> {
  const existing = await fetcher("/session", {
    credentials: "same-origin",
    headers: { accept: "application/json" },
  });
  if (existing.ok) {
    const session = await sessionFrom(existing);
    shell.hidden = false;
    return session;
  }
  if (existing.status !== 401) throw new Error("Studio session is unavailable");

  const controls = loginForm();
  shell.hidden = true;
  shell.before(controls.container);
  controls.token.focus();
  return await new Promise<StudioSession>((resolve) => {
    controls.form.addEventListener("submit", (event) => {
      event.preventDefault();
      if (!controls.form.reportValidity()) return;
      const authorization = `Bearer ${controls.token.value}`;
      controls.token.value = "";
      controls.token.disabled = true;
      controls.submit.disabled = true;
      controls.form.setAttribute("aria-busy", "true");
      controls.status.textContent = "Authenticating.";
      void fetcher("/session", {
        method: "POST",
        credentials: "same-origin",
        headers: { accept: "application/json", authorization },
      }).then(async (response) => {
        if (!response.ok) {
          controls.status.textContent = response.status === 401
            ? "Authentication failed. Check the access token."
            : "Authentication is temporarily unavailable.";
          controls.token.disabled = false;
          controls.submit.disabled = false;
          controls.form.setAttribute("aria-busy", "false");
          controls.token.focus();
          return;
        }
        try {
          const session = await sessionFrom(response);
          controls.container.remove();
          shell.hidden = false;
          resolve(session);
        } catch {
          controls.status.textContent = "Authentication returned an invalid session.";
          controls.token.disabled = false;
          controls.submit.disabled = false;
          controls.form.setAttribute("aria-busy", "false");
          controls.token.focus();
        }
      }, () => {
        controls.status.textContent = "Authentication is temporarily unavailable.";
        controls.token.disabled = false;
        controls.submit.disabled = false;
        controls.form.setAttribute("aria-busy", "false");
        controls.token.focus();
      });
    });
  });
}
