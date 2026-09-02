import { fireEvent, getByLabelText, getByRole } from "@testing-library/dom";
import { afterEach, describe, expect, it } from "vitest";
import type { OperationIntent } from "../schema/studio-types";
import { reportFixture, snapshotFixture } from "../test-fixtures";
import type { JsonEditorElement } from "../editors/json-editor";
import type { MapEditorElement } from "../editors/map-editor";
import type { StudioShellElement } from "./studio-shell";
import "./studio-shell";

afterEach(() => document.body.replaceChildren());

describe("studio shell integration", () => {
  it("loads only the selected heavy editor while preserving semantic fallbacks", () => {
    const shell = document.createElement("studio-shell") as StudioShellElement;
    shell.snapshot = snapshotFixture();
    document.body.append(shell);
    expect(shell.querySelector("graph-editor")?.getAttribute("data-mode")).toBe("physical");
    expect(shell.querySelector("map-editor")).toBeNull();
    expect(shell.querySelector("json-editor")).toBeNull();

    fireEvent.click(getByRole(shell, "tab", { name: "Map" }));
    expect((shell.querySelector("map-editor") as MapEditorElement | null)?.active).toBe(true);
    expect(shell.querySelector("graph-editor")).toBeNull();

    fireEvent.click(getByRole(shell, "tab", { name: "Raw JSON" }));
    const json = shell.querySelector("json-editor") as JsonEditorElement | null;
    expect(json?.active).toBe(true);
    expect(getByLabelText(json!, "Document JSON")).toBeTruthy();
  });

  it("forwards exact semantic operations to the project store", async () => {
    const applied: OperationIntent[] = [];
    const shell = document.createElement("studio-shell") as StudioShellElement;
    shell.snapshot = snapshotFixture();
    shell.store = {
      apply: async (intent: OperationIntent) => {
        applied.push(intent);
        return undefined;
      },
    };
    document.body.append(shell);
    const operation: OperationIntent = {
      document_role: "physical.world",
      logical_id: null,
      kind: "set_value",
      pointer: "/places/0/label",
      value: "Reception",
    };
    shell.querySelector("graph-editor")?.dispatchEvent(new CustomEvent("studio-operation", {
      bubbles: true,
      detail: operation,
    }));
    await Promise.resolve();
    expect(applied).toEqual([operation]);
  });

  it("routes a selected diagnostic into the active raw JSON pointer and inspector", () => {
    const report = reportFixture(2, [{
      severity: "error",
      code: "unknown_place",
      document_role: "physical.world",
      logical_id: null,
      pointer: "/passages/0/source_place_id",
      related_ids: ["missing"],
      message_key: "passage_source_unknown",
    }]);
    const shell = document.createElement("studio-shell") as StudioShellElement;
    shell.snapshot = snapshotFixture();
    shell.diagnostics = report;
    document.body.append(shell);
    fireEvent.click(getByRole(shell, "tab", { name: "Raw JSON" }));
    fireEvent.click(getByRole(shell, "button", { name: "Error: passage source unknown" }));

    const textarea = getByLabelText(shell, "Document JSON") as HTMLTextAreaElement;
    expect(document.activeElement).toBe(textarea);
    expect(textarea.dataset.pointer).toBe("/passages/0/source_place_id");
    expect(getByRole(shell, "region", { name: "Property inspector" }).textContent).toContain("/passages/0/source_place_id");
  });

  it("keeps the skip link first and has no positive tabindex or unlabeled form controls", () => {
    const shell = document.createElement("studio-shell") as StudioShellElement;
    shell.snapshot = snapshotFixture();
    document.body.append(shell);
    expect(shell.querySelector("a,button,input,select,textarea")?.classList.contains("skip-link")).toBe(true);
    expect(shell.querySelector("[tabindex='1'],[tabindex='2'],[tabindex='3']")).toBeNull();
    for (const control of shell.querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>("input,select,textarea")) {
      expect(control.labels?.length || control.getAttribute("aria-label")).toBeTruthy();
    }
    expect(getByRole(shell, "region", { name: "Property inspector" }).querySelector("details[open] > summary")).toBeTruthy();
    expect(getByRole(shell, "region", { name: "Timeline" }).querySelector("details[open] > summary")).toBeTruthy();
  });
});
