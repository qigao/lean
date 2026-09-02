import { fireEvent, getByRole } from "@testing-library/dom";
import { afterEach, describe, expect, it } from "vitest";
import { reportFixture } from "../test-fixtures";
import type { DiagnosticListElement } from "./diagnostic-list";
import "./diagnostic-list";

afterEach(() => document.body.replaceChildren());

describe("diagnostic list", () => {
  it("announces counts and exposes diagnostic pointer selection as a named button", () => {
    const diagnostics = document.createElement("diagnostic-list") as DiagnosticListElement;
    diagnostics.report = reportFixture(2, [{
      severity: "error",
      code: "unknown_place",
      document_role: "physical.world",
      logical_id: null,
      pointer: "/passages/0/source_place_id",
      related_ids: ["missing"],
      message_key: "passage_source_unknown",
    }]);
    document.body.append(diagnostics);

    expect(getByRole(diagnostics, "status").textContent).toBe("1 diagnostic: 1 error, 0 warnings.");
    const button = getByRole(diagnostics, "button", { name: "Error: passage source unknown" });
    let selected: unknown;
    diagnostics.addEventListener("studio-diagnostic-select", ((event: CustomEvent) => {
      selected = event.detail;
    }) as EventListener);
    fireEvent.click(button);
    expect(selected).toEqual({
      severity: "error",
      code: "unknown_place",
      document_role: "physical.world",
      logical_id: null,
      pointer: "/passages/0/source_place_id",
      related_ids: ["missing"],
      message_key: "passage_source_unknown",
    });
  });

  it("keeps its polite status node connected while report counts change", () => {
    const diagnostics = document.createElement("diagnostic-list") as DiagnosticListElement;
    document.body.append(diagnostics);
    const status = getByRole(diagnostics, "status");

    diagnostics.report = reportFixture(2, [{
      severity: "warning",
      code: "unused_place",
      document_role: "physical.world",
      logical_id: null,
      pointer: "/places/0",
      related_ids: ["lobby"],
      message_key: "place_unused",
    }]);

    expect(getByRole(diagnostics, "status")).toBe(status);
    expect(status.textContent).toBe("1 diagnostic: 0 errors, 1 warning.");
    expect(status.isConnected).toBe(true);
  });
});
