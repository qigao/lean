import { fireEvent, getByLabelText, getByRole } from "@testing-library/dom";
import { afterEach, describe, expect, it } from "vitest";
import type { OperationIntent } from "../schema/studio-types";
import type { PropertyInspectorElement } from "./property-inspector";
import "./property-inspector";

afterEach(() => document.body.replaceChildren());

describe("property inspector fallback", () => {
  it("emits an RFC 6901 set operation and focuses malformed field input", () => {
    const inspector = document.createElement("property-inspector") as PropertyInspectorElement;
    inspector.selection = {
      label: "Public lobby",
      documentRole: "physical.world",
      logicalId: null,
      pointer: "/places/0/label",
      value: "Public lobby",
    };
    document.body.append(inspector);
    const value = getByLabelText(inspector, "JSON value") as HTMLTextAreaElement;
    expect(value.value).toBe('"Public lobby"');
    let operation: OperationIntent | undefined;
    inspector.addEventListener("studio-operation", ((event: CustomEvent<OperationIntent>) => {
      operation = event.detail;
    }) as EventListener);
    value.value = '"Reception"';
    fireEvent.submit(getByRole(inspector, "form", { name: "Edit selected property" }));
    expect(operation).toEqual({
      document_role: "physical.world",
      logical_id: null,
      kind: "set_value",
      pointer: "/places/0/label",
      value: "Reception",
    });

    value.value = "{";
    fireEvent.submit(getByRole(inspector, "form", { name: "Edit selected property" }));
    expect(value.getAttribute("aria-invalid")).toBe("true");
    expect(value.getAttribute("aria-describedby")).toBe("inspector-error");
    expect(document.activeElement).toBe(value);
  });
});
