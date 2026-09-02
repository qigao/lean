import { fireEvent, getAllByRole, getByRole } from "@testing-library/dom";
import { afterEach, describe, expect, it } from "vitest";
import { snapshotFixture } from "../test-fixtures";
import type { ProjectTreeElement } from "./project-tree";
import "./project-tree";

afterEach(() => document.body.replaceChildren());

describe("project document tree", () => {
  it("renders authoritative documents and uses roving keyboard activation", () => {
    const tree = document.createElement("project-tree") as ProjectTreeElement;
    tree.snapshot = snapshotFixture();
    document.body.append(tree);
    const items = getAllByRole(tree, "treeitem");
    expect(items.map((item) => item.textContent?.trim())).toEqual(["physical.world"]);
    expect(items.map((item) => item.getAttribute("tabindex"))).toEqual(["0"]);

    let selected: { role: string; logicalId: string | null } | undefined;
    tree.addEventListener("studio-document-select", ((event: CustomEvent) => {
      selected = event.detail;
    }) as EventListener);
    fireEvent.keyDown(items[0]!, { key: "Enter" });
    expect(selected).toEqual({ role: "physical.world", logicalId: null });
    expect(getByRole(tree, "tree").getAttribute("aria-label")).toBe("Scenario documents");
  });
});
