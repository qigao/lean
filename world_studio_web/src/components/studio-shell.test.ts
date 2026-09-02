import { fireEvent, getAllByRole, getByRole, queryAllByRole } from "@testing-library/dom";
import { afterEach, describe, expect, it } from "vitest";
import "./studio-shell";

afterEach(() => {
  document.body.replaceChildren();
});

function renderShell(): HTMLElement {
  const shell = document.createElement("studio-shell") as HTMLElement & {
    projectId: string;
  };
  shell.projectId = "law-firm";
  document.body.append(shell);
  return shell;
}

describe("studio shell", () => {
  it("keeps the authoring regions in useful DOM reading order", () => {
    const shell = renderShell();
    const skip = shell.querySelector("a");
    const header = shell.querySelector("header");
    const navigation = getByRole(shell, "navigation", { name: "Project documents" });
    const main = getByRole(shell, "main");
    const editor = getByRole(shell, "region", { name: "Primary editor" });
    const inspector = getByRole(shell, "region", { name: "Property inspector" });
    const diagnostics = getByRole(shell, "region", { name: "Diagnostics" });
    const timeline = getByRole(shell, "region", { name: "Timeline" });

    expect(skip?.textContent).toBe("Skip to content");
    expect(skip?.getAttribute("href")).toBe("#studio-main");
    expect(queryAllByRole(shell, "main")).toHaveLength(1);
    expect(getByRole(shell, "toolbar", { name: "Run controls" })).toBeTruthy();
    expect((header?.compareDocumentPosition(navigation) ?? 0) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(navigation.compareDocumentPosition(main) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(editor.compareDocumentPosition(inspector) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(inspector.compareDocumentPosition(diagnostics) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(diagnostics.compareDocumentPosition(timeline) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it("implements roving APG tabs without changing the document tab order", () => {
    const shell = renderShell();
    const tabs = getAllByRole(shell, "tab");
    expect(tabs.map((tab) => tab.getAttribute("tabindex"))).toEqual(["0", "-1", "-1", "-1", "-1", "-1"]);

    fireEvent.keyDown(tabs[0]!, { key: "End" });
    expect(tabs[5]?.getAttribute("aria-selected")).toBe("true");
    expect(document.activeElement).toBe(tabs[5]);

    fireEvent.keyDown(tabs[5]!, { key: "ArrowRight" });
    expect(tabs[0]?.getAttribute("aria-selected")).toBe("true");
    expect(getByRole(shell, "tabpanel").getAttribute("aria-labelledby")).toBe(tabs[0]?.id);
  });

  it("exposes raw JSON and timeline structure without a Task 5 live console", () => {
    const shell = renderShell();
    const jsonTab = getByRole(shell, "tab", { name: "Raw JSON" });
    fireEvent.click(jsonTab);
    expect(getByRole(shell, "tabpanel", { name: "Raw JSON" })).toBeTruthy();
    expect(shell.querySelector("live-run-console")).toBeNull();
    expect(getByRole(shell, "region", { name: "Timeline" }).textContent).toContain("Run events will appear here");
  });
});
