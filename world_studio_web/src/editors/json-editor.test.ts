import { fireEvent, getByLabelText, getByRole } from "@testing-library/dom";
import { afterEach, describe, expect, it } from "vitest";
import type { DiagnosticReport, DraftDocument, OperationIntent } from "../schema/studio-types";
import { reportFixture, snapshotFixture } from "../test-fixtures";
import {
  JsonEditorElement,
  markersForDocument,
  replaceDocumentIntent,
} from "./json-editor";

afterEach(() => document.body.replaceChildren());

function physicalDocument(): DraftDocument {
  return snapshotFixture().documents[0]!;
}

describe("raw JSON authority adapter", () => {
  it("replaces one whole server document at the root pointer", () => {
    const document = physicalDocument();
    expect(replaceDocumentIntent(document, { model_id: "replacement", places: [] })).toEqual({
      document_role: "physical.world",
      logical_id: null,
      kind: "replace_document",
      pointer: "",
      value: { model_id: "replacement", places: [] },
    });
  });

  it("maps exact server message keys and RFC 6901 pointers into markers", () => {
    const report: DiagnosticReport = reportFixture(2, [{
      severity: "error",
      code: "unknown_place",
      document_role: "physical.world",
      logical_id: null,
      pointer: "/passages/0/source_place_id",
      related_ids: ["missing"],
      message_key: "passage_source_unknown",
    }]);
    expect(markersForDocument(report, physicalDocument())).toEqual([{
      severity: "error",
      code: "unknown_place",
      message: "passage_source_unknown",
      pointer: "/passages/0/source_place_id",
    }]);
  });

  it("keeps a labeled text fallback, emits replacement, and focuses invalid JSON", () => {
    const editor = document.createElement("json-editor") as JsonEditorElement;
    editor.document = physicalDocument();
    document.body.append(editor);
    const textarea = getByLabelText(editor, "Document JSON") as HTMLTextAreaElement;
    expect(textarea.value).toContain('"model_id": "law-firm-world"');

    let emitted: OperationIntent | undefined;
    editor.addEventListener("studio-operation", ((event: CustomEvent<OperationIntent>) => {
      emitted = event.detail;
    }) as EventListener);
    textarea.value = '{"model_id":"replacement","places":[]}';
    fireEvent.submit(getByRole(editor, "form", { name: "Edit raw JSON" }));
    expect(emitted).toEqual(expect.objectContaining({ kind: "replace_document", pointer: "" }));

    textarea.value = "{";
    fireEvent.submit(getByRole(editor, "form", { name: "Edit raw JSON" }));
    expect(textarea.getAttribute("aria-invalid")).toBe("true");
    expect(textarea.getAttribute("aria-describedby")).toBe("json-error");
    expect(document.activeElement).toBe(textarea);
    expect(editor.querySelector("#json-error")?.textContent).toBe("Enter a valid JSON object.");
  });

  it("focuses the raw fallback at a selected diagnostic pointer", () => {
    const editor = document.createElement("json-editor") as JsonEditorElement;
    editor.document = physicalDocument();
    document.body.append(editor);
    editor.focusPointer("/places/0/label");
    const textarea = getByLabelText(editor, "Document JSON") as HTMLTextAreaElement;
    expect(document.activeElement).toBe(textarea);
    expect(textarea.dataset.pointer).toBe("/places/0/label");
    expect(textarea.selectionEnd).toBeGreaterThan(textarea.selectionStart);
  });

  it("traverses arrays and decoded RFC 6901 tokens to select the target object", () => {
    const editor = document.createElement("json-editor") as JsonEditorElement;
    editor.document = physicalDocument();
    document.body.append(editor);
    const textarea = getByLabelText(editor, "Document JSON") as HTMLTextAreaElement;
    textarea.value = '{"rows":[{"label":"first"},{"label":"second","a/b":{"~key":[1,{"ok":true}]}}]}';

    editor.focusPointer("/rows/1/a~1b/~0key/1");

    expect(textarea.value.slice(textarea.selectionStart, textarea.selectionEnd)).toBe('{"ok":true}');
  });

  it("uses the last duplicate object member and selects its complete array value", () => {
    const editor = document.createElement("json-editor") as JsonEditorElement;
    editor.document = physicalDocument();
    document.body.append(editor);
    const textarea = getByLabelText(editor, "Document JSON") as HTMLTextAreaElement;
    textarea.value = '{"target":"obsolete","target":{"items":[10,20]}}';

    editor.focusPointer("/target/items");

    expect(textarea.value.slice(textarea.selectionStart, textarea.selectionEnd)).toBe("[10,20]");
  });
});
