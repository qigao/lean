import { fireEvent, getAllByRole, getByLabelText, getByRole } from "@testing-library/dom";
import { afterEach, describe, expect, it } from "vitest";
import type { DraftDocument, OperationIntent, ProjectSnapshot } from "../schema/studio-types";
import { snapshotFixture } from "../test-fixtures";
import {
  connectPhysicalPlacesIntent,
  createPhysicalPlaceIntent,
  deleteGraphCellIntent,
  GraphEditorElement,
  projectGraph,
  setGraphPositionIntent,
  setGraphPropertyIntent,
} from "./graph-editor";

afterEach(() => document.body.replaceChildren());

const HASH = `sha256:${"f".repeat(64)}`;

function graphSnapshot(): ProjectSnapshot {
  const base = snapshotFixture();
  const documents: DraftDocument[] = [
    ...base.documents,
    {
      role: "social.relationships",
      logical_id: null,
      value: {
        model_id: "law-firm-social",
        relationships: [
          { source_agent_id: "alice", relationship_type: "supervises", target_agent_id: "bob", strength: 0.8 },
        ],
      },
      content_hash: HASH,
    },
    {
      role: "story.outline",
      logical_id: null,
      value: {
        acts: [{ act_id: "case", scene_ids: ["discover", "confront"] }],
        scenes: [{ scene_id: "discover" }, { scene_id: "confront" }],
        dependencies: [{ predecessor_scene_id: "discover", successor_scene_id: "confront" }],
      },
      content_hash: HASH,
    },
    {
      role: "knowledge.catalog",
      logical_id: null,
      value: {
        resources: [{
          resource_id: "statute",
          kind: "document",
          entitlements: [{ subject_scope: "public", subject_id: null }],
        }],
      },
      content_hash: HASH,
    },
    {
      role: "asset.catalog",
      logical_id: null,
      value: {
        resources: [{ resource_id: "archive-cabinet", kind: "model_3d" }],
        grants: [{ resource_ids: ["archive-cabinet"], subject_scope: "role", subject_id: "lawyer" }],
      },
      content_hash: HASH,
    },
  ];
  return { ...base, documents };
}

describe("pure graph projections", () => {
  it("derives physical cells from stable domain IDs", () => {
    const graph = projectGraph(graphSnapshot(), "physical");
    expect(graph.nodes.map(({ id }) => id)).toEqual(["lobby", "meeting", "archive"]);
    expect(graph.edges).toEqual([
      expect.objectContaining({
        id: "lobby-meeting",
        source: "lobby",
        target: "meeting",
        documentRole: "physical.world",
        pointer: "/passages/0",
      }),
      expect.objectContaining({
        id: "meeting-archive",
        source: "meeting",
        target: "archive",
        documentRole: "physical.world",
        pointer: "/passages/1",
      }),
    ]);
  });

  it("derives social, story, and resource identities without library cell JSON", () => {
    const social = projectGraph(graphSnapshot(), "social");
    expect(social.edges[0]).toEqual(expect.objectContaining({
      id: "relationship:0:alice:supervises:bob",
      source: "alice",
      target: "bob",
      pointer: "/relationships/0",
    }));

    const story = projectGraph(graphSnapshot(), "story");
    expect(story.nodes.map(({ id }) => id)).toEqual(["case", "discover", "confront"]);
    expect(story.edges[0]).toEqual(expect.objectContaining({
      id: "dependency:discover:confront",
      source: "discover",
      target: "confront",
    }));

    const resources = projectGraph(graphSnapshot(), "resources");
    expect(resources.nodes.map(({ id }) => id)).toEqual([
      "statute",
      "subject:public:public",
      "archive-cabinet",
      "subject:role:lawyer",
    ]);
    expect(resources.edges.map(({ id }) => id)).toEqual([
      "entitlement:statute:public:public",
      "entitlement:archive-cabinet:role:lawyer",
    ]);
    expect(JSON.stringify(resources)).not.toContain("positionAbsolute");
  });
});

describe("graph domain operations", () => {
  it("creates, changes, deletes, and positions through exact scenario operations", () => {
    const snapshot = graphSnapshot();
    const graph = projectGraph(snapshot, "physical");
    expect(createPhysicalPlaceIntent(snapshot, { place_id: "copy-room", label: "Copy room" })).toEqual({
      document_role: "physical.world",
      logical_id: null,
      kind: "insert_value",
      pointer: "/places/3",
      value: { place_id: "copy-room", label: "Copy room" },
    });
    expect(setGraphPropertyIntent(graph.nodes[0]!, "label", "Reception")).toEqual({
      document_role: "physical.world",
      logical_id: null,
      kind: "set_value",
      pointer: "/places/0/label",
      value: "Reception",
    });
    expect(deleteGraphCellIntent(graph.edges[0]!)).toEqual({
      document_role: "physical.world",
      logical_id: null,
      kind: "remove_value",
      pointer: "/passages/0",
    });
    expect(setGraphPositionIntent(snapshot, "physical", "lobby", { x: 10, y: 20 })).toEqual({
      document_role: "layout",
      logical_id: null,
      kind: "set_layout",
      pointer: "/graphs",
      value: { physical: { lobby: { x: 10, y: 20 } } },
    });
  });

  it("connects only existing physical endpoints using stable passage identity", () => {
    const snapshot = graphSnapshot();
    expect(connectPhysicalPlacesIntent(snapshot, {
      passage_id: "lobby-archive",
      source_place_id: "lobby",
      target_place_id: "archive",
      initially_open: true,
    })).toEqual({
      document_role: "physical.world",
      logical_id: null,
      kind: "insert_value",
      pointer: "/passages/2",
      value: {
        passage_id: "lobby-archive",
        source_place_id: "lobby",
        target_place_id: "archive",
        initially_open: true,
      },
    });
    expect(() => connectPhysicalPlacesIntent(snapshot, {
      passage_id: "dangling",
      source_place_id: "missing",
      target_place_id: "archive",
      initially_open: false,
    })).toThrow("existing places");
  });
});

describe("graph semantic fallback", () => {
  it("selects nodes by keyboard and creates an edge through a labeled form", () => {
    const editor = document.createElement("graph-editor") as GraphEditorElement;
    editor.snapshot = graphSnapshot();
    editor.mode = "physical";
    document.body.append(editor);

    const nodes = getAllByRole(editor, "button", { name: /Select place/ });
    nodes[0]?.focus();
    fireEvent.keyDown(nodes[0]!, { key: "ArrowDown" });
    expect(document.activeElement).toBe(nodes[1]);
    expect(nodes[1]?.getAttribute("aria-pressed")).toBe("true");

    let emitted: OperationIntent | undefined;
    editor.addEventListener("studio-operation", ((event: CustomEvent<OperationIntent>) => {
      emitted = event.detail;
    }) as EventListener, { once: true });
    fireEvent.input(getByLabelText(editor, "Passage ID"), { target: { value: "lobby-archive" } });
    fireEvent.change(getByLabelText(editor, "Source place"), { target: { value: "lobby" } });
    fireEvent.change(getByLabelText(editor, "Target place"), { target: { value: "archive" } });
    fireEvent.submit(getByRole(editor, "form", { name: "Create passage" }));

    expect(emitted).toEqual(expect.objectContaining({
      kind: "insert_value",
      pointer: "/passages/2",
      value: expect.objectContaining({ passage_id: "lobby-archive", source_place_id: "lobby", target_place_id: "archive" }),
    }));
  });

  it("translates canvas authoring events and toolbar controls without accepting canvas state as authority", async () => {
    interface CanvasActions {
      select: (id: string) => void;
      move: (id: string, position: { x: number; y: number }) => void;
      connect: (source: string, target: string) => void;
      delete: (id: string) => void;
    }
    let actions: CanvasActions | undefined;
    const controlCalls: string[] = [];
    const editor = document.createElement("graph-editor") as GraphEditorElement;
    editor.snapshot = graphSnapshot();
    editor.mode = "physical";
    editor.canvasLoader = async (_container, _graph, _layout, canvasActions: CanvasActions) => {
      actions = canvasActions;
      return {
        dispose: () => undefined,
        zoomIn: () => controlCalls.push("zoom-in"),
        zoomOut: () => controlCalls.push("zoom-out"),
        fit: () => controlCalls.push("fit"),
        setMinimapVisible: (visible: boolean) => controlCalls.push(`minimap:${visible}`),
      };
    };
    editor.active = true;
    document.body.append(editor);
    await Promise.resolve();
    await Promise.resolve();

    const selections: unknown[] = [];
    const operations: OperationIntent[] = [];
    editor.addEventListener("studio-select", ((event: CustomEvent) => { selections.push(event.detail); }) as EventListener);
    editor.addEventListener("studio-operation", ((event: CustomEvent<OperationIntent>) => { operations.push(event.detail); }) as EventListener);
    actions!.select("meeting");
    actions!.move("lobby", { x: 12, y: 34 });
    actions!.delete("lobby-meeting");
    actions!.connect("lobby", "archive");

    expect(selections).toEqual([{ mode: "physical", id: "meeting" }]);
    expect(getByRole(editor, "button", { name: "Select place Meeting room" }).getAttribute("aria-pressed")).toBe("true");
    expect(operations).toEqual([
      {
        document_role: "layout",
        logical_id: null,
        kind: "set_layout",
        pointer: "/graphs",
        value: { physical: { lobby: { x: 12, y: 34 } } },
      },
      {
        document_role: "physical.world",
        logical_id: null,
        kind: "remove_value",
        pointer: "/passages/0",
      },
    ]);
    expect((getByLabelText(editor, "Source place") as HTMLSelectElement).value).toBe("lobby");
    expect((getByLabelText(editor, "Target place") as HTMLSelectElement).value).toBe("archive");
    expect(document.activeElement).toBe(getByLabelText(editor, "Passage ID"));

    fireEvent.input(getByLabelText(editor, "Passage ID"), { target: { value: "lobby-archive" } });
    fireEvent.submit(getByRole(editor, "form", { name: "Create passage" }));
    expect(operations[2]).toEqual({
      document_role: "physical.world",
      logical_id: null,
      kind: "insert_value",
      pointer: "/passages/2",
      value: {
        passage_id: "lobby-archive",
        source_place_id: "lobby",
        target_place_id: "archive",
        initially_open: false,
      },
    });

    fireEvent.click(getByRole(editor, "button", { name: "Zoom in" }));
    fireEvent.click(getByRole(editor, "button", { name: "Zoom out" }));
    fireEvent.click(getByRole(editor, "button", { name: "Fit graph" }));
    fireEvent.click(getByRole(editor, "button", { name: "Minimap" }));
    expect(controlCalls).toEqual(["zoom-in", "zoom-out", "fit", "minimap:false"]);
  });
});
