import { fireEvent, getByLabelText, getByRole } from "@testing-library/dom";
import { afterEach, describe, expect, it } from "vitest";
import type { DraftDocument, OperationIntent, ProjectSnapshot } from "../schema/studio-types";
import { snapshotFixture } from "../test-fixtures";
import { MapEditorElement, mapRenderItems, moveMapPlaceIntents, projectMap } from "./map-editor";

afterEach(() => document.body.replaceChildren());

const HASH = `sha256:${"9".repeat(64)}`;

function mapSnapshot(): ProjectSnapshot {
  const base = snapshotFixture();
  const documents: DraftDocument[] = [
    ...base.documents,
    {
      role: "physical.map",
      logical_id: null,
      value: {
        orientation: "orthogonal",
        tilewidth: 10,
        tileheight: 10,
        layers: [
          {
            name: "places",
            type: "objectgroup",
            objects: [
              { id: 1, name: "lobby", class: "place", x: -40, y: -40, width: 80, height: 80 },
              { id: 2, name: "meeting", class: "place", x: 50, y: -40, width: 100, height: 80 },
              { id: 3, name: "archive", class: "place", x: 180, y: -30, width: 80, height: 60 },
            ],
          },
          {
            name: "passages",
            type: "objectgroup",
            objects: [
              { id: 4, name: "lobby-meeting", class: "passage", x: 40, y: -12, width: 10, height: 24 },
            ],
          },
        ],
      },
      content_hash: HASH,
    },
    {
      role: "physical.initial_state",
      logical_id: null,
      value: {
        agents: [{ agent_id: "alice", place_id: "meeting" }],
        objects: [{ object_id: "case-file", place_id: "archive", holder_agent_id: null }],
        passages: [{ passage_id: "lobby-meeting", open: true }],
      },
      content_hash: HASH,
    },
    {
      role: "physical.perception",
      logical_id: null,
      value: {
        edges: [{ edge_id: "visual-lobby-meeting", layer: "visibility", source_place_id: "lobby", target_place_id: "meeting", passage_id: "lobby-meeting" }],
      },
      content_hash: HASH,
    },
  ];
  return { ...base, documents };
}

describe("map projection and synchronization", () => {
  it("derives Tiled geometry while preserving domain passage endpoints and identities", () => {
    const map = projectMap(mapSnapshot());
    expect(map.orientation).toBe("orthogonal");
    expect(map.places[0]).toEqual({
      id: "lobby",
      label: "Public lobby",
      x: -40,
      y: -40,
      width: 80,
      height: 80,
      layerIndex: 0,
      objectIndex: 0,
      pointer: "/layers/0/objects/0",
    });
    expect(map.passages[0]).toEqual(expect.objectContaining({
      id: "lobby-meeting",
      sourcePlaceId: "lobby",
      targetPlaceId: "meeting",
      x: 40,
      y: -12,
    }));
    expect(map.agents).toEqual([{ id: "alice", placeId: "meeting" }]);
    expect(map.objects).toEqual([{ id: "case-file", placeId: "archive", holderAgentId: null }]);
    expect(map.perceptionLayers).toEqual([expect.objectContaining({ id: "visual-lobby-meeting", layer: "visibility" })]);
  });

  it("builds render items for geometry, agents, objects, and perception overlays", () => {
    const items = mapRenderItems(projectMap(mapSnapshot()));
    expect(items.map(({ kind }) => kind)).toEqual([
      "place", "place", "place", "passage", "agent", "object", "perception",
    ]);
    expect(items[4]).toEqual(expect.objectContaining({ kind: "agent", id: "alice", x: 100, y: 0 }));
    expect(items[5]).toEqual(expect.objectContaining({ kind: "object", id: "case-file", x: 220, y: 0 }));
    expect(items[6]).toEqual(expect.objectContaining({
      kind: "perception",
      id: "visual-lobby-meeting",
      sourceX: 0,
      sourceY: 0,
      targetX: 100,
      targetY: 0,
    }));
  });

  it("moves a Tiled place using only its declared x and y geometry fields", () => {
    expect(moveMapPlaceIntents(mapSnapshot(), "lobby", 12, 34)).toEqual([
      {
        document_role: "physical.map",
        logical_id: null,
        kind: "set_value",
        pointer: "/layers/0/objects/0/x",
        value: 12,
      },
      {
        document_role: "physical.map",
        logical_id: null,
        kind: "set_value",
        pointer: "/layers/0/objects/0/y",
        value: 34,
      },
    ]);
  });

  it("keeps the form fallback operable when the Pixi adapter fails", async () => {
    const editor = document.createElement("map-editor") as MapEditorElement;
    editor.snapshot = mapSnapshot();
    editor.canvasLoader = async () => { throw new Error("no WebGL"); };
    editor.active = true;
    document.body.append(editor);
    await Promise.resolve();
    await Promise.resolve();
    expect(getByRole(editor, "status").textContent).toContain("Map canvas unavailable");

    let emitted: OperationIntent[] | undefined;
    editor.addEventListener("studio-operations", ((event: CustomEvent<OperationIntent[]>) => {
      emitted = event.detail;
    }) as EventListener);
    fireEvent.change(getByLabelText(editor, "Place"), { target: { value: "lobby" } });
    fireEvent.input(getByLabelText(editor, "X coordinate"), { target: { value: "12" } });
    fireEvent.input(getByLabelText(editor, "Y coordinate"), { target: { value: "34" } });
    fireEvent.submit(getByRole(editor, "form", { name: "Move map place" }));
    expect(emitted).toEqual(moveMapPlaceIntents(mapSnapshot(), "lobby", 12, 34));
  });

  it("rejects a blank coordinate before numeric conversion and focuses its described error", () => {
    const editor = document.createElement("map-editor") as MapEditorElement;
    editor.snapshot = mapSnapshot();
    document.body.append(editor);
    const x = getByLabelText(editor, "X coordinate") as HTMLInputElement;
    const y = getByLabelText(editor, "Y coordinate") as HTMLInputElement;
    expect(x.required).toBe(true);
    expect(y.required).toBe(true);
    let emitted = false;
    editor.addEventListener("studio-operations", () => { emitted = true; });

    fireEvent.input(x, { target: { value: "" } });
    fireEvent.input(y, { target: { value: "34" } });
    fireEvent.submit(getByRole(editor, "form", { name: "Move map place" }));

    expect(emitted).toBe(false);
    expect(x.getAttribute("aria-invalid")).toBe("true");
    expect(x.getAttribute("aria-describedby")).toBe("map-error");
    expect(document.activeElement).toBe(x);
    expect(editor.querySelector("#map-error")?.textContent).toContain("X coordinate is required");
  });

  it("translates Pixi selection, drag, and view controls into semantic editor behavior", async () => {
    interface CanvasActions {
      select: (id: string, pointer: string) => void;
      move: (id: string, x: number, y: number) => void;
    }
    let actions: CanvasActions | undefined;
    const controlCalls: string[] = [];
    const editor = document.createElement("map-editor") as MapEditorElement;
    editor.snapshot = mapSnapshot();
    editor.canvasLoader = async (_container, map, canvasActions: CanvasActions) => {
      expect(map.agents.map(({ id }) => id)).toEqual(["alice"]);
      expect(map.objects.map(({ id }) => id)).toEqual(["case-file"]);
      expect(map.perceptionLayers.map(({ id }) => id)).toEqual(["visual-lobby-meeting"]);
      actions = canvasActions;
      return {
        dispose: () => undefined,
        zoomIn: () => controlCalls.push("zoom-in"),
        zoomOut: () => controlCalls.push("zoom-out"),
        fit: () => controlCalls.push("fit"),
        setPerceptionVisible: (visible: boolean) => controlCalls.push(`perception:${visible}`),
      };
    };
    const selections: unknown[] = [];
    const operations: OperationIntent[][] = [];
    editor.addEventListener("studio-map-select", ((event: CustomEvent) => { selections.push(event.detail); }) as EventListener);
    editor.addEventListener("studio-operations", ((event: CustomEvent<OperationIntent[]>) => { operations.push(event.detail); }) as EventListener);
    editor.active = true;
    document.body.append(editor);
    await Promise.resolve();
    await Promise.resolve();

    actions!.select("lobby", "/layers/0/objects/0");
    actions!.move("lobby", 12, 34);
    expect(selections).toEqual([{ id: "lobby", pointer: "/layers/0/objects/0" }]);
    expect(operations).toEqual([moveMapPlaceIntents(mapSnapshot(), "lobby", 12, 34)]);

    fireEvent.click(getByRole(editor, "button", { name: "Zoom in" }));
    fireEvent.click(getByRole(editor, "button", { name: "Zoom out" }));
    fireEvent.click(getByRole(editor, "button", { name: "Fit map" }));
    const perception = getByRole(editor, "button", { name: "Perception layers" });
    fireEvent.click(perception);
    expect(perception.getAttribute("aria-pressed")).toBe("true");
    expect(controlCalls).toEqual(["zoom-in", "zoom-out", "fit", "perception:true"]);
  });
});
