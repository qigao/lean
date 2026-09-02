import { fireEvent, getByLabelText, getByRole } from "@testing-library/dom";
import { afterEach, describe, expect, it } from "vitest";
import type { DraftDocument, OperationIntent, ProjectSnapshot } from "../schema/studio-types";
import { snapshotFixture } from "../test-fixtures";
import { MapEditorElement, moveMapPlaceIntents, projectMap } from "./map-editor";

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
});
