import type { DraftDocument, JsonObject, JsonValue, OperationIntent, ProjectSnapshot } from "../schema/studio-types";

export interface MapPlace {
  id: string;
  label: string;
  x: number;
  y: number;
  width: number;
  height: number;
  layerIndex: number;
  objectIndex: number;
  pointer: string;
}

export interface MapPassage {
  id: string;
  sourcePlaceId: string;
  targetPlaceId: string;
  x: number;
  y: number;
  width: number;
  height: number;
  pointer: string;
}

export interface MapProjection {
  orientation: string;
  tileWidth: number;
  tileHeight: number;
  places: MapPlace[];
  passages: MapPassage[];
  agents: Array<{ id: string; placeId: string }>;
  objects: Array<{ id: string; placeId: string; holderAgentId: string | null }>;
  perceptionLayers: Array<{ id: string; layer: string; sourcePlaceId: string; targetPlaceId: string; passageId: string | null }>;
}

export type MapRenderItem =
  | { kind: "place"; id: string; x: number; y: number; width: number; height: number; pointer: string }
  | { kind: "passage"; id: string; x: number; y: number; width: number; height: number }
  | { kind: "agent" | "object"; id: string; x: number; y: number }
  | { kind: "perception"; id: string; layer: string; sourceX: number; sourceY: number; targetX: number; targetY: number };

export interface MapCanvasActions {
  select: (id: string, pointer: string) => void;
  move: (id: string, x: number, y: number) => void;
}

export interface MapCanvasAdapter {
  dispose: () => void;
  zoomIn: () => void;
  zoomOut: () => void;
  fit: () => void;
  setPerceptionVisible: (visible: boolean) => void;
}

export type MapCanvasLoader = (
  container: HTMLElement,
  map: MapProjection,
  actions: MapCanvasActions,
) => Promise<MapCanvasAdapter>;

function record(value: JsonValue | undefined): JsonObject | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value) ? value : undefined;
}

function list(value: JsonValue | undefined): JsonValue[] {
  return Array.isArray(value) ? value : [];
}

function text(value: JsonValue | undefined): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function number(value: JsonValue | undefined): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function documentFor(snapshot: ProjectSnapshot, role: string): DraftDocument | undefined {
  return snapshot.documents.find((document) => document.role === role && document.logical_id === null);
}

export function projectMap(snapshot: ProjectSnapshot): MapProjection {
  const world = documentFor(snapshot, "physical.world")?.value;
  const tiled = documentFor(snapshot, "physical.map")?.value;
  const initial = documentFor(snapshot, "physical.initial_state")?.value;
  const perception = documentFor(snapshot, "physical.perception")?.value;
  const placeLabels = new Map<string, string>();
  list(world?.places).forEach((item) => {
    const place = record(item);
    const id = text(place?.place_id);
    if (id) placeLabels.set(id, text(place?.label) ?? id);
  });
  const passageEndpoints = new Map<string, { source: string; target: string }>();
  list(world?.passages).forEach((item) => {
    const passage = record(item);
    const id = text(passage?.passage_id);
    const source = text(passage?.source_place_id);
    const target = text(passage?.target_place_id);
    if (id && source && target) passageEndpoints.set(id, { source, target });
  });

  const places: MapPlace[] = [];
  const passages: MapPassage[] = [];
  list(tiled?.layers).forEach((layerValue, layerIndex) => {
    const layer = record(layerValue);
    list(layer?.objects).forEach((objectValue, objectIndex) => {
      const object = record(objectValue);
      const id = text(object?.name);
      const objectClass = text(object?.class);
      const x = number(object?.x);
      const y = number(object?.y);
      if (!id || x === undefined || y === undefined) return;
      const width = number(object?.width) ?? 0;
      const height = number(object?.height) ?? 0;
      const pointer = `/layers/${layerIndex}/objects/${objectIndex}`;
      if (objectClass === "place") {
        places.push({ id, label: placeLabels.get(id) ?? id, x, y, width, height, layerIndex, objectIndex, pointer });
      } else if (objectClass === "passage") {
        const endpoints = passageEndpoints.get(id);
        if (endpoints) {
          passages.push({
            id,
            sourcePlaceId: endpoints.source,
            targetPlaceId: endpoints.target,
            x,
            y,
            width,
            height,
            pointer,
          });
        }
      }
    });
  });

  const agents = list(initial?.agents).flatMap((item) => {
    const agent = record(item);
    const id = text(agent?.agent_id);
    const placeId = text(agent?.place_id);
    return id && placeId ? [{ id, placeId }] : [];
  });
  const objects = list(initial?.objects).flatMap((item) => {
    const object = record(item);
    const id = text(object?.object_id);
    const placeId = text(object?.place_id);
    const holderAgentId = object?.holder_agent_id === null ? null : text(object?.holder_agent_id);
    return id && placeId && (holderAgentId !== undefined || object?.holder_agent_id === null)
      ? [{ id, placeId, holderAgentId: holderAgentId ?? null }]
      : [];
  });
  const perceptionLayers = list(perception?.edges).flatMap((item) => {
    const edge = record(item);
    const id = text(edge?.edge_id);
    const layer = text(edge?.layer);
    const sourcePlaceId = text(edge?.source_place_id);
    const targetPlaceId = text(edge?.target_place_id);
    const passageId = edge?.passage_id === null ? null : text(edge?.passage_id);
    return id && layer && sourcePlaceId && targetPlaceId && (passageId !== undefined || edge?.passage_id === null)
      ? [{ id, layer, sourcePlaceId, targetPlaceId, passageId: passageId ?? null }]
      : [];
  });

  return {
    orientation: text(tiled?.orientation) ?? "orthogonal",
    tileWidth: number(tiled?.tilewidth) ?? 1,
    tileHeight: number(tiled?.tileheight) ?? 1,
    places,
    passages,
    agents,
    objects,
    perceptionLayers,
  };
}

export function mapRenderItems(map: MapProjection): MapRenderItem[] {
  const centers = new Map(map.places.map((place) => [
    place.id,
    { x: place.x + place.width / 2, y: place.y + place.height / 2 },
  ]));
  const items: MapRenderItem[] = [
    ...map.places.map((place): MapRenderItem => ({
      kind: "place",
      id: place.id,
      x: place.x,
      y: place.y,
      width: place.width,
      height: place.height,
      pointer: place.pointer,
    })),
    ...map.passages.map((passage): MapRenderItem => ({
      kind: "passage",
      id: passage.id,
      x: passage.x,
      y: passage.y,
      width: passage.width,
      height: passage.height,
    })),
  ];
  for (const agent of map.agents) {
    const center = centers.get(agent.placeId);
    if (center) items.push({ kind: "agent", id: agent.id, ...center });
  }
  for (const object of map.objects) {
    const center = centers.get(object.placeId);
    if (center) items.push({ kind: "object", id: object.id, ...center });
  }
  for (const perception of map.perceptionLayers) {
    const source = centers.get(perception.sourcePlaceId);
    const target = centers.get(perception.targetPlaceId);
    if (source && target) {
      items.push({
        kind: "perception",
        id: perception.id,
        layer: perception.layer,
        sourceX: source.x,
        sourceY: source.y,
        targetX: target.x,
        targetY: target.y,
      });
    }
  }
  return items;
}

export function moveMapPlaceIntents(
  snapshot: ProjectSnapshot,
  placeId: string,
  x: number,
  y: number,
): OperationIntent[] {
  if (!Number.isFinite(x) || !Number.isFinite(y)) throw new Error("Map coordinates must be finite");
  const place = projectMap(snapshot).places.find(({ id }) => id === placeId);
  if (!place) throw new Error("Map place must have declared Tiled geometry");
  return [
    {
      document_role: "physical.map",
      logical_id: null,
      kind: "set_value",
      pointer: `${place.pointer}/x`,
      value: x,
    },
    {
      document_role: "physical.map",
      logical_id: null,
      kind: "set_value",
      pointer: `${place.pointer}/y`,
      value: y,
    },
  ];
}

function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[character] ?? character);
}

interface PixiPointerEvent {
  global: { x: number; y: number };
  stopPropagation: () => void;
}

async function loadPixiCanvas(
  container: HTMLElement,
  map: MapProjection,
  actions: MapCanvasActions,
): Promise<MapCanvasAdapter> {
  const { Application, Graphics } = await import("pixi.js");
  const application = new Application();
  await application.init({ resizeTo: container, backgroundAlpha: 0, antialias: true });
  container.replaceChildren(application.canvas);
  const minimumX = Math.min(0, ...map.places.map(({ x }) => x));
  const minimumY = Math.min(0, ...map.places.map(({ y }) => y));
  const offsetX = 24 - minimumX;
  const offsetY = 24 - minimumY;
  const perceptionShapes: Array<{ visible: boolean }> = [];
  let dragging: {
    shape: { x: number; y: number };
    place: MapPlace;
    deltaX: number;
    deltaY: number;
  } | null = null;
  for (const item of mapRenderItems(map)) {
    if (item.kind === "place") {
      const place = map.places.find(({ id }) => id === item.id)!;
    const shape = new Graphics()
      .roundRect(0, 0, place.width, place.height, 8)
      .fill({ color: 0x254a76, alpha: 0.2 })
      .stroke({ color: 0x254a76, width: 2 });
      shape.position.set(place.x + offsetX, place.y + offsetY);
      shape.eventMode = "static";
      shape.cursor = "grab";
      shape.on("pointertap", () => actions.select(place.id, place.pointer));
      shape.on("pointerdown", (event: PixiPointerEvent) => {
        const local = application.stage.toLocal(event.global);
        dragging = {
          shape,
          place,
          deltaX: local.x - shape.x,
          deltaY: local.y - shape.y,
        };
        shape.cursor = "grabbing";
        event.stopPropagation();
      });
    application.stage.addChild(shape);
    } else if (item.kind === "passage") {
      application.stage.addChild(new Graphics()
        .rect(item.x + offsetX, item.y + offsetY, item.width, item.height)
        .fill({ color: 0xca6f2d, alpha: 0.8 }));
    } else if (item.kind === "agent") {
      application.stage.addChild(new Graphics()
        .circle(item.x + offsetX, item.y + offsetY, 8)
        .fill({ color: 0x2d7a4b, alpha: 1 })
        .stroke({ color: 0xffffff, width: 2 }));
    } else if (item.kind === "object") {
      application.stage.addChild(new Graphics()
        .rect(item.x + offsetX - 6, item.y + offsetY - 6, 12, 12)
        .fill({ color: 0x6b4ba1, alpha: 1 })
        .stroke({ color: 0xffffff, width: 2 }));
    } else if (item.kind === "perception") {
      const shape = new Graphics()
        .moveTo(item.sourceX + offsetX, item.sourceY + offsetY)
        .lineTo(item.targetX + offsetX, item.targetY + offsetY)
        .stroke({ color: 0x2f7f8f, width: 3, alpha: 0.8 });
      shape.visible = false;
      perceptionShapes.push(shape);
      application.stage.addChild(shape);
    }
  }
  application.stage.eventMode = "static";
  application.stage.hitArea = application.screen;
  application.stage.on("pointermove", (event: PixiPointerEvent) => {
    if (!dragging) return;
    const local = application.stage.toLocal(event.global);
    dragging.shape.x = local.x - dragging.deltaX;
    dragging.shape.y = local.y - dragging.deltaY;
  });
  const finishDrag = (): void => {
    if (!dragging) return;
    const { shape, place } = dragging;
    const x = shape.x - offsetX;
    const y = shape.y - offsetY;
    shape.x = place.x + offsetX;
    shape.y = place.y + offsetY;
    dragging = null;
    actions.move(place.id, x, y);
  };
  application.stage.on("pointerup", finishDrag);
  application.stage.on("pointerupoutside", finishDrag);

  const fit = (): void => {
    const bounds = application.stage.getLocalBounds();
    const width = Math.max(1, bounds.width);
    const height = Math.max(1, bounds.height);
    const scale = Math.min(1, (application.renderer.width - 48) / width, (application.renderer.height - 48) / height);
    application.stage.scale.set(Math.max(0.1, scale));
    application.stage.position.set(24 - bounds.x * scale, 24 - bounds.y * scale);
  };
  return {
    dispose: () => application.destroy(true, { children: true }),
    zoomIn: () => { application.stage.scale.set(Math.min(4, application.stage.scale.x * 1.1)); },
    zoomOut: () => { application.stage.scale.set(Math.max(0.1, application.stage.scale.x / 1.1)); },
    fit,
    setPerceptionVisible: (visible) => perceptionShapes.forEach((shape) => { shape.visible = visible; }),
  };
}

export class MapEditorElement extends HTMLElement {
  private project: ProjectSnapshot | null = null;
  private enabled = false;
  private canvasAdapter: MapCanvasAdapter | undefined;
  canvasLoader: MapCanvasLoader = loadPixiCanvas;

  set snapshot(value: ProjectSnapshot | null) {
    this.project = value;
    if (this.isConnected) this.render();
  }

  get snapshot(): ProjectSnapshot | null {
    return this.project;
  }

  set active(value: boolean) {
    this.enabled = value;
    if (this.isConnected && value) void this.initializeCanvas();
  }

  get active(): boolean {
    return this.enabled;
  }

  connectedCallback(): void {
    this.render();
  }

  disconnectedCallback(): void {
    this.canvasAdapter?.dispose();
    this.canvasAdapter = undefined;
  }

  focusPointer(pointer: string): void {
    this.querySelector<HTMLElement>(`[data-pointer="${CSS.escape(pointer)}"]`)?.focus();
  }

  private async initializeCanvas(): Promise<void> {
    const container = this.querySelector<HTMLElement>(".map-canvas");
    const status = this.querySelector<HTMLElement>(".canvas-status");
    if (!container || !status || !this.project || !this.enabled || this.canvasAdapter) return;
    if (this.canvasLoader === loadPixiCanvas && navigator.userAgent.includes("jsdom")) {
      status.textContent = "Map canvas unavailable. The map object list and form remain available.";
      return;
    }
    status.textContent = "Loading map canvas.";
    try {
      this.canvasAdapter = await this.canvasLoader(container, projectMap(this.project), {
        select: (id, pointer) => this.dispatchEvent(new CustomEvent("studio-map-select", {
          bubbles: true,
          composed: true,
          detail: { id, pointer },
        })),
        move: (id, x, y) => {
          if (!this.project) return;
          this.dispatchEvent(new CustomEvent<OperationIntent[]>("studio-operations", {
            bubbles: true,
            composed: true,
            detail: moveMapPlaceIntents(this.project, id, x, y),
          }));
        },
      });
      status.textContent = "Map canvas ready. The map object list and form remain available.";
    } catch {
      status.textContent = "Map canvas unavailable. The map object list and form remain available.";
    }
  }

  private bind(): void {
    const status = this.querySelector<HTMLElement>(".canvas-status");
    const runControl = (action: keyof Pick<MapCanvasAdapter, "zoomIn" | "zoomOut" | "fit">): void => {
      if (this.canvasAdapter) this.canvasAdapter[action]();
      else if (status) status.textContent = "Map canvas unavailable. The map object list and form remain available.";
    };
    this.querySelector<HTMLButtonElement>("[data-map-action=zoom-in]")?.addEventListener("click", () => runControl("zoomIn"));
    this.querySelector<HTMLButtonElement>("[data-map-action=zoom-out]")?.addEventListener("click", () => runControl("zoomOut"));
    this.querySelector<HTMLButtonElement>("[data-map-action=fit]")?.addEventListener("click", () => runControl("fit"));
    this.querySelector<HTMLButtonElement>("[data-map-action=perception]")?.addEventListener("click", (event) => {
      const button = event.currentTarget as HTMLButtonElement;
      const visible = button.getAttribute("aria-pressed") !== "true";
      button.setAttribute("aria-pressed", String(visible));
      this.canvasAdapter?.setPerceptionVisible(visible);
      const fallback = this.querySelector<HTMLElement>(".perception-fallback");
      if (fallback) fallback.hidden = !visible;
    });
    this.querySelectorAll<HTMLButtonElement>("[data-map-id]").forEach((button) => {
      button.addEventListener("click", () => {
        const id = button.dataset.mapId ?? "";
        const pointer = button.dataset.pointer ?? "";
        this.dispatchEvent(new CustomEvent("studio-map-select", {
          bubbles: true,
          composed: true,
          detail: { id, pointer },
        }));
      });
    });
    this.querySelector<HTMLFormElement>("form")?.addEventListener("submit", (event) => {
      event.preventDefault();
      if (!this.project) return;
      const place = this.querySelector<HTMLSelectElement>("#map-place");
      const x = this.querySelector<HTMLInputElement>("#map-x");
      const y = this.querySelector<HTMLInputElement>("#map-y");
      const error = this.querySelector<HTMLElement>("#map-error");
      const rejectField = (field: HTMLInputElement | HTMLSelectElement | null, message: string): void => {
        field?.setAttribute("aria-invalid", "true");
        field?.setAttribute("aria-describedby", "map-error");
        if (error) error.textContent = message;
        field?.focus();
      };
      if (!place?.value) {
        rejectField(place, "Choose a mapped place.");
        return;
      }
      if (!x || x.value.trim() === "") {
        rejectField(x, "X coordinate is required.");
        return;
      }
      if (!y || y.value.trim() === "") {
        rejectField(y, "Y coordinate is required.");
        return;
      }
      try {
        const intents = moveMapPlaceIntents(
          this.project,
          place.value,
          x.valueAsNumber,
          y.valueAsNumber,
        );
        place.removeAttribute("aria-invalid");
        x?.removeAttribute("aria-invalid");
        y?.removeAttribute("aria-invalid");
        if (error) error.textContent = "";
        this.dispatchEvent(new CustomEvent<OperationIntent[]>("studio-operations", {
          bubbles: true,
          composed: true,
          detail: intents,
        }));
      } catch {
        for (const field of [x, y]) {
          field?.setAttribute("aria-invalid", "true");
          field?.setAttribute("aria-describedby", "map-error");
        }
        if (error) error.textContent = "Choose a mapped place and enter finite coordinates.";
        x?.focus();
      }
    });
  }

  render(): void {
    this.canvasAdapter?.dispose();
    this.canvasAdapter = undefined;
    const map = this.project ? projectMap(this.project) : null;
    const placeOptions = (map?.places ?? []).map((place) => `<option value="${escapeHtml(place.id)}">${escapeHtml(place.label)}</option>`).join("");
    const places = (map?.places ?? []).map((place) => `<li><button type="button" data-map-id="${escapeHtml(place.id)}" data-pointer="${escapeHtml(place.pointer)}">${escapeHtml(place.label)} at ${place.x}, ${place.y}</button></li>`).join("");
    const agents = (map?.agents ?? []).map((agent) => `<li>${escapeHtml(agent.id)} at ${escapeHtml(agent.placeId)}</li>`).join("");
    const objects = (map?.objects ?? []).map((object) => `<li>${escapeHtml(object.id)} at ${escapeHtml(object.placeId)}</li>`).join("");
    const perceptions = (map?.perceptionLayers ?? []).map((layer) => `<li>${escapeHtml(layer.layer)}: ${escapeHtml(layer.sourcePlaceId)} to ${escapeHtml(layer.targetPlaceId)}</li>`).join("");
    this.innerHTML = `
      <section class="map-editor" role="region" aria-label="Map editor">
        <div class="editor-toolbar" role="toolbar" aria-label="Map view controls">
          <button type="button" data-map-action="zoom-in">Zoom in</button>
          <button type="button" data-map-action="zoom-out">Zoom out</button>
          <button type="button" data-map-action="fit">Fit map</button>
          <button type="button" data-map-action="perception" aria-pressed="false">Perception layers</button>
        </div>
        <div class="map-canvas" role="img" aria-label="Orthogonal map visual canvas"></div>
        <p class="canvas-status" role="status" aria-live="polite">Semantic map controls ready.</p>
        <section aria-labelledby="map-objects-heading">
          <h3 id="map-objects-heading">Map objects</h3>
          <h4>Places</h4><ul>${places || "<li>No mapped places.</li>"}</ul>
          <h4>Agents</h4><ul>${agents || "<li>No agents.</li>"}</ul>
          <h4>Objects</h4><ul>${objects || "<li>No objects.</li>"}</ul>
          <section class="perception-fallback" aria-label="Perception layers" hidden><h4>Perception</h4><ul>${perceptions || "<li>No perception layers.</li>"}</ul></section>
        </section>
        <form aria-label="Move map place" novalidate>
          <h3>Move place</h3>
          <label for="map-place">Place</label>
          <select id="map-place" name="place_id" required>${placeOptions}</select>
          <label for="map-x">X coordinate</label>
          <input id="map-x" name="x" type="number" inputmode="decimal" value="0" required>
          <label for="map-y">Y coordinate</label>
          <input id="map-y" name="y" type="number" inputmode="decimal" value="0" required>
          <p id="map-error" class="field-error"></p>
          <button type="submit">Move place</button>
        </form>
      </section>
    `;
    this.bind();
    if (this.enabled) void this.initializeCanvas();
  }
}

if (!customElements.get("map-editor")) {
  customElements.define("map-editor", MapEditorElement);
}
