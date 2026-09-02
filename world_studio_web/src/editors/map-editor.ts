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

export type MapCanvasLoader = (container: HTMLElement, map: MapProjection) => Promise<() => void>;

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

async function loadPixiCanvas(container: HTMLElement, map: MapProjection): Promise<() => void> {
  const { Application, Graphics } = await import("pixi.js");
  const application = new Application();
  await application.init({ resizeTo: container, backgroundAlpha: 0, antialias: true });
  container.replaceChildren(application.canvas);
  const minimumX = Math.min(0, ...map.places.map(({ x }) => x));
  const minimumY = Math.min(0, ...map.places.map(({ y }) => y));
  const offsetX = 24 - minimumX;
  const offsetY = 24 - minimumY;
  for (const place of map.places) {
    const shape = new Graphics()
      .roundRect(place.x + offsetX, place.y + offsetY, place.width, place.height, 8)
      .fill({ color: 0x254a76, alpha: 0.2 })
      .stroke({ color: 0x254a76, width: 2 });
    application.stage.addChild(shape);
  }
  for (const passage of map.passages) {
    const shape = new Graphics()
      .rect(passage.x + offsetX, passage.y + offsetY, passage.width, passage.height)
      .fill({ color: 0xca6f2d, alpha: 0.8 });
    application.stage.addChild(shape);
  }
  return () => application.destroy(true, { children: true });
}

export class MapEditorElement extends HTMLElement {
  private project: ProjectSnapshot | null = null;
  private enabled = false;
  private disposeCanvas: (() => void) | undefined;
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
    this.disposeCanvas?.();
  }

  focusPointer(pointer: string): void {
    this.querySelector<HTMLElement>(`[data-pointer="${CSS.escape(pointer)}"]`)?.focus();
  }

  private async initializeCanvas(): Promise<void> {
    const container = this.querySelector<HTMLElement>(".map-canvas");
    const status = this.querySelector<HTMLElement>(".canvas-status");
    if (!container || !status || !this.project || !this.enabled || this.disposeCanvas) return;
    if (this.canvasLoader === loadPixiCanvas && navigator.userAgent.includes("jsdom")) {
      status.textContent = "Map canvas unavailable. The map object list and form remain available.";
      return;
    }
    status.textContent = "Loading map canvas.";
    try {
      this.disposeCanvas = await this.canvasLoader(container, projectMap(this.project));
      status.textContent = "Map canvas ready. The map object list and form remain available.";
    } catch {
      status.textContent = "Map canvas unavailable. The map object list and form remain available.";
    }
  }

  private bind(): void {
    this.querySelector<HTMLFormElement>("form")?.addEventListener("submit", (event) => {
      event.preventDefault();
      if (!this.project) return;
      const place = this.querySelector<HTMLSelectElement>("#map-place");
      const x = this.querySelector<HTMLInputElement>("#map-x");
      const y = this.querySelector<HTMLInputElement>("#map-y");
      const error = this.querySelector<HTMLElement>("#map-error");
      try {
        const intents = moveMapPlaceIntents(
          this.project,
          place?.value ?? "",
          Number(x?.value),
          Number(y?.value),
        );
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
    this.disposeCanvas?.();
    this.disposeCanvas = undefined;
    const map = this.project ? projectMap(this.project) : null;
    const placeOptions = (map?.places ?? []).map((place) => `<option value="${escapeHtml(place.id)}">${escapeHtml(place.label)}</option>`).join("");
    const places = (map?.places ?? []).map((place) => `<li><button type="button" data-pointer="${escapeHtml(place.pointer)}">${escapeHtml(place.label)} at ${place.x}, ${place.y}</button></li>`).join("");
    const agents = (map?.agents ?? []).map((agent) => `<li>${escapeHtml(agent.id)} at ${escapeHtml(agent.placeId)}</li>`).join("");
    const objects = (map?.objects ?? []).map((object) => `<li>${escapeHtml(object.id)} at ${escapeHtml(object.placeId)}</li>`).join("");
    this.innerHTML = `
      <section class="map-editor" role="region" aria-label="Map editor">
        <div class="editor-toolbar" role="toolbar" aria-label="Map view controls">
          <button type="button">Zoom in</button>
          <button type="button">Zoom out</button>
          <button type="button">Fit map</button>
          <button type="button" aria-pressed="false">Perception layers</button>
        </div>
        <div class="map-canvas" aria-label="Orthogonal map visual canvas"></div>
        <p class="canvas-status" role="status" aria-live="polite">Semantic map controls ready.</p>
        <section aria-labelledby="map-objects-heading">
          <h3 id="map-objects-heading">Map objects</h3>
          <h4>Places</h4><ul>${places || "<li>No mapped places.</li>"}</ul>
          <h4>Agents</h4><ul>${agents || "<li>No agents.</li>"}</ul>
          <h4>Objects</h4><ul>${objects || "<li>No objects.</li>"}</ul>
        </section>
        <form aria-label="Move map place" novalidate>
          <h3>Move place</h3>
          <label for="map-place">Place</label>
          <select id="map-place" name="place_id">${placeOptions}</select>
          <label for="map-x">X coordinate</label>
          <input id="map-x" name="x" type="number" inputmode="decimal" value="0">
          <label for="map-y">Y coordinate</label>
          <input id="map-y" name="y" type="number" inputmode="decimal" value="0">
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
