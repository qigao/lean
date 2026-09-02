import type {
  DraftDocument,
  JsonObject,
  JsonValue,
  OperationIntent,
  ProjectSnapshot,
} from "../schema/studio-types";

export type GraphMode = "physical" | "social" | "story" | "resources";

export interface GraphNode {
  kind: "node";
  id: string;
  label: string;
  domainKind: string;
  documentRole: string;
  logicalId: string | null;
  pointer: string;
}

export interface GraphEdge {
  kind: "edge";
  id: string;
  label: string;
  domainKind: string;
  source: string;
  target: string;
  documentRole: string;
  logicalId: string | null;
  pointer: string;
}

export type GraphCell = GraphNode | GraphEdge;

export interface DomainGraph {
  mode: GraphMode;
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface GraphPosition {
  x: number;
  y: number;
}

export interface GraphCanvasActions {
  select: (id: string) => void;
  move: (id: string, position: GraphPosition) => void;
  connect: (source: string, target: string) => void;
  delete: (id: string) => void;
}

export interface GraphCanvasAdapter {
  dispose: () => void;
  zoomIn: () => void;
  zoomOut: () => void;
  fit: () => void;
  setMinimapVisible: (visible: boolean) => void;
}

export type GraphCanvasLoader = (
  container: HTMLElement,
  graph: DomainGraph,
  layout: JsonObject,
  actions: GraphCanvasActions,
) => Promise<GraphCanvasAdapter>;

function record(value: JsonValue | undefined): JsonObject | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value) ? value : undefined;
}

function list(value: JsonValue | undefined): JsonValue[] {
  return Array.isArray(value) ? value : [];
}

function text(value: JsonValue | undefined): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function documentFor(snapshot: ProjectSnapshot, role: string): DraftDocument | undefined {
  return snapshot.documents.find((document) => document.role === role && document.logical_id === null);
}

function node(
  id: string,
  label: string,
  domainKind: string,
  documentRole: string,
  pointer: string,
): GraphNode {
  return { kind: "node", id, label, domainKind, documentRole, logicalId: null, pointer };
}

function edge(
  id: string,
  label: string,
  domainKind: string,
  source: string,
  target: string,
  documentRole: string,
  pointer: string,
): GraphEdge {
  return { kind: "edge", id, label, domainKind, source, target, documentRole, logicalId: null, pointer };
}

function physicalGraph(snapshot: ProjectSnapshot): DomainGraph {
  const role = "physical.world";
  const value = documentFor(snapshot, role)?.value;
  const nodes = list(value?.places).flatMap((item, index) => {
    const place = record(item);
    const id = text(place?.place_id);
    return id ? [node(id, text(place?.label) ?? id, "place", role, `/places/${index}`)] : [];
  });
  const edges = list(value?.passages).flatMap((item, index) => {
    const passage = record(item);
    const id = text(passage?.passage_id);
    const source = text(passage?.source_place_id);
    const target = text(passage?.target_place_id);
    return id && source && target
      ? [edge(id, id, "passage", source, target, role, `/passages/${index}`)]
      : [];
  });
  return { mode: "physical", nodes, edges };
}

function socialGraph(snapshot: ProjectSnapshot): DomainGraph {
  const role = "social.relationships";
  const relationships = list(documentFor(snapshot, role)?.value.relationships);
  const nodes: GraphNode[] = [];
  const seen = new Set<string>();
  const edges: GraphEdge[] = [];
  relationships.forEach((item, index) => {
    const relationship = record(item);
    const source = text(relationship?.source_agent_id);
    const target = text(relationship?.target_agent_id);
    const relationshipType = text(relationship?.relationship_type);
    if (!source || !target || !relationshipType) return;
    for (const id of [source, target]) {
      if (!seen.has(id)) {
        seen.add(id);
        nodes.push(node(id, id, "agent", role, ""));
      }
    }
    edges.push(edge(
      `relationship:${index}:${source}:${relationshipType}:${target}`,
      relationshipType,
      "relationship",
      source,
      target,
      role,
      `/relationships/${index}`,
    ));
  });
  return { mode: "social", nodes, edges };
}

function storyGraph(snapshot: ProjectSnapshot): DomainGraph {
  const role = "story.outline";
  const value = documentFor(snapshot, role)?.value;
  const nodes: GraphNode[] = [];
  list(value?.acts).forEach((item, index) => {
    const act = record(item);
    const id = text(act?.act_id);
    if (id) nodes.push(node(id, id, "act", role, `/acts/${index}`));
  });
  list(value?.scenes).forEach((item, index) => {
    const scene = record(item);
    const id = text(scene?.scene_id);
    if (id) nodes.push(node(id, id, "scene", role, `/scenes/${index}`));
  });
  const edges = list(value?.dependencies).flatMap((item, index) => {
    const dependency = record(item);
    const predecessor = text(dependency?.predecessor_scene_id);
    const successor = text(dependency?.successor_scene_id);
    return predecessor && successor
      ? [edge(
          `dependency:${predecessor}:${successor}`,
          "precedes",
          "dependency",
          predecessor,
          successor,
          role,
          `/dependencies/${index}`,
        )]
      : [];
  });
  return { mode: "story", nodes, edges };
}

function resourceGraph(snapshot: ProjectSnapshot): DomainGraph {
  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];
  const subjects = new Set<string>();
  const resources = new Set<string>();

  const addEntitlement = (
    role: string,
    resourceId: string,
    scope: string,
    subject: string | null,
    pointer: string,
  ): void => {
    const subjectIdentity = subject ?? "public";
    const subjectId = `subject:${scope}:${subjectIdentity}`;
    if (!subjects.has(subjectId)) {
      subjects.add(subjectId);
      nodes.push(node(subjectId, `${scope}: ${subjectIdentity}`, "subject", role, pointer));
    }
    const id = `entitlement:${resourceId}:${scope}:${subjectIdentity}`;
    if (!edges.some((item) => item.id === id)) {
      edges.push(edge(id, "may use", "entitlement", subjectId, resourceId, role, pointer));
    }
  };

  for (const role of ["knowledge.catalog", "asset.catalog"] as const) {
    const value = documentFor(snapshot, role)?.value;
    list(value?.resources).forEach((item, resourceIndex) => {
      const resource = record(item);
      const resourceId = text(resource?.resource_id);
      if (!resourceId) return;
      if (!resources.has(resourceId)) {
        resources.add(resourceId);
        nodes.push(node(
          resourceId,
          resourceId,
          role === "asset.catalog" ? "asset" : "resource",
          role,
          `/resources/${resourceIndex}`,
        ));
      }
      list(resource?.entitlements).forEach((item, entitlementIndex) => {
        const entitlement = record(item);
        const scope = text(entitlement?.subject_scope);
        const subject = entitlement?.subject_id === null ? null : text(entitlement?.subject_id);
        if (scope && (subject !== undefined || entitlement?.subject_id === null)) {
          addEntitlement(role, resourceId, scope, subject ?? null, `/resources/${resourceIndex}/entitlements/${entitlementIndex}`);
        }
      });
    });
    list(value?.grants).forEach((item, grantIndex) => {
      const grant = record(item);
      const scope = text(grant?.subject_scope);
      const subject = grant?.subject_id === null ? null : text(grant?.subject_id);
      if (!scope || (subject === undefined && grant?.subject_id !== null)) return;
      list(grant?.resource_ids).forEach((resourceValue) => {
        const resourceId = text(resourceValue);
        if (resourceId) addEntitlement(role, resourceId, scope, subject ?? null, `/grants/${grantIndex}`);
      });
    });
  }
  return { mode: "resources", nodes, edges };
}

export function projectGraph(snapshot: ProjectSnapshot, mode: GraphMode): DomainGraph {
  if (mode === "physical") return physicalGraph(snapshot);
  if (mode === "social") return socialGraph(snapshot);
  if (mode === "story") return storyGraph(snapshot);
  return resourceGraph(snapshot);
}

function pointerToken(value: string): string {
  return value.replaceAll("~", "~0").replaceAll("/", "~1");
}

export function createPhysicalPlaceIntent(
  snapshot: ProjectSnapshot,
  place: { place_id: string; label: string },
): OperationIntent {
  const graph = physicalGraph(snapshot);
  if (!place.place_id.trim() || graph.nodes.some(({ id }) => id === place.place_id)) {
    throw new Error("Place ID must be non-empty and unique");
  }
  return {
    document_role: "physical.world",
    logical_id: null,
    kind: "insert_value",
    pointer: `/places/${graph.nodes.length}`,
    value: { place_id: place.place_id, label: place.label },
  };
}

export function connectPhysicalPlacesIntent(
  snapshot: ProjectSnapshot,
  passage: {
    passage_id: string;
    source_place_id: string;
    target_place_id: string;
    initially_open: boolean;
  },
): OperationIntent {
  const graph = physicalGraph(snapshot);
  const nodeIds = new Set(graph.nodes.map(({ id }) => id));
  if (!nodeIds.has(passage.source_place_id) || !nodeIds.has(passage.target_place_id)) {
    throw new Error("Passage endpoints must be existing places");
  }
  if (!passage.passage_id.trim() || graph.edges.some(({ id }) => id === passage.passage_id)) {
    throw new Error("Passage ID must be non-empty and unique");
  }
  return {
    document_role: "physical.world",
    logical_id: null,
    kind: "insert_value",
    pointer: `/passages/${graph.edges.length}`,
    value: { ...passage },
  };
}

export function deleteGraphCellIntent(cell: GraphCell): OperationIntent {
  if (!cell.pointer) throw new Error("This derived cell has no independently removable domain value");
  return {
    document_role: cell.documentRole,
    logical_id: cell.logicalId,
    kind: "remove_value",
    pointer: cell.pointer,
  };
}

export function setGraphPropertyIntent(cell: GraphCell, property: string, value: JsonValue): OperationIntent {
  if (!cell.pointer || !property.trim()) throw new Error("Property edit requires a domain pointer");
  return {
    document_role: cell.documentRole,
    logical_id: cell.logicalId,
    kind: "set_value",
    pointer: `${cell.pointer}/${pointerToken(property)}`,
    value,
  };
}

export function setGraphPositionIntent(
  snapshot: ProjectSnapshot,
  mode: GraphMode,
  domainId: string,
  position: GraphPosition,
): OperationIntent {
  if (!Number.isFinite(position.x) || !Number.isFinite(position.y)) throw new Error("Graph position must be finite");
  const graphs = record(snapshot.layout.graphs) ?? {};
  const modeLayout = record(graphs[mode]) ?? {};
  return {
    document_role: "layout",
    logical_id: null,
    kind: "set_layout",
    pointer: "/graphs",
    value: {
      ...graphs,
      [mode]: {
        ...modeLayout,
        [domainId]: { x: position.x, y: position.y },
      },
    },
  };
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

export function installX6DeletionKeyboard<TPlugin>(
  canvas: {
    use: (plugin: TPlugin) => unknown;
    bindKey: (keys: string[], callback: () => unknown) => unknown;
    getSelectedCells: () => Array<{ id: string }>;
  },
  KeyboardPlugin: new (options: { enabled: boolean }) => TPlugin,
  onDelete: (id: string) => void,
): void {
  canvas.use(new KeyboardPlugin({ enabled: true }));
  canvas.bindKey(["backspace", "delete"], () => {
    for (const cell of canvas.getSelectedCells()) onDelete(cell.id);
    return false;
  });
}

async function loadX6Canvas(
  container: HTMLElement,
  graph: DomainGraph,
  layout: JsonObject,
  actions: GraphCanvasActions,
): Promise<GraphCanvasAdapter> {
  const { Graph, Keyboard, MiniMap, Selection, Snapline } = await import("@antv/x6");
  const surface = document.createElement("div");
  surface.className = "graph-surface";
  const miniMap = document.createElement("div");
  miniMap.className = "graph-minimap";
  miniMap.setAttribute("aria-hidden", "true");
  container.replaceChildren(surface, miniMap);
  const canvas = new Graph({
    container: surface,
    background: { color: "transparent" },
    grid: true,
    panning: { enabled: true },
    mousewheel: { enabled: true, modifiers: ["ctrl", "meta"] },
    connecting: {
      snap: true,
      allowBlank: false,
      allowLoop: false,
      allowNode: graph.mode === "physical",
      allowEdge: false,
    },
    interacting: true,
  });
  canvas.use(new Selection({ enabled: true, multiple: false, rubberband: true, movable: true }));
  canvas.use(new Snapline({ enabled: true, sharp: true }));
  canvas.use(new MiniMap({ container: miniMap, width: 160, height: 100, padding: 8, scalable: true }));
  installX6DeletionKeyboard(canvas, Keyboard, actions.delete);
  const graphs = record(layout.graphs) ?? {};
  const positions = record(graphs[graph.mode]) ?? {};
  graph.nodes.forEach((item, index) => {
    const position = record(positions[item.id]);
    canvas.addNode({
      id: item.id,
      x: typeof position?.x === "number" ? position.x : 40 + (index % 3) * 180,
      y: typeof position?.y === "number" ? position.y : 40 + Math.floor(index / 3) * 100,
      width: 140,
      height: 44,
      label: item.label,
      attrs: { body: { magnet: graph.mode === "physical" } },
    });
  });
  graph.edges.forEach((item) => canvas.addEdge({ id: item.id, source: item.source, target: item.target, label: item.label }));
  canvas.on("cell:click", ({ cell }) => actions.select(cell.id));
  canvas.on("node:moved", ({ node }) => actions.move(node.id, node.position()));
  canvas.on("edge:connected", ({ edge }) => {
    const source = edge.getSourceCellId();
    const target = edge.getTargetCellId();
    canvas.removeCell(edge, { silent: true });
    if (source && target) actions.connect(source, target);
  });
  return {
    dispose: () => canvas.dispose(),
    zoomIn: () => { canvas.zoom(0.1); },
    zoomOut: () => { canvas.zoom(-0.1); },
    fit: () => { canvas.zoomToFit({ padding: 24, maxScale: 1 }); },
    setMinimapVisible: (visible) => { miniMap.hidden = !visible; },
  };
}

export class GraphEditorElement extends HTMLElement {
  private project: ProjectSnapshot | null = null;
  private graphMode: GraphMode = "physical";
  private selectedId: string | null = null;
  private enabled = false;
  private canvasAdapter: GraphCanvasAdapter | undefined;
  canvasLoader: GraphCanvasLoader = loadX6Canvas;

  set snapshot(value: ProjectSnapshot | null) {
    this.project = value;
    if (this.isConnected) this.render();
  }

  get snapshot(): ProjectSnapshot | null {
    return this.project;
  }

  set mode(value: GraphMode) {
    this.graphMode = value;
    this.selectedId = null;
    if (this.isConnected) this.render();
  }

  get mode(): GraphMode {
    return this.graphMode;
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

  private emitOperation(intent: OperationIntent): void {
    this.dispatchEvent(new CustomEvent<OperationIntent>("studio-operation", {
      bubbles: true,
      composed: true,
      detail: intent,
    }));
  }

  private selectId(id: string, focus: boolean): void {
    const cell = this.project
      ? [...projectGraph(this.project, this.graphMode).nodes, ...projectGraph(this.project, this.graphMode).edges]
          .find((item) => item.id === id)
      : undefined;
    if (!cell) return;
    this.selectedId = id;
    this.querySelectorAll<HTMLElement>("[data-node-id],[data-cell-id]").forEach((item) => {
      item.setAttribute("aria-pressed", String(item.dataset.nodeId === id || item.dataset.cellId === id));
    });
    const target = Array.from(this.querySelectorAll<HTMLElement>("[data-node-id],[data-cell-id]"))
      .find((item) => item.dataset.nodeId === id || item.dataset.cellId === id);
    if (focus) target?.focus();
    this.dispatchEvent(new CustomEvent("studio-select", {
      bubbles: true,
      composed: true,
      detail: { mode: this.graphMode, id },
    }));
  }

  private select(nodes: HTMLButtonElement[], index: number): void {
    const nextIndex = (index + nodes.length) % nodes.length;
    const next = nodes[nextIndex];
    if (!next) return;
    this.selectedId = next.dataset.nodeId ?? null;
    nodes.forEach((item) => {
      const selected = item === next;
      item.tabIndex = selected ? 0 : -1;
      item.setAttribute("aria-pressed", String(selected));
    });
    next.focus();
    this.dispatchEvent(new CustomEvent("studio-select", {
      bubbles: true,
      composed: true,
      detail: { mode: this.graphMode, id: this.selectedId },
    }));
  }

  private bind(graph: DomainGraph): void {
    const nodeButtons = Array.from(this.querySelectorAll<HTMLButtonElement>("[data-node-id]"));
    nodeButtons.forEach((button, index) => {
      button.addEventListener("click", () => this.select(nodeButtons, index));
      button.addEventListener("keydown", (event) => {
        if (event.key === "ArrowDown" || event.key === "ArrowRight") {
          event.preventDefault();
          this.select(nodeButtons, index + 1);
        } else if (event.key === "ArrowUp" || event.key === "ArrowLeft") {
          event.preventDefault();
          this.select(nodeButtons, index - 1);
        } else if (event.key === "Home") {
          event.preventDefault();
          this.select(nodeButtons, 0);
        } else if (event.key === "End") {
          event.preventDefault();
          this.select(nodeButtons, nodeButtons.length - 1);
        }
      });
    });
    this.querySelectorAll<HTMLButtonElement>("[data-cell-id]").forEach((button) => {
      button.addEventListener("click", () => this.selectId(button.dataset.cellId ?? "", true));
    });
    this.querySelectorAll<HTMLButtonElement>("[data-delete-id]").forEach((button) => {
      button.addEventListener("click", () => {
        const cell = [...graph.nodes, ...graph.edges].find(({ id }) => id === button.dataset.deleteId);
        if (cell) this.emitOperation(deleteGraphCellIntent(cell));
      });
    });

    const status = this.querySelector<HTMLElement>(".canvas-status");
    const runControl = (action: keyof Pick<GraphCanvasAdapter, "zoomIn" | "zoomOut" | "fit">): void => {
      if (this.canvasAdapter) this.canvasAdapter[action]();
      else if (status) status.textContent = "Visual graph unavailable. Semantic graph controls remain available.";
    };
    this.querySelector<HTMLButtonElement>("[data-graph-action=zoom-in]")?.addEventListener("click", () => runControl("zoomIn"));
    this.querySelector<HTMLButtonElement>("[data-graph-action=zoom-out]")?.addEventListener("click", () => runControl("zoomOut"));
    this.querySelector<HTMLButtonElement>("[data-graph-action=fit]")?.addEventListener("click", () => runControl("fit"));
    this.querySelector<HTMLButtonElement>("[data-graph-action=minimap]")?.addEventListener("click", (event) => {
      const button = event.currentTarget as HTMLButtonElement;
      const visible = button.getAttribute("aria-pressed") !== "true";
      button.setAttribute("aria-pressed", String(visible));
      this.canvasAdapter?.setMinimapVisible(visible);
      if (!this.canvasAdapter && status) status.textContent = "Visual graph unavailable. Semantic graph controls remain available.";
    });
    this.querySelector<HTMLButtonElement>("[data-graph-action=delete]")?.addEventListener("click", () => {
      const cell = [...graph.nodes, ...graph.edges].find(({ id }) => id === this.selectedId);
      if (cell) this.emitOperation(deleteGraphCellIntent(cell));
      else if (status) status.textContent = "Select a removable graph item first.";
    });

    this.querySelector<HTMLFormElement>("form")?.addEventListener("submit", (event) => {
      event.preventDefault();
      if (!this.project) return;
      const id = this.querySelector<HTMLInputElement>("#passage-id");
      const source = this.querySelector<HTMLSelectElement>("#passage-source");
      const target = this.querySelector<HTMLSelectElement>("#passage-target");
      const error = this.querySelector<HTMLElement>("#passage-error");
      try {
        const intent = connectPhysicalPlacesIntent(this.project, {
          passage_id: id?.value.trim() ?? "",
          source_place_id: source?.value ?? "",
          target_place_id: target?.value ?? "",
          initially_open: this.querySelector<HTMLInputElement>("#passage-open")?.checked ?? false,
        });
        id?.removeAttribute("aria-invalid");
        if (error) error.textContent = "";
        this.emitOperation(intent);
      } catch {
        if (id) {
          id.setAttribute("aria-invalid", "true");
          id.setAttribute("aria-describedby", "passage-error");
          id.focus();
        }
        if (error) error.textContent = "Enter a unique passage ID and choose existing endpoints.";
      }
    });
  }

  private async initializeCanvas(): Promise<void> {
    const container = this.querySelector<HTMLElement>(".graph-canvas");
    const status = this.querySelector<HTMLElement>(".canvas-status");
    if (!container || !status || !this.project || !this.enabled || this.canvasAdapter) return;
    if (this.canvasLoader === loadX6Canvas && navigator.userAgent.includes("jsdom")) {
      status.textContent = "Visual graph unavailable. Semantic graph controls remain available.";
      return;
    }
    status.textContent = "Loading visual graph.";
    try {
      const graph = projectGraph(this.project, this.graphMode);
      this.canvasAdapter = await this.canvasLoader(container, graph, this.project.layout, {
        select: (id) => this.selectId(id, false),
        move: (id, position) => {
          if (graph.nodes.some((node) => node.id === id) && this.project) {
            this.emitOperation(setGraphPositionIntent(this.project, this.graphMode, id, position));
          }
        },
        connect: (source, target) => {
          if (this.graphMode !== "physical" ||
              !graph.nodes.some((node) => node.id === source) ||
              !graph.nodes.some((node) => node.id === target)) return;
          const sourceField = this.querySelector<HTMLSelectElement>("#passage-source");
          const targetField = this.querySelector<HTMLSelectElement>("#passage-target");
          if (sourceField) sourceField.value = source;
          if (targetField) targetField.value = target;
          this.querySelector<HTMLInputElement>("#passage-id")?.focus();
          status.textContent = "Canvas connection selected. Enter a passage ID to submit the authoritative operation.";
        },
        delete: (id) => {
          const cell = [...graph.nodes, ...graph.edges].find((item) => item.id === id);
          if (cell) this.emitOperation(deleteGraphCellIntent(cell));
        },
      });
      status.textContent = "Visual graph ready. Semantic graph controls remain available below.";
    } catch {
      status.textContent = "Visual graph unavailable. Semantic graph controls remain available.";
    }
  }

  render(): void {
    this.canvasAdapter?.dispose();
    this.canvasAdapter = undefined;
    const graph = this.project ? projectGraph(this.project, this.graphMode) : { mode: this.graphMode, nodes: [], edges: [] };
    const title = `${this.graphMode[0]?.toUpperCase() ?? ""}${this.graphMode.slice(1)} graph`;
    const nodeOptions = graph.nodes.map((item) => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.label)}</option>`).join("");
    const nodeItems = graph.nodes.map((item, index) => {
      const selected = this.selectedId === item.id || (this.selectedId === null && index === 0);
      return `<li><button type="button" data-node-id="${escapeHtml(item.id)}" tabindex="${selected ? "0" : "-1"}" aria-pressed="${selected}" aria-label="Select ${escapeHtml(item.domainKind)} ${escapeHtml(item.label)}">${escapeHtml(item.label)}</button></li>`;
    }).join("");
    const edgeItems = graph.edges.map((item) => `<li><button type="button" data-cell-id="${escapeHtml(item.id)}" aria-pressed="${this.selectedId === item.id}" aria-label="Select ${escapeHtml(item.domainKind)} ${escapeHtml(item.label)}">${escapeHtml(item.label)}: ${escapeHtml(item.source)} → ${escapeHtml(item.target)}</button><button type="button" data-delete-id="${escapeHtml(item.id)}" aria-label="Delete ${escapeHtml(item.domainKind)} ${escapeHtml(item.label)}">Delete</button></li>`).join("");

    this.innerHTML = `
      <section class="graph-editor" role="region" aria-label="${title}">
        <div class="editor-toolbar" role="toolbar" aria-label="Graph view controls">
          <button type="button" data-graph-action="zoom-in" aria-label="Zoom in">Zoom in</button>
          <button type="button" data-graph-action="zoom-out" aria-label="Zoom out">Zoom out</button>
          <button type="button" data-graph-action="fit">Fit graph</button>
          <button type="button" data-graph-action="minimap" aria-pressed="true">Minimap</button>
          <button type="button" data-graph-action="delete">Delete selected</button>
        </div>
        <div class="graph-canvas" role="img" aria-label="${title} visual canvas"></div>
        <p class="canvas-status" role="status" aria-live="polite">Semantic graph controls ready.</p>
        <section class="graph-fallback" aria-labelledby="graph-items-title">
          <h3 id="graph-items-title">Graph items</h3>
          <h4>Nodes</h4>
          <ul>${nodeItems || "<li>No nodes.</li>"}</ul>
          <h4>Edges</h4>
          <ul>${edgeItems || "<li>No edges.</li>"}</ul>
        </section>
        ${this.graphMode === "physical" ? `
          <form aria-label="Create passage" novalidate>
            <h3>Create passage</h3>
            <label for="passage-id">Passage ID</label>
            <input id="passage-id" name="passage_id" required>
            <label for="passage-source">Source place</label>
            <select id="passage-source" name="source_place_id">${nodeOptions}</select>
            <label for="passage-target">Target place</label>
            <select id="passage-target" name="target_place_id">${nodeOptions}</select>
            <label class="checkbox-label"><input id="passage-open" name="initially_open" type="checkbox"> Initially open</label>
            <p id="passage-error" class="field-error"></p>
            <button type="submit">Create passage</button>
          </form>
        ` : ""}
      </section>
    `;
    this.bind(graph);
    if (this.enabled) void this.initializeCanvas();
  }
}

if (!customElements.get("graph-editor")) {
  customElements.define("graph-editor", GraphEditorElement);
}
