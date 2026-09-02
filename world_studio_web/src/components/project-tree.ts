import type { ProjectSnapshot } from "../schema/studio-types";

function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character] ?? character);
}

export class ProjectTreeElement extends HTMLElement {
  projectId = "Untitled project";
  private projectSnapshot: ProjectSnapshot | null = null;
  private selectedIndex = 0;

  set snapshot(value: ProjectSnapshot | null) {
    this.projectSnapshot = value;
    this.projectId = value?.project_id ?? this.projectId;
    this.selectedIndex = 0;
    if (this.isConnected) this.render();
  }

  get snapshot(): ProjectSnapshot | null { return this.projectSnapshot; }

  connectedCallback(): void { this.render(); }

  private select(items: HTMLButtonElement[], index: number, activate: boolean): void {
    if (items.length === 0) return;
    const nextIndex = (index + items.length) % items.length;
    const next = items[nextIndex];
    if (!next) return;
    this.selectedIndex = nextIndex;
    items.forEach((item, itemIndex) => {
      const selected = itemIndex === nextIndex;
      item.tabIndex = selected ? 0 : -1;
      item.setAttribute("aria-selected", String(selected));
    });
    next.focus();
    if (activate && next.dataset.role) {
      this.dispatchEvent(new CustomEvent("studio-document-select", {
        bubbles: true,
        composed: true,
        detail: { role: next.dataset.role, logicalId: next.dataset.logicalId || null },
      }));
    }
  }

  private bind(): void {
    const items = Array.from(this.querySelectorAll<HTMLButtonElement>("[role=treeitem]"));
    items.forEach((item, index) => {
      item.addEventListener("click", () => this.select(items, index, true));
      item.addEventListener("keydown", (event) => {
        if (event.key === "ArrowDown" || event.key === "ArrowRight") {
          event.preventDefault(); this.select(items, index + 1, false);
        } else if (event.key === "ArrowUp" || event.key === "ArrowLeft") {
          event.preventDefault(); this.select(items, index - 1, false);
        } else if (event.key === "Home") {
          event.preventDefault(); this.select(items, 0, false);
        } else if (event.key === "End") {
          event.preventDefault(); this.select(items, items.length - 1, false);
        } else if (event.key === "Enter" || event.key === " ") {
          event.preventDefault(); this.select(items, index, true);
        }
      });
    });
  }

  render(): void {
    const documents = this.projectSnapshot?.documents ?? [];
    const items = documents.length > 0
      ? documents.map((document, index) => {
          const identity = document.logical_id ? `${document.role}: ${document.logical_id}` : document.role;
          return `<li role="none"><button role="treeitem" type="button" data-role="${escapeHtml(document.role)}" data-logical-id="${escapeHtml(document.logical_id ?? "")}" tabindex="${index === this.selectedIndex ? "0" : "-1"}" aria-selected="${index === this.selectedIndex}">${escapeHtml(identity)}</button></li>`;
        }).join("")
      : `<li role="none"><button role="treeitem" type="button" tabindex="0" aria-selected="true">Scenario manifest</button></li>`;
    this.innerHTML = `
      <nav class="panel project-navigation" aria-label="Project documents">
        <h2>Project</h2>
        <p class="project-name">${escapeHtml(this.projectId)}</p>
        <ul role="tree" aria-label="Scenario documents">${items}</ul>
      </nav>
    `;
    this.bind();
  }
}

if (!customElements.get("project-tree")) customElements.define("project-tree", ProjectTreeElement);
