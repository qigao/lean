"""Host-derived, transport-neutral World Studio authority."""

from __future__ import annotations

from dataclasses import dataclass
import re

from narrative_dynamics.contracts import stable_content_hash


STUDIO_PERMISSIONS = (
    "command.audit",
    "command.read",
    "output.read",
    "project.create",
    "project.read",
    "project.write",
    "run.command",
    "run.create",
    "run.fork",
    "run.read",
    "scenario.compile",
    "state.agent",
    "state.network",
    "state.public",
)

_PERMISSION_SET = frozenset(STUDIO_PERMISSIONS)
_IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")


def _identity(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or _IDENTITY.fullmatch(value) is None
        or value in {".", ".."}
        or len(value) > 256
    ):
        raise ValueError(f"{label} must be a bounded path-free identity")
    return value


def _roster(values: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{label} must be a tuple")
    return tuple(sorted({_identity(value, label=label[:-1]) for value in values}))


@dataclass(frozen=True)
class StudioCapability:
    """Canonical authority value supplied separately by the authenticated host."""

    authority_id: str
    project_ids: tuple[str, ...] = ()
    run_ids: tuple[str, ...] = ()
    agent_ids: tuple[str, ...] = ()
    permissions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "authority_id",
            _identity(self.authority_id, label="studio authority ID"),
        )
        for name in ("project_ids", "run_ids", "agent_ids"):
            object.__setattr__(
                self,
                name,
                _roster(getattr(self, name), label=name.replace("_", " ")),
            )
        if not isinstance(self.permissions, tuple):
            raise TypeError("studio permissions must be a tuple")
        permissions = tuple(sorted(set(self.permissions)))
        if any(not isinstance(item, str) or item not in _PERMISSION_SET for item in permissions):
            raise ValueError("studio permission must belong to the closed permission set")
        object.__setattr__(self, "permissions", permissions)

    @property
    def content_hash(self) -> str:
        return stable_content_hash(
            {
                "authority_id": self.authority_id,
                "project_ids": list(self.project_ids),
                "run_ids": list(self.run_ids),
                "agent_ids": list(self.agent_ids),
                "permissions": list(self.permissions),
            }
        )

    def __hash__(self) -> int:
        return hash(self.content_hash)


__all__ = ("STUDIO_PERMISSIONS", "StudioCapability")
