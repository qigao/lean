"""Optional Blender replay compilation and export integration."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.situated_network_contracts import (
    SituatedNetworkRuntimeModel,
    SituatedNetworkTrajectory,
)
from narrative_dynamics.abm.situated_spatial_map_contracts import (
    SituatedSpatialMap,
)
from narrative_dynamics.abm.situated_spatial_map import (
    auto_layout_situated_spatial_map,
)


_SCHEMA = "narrative-dynamics.situated-blend-replay/v1"
_COLORS = (
    (0.22, 0.55, 0.95, 1.0),
    (0.95, 0.42, 0.28, 1.0),
    (0.35, 0.75, 0.40, 1.0),
    (0.75, 0.40, 0.90, 1.0),
    (0.95, 0.72, 0.22, 1.0),
    (0.25, 0.78, 0.78, 1.0),
)


class BlenderExportError(RuntimeError):
    """Raised when Blender cannot publish one validated replay scene."""


@dataclass(frozen=True)
class BlenderExportReport:
    output_path: str
    runtime_model_hash: str
    trajectory_hash: str
    spatial_map_hash: str
    replay_hash: str
    first_frame: int
    last_frame: int

    def __post_init__(self) -> None:
        if not isinstance(self.output_path, str) or not self.output_path.strip():
            raise ValueError("Blender export report output path must be non-empty")
        for name in (
            "runtime_model_hash",
            "trajectory_hash",
            "spatial_map_hash",
            "replay_hash",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value.startswith("sha256:")
                or len(value) != 71
            ):
                raise ValueError(f"Blender export report {name.replace('_', ' ')} must be a content hash")
        for name in ("first_frame", "last_frame"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"Blender export report {name.replace('_', ' ')} must be positive")
        if self.last_frame < self.first_frame:
            raise ValueError("Blender export report frame range must be ordered")


def _positive_integer(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _runtime_states(trajectory: SituatedNetworkTrajectory) -> tuple[object, ...]:
    return (trajectory.initial_state,) + tuple(
        item.next_state for item in trajectory.rounds
    )


def _state_positions(state: object, centers: dict[str, tuple[float, float]]) -> dict[str, list[float]]:
    by_place: dict[str, list[str]] = {}
    for node in state.snapshot.nodes:
        by_place.setdefault(node.place_id, []).append(node.agent_id)
    positions: dict[str, list[float]] = {}
    for place_id, agent_ids in sorted(by_place.items()):
        ordered = sorted(agent_ids)
        center_x, center_y = centers[place_id]
        for index, agent_id in enumerate(ordered):
            offset = (float(index) - (len(ordered) - 1) / 2.0) * 0.7
            positions[agent_id] = [center_x + offset, center_y, 0.05]
    return positions


def compile_situated_blend_replay(
    model: SituatedNetworkRuntimeModel,
    trajectory: SituatedNetworkTrajectory,
    spatial_map: SituatedSpatialMap,
    *,
    frames_per_round: int = 24,
) -> dict[str, object]:
    """Compile public V19 state and objective event metadata for Blender."""

    if not isinstance(model, SituatedNetworkRuntimeModel):
        raise TypeError("Blender replay requires a SituatedNetworkRuntimeModel")
    if not isinstance(trajectory, SituatedNetworkTrajectory):
        raise TypeError("Blender replay requires a SituatedNetworkTrajectory")
    if not isinstance(spatial_map, SituatedSpatialMap):
        raise TypeError("Blender replay requires a SituatedSpatialMap")
    step = _positive_integer(frames_per_round, label="Blender frames per round")
    if trajectory.model_id != model.model_id or trajectory.model_hash != model.content_hash:
        raise ValueError("Blender replay trajectory must bind the exact runtime model")
    world = model.percept_memory_model.cognitive_model.world_model
    if spatial_map.world_model != world:
        raise ValueError("Blender replay spatial map must bind the exact world model")

    states = _runtime_states(trajectory)
    frame_by_position = tuple(1 + index * step for index in range(len(states)))
    centers = {
        item.place_id: (item.center_x, item.center_y)
        for item in spatial_map.places
    }
    positions = tuple(_state_positions(state, centers) for state in states)
    agent_specs = {item.agent_id: item for item in world.agents}
    actors = []
    for color_index, agent_id in enumerate(sorted(agent_specs)):
        actor_states = []
        for state, frame, state_positions in zip(states, frame_by_position, positions):
            node = next(item for item in state.snapshot.nodes if item.agent_id == agent_id)
            actor_states.append(
                {
                    "round_index": state.round_index,
                    "frame": frame,
                    "place_id": node.place_id,
                    "position": state_positions[agent_id],
                    "tracked_belief_probability": node.tracked_belief_probability,
                    "active_claim_count": node.active_claim_count,
                }
            )
        actors.append(
            {
                "agent_id": agent_id,
                "role": agent_specs[agent_id].role,
                "color": list(_COLORS[color_index % len(_COLORS)]),
                "states": actor_states,
            }
        )

    world_passages = {item.passage_id: item for item in world.passages}
    passages = []
    for spatial_passage in spatial_map.passages:
        passage_states = []
        for state, frame in zip(states, frame_by_position):
            physical = next(
                item
                for item in state.story.current_state.passages
                if item.passage_id == spatial_passage.passage_id
            )
            passage_states.append(
                {
                    "round_index": state.round_index,
                    "frame": frame,
                    "open": physical.open,
                }
            )
        passage = spatial_passage.to_dict()
        passage["initially_open"] = world_passages[spatial_passage.passage_id].initially_open
        passage["states"] = passage_states
        passages.append(passage)

    events: list[dict[str, object]] = []
    transmissions: list[dict[str, object]] = []
    for position, round_result in enumerate(trajectory.rounds, 1):
        frame = frame_by_position[position]
        objective_round = round_result.next_state.story.rounds[-1]
        for event in objective_round.events:
            events.append(
                {
                    "event_id": event.event_id,
                    "event_hash": event.content_hash,
                    "round_index": event.round_index,
                    "sequence": event.sequence,
                    "frame": frame,
                    "action_id": event.action_id,
                    "kind": event.kind.value,
                    "actor_agent_id": event.actor_agent_id,
                    "place_id": event.place_id,
                    "target_id": event.target_id,
                    "success": event.success,
                    "outcome": event.outcome,
                    "cause_event_ids": list(event.cause_event_ids),
                }
            )
        for transmission in round_result.next_state.snapshot.transmissions:
            transmissions.append(
                {
                    "event_id": transmission.event_id,
                    "frame": frame,
                    "source_agent_id": transmission.source_agent_id,
                    "observer_agent_id": transmission.observer_agent_id,
                    "fidelity": transmission.fidelity.value,
                    "channels": [item.value for item in transmission.channels],
                }
            )

    place_labels = {item.place_id: item.label for item in world.places}
    packet: dict[str, object] = {
        "schema": _SCHEMA,
        "runtime_model_id": model.model_id,
        "runtime_model_hash": model.content_hash,
        "trajectory_hash": trajectory.content_hash,
        "world_model_id": world.model_id,
        "world_model_hash": world.content_hash,
        "spatial_map_id": spatial_map.map_id,
        "spatial_map_hash": spatial_map.content_hash,
        "frames_per_round": step,
        "first_frame": frame_by_position[0],
        "last_frame": frame_by_position[-1],
        "places": [
            {**item.to_dict(), "label": place_labels[item.place_id]}
            for item in spatial_map.places
        ],
        "passages": passages,
        "actors": actors,
        "events": sorted(events, key=lambda item: (item["frame"], item["sequence"])),
        "transmissions": sorted(
            transmissions,
            key=lambda item: (
                item["frame"],
                item["event_id"],
                item["source_agent_id"],
                item["observer_agent_id"],
            ),
        ),
    }
    packet["content_hash"] = stable_content_hash(packet)
    return packet


def _path(value: str | os.PathLike[str], *, label: str) -> Path:
    try:
        path = Path(value)
    except TypeError:
        raise TypeError(f"{label} must be a filesystem path") from None
    if not str(path).strip():
        raise ValueError(f"{label} must be non-empty")
    return path


def _timeout(value: object) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0.0
    ):
        raise ValueError("Blender export timeout must be a positive finite number")
    return float(value)


def export_situated_network_blend(
    blender_executable: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    model: SituatedNetworkRuntimeModel,
    trajectory: SituatedNetworkTrajectory,
    spatial_map: SituatedSpatialMap | None = None,
    *,
    frames_per_round: int = 24,
    timeout_seconds: float = 120.0,
) -> BlenderExportReport:
    """Run Blender headlessly and atomically publish one `.blend` replay."""

    output = _path(output_path, label="Blender output path").resolve()
    if output.suffix.lower() != ".blend":
        raise ValueError("Blender output path must use the .blend suffix")
    executable = _path(
        blender_executable,
        label="Blender executable path",
    ).resolve()
    if not executable.is_file():
        raise BlenderExportError("Blender export executable is unavailable")
    timeout = _timeout(timeout_seconds)
    selected_map = (
        auto_layout_situated_spatial_map(
            model.percept_memory_model.cognitive_model.world_model
        )
        if spatial_map is None
        else spatial_map
    )
    packet = compile_situated_blend_replay(
        model,
        trajectory,
        selected_map,
        frames_per_round=frames_per_round,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    builder = Path(__file__).with_name("blender_scene_builder.py").resolve()
    if not builder.is_file():
        raise BlenderExportError("Blender export scene builder is unavailable")

    try:
        with TemporaryDirectory(
            prefix=".nd-blender-",
            dir=output.parent,
        ) as temporary:
            root = Path(temporary)
            packet_path = root / "replay.json"
            staged_path = root / "scene.blend"
            log_path = root / "blender.log"
            packet_path.write_text(
                json.dumps(
                    packet,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            command = (
                str(executable),
                "--background",
                "--factory-startup",
                "--python-exit-code",
                "21",
                "--python",
                str(builder),
                "--",
                "--input",
                str(packet_path),
                "--output",
                str(staged_path),
            )
            with log_path.open("wb") as log:
                completed = subprocess.run(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=timeout,
                )
            if completed.returncode != 0:
                raise BlenderExportError("Blender export process failed")
            if not staged_path.is_file():
                raise BlenderExportError("Blender export did not produce a scene")
            with staged_path.open("rb") as stream:
                if stream.read(7) != b"BLENDER":
                    raise BlenderExportError("Blender export produced an invalid scene")
            os.replace(staged_path, output)
    except BlenderExportError:
        raise
    except (OSError, subprocess.SubprocessError):
        raise BlenderExportError("Blender export process failed") from None

    return BlenderExportReport(
        str(output),
        model.content_hash,
        trajectory.content_hash,
        selected_map.content_hash,
        packet["content_hash"],
        packet["first_frame"],
        packet["last_frame"],
    )


__all__ = (
    "BlenderExportError",
    "BlenderExportReport",
    "compile_situated_blend_replay",
    "export_situated_network_blend",
)
