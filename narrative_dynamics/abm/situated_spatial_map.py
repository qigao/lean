"""Import and deterministic layout helpers for situated spatial maps."""

from __future__ import annotations

import json
import math
from pathlib import Path

from narrative_dynamics.abm.situated_contracts import SituatedWorldModel
from narrative_dynamics.abm.situated_spatial_map_contracts import (
    SpatialPassage,
    SpatialPlace,
    SituatedSpatialMap,
)


def _positive(value: object, *, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0.0
    ):
        raise ValueError(f"{label} must be a positive finite number")
    return float(value)


def _number(value: object, *, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _object_layers(
    layers: object,
    *,
    parent_offset_x: float = 0.0,
    parent_offset_y: float = 0.0,
) -> tuple[tuple[dict[str, object], float, float], ...]:
    if not isinstance(layers, list):
        raise ValueError("Tiled map layers must be an array")
    found: list[tuple[dict[str, object], float, float]] = []
    for layer in layers:
        if not isinstance(layer, dict):
            raise ValueError("Tiled map layers must contain objects")
        for axis in ("x", "y"):
            if _number(
                layer.get(axis, 0.0),
                label=f"Tiled layer {axis} offset",
            ) != 0.0:
                raise ValueError(
                    "Tiled spatial layers do not support tile-coordinate offsets"
                )
        offset_x = parent_offset_x + _number(
            layer.get("offsetx", 0.0),
            label="Tiled layer pixel x offset",
        )
        offset_y = parent_offset_y + _number(
            layer.get("offsety", 0.0),
            label="Tiled layer pixel y offset",
        )
        layer_type = layer.get("type")
        if layer_type == "objectgroup":
            found.append((layer, offset_x, offset_y))
        elif layer_type == "group":
            found.extend(
                _object_layers(
                    layer.get("layers"),
                    parent_offset_x=offset_x,
                    parent_offset_y=offset_y,
                )
            )
    return tuple(found)


def _classified_rectangles(
    document: dict[str, object],
) -> tuple[tuple[dict[str, object], float, float], ...]:
    selected: list[tuple[dict[str, object], float, float]] = []
    for layer, offset_x, offset_y in _object_layers(document.get("layers")):
        objects = layer.get("objects")
        if not isinstance(objects, list):
            raise ValueError("Tiled object layer objects must be an array")
        for item in objects:
            if not isinstance(item, dict):
                raise ValueError("Tiled object layer must contain objects")
            classification = item.get("class") or item.get("type")
            if classification not in {"place", "passage"}:
                continue
            if _number(item.get("rotation", 0.0), label="Tiled object rotation") != 0.0:
                raise ValueError("Tiled spatial object rotation must be zero")
            if any(
                key in item
                for key in ("polygon", "polyline", "ellipse", "point", "gid")
            ):
                raise ValueError("Tiled spatial objects must be plain rectangles")
            selected.append((item, offset_x, offset_y))
    return tuple(selected)


def load_tiled_situated_spatial_map(
    path: str | Path,
    world_model: SituatedWorldModel,
    *,
    meters_per_pixel: float = 0.05,
) -> SituatedSpatialMap:
    """Compile strict Tiled rectangle objects into one path-independent map."""

    if not isinstance(world_model, SituatedWorldModel):
        raise TypeError("Tiled spatial import requires a SituatedWorldModel")
    scale = _positive(meters_per_pixel, label="Tiled meters per pixel")
    try:
        decoded = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        raise ValueError("Tiled spatial map could not be decoded") from None
    if not isinstance(decoded, dict):
        raise ValueError("Tiled spatial map root must be an object")
    if decoded.get("orientation") != "orthogonal":
        raise ValueError("Tiled spatial map must use orthogonal orientation")

    world_passages = {item.passage_id: item for item in world_model.passages}
    places: list[SpatialPlace] = []
    passages: list[SpatialPassage] = []
    for item, offset_x, offset_y in _classified_rectangles(decoded):
        classification = item.get("class") or item.get("type")
        identity = item.get("name")
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("Tiled spatial object name must be a non-empty world id")
        x = _number(item.get("x"), label="Tiled object x") + offset_x
        y = _number(item.get("y"), label="Tiled object y") + offset_y
        width = _positive(item.get("width"), label="Tiled object width")
        depth = _positive(item.get("height"), label="Tiled object height")
        center_x = (x + width / 2.0) * scale
        center_y = -(y + depth / 2.0) * scale
        if classification == "place":
            places.append(
                SpatialPlace(
                    identity,
                    center_x,
                    center_y,
                    width * scale,
                    depth * scale,
                    2.8,
                )
            )
        else:
            passage = world_passages.get(identity)
            if passage is None:
                raise ValueError("Tiled passage must reference a world passage")
            passages.append(
                SpatialPassage(
                    identity,
                    passage.source_place_id,
                    passage.target_place_id,
                    center_x,
                    center_y,
                    width * scale,
                    depth * scale,
                    2.1,
                )
            )
    return SituatedSpatialMap(
        f"{world_model.model_id}:tiled",
        "1",
        world_model,
        tuple(places),
        tuple(passages),
    )


def auto_layout_situated_spatial_map(
    world_model: SituatedWorldModel,
    *,
    room_size: float = 6.0,
    gap: float = 3.0,
) -> SituatedSpatialMap:
    """Place sorted rooms on one deterministic square grid."""

    if not isinstance(world_model, SituatedWorldModel):
        raise TypeError("spatial auto-layout requires a SituatedWorldModel")
    size = _positive(room_size, label="spatial auto-layout room size")
    spacing = _positive(gap, label="spatial auto-layout gap")
    columns = math.ceil(math.sqrt(len(world_model.places)))
    pitch = size + spacing
    places = tuple(
        SpatialPlace(
            place.place_id,
            float(index % columns) * pitch,
            -float(index // columns) * pitch,
            size,
            size,
            2.8,
        )
        for index, place in enumerate(world_model.places)
    )
    centers = {
        item.place_id: (item.center_x, item.center_y) for item in places
    }
    passages = tuple(
        SpatialPassage(
            passage.passage_id,
            passage.source_place_id,
            passage.target_place_id,
            (centers[passage.source_place_id][0] + centers[passage.target_place_id][0]) / 2.0,
            (centers[passage.source_place_id][1] + centers[passage.target_place_id][1]) / 2.0,
            1.0,
            0.3,
            2.1,
        )
        for passage in world_model.passages
    )
    return SituatedSpatialMap(
        f"{world_model.model_id}:auto-layout",
        "1",
        world_model,
        places,
        passages,
    )


__all__ = (
    "load_tiled_situated_spatial_map",
    "auto_layout_situated_spatial_map",
)
