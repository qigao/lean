"""Content-addressed spatial geometry for situated worlds."""

from __future__ import annotations

from dataclasses import dataclass
import math

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.situated_contracts import SituatedWorldModel


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _finite(value: object, *, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _positive(value: object, *, label: str) -> float:
    result = _finite(value, label=label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


@dataclass(frozen=True)
class SpatialPlace:
    place_id: str
    center_x: float
    center_y: float
    width: float
    depth: float
    height: float = 2.8

    def __post_init__(self) -> None:
        object.__setattr__(self, "place_id", _text(self.place_id, label="spatial place id"))
        for name in ("center_x", "center_y"):
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), label=f"spatial place {name.replace('_', ' ')}"),
            )
        for name in ("width", "depth", "height"):
            object.__setattr__(
                self,
                name,
                _positive(getattr(self, name), label=f"spatial place {name}"),
            )

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SpatialPassage:
    passage_id: str
    source_place_id: str
    target_place_id: str
    center_x: float
    center_y: float
    width: float
    depth: float
    height: float = 2.1

    def __post_init__(self) -> None:
        for name in ("passage_id", "source_place_id", "target_place_id"):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), label=f"spatial passage {name.replace('_', ' ')}"),
            )
        if self.source_place_id == self.target_place_id:
            raise ValueError("spatial passage endpoints must differ")
        for name in ("center_x", "center_y"):
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), label=f"spatial passage {name.replace('_', ' ')}"),
            )
        for name in ("width", "depth", "height"):
            object.__setattr__(
                self,
                name,
                _positive(getattr(self, name), label=f"spatial passage {name}"),
            )

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedSpatialMap:
    map_id: str
    version: str
    world_model: SituatedWorldModel
    places: tuple[SpatialPlace, ...]
    passages: tuple[SpatialPassage, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "map_id", _text(self.map_id, label="spatial map id"))
        object.__setattr__(self, "version", _text(self.version, label="spatial map version"))
        if not isinstance(self.world_model, SituatedWorldModel):
            raise TypeError("spatial map world model must be SituatedWorldModel")
        for name, values, expected, identity in (
            ("places", self.places, SpatialPlace, lambda item: item.place_id),
            ("passages", self.passages, SpatialPassage, lambda item: item.passage_id),
        ):
            if not isinstance(values, tuple) or any(
                not isinstance(item, expected) for item in values
            ):
                raise TypeError(
                    f"spatial map {name} must be a tuple of {expected.__name__} values"
                )
            identities = tuple(identity(item) for item in values)
            if len(set(identities)) != len(identities):
                raise ValueError(f"spatial map {name} ids must be unique")
            object.__setattr__(self, name, tuple(sorted(values, key=identity)))

        world_place_ids = {item.place_id for item in self.world_model.places}
        map_place_ids = {item.place_id for item in self.places}
        if map_place_ids != world_place_ids:
            raise ValueError("spatial map must cover every world place exactly once")
        world_passages = {
            item.passage_id: (item.source_place_id, item.target_place_id)
            for item in self.world_model.passages
        }
        map_passage_ids = {item.passage_id for item in self.passages}
        if map_passage_ids != set(world_passages):
            raise ValueError("spatial map must cover every world passage exactly once")
        for passage in self.passages:
            if (
                passage.source_place_id,
                passage.target_place_id,
            ) != world_passages[passage.passage_id]:
                raise ValueError("spatial passage endpoints must match the world model")

    def to_dict(self) -> dict[str, object]:
        return {
            "map_id": self.map_id,
            "version": self.version,
            "world_model_id": self.world_model.model_id,
            "world_model_hash": self.world_model.content_hash,
            "places": [item.to_dict() for item in self.places],
            "passages": [item.to_dict() for item in self.passages],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


__all__ = ("SpatialPlace", "SpatialPassage", "SituatedSpatialMap")
