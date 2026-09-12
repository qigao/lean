from __future__ import annotations

from dataclasses import dataclass


Point = tuple[float, float, float]


@dataclass(frozen=True)
class KeypointFrame:
    """One frame of ordered (x, y, confidence) keypoints."""

    points: tuple[Point, ...]

    def __post_init__(self) -> None:
        for point in self.points:
            if len(point) != 3:
                raise ValueError("each keypoint must contain x, y, confidence")


@dataclass(frozen=True)
class PointSequence:
    """A non-empty temporal sequence with a stable keypoint schema."""

    frames: tuple[KeypointFrame, ...]

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("point sequence must contain at least one frame")
        expected = len(self.frames[0].points)
        if any(len(frame.points) != expected for frame in self.frames[1:]):
            raise ValueError("keypoint count must remain constant across frames")

    def shifted(self, dx: float, dy: float) -> "PointSequence":
        return PointSequence(
            frames=tuple(
                KeypointFrame(
                    points=tuple((x + dx, y + dy, confidence) for x, y, confidence in frame.points)
                )
                for frame in self.frames
            )
        )


@dataclass(frozen=True)
class LabeledSequence:
    """A behavior label attached to one immutable observation sequence."""

    sequence: PointSequence
    label: str
    sample_id: str
