"""Collate timed COCO17 features without confusing missing frames with padding."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class PoseBatch:
    """Observation tensors in caller order; frozen fields, mutable tensor storage.

    Use collate_pose_features to validate numeric inputs. This object does not
    authenticate provenance, authorize final-test access or enforce split membership.
    Every true time step is retained, including observed all-missing frames.
    """

    features: torch.Tensor
    lengths: torch.Tensor
    time_mask: torch.Tensor


def _float32_sequence(sequence: np.ndarray) -> np.ndarray:
    if (type(sequence) is not np.ndarray or sequence.dtype != np.dtype(np.float64)
            or sequence.ndim != 2 or sequence.shape[0] == 0 or sequence.shape[1] != 121):
        raise ValueError("expected a nonempty float64 array [frames,121]")
    if not np.isfinite(sequence).all():
        raise ValueError("features must be finite")

    positions = sequence[:, :34].reshape(-1, 17, 2)
    velocities = sequence[:, 34:68].reshape(-1, 17, 2)
    confidence = sequence[:, 68:85]
    masks = sequence[:, 85:119]
    person = sequence[:, 119]
    intervals = sequence[:, 120]
    if (np.any((confidence < 0) | (confidence > 1))
            or np.any((person < 0) | (person > 1))):
        raise ValueError("confidence must lie in [0,1]")
    if np.any((masks != 0) & (masks != 1)):
        raise ValueError("position and velocity masks must be binary")
    position_mask = sequence[:, 85:102] == 1
    velocity_mask = sequence[:, 102:119] == 1
    if (velocity_mask[0].any()
            or not np.array_equal(velocity_mask[1:], position_mask[1:] & position_mask[:-1])):
        raise ValueError("velocity masks must match both adjacent position masks, initially zero")
    if np.any(positions[~position_mask] != 0) or np.any(velocities[~velocity_mask] != 0):
        raise ValueError("masked positions and velocities must be zero")
    if np.any(position_mask & (confidence == 0)):
        raise ValueError("valid positions require positive joint confidence")
    if np.any(sequence[person == 0, :120] != 0):
        raise ValueError("missing-person observations must have zero non-time channels")
    if intervals[0] != 0 or np.any(intervals[1:] <= 0):
        raise ValueError("first interval must be zero and later intervals positive")

    # Preserve the encoder's masks; rethresholding rounded confidences is incorrect.
    # Underflow warnings also cover representable subnormals, so test zero loss
    # explicitly instead of rejecting every operation flagged as underflow.
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            converted = sequence.astype(np.float32, order="C", copy=True)
    except FloatingPointError as exc:
        raise ValueError("features overflow float32") from exc
    if not np.isfinite(converted).all():
        raise ValueError("float32 features must remain finite")
    if np.any((sequence != 0) & (converted == 0)):
        raise ValueError("nonzero feature would become zero in float32")
    return converted


def collate_pose_features(sequences: tuple[np.ndarray, ...]) -> PoseBatch:
    """Copy complete encoded clips into one CPU float32 batch with trailing padding.

    Inputs must share the independently frozen encoder/configuration. Structural
    validation here cannot establish that origin. Labels and other metadata stay
    outside this API. Caller-owned inputs must not be modified during this call.
    Memory grows with batch size times longest clip; no truncation is performed.
    """
    if type(sequences) is not tuple or not sequences:
        raise ValueError("sequences must be a nonempty tuple")
    converted = []
    for index, sequence in enumerate(sequences):
        try:
            converted.append(_float32_sequence(sequence))
        except ValueError as exc:
            raise ValueError(f"sequence {index}: {exc}") from exc

    sizes = [len(sequence) for sequence in converted]
    maximum = max(sizes)
    lengths = torch.tensor(sizes, dtype=torch.int64, device="cpu")
    features = torch.zeros((len(sizes), maximum, 121), dtype=torch.float32, device="cpu")
    time_mask = torch.arange(maximum, dtype=torch.int64, device="cpu")[None, :] < lengths[:, None]
    for index, sequence in enumerate(converted):
        features[index, :len(sequence)].copy_(torch.from_numpy(sequence))
    return PoseBatch(features=features, lengths=lengths, time_mask=time_mask)
