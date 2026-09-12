"""Shared structural gate for the explicit CPU pose-batch model interface."""
from __future__ import annotations

import torch

from ..pose_batches import PoseBatch


def validate_pose_batch(batch: PoseBatch, *, input_dim: int, model: torch.nn.Module) -> None:
    """Reject malformed mutable tensors before recurrence; never infer lengths.

    This checks padding structure, not source provenance or the float64 encoder's
    confidence decisions. Do not cast, detach, repair, or rewrite observations.
    """
    if type(batch) is not PoseBatch:
        raise ValueError("expected an explicit PoseBatch with lengths and time_mask")
    for name, dtype in (("features", torch.float32), ("lengths", torch.int64),
                        ("time_mask", torch.bool)):
        tensor = getattr(batch, name)
        if (type(tensor) is not torch.Tensor or tensor.layout != torch.strided
                or tensor.device.type != "cpu" or tensor.dtype != dtype):
            raise ValueError(f"{name} must be a dense CPU {dtype} tensor")
    features, lengths, mask = batch.features, batch.lengths, batch.time_mask
    if (features.ndim != 3 or features.shape[0] == 0 or features.shape[1] == 0
            or features.shape[2] != 121 or input_dim != 121):
        raise ValueError("expected nonempty [batch,time,121] features and a 121-input model")
    size, steps, _ = features.shape
    if lengths.shape != (size,) or mask.shape != (size, steps):
        raise ValueError("lengths or time_mask shape does not match features")
    if torch.any((lengths < 1) | (lengths > steps)).item():
        raise ValueError("each observed length must lie in [1,time]")
    expected = torch.arange(steps, device="cpu")[None, :] < lengths[:, None]
    if not torch.equal(mask, expected):
        raise ValueError("time_mask must be the exact observed prefix defined by lengths")
    if not torch.isfinite(features).all().item():
        raise ValueError("features must be finite, including padding")
    if torch.any(features[~mask] != 0).item():
        raise ValueError("feature rows outside the observed prefix must be zero padding")
    # Model state is mutable too: check every parameter and buffer, including
    # the readout and nonpersistent graph adjacency, before any recurrent work.
    for tensor in (*model.parameters(), *model.buffers()):
        if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
            raise ValueError("padded execution requires a CPU float32 model")
