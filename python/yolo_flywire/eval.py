from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MetricBundle:
    macro_f1: float
    balanced_accuracy: float


@dataclass(frozen=True)
class RobustnessReport:
    clean: MetricBundle
    perturbed: MetricBundle

    @classmethod
    def example(cls) -> "RobustnessReport":
        return cls(
            clean=MetricBundle(macro_f1=0.0, balanced_accuracy=0.0),
            perturbed=MetricBundle(macro_f1=0.0, balanced_accuracy=0.0),
        )


def _feature_keypoint_count(x: torch.Tensor) -> int:
    if x.ndim != 3:
        raise ValueError("expected [batch, time, feature] encoded observations")
    if x.shape[-1] == 0 or x.shape[-1] % 6 != 0:
        raise ValueError("feature dimension must match the V0 6K encoded keypoint layout")
    return x.shape[-1] // 6


def _recompute_velocity(result: torch.Tensor, keypoints: int) -> None:
    coordinates = result[..., : 2 * keypoints].reshape(
        result.shape[0], result.shape[1], keypoints, 2
    )
    velocity = torch.zeros_like(coordinates)
    if result.shape[1] > 1:
        velocity[:, 1:] = coordinates[:, 1:] - coordinates[:, :-1]
    result[..., 2 * keypoints : 4 * keypoints] = velocity.reshape(
        result.shape[0], result.shape[1], 2 * keypoints
    )


def add_keypoint_noise(x: torch.Tensor, sigma: float, seed: int) -> torch.Tensor:
    """Add deterministic Gaussian noise to encoded xy observations without mutating input."""
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    keypoints = _feature_keypoint_count(x)
    result = x.clone()
    if sigma == 0:
        return result

    shape = (x.shape[0], x.shape[1], keypoints, 2)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(shape, generator=generator, dtype=torch.float32).to(
        device=x.device, dtype=x.dtype
    ) * sigma
    coordinates = result[..., : 2 * keypoints].reshape(shape)
    coordinates.add_(noise)
    _recompute_velocity(result, keypoints)
    return result


def mask_keypoints(x: torch.Tensor, probability: float, seed: int) -> torch.Tensor:
    """Deterministically mask encoded keypoints and their derived velocity/confidence/mask fields."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    keypoints = _feature_keypoint_count(x)
    result = x.clone()
    if probability == 0.0:
        return result

    generator = torch.Generator(device="cpu").manual_seed(seed)
    sampled = torch.rand(
        (x.shape[0], x.shape[1], keypoints),
        generator=generator,
        dtype=torch.float32,
    )
    masked = (sampled < probability).to(device=x.device)

    coordinates = result[..., : 2 * keypoints].reshape(
        x.shape[0], x.shape[1], keypoints, 2
    )
    coordinates.masked_fill_(masked.unsqueeze(-1), 0.0)
    _recompute_velocity(result, keypoints)

    velocity = result[..., 2 * keypoints : 4 * keypoints].reshape(
        x.shape[0], x.shape[1], keypoints, 2
    )
    invalid_transition = masked.clone()
    if x.shape[1] > 1:
        invalid_transition[:, 1:] |= masked[:, :-1]
    velocity.masked_fill_(invalid_transition.unsqueeze(-1), 0.0)

    confidence = result[..., 4 * keypoints : 5 * keypoints]
    visibility = result[..., 5 * keypoints : 6 * keypoints]
    confidence.masked_fill_(masked, 0.0)
    visibility.masked_fill_(masked, 0.0)
    return result


def evaluate(model: torch.nn.Module, test: tuple[torch.Tensor, torch.Tensor]) -> MetricBundle:
    """Evaluate a frozen model on a held-out tensor pair; never used for selection."""
    x, y = test
    model.eval()
    with torch.no_grad():
        predicted = model(x).argmax(dim=1)

    labels = torch.unique(y).tolist()
    recalls: list[float] = []
    f1s: list[float] = []
    for label in labels:
        true_positive = int(((predicted == label) & (y == label)).sum().item())
        false_positive = int(((predicted == label) & (y != label)).sum().item())
        false_negative = int(((predicted != label) & (y == label)).sum().item())
        recall = true_positive / max(1, true_positive + false_negative)
        precision = true_positive / max(1, true_positive + false_positive)
        recalls.append(recall)
        f1s.append(0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall))

    return MetricBundle(
        macro_f1=sum(f1s) / max(1, len(f1s)),
        balanced_accuracy=sum(recalls) / max(1, len(recalls)),
    )
