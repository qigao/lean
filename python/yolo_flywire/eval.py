from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MetricBundle:
    macro_f1: float
    balanced_accuracy: float


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
