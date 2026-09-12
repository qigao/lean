import torch

from yolo_flywire.eval import evaluate
from yolo_flywire.manifests import RunManifest
from yolo_flywire.models import GRUClassifier
from yolo_flywire.train import TrainConfig, train_model


def test_manifest_hash_changes_when_split_changes():
    a = RunManifest.example(split_hash="a")
    b = RunManifest.example(split_hash="b")
    assert a.content_hash() != b.content_hash()


def test_manifest_rejects_missing_rewired_control_for_topology_claim():
    manifest = RunManifest.example(
        claim="topology_specific_advantage",
        compared_families=("flywire",),
    )
    try:
        manifest.validate_claim_boundary()
    except ValueError as exc:
        assert "rewired" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_training_seed_is_reproducible_and_test_evaluation_is_separate():
    x = torch.tensor(
        [
            [[-1.0], [-0.8], [-0.6]],
            [[-0.9], [-0.7], [-0.5]],
            [[1.0], [0.8], [0.6]],
            [[0.9], [0.7], [0.5]],
        ],
        dtype=torch.float32,
    )
    y = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    train = (x, y)
    validation = (x, y)
    config = TrainConfig(seed=11, epochs=25, lr=0.05, batch_size=4, parameter_ceiling=10000)

    torch.manual_seed(123)
    first = train_model(GRUClassifier(1, 4, 2), train, validation, config)
    torch.manual_seed(999)
    second = train_model(GRUClassifier(1, 4, 2), train, validation, config)

    assert first.state_hash == second.state_hash
    metrics = evaluate(first.model, (x, y))
    assert metrics.macro_f1 >= 0.50
    assert metrics.balanced_accuracy >= 0.50
