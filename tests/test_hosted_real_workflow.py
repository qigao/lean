from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "yolo-flywire-hosted-real.yml"


def _workflow():
    assert WORKFLOW.is_file()
    return WORKFLOW.read_text(encoding="utf-8")


def test_hosted_real_workflow_is_manual_and_github_hosted():
    text = _workflow()
    assert "workflow_dispatch:" in text
    assert "\n  push:" not in text
    assert "\n  pull_request:" not in text
    assert "runs-on: ubuntu-latest" in text
    assert "self-hosted" not in text
    assert "REQUIRED_FREE_GIB" not in text
    assert "timeout-minutes: 720" not in text


def test_hosted_real_workflow_uses_remote_manifest_and_yolo11_shards():
    text = _workflow()
    assert "NTU120_REMOTE_MANIFEST_URL: ${{ secrets.NTU120_REMOTE_MANIFEST_URL }}" in text
    assert "yolo11n-pose.pt" in text
    assert "strategy:" in text and "matrix:" in text
    assert "python -m yolo_flywire.hosted_extract shard" in text
    assert "python -m yolo_flywire.hosted_aggregate" in text
    assert "python -m yolo_flywire.hosted_development run" in text
    assert "ntu120-rgb-part1.zip" not in text
    assert "ntu120-rgb-part2.zip" not in text


def test_hosted_real_workflow_never_exposes_final_test_or_private_assets():
    text = _workflow()
    assert "--final-test" not in text
    assert "--test" not in text
    assert "transport-manifest.json" not in text.split("Upload final hosted development evidence", 1)[1]
    assert "yolo11n-pose.pt" not in text.split("Upload final hosted development evidence", 1)[1]
    assert "final-evidence" in text
