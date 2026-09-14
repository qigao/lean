from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "yolo-flywire-real.yml"


def _workflow() -> str:
    assert WORKFLOW.is_file(), "real-data CI workflow is missing"
    return WORKFLOW.read_text(encoding="utf-8")


def test_real_workflow_is_manual_and_uses_dedicated_self_hosted_runner():
    text = _workflow()
    assert "workflow_dispatch:" in text
    assert "\n  push:" not in text
    assert "\n  pull_request:" not in text
    assert "runs-on: [self-hosted, linux, x64, yolo-flywire-real]" in text
    assert 'REQUIRED_FREE_GIB: "350"' in text


def test_real_workflow_requires_authorized_ntu_urls_and_downloads_pinned_yolo_family():
    text = _workflow()
    assert "NTU120_RGB_ARCHIVE_URL_1: ${{ secrets.NTU120_RGB_ARCHIVE_URL_1 }}" in text
    assert "NTU120_RGB_ARCHIVE_URL_2: ${{ secrets.NTU120_RGB_ARCHIVE_URL_2 }}" in text
    assert "curl --fail --location --retry 5" in text
    assert 'ultralytics==8.4.146' in text
    assert 'YOLO("yolo26n-pose.pt")' in text
    assert "sha256sum yolo26n-pose.pt" in text


def test_real_workflow_executes_freeze_provenance_and_development_only_pipeline():
    text = _workflow()
    for command in (
        "python -m yolo_flywire.ntu_io freeze",
        "python -m yolo_flywire.real_freeze freeze",
        "python -m yolo_flywire.provenance generate",
        "python -m yolo_flywire.provenance verify",
        "python -m yolo_flywire.real_development run",
    ):
        assert command in text
    assert "control-bundle.json" in text
    assert "development_report.json" in text
    assert "final-test" not in text.lower()


def test_real_workflow_uploads_evidence_only_not_licensed_media_or_model_assets():
    text = _workflow()
    assert "actions/upload-artifact@v4" in text
    upload = text.split("actions/upload-artifact@v4", 1)[1]
    assert "path: .evidence" in upload
    assert ".real-input" not in upload
    assert "yolo26n-pose.pt" not in upload
    assert "extraction/observations.jsonl" not in upload
    assert "extraction/timing.jsonl" not in upload
