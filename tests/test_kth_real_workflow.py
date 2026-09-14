from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "yolo-flywire-kth-real.yml"


def _workflow() -> str:
    assert WORKFLOW.is_file(), "KTH real-CI workflow is missing"
    return WORKFLOW.read_text(encoding="utf-8")


def test_kth_real_workflow_runs_publicly_on_research_branch():
    text = _workflow()
    assert "workflow_dispatch:" in text
    assert "push:" in text
    assert "research/yolo-flywire-behavior-v0" in text
    assert "runs-on: ubuntu-latest" in text
    assert "self-hosted" not in text
    assert "secrets." not in text


def test_kth_real_workflow_uses_only_official_kth_and_yolo_sources():
    text = _workflow()
    assert "https://www.csc.kth.se/cvap/actions" in text
    assert "00sequences.txt" in text
    for action in ("boxing", "handclapping", "handwaving", "jogging", "running", "walking"):
        assert action in text
    assert "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-pose.pt" in text
    assert 'YOLO("yolo11n-pose.pt")' not in text
    assert "mirror" not in text.lower()


def test_kth_real_workflow_executes_real_extract_aggregate_and_existing_comparison():
    text = _workflow()
    for command in (
        "python -m yolo_flywire.kth_extract freeze",
        "python -m yolo_flywire.kth_extract shard",
        "python -m yolo_flywire.kth_aggregate",
        "python -m yolo_flywire.provenance generate",
        "python -m yolo_flywire.provenance verify",
        "python -m yolo_flywire.kth_development run",
    ):
        assert command in text
    assert "max-parallel: 6" in text
    assert "timeout-minutes: 720" not in text


def test_kth_real_workflow_keeps_all_shards_and_diagnoses_boxing_decode_failures():
    text = _workflow()
    assert "fail-fast: false" in text
    assert "Diagnose boxing decoder boundary on failure" in text
    assert "if: failure() && matrix.action == 'boxing'" in text
    assert "person01_boxing_d4_uncomp.avi" in text
    assert "KTH_PYAV_DIAGNOSTIC" in text
    assert "successful_frames=" in text
    assert "last_pts=" in text
    assert "exception_type=" in text
    assert "exception_repr=" in text
    assert "kth-decoder-diagnostic" not in text


def test_kth_real_workflow_never_uploads_rgb_model_or_final_test_observations():
    text = _workflow()
    final_upload = text.split("Upload real KTH development report", 1)[1]
    assert "path: final-evidence" in final_upload
    assert ".zip" not in final_upload
    assert ".avi" not in final_upload
    assert "yolo11n-pose.pt" not in final_upload
    assert "--final-test" not in text
    assert "--test" not in text
