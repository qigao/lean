from pathlib import Path
from zipfile import ZipFile

import pytest

from yolo_flywire.kth_source import _ACTIONS, build_kth_source, parse_sequence_file


def _sequence_text() -> str:
    lines = [
        "This database contains sequences of six classes of actions performed by",
        "25 subjects in four different conditions d1-d4 (see below).",
        "Training:   person11, 12, 13, 14, 15, 16, 17, 18",
        "Validation: person19, 20, 21, 23, 24, 25, 01, 04",
        "Test:       person22, 02, 03, 05, 06, 07, 08, 09, 10",
        "",
    ]
    reduced = 5
    for subject in range(1, 26):
        for action in _ACTIONS:
            for scenario in range(1, 5):
                key = f"person{subject:02d}_{action}_d{scenario}"
                if key == "person13_handclapping_d3":
                    lines.append(f"{key}\t*missing*")
                    continue
                ranges = ["1-2", "3-4", "5-6", "7-8"]
                if reduced:
                    ranges.pop()
                    reduced -= 1
                lines.append(key + "\tframes\t" + ", ".join(ranges))
    return "\n".join(lines) + "\n"


def _archives(tmp_path: Path) -> dict[str, Path]:
    result = {}
    for action in _ACTIONS:
        path = tmp_path / f"{action}.zip"
        with ZipFile(path, "w") as zipped:
            for subject in range(1, 26):
                for scenario in range(1, 5):
                    if action == "handclapping" and subject == 13 and scenario == 3:
                        continue
                    name = f"person{subject:02d}_{action}_d{scenario}_uncomp.avi"
                    zipped.writestr(name, f"{action}:{subject}:{scenario}".encode())
        result[action] = path
    return result


def test_parse_kth_official_shape_and_subject_split():
    plan = parse_sequence_file(_sequence_text())
    assert len(plan.videos) == 600
    assert sum(not row["missing"] for row in plan.videos) == 599
    assert [row["video_key"] for row in plan.videos if row["missing"]] == ["person13_handclapping_d3"]
    assert len(plan.subsequences) == 2391
    assert plan.videos[0]["video_key"] == "person01_boxing_d1"
    assert plan.subsequences[0]["sample_id"] == "person01_boxing_d1#01"
    assert {row["split"] for row in plan.subsequences if row["subject"] == 11} == {"train"}
    assert {row["split"] for row in plan.subsequences if row["subject"] == 19} == {"validation"}
    assert {row["split"] for row in plan.subsequences if row["subject"] == 22} == {"final_test"}


def test_build_kth_source_hashes_exact_six_archive_roster(tmp_path):
    sequence = tmp_path / "00sequences.txt"
    sequence.write_text(_sequence_text(), encoding="utf-8")
    source = build_kth_source(_archives(tmp_path), sequence)
    assert source["kind"] == "kth_rgb_source_manifest"
    assert source["classes"] == list(_ACTIONS)
    assert len(source["archives"]) == 6
    assert len(source["videos"]) == 599
    assert source["missing_videos"] == [{
        "video_key": "person13_handclapping_d3",
        "filename": "person13_handclapping_d3_uncomp.avi",
        "action": "handclapping",
        "subject": 13,
        "scenario": 3,
        "split": "train",
    }]
    assert len(source["subsequences"]) == 2391
    for name in ("dataset_content_hash", "split_hash", "source_manifest_hash", "input_inventory_hash"):
        assert len(source[name]) == 64


def test_kth_source_rejects_split_drift_and_missing_video(tmp_path):
    broken = _sequence_text().replace(
        "Training:   person11, 12, 13, 14, 15, 16, 17, 18",
        "Training:   person10, 12, 13, 14, 15, 16, 17, 18",
        1,
    )
    with pytest.raises(ValueError, match="subject split"):
        parse_sequence_file(broken)
    lines = _sequence_text().splitlines()
    missing = "\n".join(line for line in lines if not line.startswith("person25_walking_d4")) + "\n"
    with pytest.raises(ValueError, match="600 logical video slots"):
        parse_sequence_file(missing)
