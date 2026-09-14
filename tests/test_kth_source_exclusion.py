import hashlib
from copy import deepcopy

import pytest

from yolo_flywire.kth_aggregate import _source_record
from yolo_flywire.kth_pose_index import verify_kth_source
from yolo_flywire.kth_source import (
    _ACTIONS,
    _CORRUPT_BOXING_ARCHIVE_SHA256,
    _CORRUPT_BOXING_MEMBER_SHA256,
    _MISSING_VIDEO_KEY,
    _partition_source_corrupt_rows,
    development_subsequences,
    expected_source_exclusions,
    parse_sequence_file,
)


def _sequence_text() -> str:
    lines = [
        "Training:   person11, 12, 13, 14, 15, 16, 17, 18",
        "Validation: person19, 20, 21, 23, 24, 25, 01, 04",
        "Test:       person22, 02, 03, 05, 06, 07, 08, 09, 10",
    ]
    reduced = 5
    for subject in range(1, 26):
        for action in _ACTIONS:
            for scenario in range(1, 5):
                key = f"person{subject:02d}_{action}_d{scenario}"
                if key == _MISSING_VIDEO_KEY:
                    lines.append(f"{key} *missing*")
                    continue
                if key == "person01_boxing_d4":
                    ranges = ["1-106", "107-170", "171-245", "246-370"]
                else:
                    ranges = ["1-2", "3-4", "5-6", "7-8"]
                    if reduced:
                        ranges.pop()
                        reduced -= 1
                lines.append(key + " frames " + ", ".join(ranges))
    return "\n".join(lines) + "\n"


def _target_rows(plan):
    return [row for row in plan.subsequences if row["video_key"] == "person01_boxing_d4"]


def test_exact_corrupt_parent_keeps_first_three_ranges_and_excludes_only_fourth():
    plan = parse_sequence_file(_sequence_text())
    rows = _target_rows(plan)
    kept, exclusions = _partition_source_corrupt_rows(
        "person01_boxing_d4",
        archive_sha256=_CORRUPT_BOXING_ARCHIVE_SHA256,
        member_sha256=_CORRUPT_BOXING_MEMBER_SHA256,
        sample_rows=rows,
    )
    assert [row["sample_id"] for row in kept] == [
        "person01_boxing_d4#01", "person01_boxing_d4#02", "person01_boxing_d4#03"
    ]
    assert exclusions == expected_source_exclusions()
    assert exclusions[0]["sample_id"] == "person01_boxing_d4#04"
    assert exclusions[0]["official_range"] == [246, 370]
    assert exclusions[0]["declared_frames"] == 370
    assert exclusions[0]["strict_decodable_frames"] == 304
    assert exclusions[0]["nonstrict_decodable_frames"] == 305


def test_corrupt_exclusion_is_bound_to_exact_official_archive_and_member_bytes():
    plan = parse_sequence_file(_sequence_text())
    rows = _target_rows(plan)
    with pytest.raises(ValueError, match="archive SHA"):
        _partition_source_corrupt_rows(
            "person01_boxing_d4",
            archive_sha256="0" * 64,
            member_sha256=_CORRUPT_BOXING_MEMBER_SHA256,
            sample_rows=rows,
        )
    with pytest.raises(ValueError, match="member SHA"):
        _partition_source_corrupt_rows(
            "person01_boxing_d4",
            archive_sha256=_CORRUPT_BOXING_ARCHIVE_SHA256,
            member_sha256="0" * 64,
            sample_rows=rows,
        )


def test_source_manifest_preserves_official_2391_and_development_roster_excludes_one_validation_sample():
    plan = parse_sequence_file(_sequence_text())
    videos = []
    for row in plan.videos:
        if row["missing"]:
            continue
        digest = hashlib.sha256(row["video_key"].encode()).hexdigest()
        if row["video_key"] == "person01_boxing_d4":
            digest = _CORRUPT_BOXING_MEMBER_SHA256
        videos.append({
            key: row[key] for key in ("video_key", "filename", "action", "subject", "scenario", "split")
        } | {"size_bytes": 1, "sha256": digest})
    missing = [{
        "video_key": _MISSING_VIDEO_KEY,
        "filename": _MISSING_VIDEO_KEY + "_uncomp.avi",
        "action": "handclapping", "subject": 13, "scenario": 3, "split": "train",
    }]
    archives = []
    for action in _ACTIONS:
        digest = hashlib.sha256(action.encode()).hexdigest()
        if action == "boxing":
            digest = _CORRUPT_BOXING_ARCHIVE_SHA256
        archives.append({
            "action": action, "url": f"https://www.csc.kth.se/cvap/actions/{action}.zip",
            "size_bytes": 1, "sha256": digest,
        })
    source = _source_record(
        plan, "a" * 64, archives, videos, missing, expected_source_exclusions()
    )
    assert len(source["subsequences"]) == 2391
    assert source["source_exclusions"] == expected_source_exclusions()
    verified = verify_kth_source(source)
    usable = development_subsequences(verified["subsequences"], verified["source_exclusions"])
    official_development = [row for row in plan.subsequences if row["split"] != "final_test"]
    assert len(usable) == len(official_development) - 1
    assert "person01_boxing_d4#04" not in {row["sample_id"] for row in usable}
    assert {f"person01_boxing_d4#{index:02d}" for index in (1, 2, 3)} <= {
        row["sample_id"] for row in usable
    }

    drift = deepcopy(source)
    boxing = next(row for row in drift["archives"] if row["action"] == "boxing")
    boxing["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source exclusion"):
        verify_kth_source(drift)
