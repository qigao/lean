"""Verify official KTH action metadata and archive bytes without decoding video."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any
from zipfile import BadZipFile, ZipFile

from .ntu_io import _canonical_json, _hash_file, _hash_json

_ACTIONS = ("boxing", "handclapping", "handwaving", "jogging", "running", "walking")
_ARCHIVE_URLS = {action: f"https://www.csc.kth.se/cvap/actions/{action}.zip" for action in _ACTIONS}
_SEQUENCE_URL = "https://www.csc.kth.se/cvap/actions/00sequences.txt"
_TRAIN = (11, 12, 13, 14, 15, 16, 17, 18)
_VALIDATION = (19, 20, 21, 23, 24, 25, 1, 4)
_FINAL_TEST = (22, 2, 3, 5, 6, 7, 8, 9, 10)
_SPLITS = {**{value: "train" for value in _TRAIN},
           **{value: "validation" for value in _VALIDATION},
           **{value: "final_test" for value in _FINAL_TEST}}
_MISSING_VIDEO_KEY = "person13_handclapping_d3"
_CORRUPT_BOXING_VIDEO_KEY = "person01_boxing_d4"
_CORRUPT_BOXING_SAMPLE_ID = "person01_boxing_d4#04"
_CORRUPT_BOXING_ARCHIVE_SHA256 = "09437744bd760b3ba628d0315d12e2b6a7f86f5fbd41a316583bb4e3d5f09f20"
_CORRUPT_BOXING_MEMBER_SHA256 = "9c6a972f1268aab20ee3aca6609c5686d6170a92b03a362deaa6b5c7920689d1"
_LINE = re.compile(
    r"^(person(?P<subject>[0-9]{2})_(?P<action>boxing|handclapping|handwaving|jogging|running|walking)_d(?P<scenario>[1-4]))"
    r"\s+frames\s+(?P<ranges>.+?)\s*$"
)
_MISSING_LINE = re.compile(
    r"^(person(?P<subject>[0-9]{2})_(?P<action>boxing|handclapping|handwaving|jogging|running|walking)_d(?P<scenario>[1-4]))"
    r"\s+\*missing\*\s*$"
)
_RANGE = re.compile(r"([0-9]+)-([0-9]+)")
_MEMBER = re.compile(
    r"^person(?P<subject>[0-9]{2})_(?P<action>boxing|handclapping|handwaving|jogging|running|walking)_d(?P<scenario>[1-4])_uncomp\.avi$"
)
_SPLIT_LINE = re.compile(r"^(Training|Validation|Test):\s*(.*)$")
_CHUNK = 1024 * 1024


@dataclass(frozen=True)
class KthSequencePlan:
    videos: tuple[dict[str, Any], ...]
    subsequences: tuple[dict[str, Any], ...]
    split_policy: dict[str, Any]


def expected_source_exclusions() -> list[dict[str, Any]]:
    """Return the one source-level exclusion proven against the pinned official boxing bytes."""
    return [{
        "kind": "kth_source_corrupt_subsequence_exclusion",
        "sample_id": _CORRUPT_BOXING_SAMPLE_ID,
        "video_key": _CORRUPT_BOXING_VIDEO_KEY,
        "member_filename": _CORRUPT_BOXING_VIDEO_KEY + "_uncomp.avi",
        "action": "boxing",
        "subject": 1,
        "scenario": 4,
        "split": "validation",
        "range_index": 4,
        "official_range": [246, 370],
        "archive_sha256": _CORRUPT_BOXING_ARCHIVE_SHA256,
        "member_sha256": _CORRUPT_BOXING_MEMBER_SHA256,
        "declared_frames": 370,
        "strict_decodable_frames": 304,
        "nonstrict_decodable_frames": 305,
        "reason": "official KTH AVI is truncated inside the fourth listed subsequence; no tested decoder recovers frames 306-370",
    }]


def _validate_source_exclusions(exclusions: Any) -> list[dict[str, Any]]:
    if type(exclusions) is not list:
        raise ValueError("KTH source exclusions must be a JSON array")
    if _canonical_json(exclusions) != _canonical_json(expected_source_exclusions()):
        raise ValueError("KTH source exclusion identity changed")
    return exclusions


def _partition_source_corrupt_rows(
    video_key: str, *, archive_sha256: str, member_sha256: str,
    sample_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Exclude only the unrecoverable official fourth boxing interval, bound to exact source bytes."""
    if video_key != _CORRUPT_BOXING_VIDEO_KEY:
        return list(sample_rows), []
    if archive_sha256 != _CORRUPT_BOXING_ARCHIVE_SHA256:
        raise ValueError("KTH corrupt-source exclusion archive SHA changed")
    if member_sha256 != _CORRUPT_BOXING_MEMBER_SHA256:
        raise ValueError("KTH corrupt-source exclusion member SHA changed")
    expected = [
        ("person01_boxing_d4#01", 1, 1, 106),
        ("person01_boxing_d4#02", 2, 107, 170),
        ("person01_boxing_d4#03", 3, 171, 245),
        ("person01_boxing_d4#04", 4, 246, 370),
    ]
    actual = [
        (row.get("sample_id"), row.get("range_index"), row.get("start_frame"), row.get("end_frame"))
        for row in sample_rows
    ]
    if actual != expected:
        raise ValueError("KTH corrupt-source exclusion official ranges changed")
    kept = [row for row in sample_rows if row["sample_id"] != _CORRUPT_BOXING_SAMPLE_ID]
    if len(kept) != 3:
        raise ValueError("KTH corrupt-source exclusion did not remove exactly one subsequence")
    return kept, expected_source_exclusions()


def development_subsequences(
    subsequences: Any, source_exclusions: Any,
) -> list[dict[str, Any]]:
    """Return the exact usable development roster after source exclusions; final test stays sealed."""
    if not isinstance(subsequences, (list, tuple)):
        raise ValueError("KTH subsequences must be an ordered sequence")
    exclusions = _validate_source_exclusions(source_exclusions)
    excluded = {row["sample_id"] for row in exclusions}
    rows = [dict(row) for row in subsequences
            if row.get("split") != "final_test" and row.get("sample_id") not in excluded]
    if excluded != {_CORRUPT_BOXING_SAMPLE_ID}:
        raise ValueError("KTH source exclusion sample set changed")
    if _CORRUPT_BOXING_SAMPLE_ID in {row.get("sample_id") for row in rows}:
        raise ValueError("KTH corrupt subsequence remained in development roster")
    return rows


def _split_policy() -> dict[str, Any]:
    return {
        "id": "kth-icpr2004-subject-split-v1",
        "training_subjects": list(_TRAIN),
        "validation_subjects": list(_VALIDATION),
        "final_test_subjects": list(_FINAL_TEST),
        "source": "KTH:00sequences.txt:Schuldt-Laptev-Caputo-ICPR2004",
        "missing_video_key": _MISSING_VIDEO_KEY,
        "range_index_policy": "preserve-official-list-order",
        "range_routing_policy": "frame-membership; overlapping official intervals share one predictor result",
    }


def _parse_subject_list(text: str) -> tuple[int, ...]:
    values = tuple(int(value) for value in re.findall(r"[0-9]{2}|(?<![0-9])[0-9](?![0-9])", text))
    if not values or any(not 1 <= value <= 25 for value in values) or len(set(values)) != len(values):
        raise ValueError("malformed KTH subject split declaration")
    return values


def _ranges(text: str, video_key: str) -> tuple[tuple[int, int], ...]:
    """Preserve official list order, including documented overlaps/non-monotonic entries."""
    matches = tuple((int(a), int(b)) for a, b in _RANGE.findall(text))
    residue = _RANGE.sub("", text)
    if not matches or residue.replace(",", "").strip():
        raise ValueError(f"malformed KTH frame ranges: {video_key}")
    for start, end in matches:
        if start < 1 or end < start:
            raise ValueError(f"invalid KTH frame range: {video_key}")
    return matches


def _identity(match: re.Match[str]) -> tuple[str, int, str, int]:
    key = match.group(1)
    subject = int(match.group("subject"))
    action = match.group("action")
    scenario = int(match.group("scenario"))
    if subject not in _SPLITS or action not in _ACTIONS or not 1 <= scenario <= 4:
        raise ValueError(f"unexpected KTH sequence identity: {key}")
    return key, subject, action, scenario


def parse_sequence_file(text: str) -> KthSequencePlan:
    """Parse all 600 logical KTH slots, including the one official missing parent video."""
    if type(text) is not str or not text.strip():
        raise ValueError("KTH sequence text must be nonempty")
    declared: dict[str, tuple[int, ...]] = {}
    ranges_by_key: dict[str, tuple[tuple[int, int], ...]] = {}
    missing_keys: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        split_match = _SPLIT_LINE.fullmatch(line)
        if split_match:
            declared[split_match.group(1)] = _parse_subject_list(split_match.group(2))
            continue
        missing_match = _MISSING_LINE.fullmatch(line)
        if missing_match:
            key, _, _, _ = _identity(missing_match)
            if key in ranges_by_key:
                raise ValueError(f"duplicate KTH video key: {key}")
            ranges_by_key[key] = ()
            missing_keys.add(key)
            continue
        match = _LINE.fullmatch(line)
        if not match:
            continue
        key, _, _, _ = _identity(match)
        if key in ranges_by_key:
            raise ValueError(f"duplicate KTH video key: {key}")
        ranges_by_key[key] = _ranges(match.group("ranges"), key)

    expected_declared = {"Training": _TRAIN, "Validation": _VALIDATION, "Test": _FINAL_TEST}
    if declared != expected_declared:
        raise ValueError("KTH declared subject split differs from frozen ICPR 2004 split")
    expected_keys = {
        f"person{subject:02d}_{action}_d{scenario}"
        for action in _ACTIONS for subject in range(1, 26) for scenario in range(1, 5)
    }
    if set(ranges_by_key) != expected_keys or len(ranges_by_key) != 600:
        missing = sorted(expected_keys - set(ranges_by_key))
        extra = sorted(set(ranges_by_key) - expected_keys)
        raise ValueError(
            f"KTH sequence file must describe exactly 600 logical video slots; "
            f"missing={missing[:8]} extra={extra[:8]}"
        )
    if missing_keys != {_MISSING_VIDEO_KEY}:
        raise ValueError(f"KTH official missing-video set changed: {sorted(missing_keys)}")

    videos: list[dict[str, Any]] = []
    subsequences: list[dict[str, Any]] = []
    for action in _ACTIONS:
        for subject in range(1, 26):
            split = _SPLITS[subject]
            for scenario in range(1, 5):
                key = f"person{subject:02d}_{action}_d{scenario}"
                ranges = ranges_by_key[key]
                is_missing = key == _MISSING_VIDEO_KEY
                videos.append({
                    "video_key": key, "filename": key + "_uncomp.avi", "action": action,
                    "subject": subject, "scenario": scenario, "split": split,
                    "missing": is_missing,
                    "ranges": [[start, end] for start, end in ranges],
                })
                for index, (start, end) in enumerate(ranges, 1):
                    subsequences.append({
                        "sample_id": f"{key}#{index:02d}", "video_key": key,
                        "range_index": index, "start_frame": start, "end_frame": end,
                        "label": action, "subject": subject, "scenario": scenario,
                        "split": split,
                    })
    if sum(not row["missing"] for row in videos) != 599:
        raise ValueError("KTH must contain exactly 599 present parent videos")
    if len(subsequences) != 2391:
        raise ValueError(
            f"KTH sequence file must describe exactly 2391 official subsequences; got {len(subsequences)}"
        )
    return KthSequencePlan(tuple(videos), tuple(subsequences), _split_policy())


def _member_digest(handle: Any) -> str:
    digest = hashlib.sha256()
    while chunk := handle.read(_CHUNK):
        digest.update(chunk)
    return digest.hexdigest()


def build_kth_source(archives: dict[str, str | Path], sequence_file: str | Path) -> dict[str, Any]:
    """Bind six official archive bytes, 599 present AVIs, one missing slot and official metadata."""
    if type(archives) is not dict or set(archives) != set(_ACTIONS):
        raise ValueError("KTH source requires exactly the six official action archives")
    seq_path = Path(sequence_file).absolute()
    if seq_path.is_symlink() or not seq_path.is_file():
        raise ValueError("KTH sequence file must be a regular non-symlink file")
    sequence_raw = seq_path.read_bytes()
    try:
        sequence_text = sequence_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("KTH sequence file must be UTF-8 text") from exc
    plan = parse_sequence_file(sequence_text)
    sequence_sha256 = hashlib.sha256(sequence_raw).hexdigest()

    video_plan = {row["video_key"]: row for row in plan.videos}
    archive_records: list[dict[str, Any]] = []
    video_records: list[dict[str, Any]] = []
    missing_records = [
        {key: row[key] for key in ("video_key", "filename", "action", "subject", "scenario", "split")}
        for row in plan.videos if row["missing"]
    ]
    for action in _ACTIONS:
        archive = Path(archives[action]).absolute()
        info = archive.lstat() if archive.exists() else None
        if (info is None or not stat.S_ISREG(info.st_mode) or archive.is_symlink()
                or info.st_size <= 0):
            raise ValueError(f"KTH archive must be a nonempty regular file: {action}")
        archive_size, archive_sha256, _ = _hash_file(archive)
        expected_names = {row["filename"] for row in plan.videos
                          if row["action"] == action and not row["missing"]}
        try:
            with ZipFile(archive, "r") as zipped:
                infos = [entry for entry in zipped.infolist() if not entry.is_dir()]
                names = [entry.filename for entry in infos]
                if len(names) != len(set(names)):
                    raise ValueError(f"duplicate KTH ZIP member: {action}")
                for name in names:
                    pure = PurePosixPath(name)
                    if pure.is_absolute() or ".." in pure.parts or len(pure.parts) != 1:
                        raise ValueError(f"unsafe KTH ZIP member path: {name}")
                if set(names) != expected_names or len(names) != len(expected_names):
                    raise ValueError(
                        f"KTH {action} archive roster differs from official present-video set; "
                        f"expected={len(expected_names)} actual={len(names)}"
                    )
                by_name = {entry.filename: entry for entry in infos}
                for name in sorted(expected_names):
                    match = _MEMBER.fullmatch(name)
                    assert match is not None
                    subject, scenario = int(match.group("subject")), int(match.group("scenario"))
                    key = name.removesuffix("_uncomp.avi")
                    planned = video_plan[key]
                    entry = by_name[name]
                    if entry.file_size <= 0:
                        raise ValueError(f"empty KTH AVI member: {name}")
                    with zipped.open(entry, "r") as handle:
                        digest = _member_digest(handle)
                    video_records.append({
                        "video_key": key, "filename": name, "action": action,
                        "subject": subject, "scenario": scenario, "split": planned["split"],
                        "size_bytes": entry.file_size, "sha256": digest,
                    })
        except BadZipFile as exc:
            raise ValueError(f"invalid KTH ZIP archive: {action}") from exc
        archive_records.append({
            "action": action, "url": _ARCHIVE_URLS[action], "size_bytes": archive_size,
            "sha256": archive_sha256,
        })

    videos = sorted(video_records, key=lambda row: (_ACTIONS.index(row["action"]), row["subject"], row["scenario"]))
    if len(videos) != 599 or missing_records != [{
        "video_key": _MISSING_VIDEO_KEY,
        "filename": _MISSING_VIDEO_KEY + "_uncomp.avi",
        "action": "handclapping", "subject": 13, "scenario": 3, "split": "train",
    }]:
        raise ValueError("KTH present/missing parent-video roster changed")
    subsequences = [dict(row) for row in plan.subsequences]
    boxing_archive = next(row for row in archive_records if row["action"] == "boxing")
    corrupt_parent = next(row for row in videos if row["video_key"] == _CORRUPT_BOXING_VIDEO_KEY)
    source_exclusions = (expected_source_exclusions()
                         if boxing_archive["sha256"] == _CORRUPT_BOXING_ARCHIVE_SHA256
                         and corrupt_parent["sha256"] == _CORRUPT_BOXING_MEMBER_SHA256 else [])
    dataset_content_hash = _hash_json({
        "sequence_file_sha256": sequence_sha256,
        "videos": [{key: row[key] for key in ("video_key", "size_bytes", "sha256")} for row in videos],
        "missing_videos": missing_records,
    })
    split_hash = _hash_json({
        "dataset_content_hash": dataset_content_hash,
        "classes": list(_ACTIONS), "policy": plan.split_policy,
        "assignments": [{"sample_id": row["sample_id"], "split": row["split"]} for row in subsequences],
        "source_exclusions": source_exclusions,
    })
    inventory = {
        "format_version": 1, "kind": "kth_rgb_source_manifest",
        "evidence_scope": "real_source_bytes_only_no_video_decode",
        "official_source": "https://www.csc.kth.se/cvap/actions/",
        "archive_urls": [_ARCHIVE_URLS[action] for action in _ACTIONS],
        "sequence_url": _SEQUENCE_URL, "classes": list(_ACTIONS),
        "split_policy": plan.split_policy, "sequence_file_sha256": sequence_sha256,
        "archives": archive_records, "videos": videos, "missing_videos": missing_records,
        "subsequences": subsequences, "source_exclusions": source_exclusions,
        "dataset_content_hash": dataset_content_hash, "split_hash": split_hash,
    }
    source_payload = {key: value for key, value in inventory.items() if key != "source_manifest_hash"}
    inventory["source_manifest_hash"] = _hash_json(source_payload)
    inventory["input_inventory_hash"] = _hash_json({
        "sequence_file_sha256": sequence_sha256, "videos": videos,
        "missing_videos": missing_records, "subsequences": subsequences,
        "source_exclusions": source_exclusions,
    })
    return json.loads(_canonical_json(inventory))


def read_sequence_plan(path: str | Path) -> KthSequencePlan:
    return parse_sequence_file(Path(path).read_text(encoding="utf-8"))
