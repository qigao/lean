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
_LINE = re.compile(
    r"^(person(?P<subject>[0-9]{2})_(?P<action>boxing|handclapping|handwaving|jogging|running|walking)_d(?P<scenario>[1-4]))"
    r"\s+frames\s+(?P<ranges>.+?)\s*$"
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


def _split_policy() -> dict[str, Any]:
    return {
        "id": "kth-icpr2004-subject-split-v1",
        "training_subjects": list(_TRAIN),
        "validation_subjects": list(_VALIDATION),
        "final_test_subjects": list(_FINAL_TEST),
        "source": "KTH:00sequences.txt:Schuldt-Laptev-Caputo-ICPR2004",
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


def parse_sequence_file(text: str) -> KthSequencePlan:
    """Parse the complete official KTH sequence roster and frozen subject split."""
    if type(text) is not str or not text.strip():
        raise ValueError("KTH sequence text must be nonempty")
    declared: dict[str, tuple[int, ...]] = {}
    ranges_by_key: dict[str, tuple[tuple[int, int], ...]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        split_match = _SPLIT_LINE.fullmatch(line)
        if split_match:
            declared[split_match.group(1)] = _parse_subject_list(split_match.group(2))
            continue
        match = _LINE.fullmatch(line)
        if not match:
            continue
        subject = int(match.group("subject"))
        action = match.group("action")
        scenario = int(match.group("scenario"))
        if subject not in _SPLITS:
            raise ValueError("KTH sequence references subject outside 1..25")
        key = match.group(1)
        if key in ranges_by_key:
            raise ValueError(f"duplicate KTH video key: {key}")
        if action not in _ACTIONS or not 1 <= scenario <= 4:
            raise ValueError(f"unexpected KTH action/scenario: {key}")
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
            f"KTH sequence file must describe exactly the frozen 600-video roster; "
            f"missing={missing[:8]} extra={extra[:8]}"
        )

    videos: list[dict[str, Any]] = []
    subsequences: list[dict[str, Any]] = []
    for action in _ACTIONS:
        for subject in range(1, 26):
            split = _SPLITS[subject]
            for scenario in range(1, 5):
                key = f"person{subject:02d}_{action}_d{scenario}"
                ranges = ranges_by_key[key]
                videos.append({
                    "video_key": key, "filename": key + "_uncomp.avi", "action": action,
                    "subject": subject, "scenario": scenario, "split": split,
                    "ranges": [[start, end] for start, end in ranges],
                })
                for index, (start, end) in enumerate(ranges, 1):
                    subsequences.append({
                        "sample_id": f"{key}#{index:02d}", "video_key": key,
                        "range_index": index, "start_frame": start, "end_frame": end,
                        "label": action, "subject": subject, "scenario": scenario,
                        "split": split,
                    })
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
    """Bind six official archive bytes and the official sequence file without decoding RGB."""
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
    for action in _ACTIONS:
        archive = Path(archives[action]).absolute()
        info = archive.lstat() if archive.exists() else None
        if (info is None or not stat.S_ISREG(info.st_mode) or archive.is_symlink()
                or info.st_size <= 0):
            raise ValueError(f"KTH archive must be a nonempty regular file: {action}")
        archive_size, archive_sha256, _ = _hash_file(archive)
        expected_names = {
            f"person{subject:02d}_{action}_d{scenario}_uncomp.avi"
            for subject in range(1, 26) for scenario in range(1, 5)
        }
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
                if set(names) != expected_names or len(names) != 100:
                    raise ValueError(f"KTH {action} archive must contain the exact 100-video roster")
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
    subsequences = [dict(row) for row in plan.subsequences]
    dataset_content_hash = _hash_json({
        "sequence_file_sha256": sequence_sha256,
        "videos": [{key: row[key] for key in ("video_key", "size_bytes", "sha256")} for row in videos],
    })
    split_hash = _hash_json({
        "dataset_content_hash": dataset_content_hash,
        "classes": list(_ACTIONS), "policy": plan.split_policy,
        "assignments": [{"sample_id": row["sample_id"], "split": row["split"]} for row in subsequences],
    })
    inventory = {
        "format_version": 1, "kind": "kth_rgb_source_manifest",
        "evidence_scope": "real_source_bytes_only_no_video_decode",
        "official_source": "https://www.csc.kth.se/cvap/actions/",
        "archive_urls": [_ARCHIVE_URLS[action] for action in _ACTIONS],
        "sequence_url": _SEQUENCE_URL, "classes": list(_ACTIONS),
        "split_policy": plan.split_policy, "sequence_file_sha256": sequence_sha256,
        "archives": archive_records, "videos": videos, "subsequences": subsequences,
        "dataset_content_hash": dataset_content_hash, "split_hash": split_hash,
    }
    source_payload = {key: value for key, value in inventory.items() if key != "source_manifest_hash"}
    inventory["source_manifest_hash"] = _hash_json(source_payload)
    inventory["input_inventory_hash"] = _hash_json({
        "sequence_file_sha256": sequence_sha256, "videos": videos, "subsequences": subsequences,
    })
    return json.loads(_canonical_json(inventory))


def read_sequence_plan(path: str | Path) -> KthSequencePlan:
    return parse_sequence_file(Path(path).read_text(encoding="utf-8"))
