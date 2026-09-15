from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from typing import Sequence

from .data import PreparedData, prepare_data
from .protocol import load_protocol
from .results import DevelopmentCell, reduce_development_matrix, require_development_split
from .training import train_validation_arm, training_protocol_fingerprint


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def _write_json(path: Path, value: object) -> None:
    path.write_text(_canonical_json(value), encoding="utf-8")


def _binding_payload(prepared: PreparedData) -> dict[str, object]:
    binding = prepared.binding
    return {
        "binding_fingerprint": binding.fingerprint,
        "dataset_content_hash": binding.dataset_content_hash,
        "split_hash": binding.split_hash,
        "feature_stats_hash": binding.feature_stats_hash,
        "label_map": [list(pair) for pair in binding.label_map],
        "label_map_hash": binding.label_map_hash,
        "adjacency_hash": binding.adjacency_hash,
        "ssm_spec_hash": binding.ssm_spec_hash,
        "length_quartiles": list(binding.length_quartiles),
    }


def _prepare_command(root: Path, protocol_path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"prepared output already exists: {output}")
    protocol = load_protocol(protocol_path)
    prepared = prepare_data(root, protocol)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(protocol_path, output / "protocol.json")
    _write_json(output / "binding.json", _binding_payload(prepared))
    (output / "feature_stats.json").write_text(prepared.standardizer.to_json() + "\n", encoding="utf-8")
    _write_json(
        output / "development_inventory.json",
        {
            "train": [
                {"sample_id": sample.sample_id, "label": sample.label, "frame_count": int(sample.features.shape[0])}
                for sample in prepared.train_samples
            ],
            "validation": [
                {"sample_id": sample.sample_id, "label": sample.label, "frame_count": int(sample.features.shape[0])}
                for sample in prepared.validation_samples
            ],
        },
    )
    _write_json(output / "final_test_inventory.json", list(prepared.final_test_inventory))


def _compare_validation_command(root: Path, prepared_dir: Path, output: Path) -> None:
    require_development_split("validation")
    if output.exists():
        raise FileExistsError(f"result output already exists: {output}")
    protocol_path = prepared_dir / "protocol.json"
    binding_path = prepared_dir / "binding.json"
    if not protocol_path.is_file() or not binding_path.is_file():
        raise ValueError("prepared directory is missing protocol.json or binding.json")
    protocol = load_protocol(protocol_path)
    prepared = prepare_data(root, protocol)
    try:
        recorded_binding = json.loads(binding_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("could not read prepared binding") from exc
    if recorded_binding.get("binding_fingerprint") != prepared.binding.fingerprint:
        raise ValueError("prepared binding fingerprint does not match current dataset bytes")

    training_hash = training_protocol_fingerprint(protocol)
    cells: list[DevelopmentCell] = []
    output.mkdir(parents=True, exist_ok=False)
    for seed in protocol.seeds:
        for kind in protocol.model_kinds:
            run = train_validation_arm(kind, seed, prepared, protocol)
            cell = DevelopmentCell(
                run=run,
                binding_fingerprint=prepared.binding.fingerprint,
                dataset_content_hash=prepared.binding.dataset_content_hash,
                split_hash=prepared.binding.split_hash,
                feature_stats_hash=prepared.binding.feature_stats_hash,
                training_fingerprint=training_hash,
            )
            cells.append(cell)
            _write_json(output / f"{kind}-seed-{seed}.json", asdict(cell))
    summary = reduce_development_matrix(cells, protocol)
    _write_json(output / "summary.json", asdict(summary))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pose-graph-ssm")
    subcommands = parser.add_subparsers(dest="command", required=True)

    prepare = subcommands.add_parser("prepare", help="prepare sealed development evidence")
    prepare.add_argument("--root", required=True, type=Path)
    prepare.add_argument("--protocol", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)

    compare = subcommands.add_parser("compare-validation", help="run the frozen 4x5 validation matrix")
    compare.add_argument("--root", required=True, type=Path)
    compare.add_argument("--prepared", required=True, type=Path)
    compare.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        _prepare_command(args.root, args.protocol, args.output)
        return 0
    if args.command == "compare-validation":
        _compare_validation_command(args.root, args.prepared, args.output)
        return 0
    raise ValueError(f"unsupported command: {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
