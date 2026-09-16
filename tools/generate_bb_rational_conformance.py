"""Generate the reviewed exact-rational BB conformance JSONL corpus."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tools.bb_rational_conformance import (
    CASES,
    SCHEMA,
    canonical_json_line,
    contract_hash,
    evaluate_named,
    rat,
)

GENERATOR = "tools/generate_bb_rational_conformance.py"
LEAN_CHECKER = "NarrativeDynamics/Conformance/BBRationalConformance.lean"
CASE_SET = "v1"


def _path_edges(node_count: int) -> list[list[int]]:
    return [[i, i + 1] for i in range(node_count - 1)]


def _provenance() -> dict[str, str]:
    return {
        "schema": SCHEMA,
        "generator": GENERATOR,
        "generator_contract": contract_hash(),
        "lean_checker": LEAN_CHECKER,
        "case_set": CASE_SET,
    }


def _record(case_id: str) -> dict[str, Any]:
    evaluated = evaluate_named(case_id)
    inputs = dict(evaluated["input"])
    expected = dict(evaluated["expected"])

    if evaluated["kind"] == "propagation":
        node_count = len(inputs["beliefs"])
        inputs = {
            "node_count": node_count,
            "edges": _path_edges(node_count),
            "alpha": inputs["alpha"],
            "threshold": inputs["threshold"],
            "beliefs": inputs["beliefs"],
            "exposures": inputs["exposures"],
        }
    elif case_id == "bb-successive-births":
        inputs = {
            "m": 1,
            "seed_fitness": inputs["seed_fitness"],
            "seed_edges": inputs["seed_edges"],
            "births": [
                {"fitness": rat(1), "targets": targets}
                for targets in inputs["birth_targets"]
            ],
        }
        expected.update({"birth_count": 2, "tick_count": 2})
    elif case_id == "duplicate-target-error":
        inputs = {
            "m": 2,
            "seed_fitness": inputs["seed_fitness"],
            "seed_edges": inputs["seed_edges"],
            "birth": {"fitness": rat(1), "targets": inputs["targets"]},
        }

    return {
        "schema": SCHEMA,
        "case_id": case_id,
        "kind": evaluated["kind"],
        "provenance": _provenance(),
        "input": inputs,
        "expected": expected,
    }


def generate_records() -> list[dict[str, Any]]:
    """Return the five authored v1 records in stable lexical id order."""

    case_ids = sorted(case["case_id"] for case in CASES)
    return [_record(case_id) for case_id in case_ids]


def write_corpus(path: str | Path) -> None:
    """Write canonical UTF-8 JSONL with exactly one final newline per record."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(canonical_json_line(record) for record in generate_records())
    destination.write_text(payload, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_corpus(args.output)


if __name__ == "__main__":
    main()
