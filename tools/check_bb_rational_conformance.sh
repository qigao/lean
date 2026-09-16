#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GOLDEN="conformance/bb_rational_v1.jsonl"
CHECKER="NarrativeDynamics/Conformance/BBRationalConformance.lean"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

bash -n "$0"
python3 -m unittest discover -s tests -p 'test_bb_rational_conformance.py' -v

python3 -m tools.generate_bb_rational_conformance --output "$TMP_DIR/generated.jsonl"
cmp -- "$GOLDEN" "$TMP_DIR/generated.jsonl"

export PATH="$HOME/.elan/bin:$PATH"
timeout --kill-after=10s 240s lake build \
  NarrativeDynamics.Core.FitnessABMReplay \
  NarrativeDynamics.Core.FitnessABMPathN

run_checker() {
  timeout --kill-after=10s 240s \
    lake env lean --run "$CHECKER" "$1"
}

canonical_output="$(run_checker "$GOLDEN")"
if [[ "$canonical_output" != "5 cases checked" ]]; then
  printf 'unexpected canonical checker output: %s\n' "$canonical_output" >&2
  exit 1
fi

BB_RATIONAL_TMP_DIR="$TMP_DIR" python3 - <<'PY'
import copy
import os
from pathlib import Path

from tools.bb_rational_conformance import (
    canonical_json_line,
    mutate_contract_hash,
    mutate_expected_rational,
    mutate_schema_version,
)
from tools.generate_bb_rational_conformance import generate_records

root = Path(os.environ["BB_RATIONAL_TMP_DIR"])
records = generate_records()

def case(records, case_id):
    return next(record for record in records if record["case_id"] == case_id)

extra_top_level = copy.deepcopy(records)
extra_top_level[0]["unexpected"] = 1

extra_provenance = copy.deepcopy(records)
extra_provenance[0]["provenance"]["unexpected"] = 1

extra_rational = copy.deepcopy(records)
case(extra_rational, "inclusive-threshold")["input"]["alpha"]["unexpected"] = 1

mutations = {
    "schema.jsonl": mutate_schema_version(records),
    "contract-hash.jsonl": mutate_contract_hash(records),
    "expected-rational.jsonl": mutate_expected_rational(records),
    "extra-top-level.jsonl": extra_top_level,
    "extra-provenance.jsonl": extra_provenance,
    "extra-rational.jsonl": extra_rational,
}
for name, mutated in mutations.items():
    payload = "".join(canonical_json_line(record) for record in mutated)
    (root / name).write_text(payload, encoding="utf-8", newline="\n")
PY

for name in schema contract-hash expected-rational extra-top-level extra-provenance extra-rational; do
  corpus="$TMP_DIR/$name.jsonl"
  log="$TMP_DIR/$name.log"
  if run_checker "$corpus" >"$log" 2>&1; then
    printf 'checker unexpectedly accepted %s drift\n' "$name" >&2
    exit 1
  fi
  printf 'checker rejected %s drift\n' "$name"
done

# Mutations are temporary; the reviewed golden must remain byte-identical.
cmp -- "$GOLDEN" "$TMP_DIR/generated.jsonl"
printf 'BB rational conformance gate passed\n'
