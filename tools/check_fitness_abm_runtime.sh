#!/usr/bin/env bash
set -euo pipefail
bb_runtime_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$bb_runtime_root"
bb_runtime_log="$(mktemp)"
bb_runtime_vectors="$(mktemp)"
trap 'rm -f "$bb_runtime_log" "$bb_runtime_vectors"' EXIT

python3 tools/audit_fitness_trust.py source \
  NarrativeDynamics/Conformance/FitnessABMRuntimeVectors.lean \
  NarrativeDynamics/Tests/FitnessABMRuntime.lean

/usr/bin/time -f 'FitnessABMRuntimeVectors build elapsed=%e s peak_rss=%M KiB' \
  timeout --kill-after=10s 240s \
  lake build NarrativeDynamics.Conformance.FitnessABMRuntimeVectors

/usr/bin/time -f 'FitnessABMRuntime literals elapsed=%e s peak_rss=%M KiB' \
  timeout --kill-after=10s 240s \
  lake env lean -DmaxErrors=1 NarrativeDynamics/Tests/FitnessABMRuntime.lean \
  2>&1 | tee "$bb_runtime_log"

python3 tools/audit_fitness_trust.py log "$bb_runtime_log" \
  --require NarrativeDynamics.Tests.FitnessABMRuntime.success_literals \
  --require NarrativeDynamics.Tests.FitnessABMRuntime.shared_error_literals \
  --require NarrativeDynamics.Tests.FitnessABMRuntime.transition_literals

/usr/bin/time -f 'FitnessABMRuntimeVectors run elapsed=%e s peak_rss=%M KiB' \
  timeout --kill-after=10s 240s \
  lake env lean --run NarrativeDynamics/Conformance/FitnessABMRuntimeVectors.lean \
  > "$bb_runtime_vectors"
cmp conformance/bb_abm_runtime_v1.json "$bb_runtime_vectors"
