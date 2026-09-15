#!/usr/bin/env bash
set -euo pipefail

path4_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$path4_root"

path4_time="$(type -P time)"
path4_log="$(mktemp)"
trap 'rm -f "$path4_log"' EXIT

python3 tools/audit_fitness_trust.py source \
  NarrativeDynamics/Core/FitnessABMPath4.lean \
  NarrativeDynamics/Core/FitnessABMPath4Convergence.lean \
  NarrativeDynamics/Tests/FitnessABMPath4.lean

for path4_module in NarrativeDynamics.Core.FitnessABMPath4 \
    NarrativeDynamics.Core.FitnessABMPath4Convergence; do
  "$path4_time" -f "$path4_module elapsed=%e s peak_rss=%M KiB" \
    timeout --kill-after=10s 240s lake build "$path4_module"
done

"$path4_time" -f 'FitnessABMPath4 tests elapsed=%e s peak_rss=%M KiB' \
  timeout --kill-after=10s 240s \
  lake env lean -DmaxErrors=1 NarrativeDynamics/Tests/FitnessABMPath4.lean \
  2>&1 | tee "$path4_log"

"$path4_time" -f 'FitnessABMPath5 tests elapsed=%e s peak_rss=%M KiB' \
  timeout --kill-after=10s 240s \
  lake env lean -DmaxErrors=1 NarrativeDynamics/Tests/FitnessABMPath5.lean

path4_required=()
for path4_name in propagate_independent_exposures propagate_eq_linear \
    allBroadcast_iterate mean_step iterate_closedForm trajectory_tendsto; do
  path4_required+=(--require "NarrativeDynamics.FitnessABMPath4.$path4_name")
done
for path4_name in replay_baseline raw_tail_bridge activation_bbii activation_bibi \
    activation_iibb bbii_tendsto bibi_tendsto iibb_tendsto equal_control separation_limit; do
  path4_required+=(--require "NarrativeDynamics.Tests.FitnessABMPath4.$path4_name")
done
python3 tools/audit_fitness_trust.py log "$path4_log" "${path4_required[@]}"
