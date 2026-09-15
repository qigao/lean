#!/usr/bin/env bash
set -euo pipefail

path4_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$path4_root"

path4_time="$(type -P time)"
idle_tail_log="$(mktemp)"
fitness_idle_tail_log="$(mktemp)"
path4_log="$(mktemp)"
path5_vectors="$(mktemp)"
trap 'rm -f "$idle_tail_log" "$fitness_idle_tail_log" "$path4_log" "$path5_vectors"' EXIT

python3 tools/audit_fitness_trust.py source \
  NarrativeDynamics/Core/IdleTail.lean \
  NarrativeDynamics/Tests/IdleTail.lean \
  NarrativeDynamics/Core/FitnessABMIdleTail.lean \
  NarrativeDynamics/Core/FitnessABMReplay.lean \
  NarrativeDynamics/Tests/FitnessABMIdleTail.lean \
  NarrativeDynamics/Core/FitnessABMPath4.lean \
  NarrativeDynamics/Core/FitnessABMPath4Convergence.lean \
  NarrativeDynamics/Tests/FitnessABMPath4.lean \
  NarrativeDynamics/Core/FitnessABMPath5.lean \
  NarrativeDynamics/Conformance/FitnessABMPath5Vectors.lean \
  NarrativeDynamics/Tests/FitnessABMPath5.lean

for path4_module in \
    NarrativeDynamics.Core.IdleTail \
    NarrativeDynamics.Core.FitnessABMIdleTail \
    NarrativeDynamics.Core.FitnessABMReplay \
    NarrativeDynamics.Core.FitnessABMPath4 \
    NarrativeDynamics.Core.FitnessABMPath4Convergence \
    NarrativeDynamics.Core.FitnessABMPath5 \
    NarrativeDynamics.Conformance.FitnessABMPath5Vectors; do
  "$path4_time" -f "$path4_module elapsed=%e s peak_rss=%M KiB" \
    timeout --kill-after=10s 240s lake build "$path4_module"
done

"$path4_time" -f 'IdleTail tests elapsed=%e s peak_rss=%M KiB' \
  timeout --kill-after=10s 240s \
  lake env lean -DmaxErrors=1 NarrativeDynamics/Tests/IdleTail.lean \
  2>&1 | tee "$idle_tail_log"

python3 tools/audit_fitness_trust.py log "$idle_tail_log" \
  --require NarrativeDynamics.IdleTailModel.trajectory_add \
  --require NarrativeDynamics.IdleTailModel.observedTrajectory_succ

"$path4_time" -f 'FitnessABMIdleTail tests elapsed=%e s peak_rss=%M KiB' \
  timeout --kill-after=10s 240s \
  lake env lean -DmaxErrors=1 NarrativeDynamics/Tests/FitnessABMIdleTail.lean \
  2>&1 | tee "$fitness_idle_tail_log"

python3 tools/audit_fitness_trust.py log "$fitness_idle_tail_log" \
  --require NarrativeDynamics.FitnessABM.runIdleTrajectory_eq \
  --require NarrativeDynamics.FitnessABM.runInputs_replicate_idle \
  --require NarrativeDynamics.FitnessABM.runInputs_append_idle \
  --require NarrativeDynamics.FitnessABM.replay_append_idle

if git grep -n 'private theorem run_idle' -- NarrativeDynamics/Tests/FitnessABMPath4.lean; then
  echo 'legacy test-local run_idle still exists' >&2
  exit 1
fi
git grep -n 'theorem tailState_idleTail' -- NarrativeDynamics/Tests/FitnessABMPath4.lean
git grep -n 'theorem tailBelief_idleTail' -- NarrativeDynamics/Tests/FitnessABMPath4.lean
git grep -n 'replay_append_idle' -- NarrativeDynamics/Tests/FitnessABMPath4.lean

"$path4_time" -f 'FitnessABMPath5 generated replay elapsed=%e s peak_rss=%M KiB' \
  timeout --kill-after=10s 240s \
  lake env lean --run NarrativeDynamics/Conformance/FitnessABMPath5Vectors.lean \
  > "$path5_vectors"
cat "$path5_vectors"
cmp conformance/bb_path5_runtime_v1.json "$path5_vectors"

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
for path4_name in replay_baseline raw_tail_bridge tailState_idleTail activation_bbii activation_bibi \
    activation_iibb bbii_tendsto bibi_tendsto iibb_tendsto equal_control separation_limit; do
  path4_required+=(--require "NarrativeDynamics.Tests.FitnessABMPath4.$path4_name")
done
python3 tools/audit_fitness_trust.py log "$path4_log" "${path4_required[@]}"
