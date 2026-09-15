#!/usr/bin/env bash
set -euo pipefail

pathn_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$pathn_root"

pathn_time="$(type -P time)"
consensus_log="$(mktemp)"
pathn_log="$(mktemp)"
trap 'rm -f "$consensus_log" "$pathn_log"' EXIT

python3 tools/audit_fitness_trust.py source \
  NarrativeDynamics/Core/FiniteConsensus.lean \
  NarrativeDynamics/Tests/FiniteConsensus.lean \
  NarrativeDynamics/Core/FitnessABMPathN.lean \
  NarrativeDynamics/Tests/FitnessABMPathN.lean

for pathn_module in \
    NarrativeDynamics.Core.FiniteConsensus \
    NarrativeDynamics.Core.FitnessABMPathN; do
  "$pathn_time" -f "$pathn_module elapsed=%e s peak_rss=%M KiB" \
    timeout --kill-after=10s 240s lake build "$pathn_module"
done

# The PathN compatibility consumer imports the existing fixed-size modules.
# Build them as prerequisites only; their independent replay/convergence gate
# remains tools/check_fitness_abm_path4.sh and is not replaced here.
for compat_module in \
    NarrativeDynamics.Core.FitnessABMPath4 \
    NarrativeDynamics.Core.FitnessABMPath4Convergence \
    NarrativeDynamics.Core.FitnessABMPath5; do
  timeout --kill-after=10s 240s lake build "$compat_module"
done

"$pathn_time" -f 'FiniteConsensus tests elapsed=%e s peak_rss=%M KiB' \
  timeout --kill-after=10s 240s \
  lake env lean -DmaxErrors=1 NarrativeDynamics/Tests/FiniteConsensus.lean \
  2>&1 | tee "$consensus_log"

"$pathn_time" -f 'FitnessABMPathN tests elapsed=%e s peak_rss=%M KiB' \
  timeout --kill-after=10s 240s \
  lake env lean -DmaxErrors=1 NarrativeDynamics/Tests/FitnessABMPathN.lean \
  2>&1 | tee "$pathn_log"

python3 tools/audit_fitness_trust.py log "$consensus_log" \
  --require NarrativeDynamics.FiniteConsensus.coordRange_apply_le_of_commonColumn \
  --require NarrativeDynamics.FiniteConsensus.block_contraction_tendsto

python3 tools/audit_fitness_trust.py log "$pathn_log" \
  --require NarrativeDynamics.FitnessABMPathN.propagate_independent_exposures \
  --require NarrativeDynamics.FitnessABMPathN.propagate_eq_kernel \
  --require NarrativeDynamics.FitnessABMPathN.allBroadcast_iterate \
  --require NarrativeDynamics.FitnessABMPathN.path_stationary_weights \
  --require NarrativeDynamics.FitnessABMPathN.mean_step \
  --require NarrativeDynamics.FitnessABMPathN.path_block_common_mass \
  --require NarrativeDynamics.FitnessABMPathN.trajectory_tendsto
