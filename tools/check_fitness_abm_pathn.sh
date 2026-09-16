#!/usr/bin/env bash
set -euo pipefail

pathn_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$pathn_root"

pathn_time="$(type -P time)"
consensus_log="$(mktemp)"
pathn_log="$(mktemp)"
params_log="$(mktemp)"
parameter_convergence_log="$(mktemp)"
exposure_log="$(mktemp)"
trap 'rm -f "$consensus_log" "$pathn_log" "$params_log" "$parameter_convergence_log" "$exposure_log"' EXIT

python3 tools/audit_fitness_trust.py source \
  NarrativeDynamics/Core/FiniteConsensus.lean \
  NarrativeDynamics/Tests/FiniteConsensus.lean \
  NarrativeDynamics/Core/FitnessABMPathN.lean \
  NarrativeDynamics/Tests/FitnessABMPathN.lean \
  NarrativeDynamics/Tests/FitnessABMPathNParameters.lean \
  NarrativeDynamics/Core/FitnessABMPathNParameterConvergence.lean \
  NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean \
  NarrativeDynamics/Core/FitnessABMPathNExposure.lean \
  NarrativeDynamics/Tests/FitnessABMPathNExposure.lean

for pathn_module in \
    NarrativeDynamics.Core.FiniteConsensus \
    NarrativeDynamics.Core.FitnessABMPathN \
    NarrativeDynamics.Core.FitnessABMPathNParameters \
    NarrativeDynamics.Core.FitnessABMPathNParameterConvergence \
    NarrativeDynamics.Core.FitnessABMPathNExposure; do
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

"$pathn_time" -f 'FitnessABMPathNParameters tests elapsed=%e s peak_rss=%M KiB' \
  timeout --kill-after=10s 240s \
  lake env lean -DmaxErrors=1 NarrativeDynamics/Tests/FitnessABMPathNParameters.lean \
  2>&1 | tee "$params_log"

"$pathn_time" -f 'FitnessABMPathNParameterConvergence tests elapsed=%e s peak_rss=%M KiB' \
  timeout --kill-after=10s 240s \
  lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean \
  2>&1 | tee "$parameter_convergence_log"

"$pathn_time" -f 'FitnessABMPathNExposure tests elapsed=%e s peak_rss=%M KiB' \
  timeout --kill-after=10s 240s \
  lake env lean -DmaxErrors=1 NarrativeDynamics/Tests/FitnessABMPathNExposure.lean \
  2>&1 | tee "$exposure_log"

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

python3 tools/audit_fitness_trust.py log "$parameter_convergence_log" \
  --require NarrativeDynamics.FitnessABMPathNParameterConvergence.path_stationary_weights \
  --require NarrativeDynamics.FitnessABMPathNParameterConvergence.path_block_common_mass \
  --require NarrativeDynamics.FitnessABMPathNParameterConvergence.trajectory_tendsto \
  --require zero_response_path2_iterate \
  --require one_response_path2_even

python3 tools/audit_fitness_trust.py log "$exposure_log" \
  --require NarrativeDynamics.FitnessABMPathNExposure.constant_beliefStep \
  --require NarrativeDynamics.FitnessABMPathNExposure.exposure_mono \
  --require NarrativeDynamics.FitnessABMPathNExposure.beliefs_bounded_step \
  --require exposure_history_changes_next_belief
