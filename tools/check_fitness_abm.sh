#!/usr/bin/env bash
set -euo pipefail
bb_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$bb_repo_root"
bb_audit_log="$(mktemp)"
bb_vectors="$(mktemp)"
trap 'rm -f "$bb_audit_log" "$bb_vectors"' EXIT

python3 tools/audit_fitness_trust.py source \
  NarrativeDynamics/Core/Fitness*.lean \
  NarrativeDynamics/Tests/Fitness*.lean \
  NarrativeDynamics/Core/NetworkPropagation.lean \
  NarrativeDynamics/Tests/NetworkPropagation.lean \
  NarrativeDynamics/Conformance/FitnessABMVectors.lean

/usr/bin/time -f 'FitnessABMVectors build elapsed=%e s peak_rss=%M KiB' \
  timeout --kill-after=10s 240s \
  lake build NarrativeDynamics.Conformance.FitnessABMVectors

for bb_suite in NetworkPropagation FitnessABM FitnessABMReplay FitnessABMDistribution; do
  /usr/bin/time -f "$bb_suite elapsed=%e s peak_rss=%M KiB" \
    timeout --kill-after=10s 240s \
    lake env lean -DmaxErrors=1 "NarrativeDynamics/Tests/$bb_suite.lean" \
    2>&1 | tee -a "$bb_audit_log"
done

proof_names=(
  NetworkPropagation.propagate_valid
  NetworkPropagation.transmission_iff
  NetworkPropagation.nextAgent_no_incoming
  NetworkPropagation.exposures_mono
  NetworkPropagation.nextAgent_locality
  FitnessABM.extendPopulation_valid
  FitnessABM.grow_projection
  FitnessABM.grow_old_state
  FitnessABM.grow_new_state
  FitnessABM.grow_old_profile
  FitnessABM.advance_projection
  FitnessABM.advance_profiles
  FitnessABM.runTyped_counts
  FitnessABM.runTyped_probability_pos
  FitnessABM.grow_order_irrelevant
  FitnessABM.parseAgent_sound
  FitnessABM.parseAgent_complete
  FitnessABM.parseAgents_sound
  FitnessABM.parseAgents_complete
  FitnessABM.checkedBirth_spec
  FitnessABM.replay_success_iff_valid
  FitnessABM.replay_projection
  FitnessABM.replay_counts
  FitnessABM.replay_probability_pos
  FitnessABM.runInputs_append_error
  FitnessABM.jointProbability_eq_runTyped
  FitnessABM.jointFinal_projection
  FitnessABM.jointProbability_sum_one
  FitnessABM.eventProbability_true
  FitnessABM.eventProbability_false
  FitnessABM.eventProbability_nonneg
  FitnessABM.eventProbability_le_one
  FitnessABM.DistributionFixtures.newborn_mass_unit
  FitnessABM.DistributionFixtures.newborn_mass_weighted
  FitnessABM.DistributionFixtures.newborn_mass_after_idle
)
bb_required=()
for bb_name in "${proof_names[@]}"; do
  bb_required+=(--require "NarrativeDynamics.$bb_name")
done
python3 tools/audit_fitness_trust.py log "$bb_audit_log" "${bb_required[@]}"

/usr/bin/time -f 'FitnessABMVectors elapsed=%e s peak_rss=%M KiB' \
  timeout --kill-after=10s 240s \
  lake env lean --run NarrativeDynamics/Conformance/FitnessABMVectors.lean > "$bb_vectors"
cmp conformance/bb_abm_v1.json "$bb_vectors"
