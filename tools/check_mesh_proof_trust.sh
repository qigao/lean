#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PATH="$HOME/.elan/bin:$PATH"

python3 tools/audit_fitness_trust.py source \
  NarrativeDynamics/Core/SocialMesh.lean \
  NarrativeDynamics/Core/SmallWorld.lean \
  NarrativeDynamics/Core/WattsStrogatz.lean \
  NarrativeDynamics/Tests/SocialMesh.lean \
  NarrativeDynamics/Tests/SmallWorld.lean \
  NarrativeDynamics/Tests/WattsStrogatz.lean \
  NarrativeDynamics/Tests/MeshProofTrust.lean

log="$(mktemp)"
trap 'rm -f "$log"' EXIT

lake env lean NarrativeDynamics/Tests/MeshProofTrust.lean 2>&1 | tee "$log"
python3 tools/audit_fitness_trust.py log "$log" \
  --require NarrativeDynamics.restricted_walk_lifts \
  --require NarrativeDynamics.closed_no_cross_reach \
  --require NarrativeDynamics.six_degrees_of_two_three_bridge \
  --require NarrativeDynamics.global_six_hop_bound \
  --require NarrativeDynamics.smallWorld_of_gateway_cover \
  --require NarrativeDynamics.wattsStrogatzRing_globalHopBound \
  --require NarrativeDynamics.meshDiameter_mono_edges \
  --require NarrativeDynamics.averageShortestPathLength_mono_edges
