#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.elan/bin:$PATH"

python3 tools/audit_fitness_trust.py source \
  NarrativeDynamics/Core/MeshProjection.lean \
  NarrativeDynamics/Tests/MeshProjection.lean \
  NarrativeDynamics/Tests/MeshProjectionTrust.lean

log="$(mktemp)"
trap 'rm -f "$log"' EXIT
lake env lean NarrativeDynamics/Tests/MeshProjectionTrust.lean 2>&1 | tee "$log"
python3 tools/audit_fitness_trust.py log "$log" \
  --require NarrativeDynamics.projectedMesh_source \
  --require NarrativeDynamics.projectedMesh_walk_lifts \
  --require NarrativeDynamics.projectedMesh_reach_lifts \
  --require NarrativeDynamics.projectedMesh_universal_society \
  --require NarrativeDynamics.projectedMesh_bridge_mono \
  --require NarrativeDynamics.projectedMesh_reach_mono_bridges \
  --require NarrativeDynamics.projectedMesh_selected_only \
  --require NarrativeDynamics.membershipQualifiedReach_source \
  --require NarrativeDynamics.membershipQualifiedReach_outsider \
  --require NarrativeDynamics.projectedMesh_no_unsourced_edge \
  --require NarrativeDynamics.projectedMesh_global_six_hop_bound
