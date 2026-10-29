#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.elan/bin:$PATH"

python3 tools/audit_fitness_trust.py source \
  NarrativeDynamics/Core/MeshProjection.lean \
  NarrativeDynamics/Tests/MeshProjection.lean \
  NarrativeDynamics/Tests/MeshProjectionTrust.lean

required="$(mktemp)"
log="$(mktemp)"
trap 'rm -f "$required" "$log"' EXIT

# One authoritative audit policy for both the old mesh foundations and V24.
# Any new public projection theorem without a corresponding Lean report is RED.
python3 tools/check_mesh_proof_coverage.py \
  --reports NarrativeDynamics/Tests/MeshProjectionTrust.lean \
  NarrativeDynamics/Core/MeshProjection.lean > "$required"
mapfile -t theorems < "$required"
args=()
for theorem in "${theorems[@]}"; do
  args+=(--require "$theorem")
done

lake env lean NarrativeDynamics/Tests/MeshProjectionTrust.lean 2>&1 | tee "$log"
python3 tools/audit_fitness_trust.py log "$log" "${args[@]}"
