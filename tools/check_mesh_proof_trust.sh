#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.elan/bin:$PATH"
modules=(
  NarrativeDynamics/Core/SocialMesh.lean
  NarrativeDynamics/Core/SmallWorld.lean
  NarrativeDynamics/Core/WattsStrogatz.lean
)
report="NarrativeDynamics/Tests/MeshProofTrust.lean"
python3 tools/audit_fitness_trust.py source \
  "${modules[@]}" \
  NarrativeDynamics/Tests/SocialMesh.lean \
  NarrativeDynamics/Tests/SmallWorld.lean \
  NarrativeDynamics/Tests/WattsStrogatz.lean \
  NarrativeDynamics/Tests/MeshProofTrust.lean

required="$(mktemp)"
log="$(mktemp)"
trap 'rm -f "$required" "$log"' EXIT
python3 tools/check_mesh_proof_coverage.py \
  --reports "$report" "${modules[@]}" > "$required"
mapfile -t theorems < "$required"
args=()
for theorem in "${theorems[@]}"; do
  args+=(--require "$theorem")
done
lake env lean NarrativeDynamics/Tests/MeshProofTrust.lean 2>&1 | tee "$log"
python3 tools/audit_fitness_trust.py log "$log" "${args[@]}"
