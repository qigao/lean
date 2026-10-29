#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.elan/bin:$PATH"

generated="$(mktemp --suffix=.lean)"
trap 'rm -f "$generated"' EXIT

# Build fresh checked input from the actual V19 runtime fixture, never a
# stale hand-maintained graph or an unverified external JSON attachment.
python3 tools/render_mesh_evidence_lean.py > "$generated"

/usr/bin/time -f 'V24 finite Lean elapsed=%e s peak_rss=%M KiB' \
  timeout --kill-after=10s 240s \
  lake env lean -DmaxErrors=1 "$generated"
