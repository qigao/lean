#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.elan/bin:$PATH"

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
generated="$tmpdir/authoritative.lean"

# Always regenerate from the actual V19 snapshot, then enforce the immutable
# byte-level corpus commitment. Never update the pin as part of this CI run.
python3 -m tools.render_mesh_evidence_lean > "$generated"
expected="$(tr -d '[:space:]' < tests/fixtures/mesh_evidence_lean.sha256)"
actual="$(sha256sum "$generated" | cut -d' ' -f1)"
if [[ ! "$expected" =~ ^[0-9a-f]{64}$ || "$actual" != "$expected" ]]; then
  printf 'V24 finite Lean corpus mismatch: pinned=%s actual=%s\n' "$expected" "$actual" >&2
  exit 1
fi
printf 'V24 finite Lean corpus SHA-256=%s\n' "$actual"

/usr/bin/time -f 'V24 finite Lean elapsed=%e s peak_rss=%M KiB' \
  timeout --kill-after=10s 240s \
  lake env lean -DmaxErrors=1 "$generated"

# The same Lean kernel must reject two altered topologies while keeping the
# previously validated POSITIVE projected-edge obligation untouched.
for kind in source bridge; do
  mutated="$tmpdir/$kind.lean"
  log="$tmpdir/$kind.log"
  python3 -m tools.mutate_mesh_evidence_lean --kind "$kind" > "$mutated"
  if timeout --kill-after=10s 120s lake env lean -DmaxErrors=1 "$mutated" > "$log" 2>&1; then
    printf 'FAIL: %s-mutated finite Lean obligation unexpectedly passed\n' "$kind" >&2
    exit 1
  else
    status=$?
    if [[ "$status" == 124 || "$status" == 137 ]] || ! grep -q 'error:' "$log"; then
      printf 'FAIL: %s mutation did not produce a real Lean diagnostic (exit=%s)\n' "$kind" "$status" >&2
      cat "$log" >&2
      exit 1
    fi
  fi
  printf 'Expected Lean RED confirmed for %s premise mutation\n' "$kind"
done
