#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.elan/bin:$PATH"

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

# Independent Lean-lane check: exact authoritative V19 records must be
# identical to the human-reviewable versioned input vectors.
python3 -m tools.render_mesh_proof_vectors > "$tmpdir/mesh_runtime_vectors_v1.json"
if ! cmp -s conformance/mesh_runtime_vectors_v1.json "$tmpdir/mesh_runtime_vectors_v1.json"; then
  echo 'FAIL: versioned runtime source vectors disagree with exact-head export' >&2
  diff -u conformance/mesh_runtime_vectors_v1.json "$tmpdir/mesh_runtime_vectors_v1.json" || true
  exit 1
fi
# Independent Lean-lane input check: runtime events and revision hashes must
# agree exactly with the reviewed authoritative Python replay corpus.
python3 -m tools.render_topology_revision_vectors > "$tmpdir/topology_revision_v1.json"
if ! cmp -s conformance/topology_revision_v1.json "$tmpdir/topology_revision_v1.json"; then
  echo 'FAIL: R1 topology replay vector mismatch' >&2
  diff -u conformance/topology_revision_v1.json "$tmpdir/topology_revision_v1.json" || true
  exit 1
fi
python3 -m tools.render_social_topology_checkpoint_vectors > "$tmpdir/social_topology_checkpoint_v1.json"
if ! cmp -s conformance/social_topology_checkpoint_v1.json "$tmpdir/social_topology_checkpoint_v1.json"; then
  echo 'FAIL: V19 authoritative social topology checkpoint vectors changed' >&2
  diff -u conformance/social_topology_checkpoint_v1.json "$tmpdir/social_topology_checkpoint_v1.json" || true
  exit 1
fi
# R2.M0: independently check the reviewed V19 three-driver observation
# corpus before compiling the already-maintained finite Lean contracts.
python3 -m tools.render_network_evolution_vectors > "$tmpdir/network_evolution_observation_v1.json"
if ! cmp -s conformance/network_evolution_observation_v1.json "$tmpdir/network_evolution_observation_v1.json"; then
  echo 'FAIL: R2 V19 boundary/environment/content observed vectors changed' >&2
  diff -u conformance/network_evolution_observation_v1.json "$tmpdir/network_evolution_observation_v1.json" || true
  exit 1
fi
# R2.M1: independently re-run real V19 boundary actions, Story and V24
# projection against the versioned reviewed source-derived corpus.
python3 -m tools.render_network_boundary_events > "$tmpdir/network_boundary_events_v1.json"
if ! cmp -s conformance/network_boundary_events_v1.json "$tmpdir/network_boundary_events_v1.json"; then
  echo 'FAIL: R2.M1 source-derived passage-boundary event vectors changed' >&2
  diff -u conformance/network_boundary_events_v1.json "$tmpdir/network_boundary_events_v1.json" || true
  exit 1
fi
# R2.M2: independently regenerate real V19 Agent MOVE/TAKE/DROP and
# subsequent TELL action-event effects from the authoritative story.
python3 -m tools.render_network_environment_vectors > "$tmpdir/network_environment_action_v1.json"
if ! cmp -s conformance/network_environment_action_v1.json "$tmpdir/network_environment_action_v1.json"; then
  echo 'FAIL: R2.M2 source-derived environment action vectors changed' >&2
  diff -u conformance/network_environment_action_v1.json "$tmpdir/network_environment_action_v1.json" || true
  exit 1
fi
# R2.M3: independently regenerate exact-percept V19 social testimony,
# verification, claim status and directed trust evidence against reviewed bytes.
python3 -m tools.render_agent_content_evidence_vectors > "$tmpdir/agent_content_evidence_v1.json"
if ! cmp -s conformance/agent_content_evidence_v1.json "$tmpdir/agent_content_evidence_v1.json"; then
  echo 'FAIL: R2.M3 exact-percept V19 content vectors changed' >&2
  diff -u conformance/agent_content_evidence_v1.json "$tmpdir/agent_content_evidence_v1.json" || true
  exit 1
fi
# R2.M4a: deterministic paired real V19 boundary/environment/content
# intervention, original cognitive/social and V24 selected source relations.
python3 -m tools.render_network_coupled_evolution_vectors > "$tmpdir/network_coupled_evolution_v1.json"
if ! cmp -s conformance/network_coupled_evolution_v1.json "$tmpdir/network_coupled_evolution_v1.json"; then
  echo 'FAIL: R2.M4a source-derived coupled Agent/network contrast changed' >&2
  diff -u conformance/network_coupled_evolution_v1.json "$tmpdir/network_coupled_evolution_v1.json" || true
  exit 1
fi
# R2.M4b: separate verified native BB, directed rewiring, evolving
# and V19 control reference; no cross-family graph normalization.
python3 -m tools.render_model_family_comparison_vectors > "$tmpdir/network_model_family_comparison_v1.json"
if ! cmp -s conformance/network_model_family_comparison_v1.json "$tmpdir/network_model_family_comparison_v1.json"; then
  echo 'FAIL: R2.M4b model-native reviewed source vector changed' >&2
  diff -u conformance/network_model_family_comparison_v1.json "$tmpdir/network_model_family_comparison_v1.json" || true
  exit 1
fi
# R2.M4c0: do not derive universal edge or trust units.
python3 -m tools.render_native_metric_protocol_vectors > "$tmpdir/native_metric_protocol_v1.json"
if ! cmp -s conformance/native_metric_protocol_v1.json "$tmpdir/native_metric_protocol_v1.json"; then
  echo 'FAIL: R2.M4c0 native metric denominator/eligibility source changed' >&2
  diff -u conformance/native_metric_protocol_v1.json "$tmpdir/native_metric_protocol_v1.json" || true
  exit 1
fi
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

# Independently kernel-check concrete MeshWalk paths, positive nonempty
# interaction/transmission, overlapping memberships, outsider zero-hop,
# and all premises required for the existing six-hop gateway theorem.
paths="$tmpdir/concrete-paths-gateways.lean"
python3 -m tools.render_mesh_reach_lean > "$paths"
/usr/bin/time -f 'V24 concrete ReachWithin/Gateway elapsed=%e s peak_rss=%M KiB' \
  timeout --kill-after=10s 240s \
  lake env lean -DmaxErrors=1 "$paths"

# Require exact byte identity with the reviewed finite conformance corpus.
# An intentional generator change must be separately reviewed and re-pinned.
extended_sha="$(sha256sum "$paths" | cut -d' ' -f1)"
printf 'V24 extended Lean corpus SHA-256=%s\n' "$extended_sha"

# The pin is required and never silently regenerated.
pin=tests/fixtures/mesh_reach_lean.sha256
if [[ ! -f "$pin" ]]; then
  printf 'FAIL: missing reviewed extended Lean corpus pin; actual SHA-256=%s\n' "$extended_sha" >&2
  exit 1
fi
pinned="$(tr -d '[:space:]' < "$pin")"
if [[ ! "$pinned" =~ ^[0-9a-f]{64}$ || "$pinned" != "$extended_sha" ]]; then
  printf 'FAIL: extended Lean corpus digest differs: pinned=%s actual=%s\n' "$pinned" "$extended_sha" >&2
  exit 1
fi

# Mutations must be rejected by the real Lean tactic, not by Python alone.
for kind in stale-path identity; do
  mutated="$tmpdir/reach-$kind.lean"
  log="$tmpdir/reach-$kind.log"
  python3 -m tools.mutate_mesh_reach_lean --kind "$kind" > "$mutated"
  if timeout --kill-after=10s 120s lake env lean -DmaxErrors=1 "$mutated" > "$log" 2>&1; then
    printf 'FAIL: stale %s claim unexpectedly proved in Lean\n' "$kind" >&2
    exit 1
  else
    status=$?
    expected_failure='Tactic `decide` proved that the proposition'
    if [[ "$kind" == "stale-path" ]]; then
      expected_failure='`decide_cbv` failed: the proposition evaluates to `false`'
    fi
    if [[ "$status" == 124 || "$status" == 137 ]] ||
       ! grep -q 'error:' "$log" ||
       ! grep -Fq "$expected_failure" "$log" ||
       grep -Eq 'unknown identifier|maximum recursion depth|unexpected token' "$log"; then
      printf 'FAIL: %s mutation lacked a valid Lean counterexample (exit=%s)\n' "$kind" "$status" >&2
      cat "$log" >&2
      exit 1
    fi
  fi
  grep -m1 'error:' "$log"
  printf 'Expected Lean RED confirmed for %s mutation\n' "$kind"
done

