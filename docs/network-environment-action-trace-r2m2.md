# R2.M2: source-backed V19 environment actions and interaction effects

[Issue #110](https://github.com/qigao/lean/issues/110), parent
[R2.M1 Draft PR #112](https://github.com/qigao/lean/pull/112).

## Existing ownership and real action semantics

Real physical/environmental state remains in `SituatedWorldState` and the
authoritative `SituatedStory`: **MOVE** changes the Agent body place;
**TAKE** transfers a world object to one holder; **DROP** returns it to
the actor's *current* physical location. Each accepted action is a
`SituatedWorldEvent`, with an exact world-state parent/hash and
per-Agent `SituatedObservation` channels. A rejected move or object
action may create an unsuccessful world event but **not** a world effect.

The R2.M2 `observe_environment_action_trace(before, after)` observer
only accepts source-validated V19/V24 `NetworkLayerView` inputs whose
stories have a strict, real `STORY_EXTENSION` with 1–12 new rounds
and at most 128 new events, 24 Agents and 64 declared world objects.
It **re-executes each recorded round** with the original
`resolve_situated_round` (the existing single owner) and requires
byte-equivalent typed events and next state before associating any
successful MOVE / TAKE / DROP with a changed Agent location or object
owner. Counterfactual initial conditions, forged destinations,
fabricated object ownership and disconnected source histories are
rejected, not treated as Agent actions.

## Network and content distinctions

For each real round the report records exact prior/next world hashes,
original event ID and hash, state effect from/to, failed actions and
aggregate observation channels. It reuses the original
`can_situated_agents_interact` predicate to compare **directed**
interaction opportunities before/after the actual movement. At the
final source boundary it reuses R2.M0's V19/V24 relation-scoped
source/projection and path metrics. A MOVE can change interaction
opportunity without a TELL; a TAKE/DROP can change who has a physical
object without distributing its private details. A subsequent real
TELL can produce **historically realized transmission** to nearby
observers, without silently raising social trust or changing Society
membership or physical passages.

**Important:** this demonstration uses explicit real V19 Story actions
and valid *checkpointed* cognition/social source projections. It does
not claim those actions were autonomously selected by the cognitive
policy, or that the social model has completed new evidence learning.
R2.M3 will handle genuine multi-Agent content updates/recalls/claims.

## Privacy and deterministic conformance

Only source event IDs/hashes, Agent/object identifiers, place/holder
metadata, action kinds, observation-channel *counts* and V19/V24
derived network statistics are exported. Private TELL message bodies,
INSPECT evidence, actual memory contents and per-Agent observation
payloads are excluded. The immutable `EnvironmentActionTrace` stores
a canonical JSON string; caller dictionary mutations cannot forge
source events, change `UNATTRIBUTED` causality or alter its digest.
The result proves **reproducibility given the V19 source**, not
external actor identity or historical physical fact.

## Acceptance / CI

- [ ] Real TAKE memo in records -> holder Alice; MOVE Alice from records
  to open -> directed interaction with Bob; DROP memo in open -> Bob
  can observe a local VISUAL action; only the **subsequent** actual
  TELL creates a source-backed transmission edge
- [ ] A failed TAKE (object not co-located) and closed-passage MOVE
  do not mutate Agent or object state
- [ ] Forged successful TAKE effect, forged movement destination and
  alternate initial conditions are rejected before attribution
- [ ] Report source/hash immutability, private TELL redaction and
  distinct relation-selected graph edges
- [x] Reviewed versioned two-case `conformance/network_environment_action_v1.json`
  generated from real Python fixtures, compared **byte-for-byte** in
  focused Python and independent Lean-lane acceptance
- [ ] Preserve original full Python, Lean/35+11/BB/PathN, Studio/Playwright,
  cache + singleflight; same exact-HEAD full acceptance

This is a read-only effect/evidence layer. No new mutable network
owner, no general six-hop theorem, no installers or consumer CI project.

## Reviewed native source vector — 2026-10-30

Initial focused [CI #41013905544](https://github.com/qigao/lean/actions/runs/41013905544)
had **six real behavior regressions passing** and exactly one intentional
missing-vector failure. Its canonical generated source text was inspected
for schema, case order, four actual sequential environmental rounds,
the rejected-action counterexample, UNATTRIBUTED scope and absence of the
private TELL body, then committed exactly as
`conformance/network_environment_action_v1.json`. The one-time failed
candidate transfer has been removed. Python and Lean must now compare
the checked-in full UTF-8 bytes without silently rewriting the file.

The acceptance intentionally distinguishes a MOVE changing directed
interaction opportunity from a later *real* TELL creating
`alice→bob` realized transmission, while neither change fabricates a
trust/relationship edge. No external history verification is claimed.
