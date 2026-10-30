# R2.M4a — Paired real V19 boundary → movement → content feedback

**Tracker:** [Issue #110](https://github.com/qigao/lean/issues/110).
**Parent:** [R2.M3 Draft PR #114](https://github.com/qigao/lean/pull/114).

This is the first **frozen, deterministic two-arm** experiment using
the original V19 world, story, private perception, cognition, social
memory and V24 source-selected graph contracts. It tests an integrated
network-change hypothesis, not an unconditional mathematical six-hop
theorem or a general statistical/empirical causal conclusion.

## Identical baseline, one explicit intervention

Both branches begin from an **identical** immutable initial world,
cognitive model, perception and social-memory policy. The entire
branch-specific story is reexecuted using `replay_situated_story`
and `resolve_situated_round`; no manually edited world snapshot is
accepted.

| Synchronous V19 round | Control | Intervention |
| --- | --- | --- |
| 1 | Alice WAIT at closed `records-open` | Alice real OPEN_PASSAGE from its directed source |
| 2 | Alice executes same MOVE; rejected `passage_closed` | Alice executes identical MOVE; accepted to `open` |
| 3 | Alice performs identical real TELL from `records` | Alice performs identical real TELL from `open` near Bob |
| Post-story | Exact V19 percept/cognitive/social simulation | Same V19 percept/cognitive/social model, independent SQLite memory |

The first action is the **only changed scheduled intervention**.
Subsequent MOVE and TELL `SituatedActionIntent` objects are identical
in both branches. Their world outcomes, perception/recalled testimony
and selected source edges may differ as a consequence of the differing
prior state.

## What is measured (and what is not)

- All prior, post-story and post-learning world/story/social/source
  revision hashes, actual per-round event hashes, and the real
  `MOVE` outcome.
- Source-derived **direct interaction** versus **realized transmission**
  versus **trust-derived relationship** are queried separately through
  `source_relation_edges` and `project_situated_network_snapshot`.
- Bob may detect an attenuated sound when the passage remains closed;
  after the intervention allows MOVE, both Bob and Carol can receive
  exact TELL percepts. Only eligible Bob cognitive admission creates a
  TESTIMONY and an ACTIVE claim; an unverified testimony does not
  automatically increase trust or add a social relationship edge.
- Cross-branch comparison uses the existing
  `compare_network_layers` and must remain
  `DIVERGENT_STORY_BRANCH`, `causal_attribution=UNATTRIBUTED`.
  This is a **controlled model contrast** conditional on the exact
  model and intervention, not historical proof or a population causal
  estimate.
- Event **IDs may coincide across divergent branches** even when the
  event's place/outcome/source event hash differs. Never treat a bare
  `event_id` as a globally unique historical occurrence: version it
  together with its story/source revision identity.

The report intentionally contains **no private message content**,
symbol payload, independent document store, fabricated observed
transmissions or new graph topology owner.

## Acceptance gate

- [ ] Real deterministic model baseline is common to both arms.
- [ ] Control/Intervention differ only in first-round OPEN vs WAIT;
  subsequent MOVE/TELL intents remain identical and source-owned.
- [ ] Same-round and cross-round V19 action/event hashes replay exactly.
- [ ] Control MOVE is blocked while intervention MOVE succeeds.
- [ ] Control perception can be DETECTED without testimony;
  intervention creates exact observer percepts and eligible testimony.
- [ ] Real source snapshots show different direct-interaction and
  realized-transmission edges but no invented trust edge.
- [ ] Cross-branch lineage is **DIVERGENT_STORY_BRANCH** and is never
  falsely asserted to be one real history or a universal causal claim.
- [ ] Reviewed `conformance/network_coupled_evolution_v1.json`
  committed from actual native test execution; focused Python
  and independent Lean lanes compare exact UTF-8 bytes.
- [ ] All prior V19/V24/R1/R2.M0–M3, 35+11 theorem trust,
  BB/PathN and World Studio/Playwright gates full-green
  at the exact final HEAD; no merge/release on prior-head success.

**Not in M4a:** generalized BB/rewiring/evolving empirical model
comparison, statistical significance, graph ownership replacement,
shared-document merge and external actor authentication. Compare the
existing independent ABM families only in a separately qualified
M4b slice, without substituting them as the canonical V19 source.
