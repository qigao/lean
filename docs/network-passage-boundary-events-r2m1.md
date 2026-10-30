# R2.M1 — Actual V19 physical passage boundary events

Tracked by [Issue #110](https://github.com/qigao/lean/issues/110).
Stacked on [R2.M0 PR #111](https://github.com/qigao/lean/pull/111).

## Source owner and event semantics

Passage state belongs to the **existing** immutable
`SituatedWorldState.passages`. `SituatedActionKind.OPEN_PASSAGE` and
`SituatedActionKind.CLOSE_PASSAGE` are explicit, target-bearing V19 actions
resolved by **the original** `resolve_situated_round` engine and stored as
real `SituatedWorldEvent` records in `SituatedStory`. No second physical
map/graph is built; the original world model's passage roster and
`SituatedPerceptionModel` remain unchanged.

Admission and precedence:

- Only an Agent currently at the **source** of a declared directed Passage
  may issue a boundary change; unknown passage, wrong source or a no-op
  OPEN/CLOSE are explicit unsuccessful events and cannot modify the world.
- If multiple Agents at the source attempt to modify the same passage in one
  synchronous round, **all requests fail** as `passage_conflict`. No
  last-writer-wins arbitration; order of intent submission cannot grant a
  hidden permission.
- **Snapshot-isolated round:** all actions, including MOVE, use the
  *prior-round* passage state. A successful opening becomes effective for
  subsequent rounds; an Agent cannot MOVE through a gate opened in the same
  round. Similarly a simultaneous closing cannot retroactively cancel a
  valid same-round MOVE.
- The actor has a self observation. Other observers co-located at the
  actor's prior position may observe a visual boundary event via the
  established local V19 observation projection. Remote perception follows
  only explicitly configured visual/audio signal profiles and **prior**
  environment reach. Opening a passage does not magically transmit an
  observation through it on the same round.

The `SituatedRoundResult` contract now requires every change to
`next_state.passages` to be backed by a **successful explicit boundary
event**. A forged next state with changed passages and no matching accepted
event fails closed. This check does not assert external actor authorization,
physical-world authenticity or a universal graph bound.

## Relationship, interaction and content isolation

`can_situated_agents_interact` and `derive_situated_perception_reach`
read the next immutable V19 world state and the existing
`SituatedEdgeActivation.PASSAGE_OPEN/CLOSED` edges. Only actually declared
layers may change; e.g. a closed door can still permit attenuated sound.
Real `TELL` and observer-specific perceptions remain separate historical
facts. A boundary event **never** fabricates a `TELL`, a social-memory
claim, a trust change, a transmission or a reverse directed relationship.

## Validation

- Actual OPEN/CLOSE, wrong-side/unknown/no-op and multi-Agent conflict
  cases in the V19 synchronous resolver.
- Prior-state snapshot isolation of MOVE when OPEN/CLOSE occurs in the same
  round; deterministic story replay with matching event and observation
  hashes, parent linkage and canonical outcomes.
- Directed prior/next visibility, auditory and interaction reach; no
  unintended reverse interaction.
- Real V19 model/story/cognition/social-memory and V24 projections:
  physical boundary and relevant direct-interaction edge change while
  source-backed relationship/trust and realized transmission do not.
- Actual source-derived reviewed input/output JSON, full focused Python,
  original Lean/BB/PathN, World Studio/Playwright exact-HEAD CI.
- **No** new six-degree theorem, second network authority, installer,
  unrelated consumer project or fallback engine.

This implements the **physical** boundary slice only. Society membership,
institutional policies, mutable environment models and concurrent shared
content merge remain separate follow-ons under Issue #110.
