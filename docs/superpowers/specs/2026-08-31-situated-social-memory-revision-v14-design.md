# Situated Social Memory Revision V14 Design

## Objective

Extend V13's private durable recall into an auditable social-memory lifecycle.
Agents consolidate repeated testimony, preserve explicit contradiction history,
learn directed source trust and relationship affinity when later evidence verifies a
claim, forget unresolved claims under declared bounds, and use learned trust to
temper later recalled testimony.

V14 remains a deterministic simulator rule. It adds no LLM, embedding service,
LSTM, Transformer, learned latent representation, or claim extraction from free text.

## Declarative social model

`SituatedSocialMemoryModel` binds one exact `SituatedMemoryCognitiveModel`, a set of
`SituatedClaimTopic` declarations, and one `SituatedSocialMemoryPolicy`.

A topic declares the finite observation-symbol vocabulary that counts as mutually
exclusive claims about that topic. Every symbol may belong to at most one topic.
Events acquire a symbol only through the existing V11 observation-rule matcher; V14
does not infer propositions from arbitrary language.

The policy declares:

- initial directed source trust in `[0, 1]`;
- confirmation and contradiction learning rates in `[0, 1]`;
- confirmation and contradiction affinity deltas in `[0, 1]`;
- a positive maximum unresolved-claim age;
- a positive maximum active-claim count per observer.

## State and evidence

`SituatedSocialMemoryState` is a content-addressed chain bound to the exact V13
cognitive state hash and current story round. It contains:

- one directed `SituatedSourceRelationship` for every ordered pair of distinct
  agents, with trust, affinity, confirmation count, and contradiction count;
- immutable-history `SituatedConsolidatedClaim` records;
- processed social evidence IDs so replay is idempotent.

`SituatedSocialEvidence` is the narrow input to the pure social transition. A
testimony item has an observer, distinct source agent, topic, symbol, event round,
event ID, and optional memory ID. A verification item has no source and represents
non-testimony evidence privately available to the observer.

## Consolidation and contradiction

For one observer/source/topic:

- repeated testimony with the same symbol is consolidated into the active claim;
  its event and memory IDs are appended and `support_count` increases;
- testimony with a different symbol marks the prior active claim `superseded` and
  creates a new active claim, preserving both records;
- duplicate evidence IDs are ignored through the processed-evidence ledger.

This is explicit categorical revision, not unrestricted natural-language
contradiction detection.

## Verification and relationship learning

After all same-round testimony is consolidated, verification evidence is applied to
active claims for the same observer/topic:

- matching symbols mark a claim `confirmed`, update
  `trust := trust + confirmation_rate * (1 - trust)`, increase affinity, and
  increment the confirmation count;
- different symbols mark a claim `contradicted`, update
  `trust := trust * (1 - contradiction_rate)`, decrease affinity, and increment the
  contradiction count.

Affinity is clamped to `[-1, 1]`. A terminal claim is never verified twice.
Relationships are directed: Bob's trust in Alice is independent of Alice's trust in
Bob.

## Forgetting

After verification, an unresolved active claim is marked `forgotten` when its age is
strictly greater than `max_unresolved_age_rounds`. If an observer still has more than
`max_active_claims` unresolved active claims, the oldest claims are forgotten first,
with claim ID as the deterministic tie breaker.

Forgetting changes the V14 social index only. V12 SQLite records and V13 audit history
remain intact, so analysts can explain what was forgotten without destructive data
loss.

## Trust-weighted recall

V13's default behavior remains unchanged and uses source multiplier one. The internal
V14 round path supplies the observer's current directed trust for recalled external
`TELL` events:

```text
w = confidence * salience * source_trust
tempered(h) = (1 - w) + w * L(h)
posterior(h) proportional to prior(h) * tempered(h)
```

Inspection, self, visual, and other non-testimony memories retain their V13 weight.
Trust learned after a round affects later recall, never the already-completed decision.

## Round integration

One V14 round:

1. validates the exact social model, story, V13 cognitive state, and V14 social state;
2. runs the V13 cognitive round with current directed trust multipliers;
3. projects direct and recalled admissions into declared social evidence;
4. consolidates testimony, resolves verification, learns relationships, and forgets
   over-bound unresolved claims;
5. commits a parent-linked V14 state bound to the V13 next cognitive state.

The physical story continues to resolve synchronously through V10. Database paths do
not enter content hashes. Equal inputs replay exactly.

## Acceptance scenarios

- Repeated Alice-to-Bob approval testimony becomes one claim with support count two.
- A later denial from Alice supersedes the approval claim but preserves both.
- Bob's direct inspection confirming Alice increases only Bob-to-Alice trust and
  affinity; contradictory inspection decreases them.
- With prior Bob-to-Alice trust `0.5`, recalled auditory testimony has half the V13
  evidence weight and therefore changes Bob's belief less than full-trust testimony.
- Expired or over-capacity unresolved claims become forgotten while SQLite memory rows
  remain queryable.
- Replaying the same social evidence or using another database path produces identical
  state and trajectory hashes.

## Deferred work

- free-text or semantic contradiction discovery;
- strategic deception and intent inference;
- relationship-dependent physical permissions or topology rewiring;
- stochastic forgetting, emotion, and personality learning;
- cross-world autobiographical identity.

