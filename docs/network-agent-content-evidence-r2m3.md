# R2.M3 — Source-grounded V19 Agent message, percept, claim and trust evidence

Tracked by [Issue #110](https://github.com/qigao/lean/issues/110);
stacked on [Draft PR #113](https://github.com/qigao/lean/pull/113)
(source-replayed environmental actions). No known six-hop theorem is
reproven, no second social graph/content engine and no unrelated consumer
or installer is added.

## The five different facts

| Fact | Authoritative existing owner | What R2.M3 can say |
| --- | --- | --- |
| **Message sent** | `SituatedStory` / successful actual `SituatedWorldEvent(TELL)` | Event ID, hash, actor and round; **not** who understood it |
| **Perception** | `perceptual_timeline` private `SituatedPercept` | Aggregate other-Agent counts by `EXACT / IDENTIFIED / DETECTED`; DETECTED cannot expose topic, actor or message text |
| **Cognitive admission** | `SituatedPerceptMemoryCognitiveRoundResult` / `SituatedCognitiveState` | Source-bound belief hashes and their changes, **not** a claim that belief equals ground truth |
| **Evidence and claim** | `_social_evidence` + `SituatedSocialMemoryUpdate` | Exact-only testimony vs own private exact INSPECT verification; claim ID/hash/status transition without symbol/payload |
| **Directed trust** | `SituatedSocialMemoryState.relationships` | A per-(observer,source) before/after trust revision; no automatic V19 graph edge without actual runtime threshold projection |

The current implementation does **not** introduce a shared-document owner,
merge concurrent edits or infer that a TELL becomes a verified fact. Existing
claim IDs/revisions are source-owned by V19 social memory, not a new content
table. If a shared document is later requested it must first have an explicit
existing canonical owner; message history and trust are not a document revision.

## Exact-source verification

`observe_agent_content_evidence(memory_model, social_model, prior_story,
prior_cognition, prior_social, result)` takes an **actual**
`SituatedPerceptSocialCognitiveRoundResult` from the V19 simulator.

It checks exact model, story, social and cognitive identities and the
single-round world continuation by reexecuting `resolve_situated_round`.
The accepted private evidence must be reproducible through existing
`_social_evidence` with exact observer-specific percepts from the
previous story. It then replays the entire social update through
`advance_situated_social_memory`, ensuring that claimed evidence IDs,
versioned claim status, directed trust revisions and next social state are
exactly those the real owner produces. **It does not independently re-run
the SQLite/recall cognitive decision engine**, and must not be presented as
proof that an external actor spoke or that the resulting beliefs are true.

The immutable `AgentContentEvidenceTrace` reports hashes, event IDs,
aggregate fidelity, admitted exact testimony/verification IDs, claim hashes
and status, belief revision hashes and directed trust values. It does not
serialize messages, INSPECT payloads, claim symbols or private percept
details. Callers receive detached dictionaries and cannot change its
`UNATTRIBUTED` scope or underlying digest.

## Runtime conformance scenarios

- Exact heard TELL → admitted TESTIMONY and new ACTIVE claim, with **no**
  automatic verification or trust increase.
- A DETECTED distant sound and an IDENTIFIED action without exact
  message content → **no testimony, claim or trust change**.
- Own exact private INSPECT → VERIFICATION evidence without inventing
  a second Agent's testimony.
- Real recalled contradictory TELL and later local INSPECT → claim
  becomes CONTRADICTED and only Bob's trust of Alice changes, with
  unaffected reverse direction.
- Reject fabricated admitted social evidence, stale source states,
  source-irreproducible world events, and attempts to mutate public
  report data or expose private content.

Source-derived, reviewed five-case
`conformance/agent_content_evidence_v1.json` is mandatory in focused
Python and independent Lean acceptance lanes, with an exact UTF-8 byte
comparison and the original full Python, Lean/35+11/BB/PathN and
Studio/Playwright gates. First missing pin is intentionally RED for
review of actual CI output, then its temporary handoff is removed.

## Remaining boundary

This M3 phase is **observation and verification of an existing**
content/social transition, not implementation of concurrent shared
document edits, autonomous choice correctness or causal inference. Future
agent-network coevolution experiments must preserve story and cognition
ownership and compare observed histories with frozen controls.

## Reviewed first-run conformance — 2026-10-30

Focused [CI #41016351096](https://github.com/qigao/lean/actions/runs/41016351096)
reproduced the five source-derived cases. The observed other-Agent
perception counts are **2**, not 1, in the current four-Agent fixture;
the admitted evidence count is independent: Bob alone provides one
TESTIMONY under the configured cognitive rule. This is an important
distinction between *perceived*, *semantically admitted*, and *socially
consolidated*. No additional Agent's private message content is included
in the report.

Four of seven initial behavior regressions already passed; the two
incorrect test-only fidelity counts were corrected to 2 without modifying
the V19 perception or evidence owner. The third expected first-run RED
was the mandatory missing versioned JSON. Canonical
`conformance/agent_content_evidence_v1.json` was copied exactly from
the genuine runner output (including numeric representation and source
hashes), reviewed for schema, five case identities, directed trust,
claim status and absence of private content. The one-time textual
handoff was then removed; all current-head lanes must match these exact
bytes.
