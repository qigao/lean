# Situated Memory-Augmented Cognition V13 Design

## Objective

Make durable V12 memories operational in V11 cognition without turning the SQLite
store into shared knowledge or repeatedly counting the same evidence. An agent can
resume at a later story checkpoint, retrieve only its own relevant past observations,
admit them into its private belief, and make a different POMDP action because of the
recalled evidence.

V13 is deterministic retrieval-augmented cognition. It uses declared lexical cues,
structured records, and the existing finite Bayesian/POMDP model. It adds no LLM,
embedding service, LSTM, Transformer, or learned latent state.

## Cognitive checkpoint

`initialize_situated_memory_cognition` can initialize a cognitive state at the
current round of an existing situated story. This represents a restarted process or
an agent whose active working state is reconstructed from durable storage.

Each checkpoint mind records an `observation_floor_round` equal to the current story
round. Historical perspective events at or below that floor are not automatically
replayed into cognition. New physical observations above the floor follow the V11
direct-admission path. The checkpoint knows the agent's current body place but does
not grant access to historical event IDs until their memories are actually recalled.

A non-zero checkpoint state has no parent state hash and is explicitly marked as a
checkpoint. Every later state again uses the normal parent-linked chain.

## Declarative recall model

`SituatedMemoryCognitiveModel` binds the exact V11 cognitive model, the exact V12
memory policy, and one `SituatedAgentRecallPolicy` for every agent. Each policy has a
bounded per-round result count and zero or more `SituatedMemoryRecallCue` values.

A cue declares:

- a stable cue ID and literal query text;
- optional required current places;
- optional event-kind and observation-channel filters;
- minimum memory confidence;
- a per-cue result limit.

Cues are evaluated in canonical cue-ID order. Results retain the V12 private lexical
order, are deduplicated by `(agent_id, observation_id)`, and are capped by the agent
policy. Empty cue sets provide a deterministic no-recall control arm.

Every query includes the requesting agent ID, the exact world-model hash, and a
maximum source round equal to the current cognitive state round. Memories from
another agent, another world model, an inactive row, or a future round cannot enter
the decision context.

## Direct evidence and recalled evidence

Direct V11 observations and V13 memories are separate evidence channels:

1. Admit new direct observations above the checkpoint floor.
2. Exclude every memory whose observation ID was directly processed.
3. Exclude every memory ID already present in `recalled_memory_ids`.
4. Reconstruct the exact immutable perspective event from the memory record and
   verify its event hash.
5. Apply the existing declarative observation-rule matcher.
6. Record the memory as recalled even when it matches no belief rule; a recalled
   physical event may still provide a private causal source for a later action.

This prevents direct-plus-recall double counting and repeated recall reinforcement.
Recall does not rewrite or deactivate the V12 row.

## Confidence-weighted belief admission

For a recalled memory matching likelihood `L(h)` under hypothesis `h`, let:

```text
w = memory.confidence * memory.salience
tempered(h) = (1 - w) + w * L(h)
posterior(h) ∝ prior(h) * tempered(h)
```

`w = 1` reproduces the V11 Bayesian observation update. `w = 0` records recall but
does not change belief. Every `SituatedMemoryRecallAdmission` preserves the cue,
memory/event IDs, rule/symbol/action IDs, confidence, salience, effective weight,
lexical rank, and exact prior/posterior beliefs. This is a declared simulator rule,
not a claim about human memory physiology.

## Round order

One V13 round executes deterministically:

1. Validate the exact recall model, story, and cognitive state.
2. Persist each agent's current private perspective through V12 idempotent ingestion.
3. Admit only new direct observations above the mind's checkpoint floor.
4. Run that agent's declared cues against its scoped memory rows.
5. Admit eligible memories once using the confidence-weighted rule.
6. Plan from the resulting private belief and recalled event set.
7. Resolve all physical intents synchronously through V10.
8. Commit the usual parent-linked next cognitive state.

New events from step 7 are ingested and admitted at the following decision round,
preserving V11's no-same-round-circularity rule.

## Decision and explanation audit

V11 decision records gain separate recalled-memory and recalled-symbol IDs. V13
round results additionally carry the full recall admissions. Explanations report
direct evidence IDs and recalled memory IDs separately, so an analyst can distinguish
"observed just now" from "remembered from long-term storage."

The database path is runtime configuration and never enters a content hash. Equal
model, story, memory contents, and cue inputs produce equal rounds and trajectories
across database paths and process restarts.

## Office acceptance scenario

1. Alice inspects an approved restructuring memo in round one and later moves to the
   open office.
2. Her perspective is persisted, then active cognitive state is restarted at that
   later story checkpoint.
3. Without recall cues, Alice has prior belief `0.5/0.5`, cannot cite the old
   inspection, and waits.
4. With the `restructuring` cue active in the open office, only Alice retrieves the
   private inspection.
5. The inspection has weight one, moves Alice's belief to `0.9/0.1`, restores the
   exact private source event, and makes `TELL` the selected action.
6. Bob and Dana cannot retrieve Alice's inspection. Bob may update only after hearing
   Alice's new telling in a later round.
7. Re-running recall does not move Alice to `0.9878`; the memory is admitted once.
8. A database containing later events cannot leak them into an earlier checkpoint.

## Deferred work

- semantic/vector retrieval and learned query generation;
- contradiction consolidation, source trust, relationship learning, and forgetting
  policies (V14);
- theory of mind or beliefs about another agent's memories;
- stochastic recall and capacity decay;
- cross-world autobiographical identity and explicit episode IDs.
