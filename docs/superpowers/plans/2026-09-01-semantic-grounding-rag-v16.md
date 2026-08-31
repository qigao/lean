# Situated Semantic Grounding and Private RAG V16 Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a provider-neutral semantic compiler that turns natural-language interpretations into validated, replayable grounded claims using only one agent's sanitized percepts and private FTS5 memories, then bridge eligible accepted claims into V14 social evidence.

**Architecture:** The provider receives an immutable prompt packet containing a finite vocabulary and explicitly authorized evidence, and returns strict JSON only. Deterministic code parses and validates that proposal into a content-addressed `SituatedSemanticGroundingArtifact`; replay validates the accepted artifact without calling a provider. Retrieval remains agent-scoped SQLite FTS5: a provider may propose bounded literal query expansions, but cannot read the database, supply rows, or rank authoritative results.

**Tech Stack:** Python 3 standard library, immutable dataclasses, `typing.Protocol`, SQLite FTS5, `unittest`.

---

### Task 1: Define strict V16 contracts

**Files:**
- Create: `narrative_dynamics/abm/situated_grounding_contracts.py`
- Create: `tests/test_network_abm_situated_grounding_contracts.py`

1. Write failing contract tests for finite predicate/value declarations, private evidence packets, grounded claims, provider provenance, artifacts, canonical ordering, hashes, and exact-schema parsing failures.
2. Run the focused test and confirm RED because the module is absent.
3. Implement immutable contracts and strict validation, including polarity, modality, temporal scope, bounded confidence, exact evidence IDs, and model bindings.
4. Run the focused test to GREEN and commit.

### Task 2: Build private RAG prompt construction

**Files:**
- Create: `narrative_dynamics/abm/situated_grounding.py`
- Create: `tests/test_network_abm_situated_grounding.py`
- Modify: `narrative_dynamics/abm/situated_grounding_contracts.py`

1. Write failing tests for agent-scoped FTS5 retrieval, bounded query expansion, deterministic merge/deduplication, exact-percept versus partial-percept disclosure, and rejection of unknown filters or provider-supplied memory IDs.
2. Confirm RED, then implement a retrieval-planner protocol, strict JSON plan parsing, query validation, and deterministic private-context construction over `search_situated_percept_memories`.
3. Verify the planner sees only the query and allowlisted filter vocabulary; the grounder sees only sanitized selected evidence and never the objective story ledger.
4. Run focused tests to GREEN and commit.

### Task 3: Compile, validate, replay, and bridge grounded claims

**Files:**
- Modify: `narrative_dynamics/abm/situated_grounding.py`
- Modify: `tests/test_network_abm_situated_grounding.py`

1. Write failing tests for provider invocation, strict JSON decoding, unknown entity/predicate/value rejection, unavailable-evidence rejection, detected-sound detail rejection, replay without provider calls, and deterministic artifact hashes.
2. Implement provider-neutral proposal compilation and artifact replay. Provider output remains non-authoritative until deterministic validation succeeds.
3. Add a deterministic bridge from eligible asserted, affirmed topic/value claims to `SituatedSocialEvidence`; preserve observer/source/event/memory provenance and reject non-testimony/non-verification mappings.
4. Run focused tests to GREEN and commit.

### Task 4: Public API, documentation, and acceptance scenario

**Files:**
- Modify: `narrative_dynamics/abm/__init__.py`
- Modify: `tests/test_network_abm_public_api.py`
- Modify: `README.md`

1. Write failing public-export and office acceptance tests.
2. Export the V16 contracts/functions and document a provider-neutral example: ambiguous office language → private FTS5 recall → validated grounded claim → V14 social evidence.
3. State the boundaries explicitly: no bundled LLM/vendor client, no embeddings/vector store, no provider direct state mutation, and no objective-information fallback.
4. Run focused and full network-ABM tests to GREEN and commit.

### Task 5: Final verification and branch completion

1. Read and apply `superpowers:verification-before-completion`, `superpowers:requesting-code-review`, and `superpowers:finishing-a-development-branch`.
2. Run compile checks, the complete network ABM suite under the default hash seed and at least two explicit `PYTHONHASHSEED` values, and README snippets if a repository verifier exists.
3. Review the final diff for privacy leakage, provider authority, replay behavior, canonical nondeterminism, schema looseness, and unrequested dependency/vendor coupling; correct real defects through RED→GREEN.
4. Report the exact branch, commits, tests, capability boundary, and integration options.
