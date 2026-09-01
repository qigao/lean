# Situated Memory Active-Domain Hardening V19 Design

## Goal

Close the final V19 content-addressing gap by ensuring every persisted percept-memory `active` value has exactly the same meaning in Python reconstruction, canonical hashing, and SQLite recall predicates.

## Root cause

`_row_to_memory` currently reconstructs `active` with `bool(row["active"])`, so any nonzero SQLite value becomes `True`. Canonical V19 memory hashing consumes that reconstructed boolean. Recall queries, however, expose active records with the literal SQL predicate `active = 1`. Therefore illegal `active=2` content hashes exactly like `active=1` while producing a different visible recall set.

## Required behavior

- Raw persisted `active` must be a Python `int` whose value is exactly `0` or `1` before a `SituatedPerceptMemoryRecord` is constructed.
- Any other raw value raises `SituatedPerceptMemoryConflictError` before canonical hashing can accept it.
- Legal `0` and `1` values preserve their existing inactive/active behavior and hashes.
- The validation belongs in the shared SQLite row decoder so hashing, listing selected rows, search results, and ingestion conflict checks use one interpretation.
- Existing V10-V19 public signatures and valid content hashes remain unchanged.

## Verification boundary

The regression creates valid percept memories, confirms the legal checkpoint hash, corrupts `active` to `2` through direct SQLite access, and requires `hash_situated_percept_memory_store` to reject it. The test also proves legal `0` remains hashable and reconstructs as inactive. Focused V15/V19 tests and the complete network ABM suite must remain green.

## Non-goals

- No SQLite schema migration or database repair.
- No coercion of corrupt values back to `0` or `1`.
- No change to memory activation APIs, recall ranking, FTS behavior, or V19 atomic publication.
- No Blender or map-import implementation in this hardening change.
