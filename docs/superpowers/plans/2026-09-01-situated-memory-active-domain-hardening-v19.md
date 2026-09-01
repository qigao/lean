# Situated Memory Active-Domain Hardening V19 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reject persisted percept-memory activation values outside the exact integer domain `{0, 1}` so V19 canonical memory hashes cannot collide across different recall behavior.

**Architecture:** Keep SQLite as the persistence authority and add one strict raw-value decoder at the existing row-to-contract boundary. Every consumer that reconstructs a row, including the V19 logical-store hasher, then shares the same activation semantics as the SQL predicate `active = 1`.

**Tech Stack:** Python standard library, SQLite/FTS5, `unittest`/pytest, immutable percept-memory contracts.

**Spec:** `docs/superpowers/specs/2026-09-01-situated-memory-active-domain-hardening-v19-design.md`

## Global Constraints

- Do not change valid V10-V19 contracts, public signatures, or content hashes.
- Reject corrupt activation values; never silently normalize or repair them.
- Follow RED → GREEN and verify the exact reproduced collision.
- Blender replay remains a separately versioned optional integration.

---

### Task 1: Strict persisted activation domain

**Files:**
- Modify: `narrative_dynamics/abm/situated_percept_memory.py`
- Modify: `tests/test_network_abm_situated_network.py`

**Interfaces:**
- Consumes: SQLite `percept_memory_records.active`, `_row_to_memory`, and `SituatedPerceptMemoryConflictError`.
- Produces: private `_active_from_row(value: object) -> bool`; unchanged public `hash_situated_percept_memory_store(database_path) -> str` now rejects corrupt activation values.

- [x] **Step 1: Write the failing collision regression**

Add a test that ingests a real two-row Bob percept checkpoint, records its legal hash and visible count, updates both raw SQLite values to `2`, confirms raw SQL visibility differs, and asserts `hash_situated_percept_memory_store` raises `SituatedPerceptMemoryConflictError` matching `active`. This catches any nonzero-to-true coercion in the canonical hash path.

- [x] **Step 2: Run the regression and verify RED**

Run: `python -m pytest tests/test_network_abm_situated_network.py -q -k rejects_non_binary_active`

Expected: FAIL because hashing accepts `active=2` and returns the prior legal hash instead of raising.

- [x] **Step 3: Implement the minimal strict decoder**

```python
def _active_from_row(value: object) -> bool:
    if type(value) is not int or value not in (0, 1):
        raise SituatedPerceptMemoryConflictError(
            "percept memory active value must be the integer 0 or 1"
        )
    return value == 1
```

Call it from `_row_to_memory` instead of `bool(row["active"])`. Do not alter query predicates or add coercion.

- [x] **Step 4: Add the legal-zero guard and run focused tests GREEN**

In the same regression fixture, set the records to `0`, require hashing to succeed, and require `list_situated_percept_memories(..., include_inactive=True)` to return records whose `active` fields are all `False`.

Run: `python -m pytest tests/test_network_abm_situated_network.py tests/test_network_abm_situated_percept_memory.py -q`

Expected: all focused tests pass.

- [x] **Step 5: Run authoritative verification**

Run: `python -m unittest discover -s tests -p "test_network_abm*.py"`

Run: `python -m compileall -q narrative_dynamics`

Run: `git diff --check`

Expected: all network ABM tests pass and both compilation and diff checks exit zero.

- [ ] **Step 6: Commit the hardening**

```text
git add docs/superpowers/specs/2026-09-01-situated-memory-active-domain-hardening-v19-design.md docs/superpowers/plans/2026-09-01-situated-memory-active-domain-hardening-v19.md narrative_dynamics/abm/situated_percept_memory.py tests/test_network_abm_situated_network.py
git commit -m "fix(abm): validate percept memory activation domain"
```

## Self-review

- Spec coverage: strict raw domain, shared decoder, rejection behavior, legal-zero preservation, and authoritative regression all map to Task 1.
- Placeholder scan: no deferred implementation step or undefined interface remains.
- Type consistency: `_active_from_row` consumes the exact SQLite scalar currently passed to `bool` and returns the boolean required by `SituatedPerceptMemoryRecord`.
