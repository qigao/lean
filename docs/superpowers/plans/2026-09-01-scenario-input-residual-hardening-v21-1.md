# Scenario Input Residual Hardening V21.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the two final V21.1 input blockers by making compiled identity follow schema-coerced numeric semantics and making immutable source construction reject excessive nesting through a sanitized public error.

**Architecture:** Keep raw source bytes and hashes as provenance, then normalize only the semantic identity projection with the same integer-versus-number distinction enforced by compilation. Bound recursive JSON freezing inside the public source contract so both direct construction and package loading fail before Python's recursion limit. No V21.2 runtime/output behavior changes.

**Tech Stack:** Python standard library (`dataclasses`, `math`, immutable mappings), existing scenario compiler/contracts, `unittest`/`pytest`.

**Spec:** `docs/superpowers/specs/2026-09-01-scenario-io-live-projection-v21-design.md`

## Global Constraints

- Preserve canonical `scenario.json` / `run.json` / Tiled `.tmj` package layout.
- Raw manifest and document hashes remain provenance and do not enter semantic equality or `CompiledSituatedScenario.content_hash`.
- Values accepted by the same schema numeric field and coerced to the same runtime value produce the same semantic hash.
- Distinct values of integer-only fields remain distinct, including integers above binary64's exact range.
- Public loader/source-contract failures expose no machine path, source value, or unsafe chained cause.
- Add no YAML, provider, network, Blender, LLM, RAG, or V21.2 output dependency.
- Preserve V10–V20 authority, privacy, fixed-roster, Tiled, and SQLite invariants.
- Do not run the stopped repository-wide suite; use focused tests and the authorized `test_network_abm*.py` gate only.

---

### Task 1: Schema-coerced numeric semantic identity

**Files:**
- Modify: `narrative_dynamics/abm/scenario_compiler.py`
- Modify: `tests/test_network_abm_scenario_compiler.py`

**Interfaces:**
- Consumes: canonical `ScenarioPackageSource`, `_number(...) -> float`, `_integer(...) -> int`, `_semantic_package_identity(source)`.
- Produces: role/path-aware `_semantic_source_value(...)` whose numeric projection matches compiler coercion without changing raw provenance.

- [ ] **Step 1: Add failing runtime-equivalence and integer-distinction tests**

Add two real package tests:

```python
def test_compiled_identity_normalizes_large_integer_float_runtime_equivalents(self):
    integer_root = write_law_firm_package(self.root / "large-integer")
    float_root = write_law_firm_package(self.root / "large-float")
    mutate_json(
        integer_root / "physical/map.tmj",
        "/layers/0/objects/0/x",
        9007199254740993,
    )
    mutate_json(
        float_root / "physical/map.tmj",
        "/layers/0/objects/0/x",
        9007199254740993.0,
    )
    refresh_manifest_hash(integer_root, "physical.map", "map")
    refresh_manifest_hash(float_root, "physical.map", "map")

    integer = compile_situated_scenario_package(
        load_situated_scenario_package(integer_root)
    )
    floating = compile_situated_scenario_package(
        load_situated_scenario_package(float_root)
    )

    self.assertEqual(integer.spatial_map, floating.spatial_map)
    self.assertEqual(integer.package_hash, floating.package_hash)
    self.assertEqual(integer.content_hash, floating.content_hash)
    self.assertEqual(integer, floating)
    self.assertNotEqual(integer.raw_manifest_hash, floating.raw_manifest_hash)

def test_compiled_identity_keeps_large_integer_only_values_distinct(self):
    first_root = write_law_firm_package(self.root / "integer-only-first")
    second_root = write_law_firm_package(self.root / "integer-only-second")
    mutate_json(first_root / "run.json", "/maximum_rounds", 9007199254740992)
    mutate_json(second_root / "run.json", "/maximum_rounds", 9007199254740993)
    refresh_manifest_hash(first_root, "run", "run")
    refresh_manifest_hash(second_root, "run", "run")

    first = compile_situated_scenario_package(load_situated_scenario_package(first_root))
    second = compile_situated_scenario_package(load_situated_scenario_package(second_root))

    self.assertNotEqual(first.run_policy.maximum_rounds, second.run_policy.maximum_rounds)
    self.assertNotEqual(first.package_hash, second.package_hash)
    self.assertNotEqual(first.content_hash, second.content_hash)
```

These tests catch two opposite mutations: retaining authored integer type for a float-coerced field, and globally converting integer-only fields through binary64.

- [ ] **Step 2: Run Task 1 RED**

Run:

```text
python -m pytest tests/test_network_abm_scenario_compiler.py -q -k "large_integer_float_runtime_equivalents or large_integer_only_values_distinct"
```

Expected: the runtime-equivalence case fails on package/compiled equality; the integer-only distinction case passes and remains a regression guard.

- [ ] **Step 3: Implement role/path-aware numeric normalization**

In `scenario_compiler.py`, define explicit wildcard paths for every integer-only field consumed through `_integer`, including:

```python
_SEMANTIC_INTEGER_PATHS = {
    ScenarioDocumentRole.AGENT: frozenset({
        ("body", "inventory_capacity"),
        ("memory", "recall", "cues", "*", "limit"),
        ("memory", "recall", "max_memories_per_round"),
    }),
    ScenarioDocumentRole.SOCIAL_RELATIONSHIPS: frozenset({
        ("policy", "max_unresolved_age_rounds"),
        ("policy", "max_active_claims"),
    }),
    ScenarioDocumentRole.SOCIAL_NORMS: frozenset({
        ("norms", "*", "priority"),
    }),
    ScenarioDocumentRole.STORY_OUTLINE: frozenset({
        ("scenes", "*", "maximum_rounds"),
    }),
    ScenarioDocumentRole.RUN: frozenset({
        ("deterministic_seed",),
        ("maximum_rounds",),
        ("checkpoint_interval",),
        ("maximum_output_records",),
        ("maximum_resource_bytes",),
    }),
}
```

This is the complete current `_integer` path set. Add a pure wildcard matcher:

```python
def _semantic_path_matches(path: tuple[str, ...], pattern: tuple[str, ...]) -> bool:
    return len(path) == len(pattern) and all(
        expected == "*" or expected == actual
        for actual, expected in zip(path, pattern)
    )
```

After successful compilation has already validated the package, normalize numbers as follows:

```python
if isinstance(value, (int, float)) and not isinstance(value, bool):
    if any(_semantic_path_matches(path, item) for item in integer_paths):
        return value
    result = float(value)
    if result == 0.0:
        result = 0.0
    return result
```

All Tiled numeric values use the runtime's numeric coercion and therefore take the float branch. Integer-only paths preserve exact Python integers; their float spellings are rejected earlier by `_integer`. Do not change raw hashes or provenance fields.

- [ ] **Step 4: Run Task 1 GREEN and focused regression**

Run:

```text
python -m pytest tests/test_network_abm_scenario_compiler.py -q -k "large_integer_float_runtime_equivalents or large_integer_only_values_distinct or numeric_scalars"
python -m pytest tests/test_network_abm_scenario_compiler.py -q
```

Expected: both commands pass with no warnings.

- [ ] **Step 5: Commit Task 1**

```bash
git add narrative_dynamics/abm/scenario_compiler.py tests/test_network_abm_scenario_compiler.py
git commit -m "fix(abm): align numeric scenario identity"
```

### Task 2: Bounded immutable JSON construction and final gates

**Files:**
- Modify: `narrative_dynamics/abm/scenario_package_contracts.py`
- Modify: `tests/test_network_abm_scenario_package.py`

**Interfaces:**
- Consumes: `ScenarioSourceDocument(...)`, `_freeze_json_mapping(...)`, public `load_situated_scenario_package(root)`.
- Produces: a path/value-free `ValueError` for source JSON nesting deeper than `_SCENARIO_JSON_MAX_DEPTH = 128`, with no raw `RecursionError` or chained cause.

- [ ] **Step 1: Add failing direct-contract and loader tests**

Add a helper that builds nested arrays iteratively and two behavior tests:

```python
def nested_json(depth: int) -> object:
    value: object = None
    for _ in range(depth):
        value = [value]
    return value

def test_source_document_rejects_excessive_nesting_without_recursion_leak(self):
    from narrative_dynamics.abm.scenario_package_contracts import (
        ScenarioDocumentRole,
        ScenarioSourceDocument,
    )

    with self.assertRaisesRegex(ValueError, "nesting") as raised:
        ScenarioSourceDocument(
            ScenarioDocumentRole.RUN,
            "run",
            DOCUMENT_SCHEMA,
            {"nested": nested_json(700)},
            "sha256:" + "0" * 64,
        )
    self.assertIsNone(raised.exception.__cause__)
    self.assertNotIsInstance(raised.exception, RecursionError)

def test_loader_rejects_post_parse_excessive_nesting_without_source_leak(self):
    from narrative_dynamics.abm.scenario_package import (
        load_situated_scenario_package,
    )

    root = write_minimal_package(self.root / "deep-freeze")
    run_path = root / locator_for(root, "run")["path"]
    run_document = json.loads(run_path.read_text(encoding="utf-8"))
    run_document["value"]["nested"] = nested_json(700)
    _write_json(run_path, run_document)
    refresh_locator_hash(root, "run")
    with self.assertRaisesRegex(ValueError, "nesting") as raised:
        load_situated_scenario_package(root)
    self.assertIsNone(raised.exception.__cause__)
    self.assertNotIn(str(root), str(raised.exception))
```

Use existing package-test fixture helpers and raw manifest hash refresh behavior rather than mocks. The production change that makes both tests fail is removal of the explicit depth check or its sanitized error.

- [ ] **Step 2: Run Task 2 RED**

Run:

```text
python -m pytest tests/test_network_abm_scenario_package.py -q -k "excessive_nesting"
```

Expected: direct construction exposes `RecursionError`; loader either exposes the same error or fails without the required sanitized `ValueError` contract.

- [ ] **Step 3: Implement bounded recursive freezing**

Add the internal constant and depth parameter:

```python
_SCENARIO_JSON_MAX_DEPTH = 128

def _freeze_json_value(
    value: object,
    *,
    label: str,
    depth: int = 0,
) -> object:
    if depth > _SCENARIO_JSON_MAX_DEPTH:
        raise ValueError("scenario source JSON exceeds maximum nesting depth")
```

Every recursive mapping/list call passes `depth=depth + 1`. Wrap `_freeze_json_mapping`'s call defensively:

```python
try:
    frozen = _freeze_json_value(value, label=label)
except RecursionError:
    raise ValueError("scenario source JSON exceeds maximum nesting depth") from None
```

The explicit bound must fire before Python's normal recursion limit. Keep the constant private, do not include labels, paths, or source values in the depth error, and do not change accepted shallow immutable values.

- [ ] **Step 4: Run Task 2 GREEN and shallow compatibility**

Run:

```text
python -m pytest tests/test_network_abm_scenario_package.py -q -k "excessive_nesting or loads_path_independently"
python -m pytest tests/test_network_abm_scenario_package.py -q
```

Expected: both commands pass; the platform symlink test may remain skipped.

- [ ] **Step 5: Run authorized final gates**

Run exactly:

```text
python -m pytest tests/test_network_abm_scenario_package.py tests/test_network_abm_scenario_authoring_contracts.py tests/test_network_abm_scenario_compiler.py tests/test_network_abm_situated_spatial_map.py tests/test_blender_replay.py tests/test_network_abm_public_api.py -q
python -m unittest discover -s tests -p "test_network_abm*.py" -q
python -m compileall -q narrative_dynamics tests
git diff --check
```

Expected: all executed tests pass; only the existing symlink/Blender environment skips remain; compileall and diff check exit zero. Do not run any other repository-wide suite.

- [ ] **Step 6: Commit Task 2**

```bash
git add narrative_dynamics/abm/scenario_package_contracts.py tests/test_network_abm_scenario_package.py
git commit -m "fix(abm): bound scenario JSON nesting"
```
