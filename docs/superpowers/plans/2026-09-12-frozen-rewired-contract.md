# Frozen Rewired-Control Contract Closure

> **For agentic workers:** Use superpowers:executing-plans with exact-head GitHub CI.

**Goal:** Finish the already approved #68 graph freeze without opening behavioral data.

**Architecture:** Keep the diagonal-fixed generator and cache unchanged in semantics. Promote the independently matching #71/#73 artifact fingerprints into both real protocols, and share a fail-closed control-contract validator between provenance and comparison preflight. No fallback to the old algorithm, a reduced budget, or an incomplete seed map.

**Spec:** `docs/phase2-real-data-freeze.md`, #68, and `docs/provenance-ci-performance.md`.

**Evidence:** #74 `b13f2e3b1eb4af822265f59f6b8f022766193945` is the GREEN baseline. Original artifact #71 ZIP SHA-256 `ad54d6247f86ec0fe864cfbd6acce859bbbef895e161357fa279a45e08d280d8` and optimized #73 ZIP SHA-256 `e4be8b9699dfba8fa051a9a4655522a495dedddc77a6768c5b55fe8fe31659e3` were locally reread and their five fingerprints agree.

## One bounded RED-to-GREEN task

- [ ] Add `tests/test_frozen_rewired_contract.py` with the independently measured fingerprints, real-template/preflight agreement, missing/extra/malformed seed maps, stale algorithms, reduced/noninteger budgets, and no-artifact rejection tests. Commit tests before implementation; run `python -m pytest tests -q` in GitHub CI and inspect exact-head RED.
- [ ] Add `validate_rewiring_protocol(config: dict[str, Any]) -> None` in `python/yolo_flywire/provenance.py`. Require unique integer seeds; algorithm `directed-double-edge-swap-v2-diagonal-fixed`; integer multiplier 10; exactly one canonical lowercase SHA-256 per declared seed; no control identical to the original graph. Invoke from `_context` and the real-topology branch of `_validate_frozen_protocol` in `cli.py`.
- [ ] Freeze all five measured fingerprints in `protocols/v0-real-ntu120-preflight.json`. Make `protocols/v0-real-template.json` mirror the complete experiment contract except protocol_id, frozen_preflight and evidence_scope. Leave unresolved NTU/YOLO/split/schema values null. Do not change graph selection, seed list, model budget or decision threshold.
- [ ] Update the stale source-hash/rewiring descriptions in `docs/phase2-real-data-freeze.md`; distinguish graph provenance completion from real-data readiness.
- [ ] Require exact-head Lean/Python/provenance GREEN. The changed protocol/generator identity must cause a cold generation; its outputs must equal the now frozen map. Inspect all five seed results, not only structural invariants. Record evidence and remaining external inputs in #68.

## Completion boundary

This task does not implement NTU acquisition, YOLO extraction, real classifier execution, or a recognition-performance claim. The current real `compare` branch emits protocol-validation-only records; byte-level freeze alone must not be described as an executed confirmatory experiment. Graph hashes identify inputs, not biological efficacy. Review and run the separate real-data extraction/runner work only after this gate.
