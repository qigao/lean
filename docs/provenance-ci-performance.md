# FlyWire provenance CI performance checkpoint

Tracked in [#68](https://github.com/qigao/lean/issues/68).
This report concerns graph preparation and CI only. It is not behavioral-recognition evidence, and no real-data classifier training or sealed final-test evaluation was performed.

## Root cause and fix

The original sampler computed `edge_set - {old_one, old_two}` inside every proposal's collision check. This allocated a copy of the full graph repeatedly. The optimized predicate tests membership while explicitly excluding the same two edges. RNG calls, proposal order, acceptance semantics, attempt budget, diagonal constraints, and the full successful-swap target are unchanged.

`tests/test_rewiring_performance.py` compares exact ordered graph outputs and fingerprints against a test-only copy of the original sampler for all five seeds on sparse and dense fixtures. An allocation guard rejects whole-set copying without a flaky wall-clock threshold. Progress and cache integrity tests cover successful and exhausted attempts, stale identities, missing seeds, fingerprint corruption, weight changes, and diagonal-weight changes. The copying oracle is test-only, never a production fallback.

## Exact-head RED to GREEN

- RED commit: `eab759ccd08a63ca40f36edccc990d5d2cd9d68b`.
- [CI #72 / 34688929877](https://github.com/qigao/lean/actions/runs/34688929877): 42 Python tests passed, 13 failed for the newly required allocation/logging/cache behavior; Lean succeeded. The expensive provenance job was correctly skipped while tests were RED.
- Implementation commit: `9b7aea684e9322c689bf63c079f888f9479d783c`.
- [CI #73 / 34689088871](https://github.com/qigao/lean/actions/runs/34689088871): Lean, Python, and real FlyWire provenance all succeeded. Python log: **55 passed in 12.23 seconds**.

## Measured cold-cache real-source run

The verified source is the pinned FAFB v783 type table in `protocols/v0-real-ntu120-preflight.json`, not a synthetic performance fixture. The selected graph contains 187 nodes, 14,542 directed edges, and 126 diagonal population edges. Each seed executes exactly **144,160 successful off-diagonal swaps**, preserving exact diagonal edge+weight values, directed degrees, graph size, and the weight multiset.

| Seed | Attempts | Successful swaps | Generation plus per-seed checks |
| --- | ---: | ---: | ---: |
| 7 | 921,732 | 144,160 | 1.766 s |
| 11 | 926,813 | 144,160 | 1.774 s |
| 19 | 924,900 | 144,160 | 1.772 s |
| 23 | 923,035 | 144,160 | 1.811 s |
| 31 | 921,472 | 144,160 | 1.772 s |

The complete `generate` command, including source parsing, five controls, validation, and serialization, took **9.515 seconds**. A separate `verify` command took **0.559 seconds** without calling the sampler. Timings are observations from this runner, not a universal performance guarantee or a speedup ratio against an unmeasured full original run.

The uploaded `flywire-v783-provenance` artifact for #73 has ID `10296297039` and ZIP SHA-256 `e4be8b9699dfba8fa051a9a4655522a495dedddc77a6768c5b55fe8fe31659e3`. It includes the complete control bundle, selected-graph report, and current-run verification record.

## Cache contract and routine CI

The exact cache identity binds canonical protocol contents, actual source bytes, selected graph fingerprint, seeds, successful-swap budget, generator source hashes, and Python/NumPy/platform identity. No partial-key `restore-keys` are used.

On an exact miss, CI generates and validates all controls before saving. On an exact hit, CI does not run the sampler; it checks bundle identity, exact seed coverage, graph fingerprints, directed degree sequences, weights, and fixed diagonals. Reports are rebuilt from the verified graphs rather than trusted from a cached summary. A malformed or stale restored bundle is an error, not a silently substituted result.

The provenance job now depends on Lean and Python success, has a ten-minute timeout, and logs per-seed timing plus periodic attempted/accepted swap counts. The workflow declares same-workflow/ref concurrency cancellation for superseded runs using this configuration.

This documentation-only commit does not modify any cache input. Its CI run is intended to verify actual warm-cache reuse: an exact hit, generation skipped, verification and artifact upload successful. The observed result will be recorded in #68; it is not presumed successful here.

No protocol, model, dataset, evidence threshold, or experimental budget was changed by this performance fix.
