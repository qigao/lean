# R2.M4c.0: Preregister model-native metric units before comparisons

Parent: Issue #117, stacked after Draft PR #116 (R2.M4b).

This is an immutable read-only registration on *actual* BB, rewiring,
evolving source replays.  It does not replace ABM engines or V19 social memory.

| Family | Registered local metric | Numerator | Denominator | Phase |
| --- | --- | --- | --- | --- |
| BB | Undirected birth topology density | Unique undirected native edges | n(n-1)/2 unordered pairs of current born Agents | After global tick |
| Rewiring | Directed candidate activation fraction | Active directed candidates | Fixed declared directed candidate catalog, not all possible edges | After rewiring, not earlier propagation |
| Evolving | Active catalog membership fraction | Active Agent members | Whole native catalog including inactive Agents, not potential edges | After lifecycle + feedback + rewiring |
| V19 | No metric in this M4b summary | Not measured | Eligible source relation pairs unavailable | Branch-qualified Story only |

Three fractions are **not the same metric** and must not be pooled or ranked.
All metric rows carry source run/frame hashes, native clock, phase, roster
scope, a typed numerator/denominator and exact rational fraction.

Comparison eligibility: SUPPORTED is restricted to descriptions of the
same model/run/unit/phase. Across different models the contract reports
NOT_COMPARABLE. MAPPED_WITH_ASSUMPTIONS is not admitted until a separately
reviewed mapping version specifies roster, edge direction, observation
window, feedback/trust and causal provenance. The V19 reference remains
a distinct source with no inferred relation density.

The inspected native single-run corpus is pinned in
conformance/native_metric_protocol_v1.json. Existing Python and Lean jobs
must regenerate it byte-for-byte from existing M4b native source engines.

sample_size_per_native_run=1, confidence_interval=NOT_ESTIMABLE,
external_validation=NO_EXTERNAL_VALIDATION, causal_attribution=UNATTRIBUTED.
No simulated population inference, p-values, universal trust, new six-hop
theorem, fallback model or downstream consumer project.

M4c.1 still needs preregistered within-family control/intervention schedules,
matched initial conditions and stopping policies. M4c.2 needs genuine repeated
sampling and empirical external validation where possible. Keep this PR Draft
and preserve all original Python/Lean/BB/PathN/Studio CI gates.
