# R2.M5g.3 — exact source-bound Bayesian TELL theorem (Draft)

Strict parent: [M5g.2 Draft #131](https://github.com/qigao/lean/pull/131),
HEAD \`ccd5f9c1\`, [CI #41160747432](https://github.com/qigao/lean/actions/runs/41160747432)
4/4 SUCCESS. Track research limits in [Issue #120](https://github.com/qigao/lean/issues/120).

## Model identity, not a new inference implementation

In the original V19 SQLite/cognitive/social owner, an admitted external
EXACT TELL uses \`effective_likelihood(t,L) = (1-t)+tL\` when the
source-social direct testimony policy is ON. The model's registered
neutral-audit target symbol uses \`L(approved)=18/25\` and
\`L(denied)=2/25\`; Bob's prior is (1/2, 1/2). The existing native
source replay identifies \`t=3/5,3/10\` when policy ON and \`t=1\`
for the OFF direct likelihood, while its source-social trust still
learns 3/5 or 3/10 in both policy versions.

The Lean theorem is only the finite two-hypothesis mathematical contract:

\`p(t) = (1 - (7/25)t) / (2 - (6/5)t)\`, for \`0 ≤ t ≤ 1\`.

It proves the denominator is positive, the approved/denied posteriors
sum to 1, and \`p(s)<p(t)\` when \`0 ≤ s<t ≤ 1\`. Exact rational
boundaries are \`p(0)=1/2\`, \`p(3/10)=229/410\`, \`p(3/5)=13/20\`,
\`p(1)=9/10\`. Equal, nonzero likelihoods are uninformative.
No general human/social-network monotonicity is asserted.

## Exact source association and failure discipline

Existing original V19 M5g.2 corpus is still pinned and regenerated
byte-for-byte in both the original-source Python and independent
Lean source-conformance lanes. The new *test-only* rational checker
reads that corpus and uses the same source cognitive-model hash and
native model likelihoods. It requires all four original cells, exact
new target TELL, two feasible original actions, source event identity,
reported posteriors, real MAP events and the negative no-switch result.
The checker cannot set Agent state or create messages.

Invalid/nonfinite trust, illegal or swapped likelihoods and zero
evidence are rejected by the mathematical contract. Missing/DETECTED-only
TELL or broken source provenance remain rejected by inherited native
source qualification, not by constructing fake source histories.
The Lean kernel validates algebra; only the original V19 source runner
validates real admission and behavior.

## Scientific limits

- No MAP action switch: all four real native actions are TELL, and
  all four original events succeed \`told\`.
- Distinct confirmed and contradicted histories have unmatched private
  claims/memories; within-history policy ablation identifies a finite
  *software likelihood pathway*, not an isolated human trust effect.
- \`trust_only_causal_effect=NOT_IDENTIFIED\`;
  \`sample_size_per_arm=1\`; \`NOT_ESTIMABLE\`;
  \`NO_EXTERNAL_VALIDATION\`; no universal key-player score.

CI reuses existing \`proof.yml\` jobs and tests. No added workflow,
installer/consumer project, second cognitive engine, fallback,
six-hop theorem or release. This branch stays Draft for stacked review.
