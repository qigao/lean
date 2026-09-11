# Fitness Attachment V23.5 — BA-compatible BB foundation

Status: design for review; no BB implementation or new theorem verification is claimed.
User direction: “ok，升级模型” approves adding the fitness model discussed in the conversation, while retaining the deterministic pseudofractal benchmark.

## 1. Decision and repository boundary

Add a separate finite Bianconi–Barabási (BB) attachment foundation. Recover Barabási–Albert (BA) by constant fitness; do not keep two independently evolving implementations.

Start `feature/fitness-attachment-v23-5` from `feature/mesh-feasibility-v23` at `c268e5355b14fb408f30814ed1c2bee2f41f1144` (#58). This is a sibling of #59 and #60, not stacked on their unfinished work. Do not modify, close, merge, or rebase those PRs. A later integration can reconcile additive root imports and workflow steps explicitly.

The inspected #58 describes a deterministic Watts–Strogatz foundation and explicitly excludes a BA/power-law claim. Its core tree contains no dedicated BA/BB foundation. This change adds a model with a BA compatibility theorem, not an in-place migration of an established BA runtime.

#60 remains the deterministic pseudofractal proof line. Its numbered representation is in `Pseudofractal.Internal`; do not import that internal representation into this model or inherit its formulas. Existing `MeshWalk`, `ReachWithin`, and `SmallWorldMetrics` semantics remain unchanged. In particular, do not weaken `SmallWorldCertificate` or use it as a mandatory fitness-model invariant.

Alternatives considered: replace the pseudofractal constructor (reject: changes the proof object); add time-varying fitness plus closure and society policies immediately (defer: conflates independent mechanisms); add a fixed-fitness, finite BB layer with explicit BA reduction (selected).

## 2. Exact model profile

The first profile is a finite, undirected, simple-graph BB specialization with positive rational fitness and sequential weighted selection without replacement. It is not asserted to be every multigraph or replacement convention used in the literature.

A state contains a finite graph `G : SimpleGraph (Fin n)`, computable adjacency, and fitness `eta : Fin n -> Rat`. Degree is computed from actual adjacency; there is no independent authoritative degree counter. Stored fitness is strictly positive and immutable for an existing vertex. Arbitrary-precision rational arithmetic avoids floating-point normalization and comparison in the model contract.

A valid growth seed is connected, has `n0 >= 2`, and has positive fitness at every vertex. Fix `m` with `1 <= m <= n0` for a growth trace. These assumptions imply positive old degrees and enough eligible targets at every step. A two-node edge is valid; the triangle is the default small fixture, not a compulsory seed.

The incoming fitness is an explicit positive rational input at each birth. The foundation conditions on these inputs; it does not claim that arbitrary supplied fitness traces are independent samples from a continuous distribution. No real-valued fitness distribution is silently rounded. An eventual discrete sampler must identify its distribution and quantization separately.

## 3. Attachment kernel

Freeze the old graph, its degrees, and its fitness before selecting targets. For an ordered prefix of already selected distinct old vertices with underlying set S, define

    w_i(G, eta, S) = 0                     if i is in S
                  = eta_i * degree_G(i)    otherwise
    Z(G, eta, S) = sum_i w_i
    p_i(G, eta, S) = w_i / Z               only when Z > 0.

Every draw is from remaining old vertices. The newborn is never eligible during its own birth. Its fitness affects its attractiveness on later births, not its fixed initial degree m or its own choice of targets.

The kernel is normalized separately after each selection. It is not m independent draws from the initial distribution, and probabilities must not be multiplied without conditioning on the prefix. Freezing degrees is explicit: only already selected old targets would have their degree increased, and they are excluded from later draws in this birth.

The low-level normalization helper may accept nonnegative finite weights with some zero entries. A zero entry receives zero mass. All-zero weights produce an explicit error, never a zero-sum “probability distribution” or a uniform fallback. Negative weights, invalid indices, or duplicate prefixes are errors. The valid connected positive-fitness growth state rules out zero normalization before all m draws finish.

## 4. Finite stochastic semantics, separate from entropy

Use an exact finite rational probability row with nonnegativity and sum-one evidence. Prove these fields from the raw weights and positivity of Z, rather than assuming the normalized answer in an input record. This is a finite kernel contract; an infinite random-process construction is not required for this slice.

For an ordered distinct target tuple (i_1,...,i_m), its probability is the product of the conditional rows. Prove total probability one over all ordered distinct m-tuples under the seed conditions. An unordered target set's probability is the sum over its orderings, not one selected ordering's product.

The executable boundary is deterministic replay of a supplied target trace, accompanied by its exact conditional probability. It is not yet a random network simulator. A PRNG, uniform-ticket implementation, seed format, and entropy-quality claims are out of scope. BA reduction guarantees equal conditional laws and equal replayed graph updates for the same target trace; it does not claim identical bits from unspecified random generators.

## 5. Atomic graph growth and stable identity

After all m targets have been validated, add exactly one vertex with ID n and exactly the m undirected edges from n to those targets. Old IDs are included naturally in `Fin (n+1)`. Old fitness values are retained exactly, and only vertex n receives the incoming fitness.

For the old-ID embedding e, the constructed adjacency must satisfy all four cases:

    G'.Adj (e u) (e v) <-> G.Adj u v
    G'.Adj (e u) new   <-> u is a selected target
    G'.Adj new (e v)   <-> v is a selected target
    not (G'.Adj new new).

Validate the entire request before producing a successor. Invalid fitness, m=0, m>n, wrong target count, duplicate targets, out-of-range IDs, or a disconnected/degenerate raw seed are rejected. No partial graph or partially extended fitness vector is committed on failure. Typed valid-state operations can exclude these cases by construction; a raw finite-ID adapter must return explicit errors.

Prove symmetry, looplessness, no repeated undirected edges, exact old-induced-subgraph preservation, and connectedness. Prove actual vertex count increases by one and actual undirected edge count by m, using the construction, not independent recurrence fields. The newborn has degree m; each selected old vertex gains one; each unselected old vertex keeps its degree. Consequently degree sum increases by 2m.

By iterating valid steps derive N_t = N_0 + t and E_t = E_0 + m*t. These formulas are for this growth profile, not the exponential-generation pseudofractal formulas. Existing exact and bounded walks lift through the old-ID embedding without increasing their budgets.

## 6. Compatibility and attractiveness laws

Prove the following generic contracts before any distribution fitting:

- Nonnegative probability, total mass one, and exact support on positive remaining weights.
- Constant positive fitness c reduces each conditional row to degree-only BA on the same candidate set. With constant fitness for every birth, the complete finite target-trace law and graph replay reduce to the corresponding BA profile.
- Multiplying all old fitness values by the same positive rational c leaves probabilities unchanged. Common rescaling changes representation, not the law.
- For positive eligible weights, p_i / p_j = (eta_i*k_i)/(eta_j*k_j). A less connected vertex can have higher selection probability when its fitness advantage exceeds the degree disadvantage.
- Raising one eligible vertex's fitness with graph, prefix, and other fitness values fixed cannot decrease its selection probability. Strict increase requires positive competing mass; with only one eligible vertex its probability remains one.
- Changing only the newborn's incoming fitness does not change the attachment row for that birth. Existing fitness values never change in a successor state.

These are local, exact properties. They do not prove that a particular late arrival eventually overtakes an incumbent, or establish a universal degree exponent or condensation theorem.

## 7. Small-world and epistemic boundaries

Generation, topology, and delivery remain separate. A BB-generated snapshot may be measured with existing finite mesh metrics, but no declared fitness vector is a six-hop certificate. Do not add hypothetical shortcut edges to a realized-transmission relation.

Include a negative scope fixture: start from a two-node edge, take m=1 and positive fitness, and repeatedly attach to the current endpoint until there are eight vertices. Every selected endpoint has positive probability; the resulting path has endpoints at distance seven, refuting a universal six-hop guarantee for this profile. The implementation must check both a seven-hop witness and the absence of a shorter route, not infer shortestness from a supplied walk alone.

No high-clustering guarantee, pure power-law exponent, logarithmic distance asymptotic, universal six-hop bound, condensation classification, or guaranteed local navigability is claimed. Adding triangle closure, aging, capacity constraints, deletion, directed edges, multiple relation types, or society preferences requires a separately identified extension and fresh validation.

Fitness means attractiveness for this attachment relation. It is not truth, evidence quality, observation permission, physical access, or permission to update another Agent's beliefs. This foundation neither changes Python social state nor connects itself automatically to V24 societies.

## 8. Acceptance examples and adversarial checks

For a triangle with IDs 0,1,2, degrees (2,2,2), and fitness (1,2,4), the exact weights are (2,4,8) and the first row is (1/7,2/7,4/7). After selecting 2, the second row is (1/3,2/3,0). For m=2,

    P[(2,1)] = (4/7)*(2/3) = 8/21
    P[(1,2)] = (2/7)*(4/5) = 8/35
    P[{1,2}] = 8/21 + 8/35 = 64/105.

Enumerating all six ordered pairs must sum to one. For this seed with constant fitness, the first row is uniform and every ordered pair has probability 1/6. Rescaling (1,2,4) by 3/2 must retain every conditional probability.

Replaying targets (2,1) with newborn fitness 3/2 yields four vertices, five edges, degrees (2,3,3,2), and fitness (1,2,4,3/2). The old triangle and old fitness are identical to the input. Changing the order to (1,2) yields the same graph but a different trace probability; graph output must not depend on selection order once the set is fixed.

Required negative tests cover all-zero normalization, zero/negative raw fitness, duplicate or out-of-range target IDs, wrong m/target dimensions, selecting the newborn, and invalid raw seeds. Cover m=1 and m=n, a one-positive-weight row, and a zero-degree candidate in the low-level kernel. Never add epsilon to degree or fitness as an undocumented repair.

A weighted-vector example with degrees (100,20) and fitness (1/10,1) gives p=(1/3,2/3), versus BA's (5/6,1/6). This is a kernel arithmetic fixture, not a purported two-vertex simple graph with those degrees.

## 9. Implementation surfaces and review boundaries

Proposed production module: `NarrativeDynamics/Core/FitnessAttachment.lean`, namespace `NarrativeDynamics.FitnessAttachment`. Keep finite-row and raw-validation helpers under `Internal`; do not introduce a general probability framework. Test module: `NarrativeDynamics/Tests/FitnessAttachment.lean`.

Implementation may add a root import and a named `Fitness attachment contract tests` workflow step. It must retain every existing Lean and Python check and record actual checkout SHA, event, run attempt, toolchain, and mathlib revision. No pinned dependency or research-lock edits belong to this model change.

The implementation plan follows review of this committed design. It should separate kernel normalization/BA reduction, conditional tuple law, atomic graph extension/invariants, and finite replay/counterexample verification into reviewable RED/GREEN units. This document is not approval to skip that review boundary or overwrite ongoing #60 changes.

## 10. Verification and trust statement

Use the repository's pinned Lean 4.32.0 and mathlib v4.32.0. Expected mathematical REDs are missing new declarations or violated model contracts; dependency failures and unrelated Python failures are not BB REDs. Require exact-head root build, named fitness tests, all pre-existing Lean regressions, conformance comparison, and axiom reports before marking an implementation unit complete.

Generic proofs must not use `sorry`, `admit`, new user axioms, `native_decide`, or unsafe escape hatches. Audit actual axiom dependencies; do not advertise ordinary classical proofs as axiom-free. Concrete tests should prefer ordinary kernel-checked proofs or equation-based `decide_cbv`; do not replace a failure with native evaluation without an explicit trust-boundary review. An executable replay check alone is not a generic theorem.

The existing proof workflow ignores spec/plan-only diffs. A missing run on this design is not a test pass. Any independent Python or World Studio failure remains visible; the BB change does not repair or suppress it. Keep the PR draft; no merge or auto-merge is authorized.

## 11. Sources and mathematical claims

[1] G. Bianconi and A.-L. Barabasi, “Competition and multiscaling in evolving networks,” Europhysics Letters 54, 436–442 (2001), DOI 10.1209/epl/i2001-00260-6; arXiv cond-mat/0011029. Source for fitness-weighted preferential attachment and heterogeneous growth; not a proof of the finite implementation.

[2] G. Bianconi and A.-L. Barabasi, “Bose-Einstein Condensation in Complex Networks,” Physical Review Letters 86, 5632–5635 (2001), DOI 10.1103/PhysRevLett.86.5632. Source for the need to distinguish fitness growth regimes; no condensation theorem is imported by this design.

[3] Inspected repository base: #58 at c268e5355b14fb408f30814ed1c2bee2f41f1144, especially `Core/SmallWorldMetrics.lean` and `.github/workflows/proof.yml`. Exact finite probabilities, graph-update identities, and the path counterexample above are proposed derivations/acceptance targets, not empirical fit claims or already machine-checked BB theorems.
