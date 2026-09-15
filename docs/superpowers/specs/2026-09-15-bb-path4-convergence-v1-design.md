# Exact path-four convergence after BB growth

Status: proposed design for written-spec review; no new Lean proof implemented.

## Goal and pinned evidence

Upgrade the finite observations of PR #77 into kernel-checked claims about the
existing rational Lean dynamics on this particular four-node path. Prove the
all-broadcast region is invariant, its degree-weighted mean is preserved, and
the three actual authored histories converge to two distinct rational values
when their beliefs are viewed in the reals.

Pin implementation/evidence base to #77 head
`96901a09bd2438a1829a427704b57e7d52b182a8`, tree
`7fcc32c2b508508904e8e00e88e6d26a37440127`.
The original four-round and idle-tail experiments are in
`examples/bb_birth_timing.py` and `examples/bb_birth_timing_tail.py`.
#77 remains an independent PR; this design is stacked on its branch. Retarget
only after its integration and recheck the diff. No merge is part of this design.

## Model and exact claim boundary

Use `NetworkPropagation.Population`, `propagate`, `FitnessABM.advance`, and the
existing checked BB replay interfaces. Beliefs/profiles remain `Rat`.
Use real casts only to state and prove limits of rational belief sequences.

The seed has nodes 0 and 1, unit fitness, edge (0,1), beliefs (1,0), zero
exposures, receptivity 1/2 and broadcast threshold 1/2. Births add node 2 attached
to node 1, then node 3 attached to node 2, each with unit fitness, zero belief
and the same profile. Birth ticks propagate once. BBII, BIBI and IIBB each end
at global tick 4; subsequent ticks are all idle. Profiles and graph remain fixed.

Formal conclusions apply to this rational model and these authored histories.
They do not establish arbitrary Python-float equivalence, heterogeneous fitness
effects, arbitrary graphs, random sampling, universal consensus, small-world
claims or real-world social prediction. Exposure counters may keep increasing;
only belief coordinates converge. Conditional attachment mass 1/8 is not a
probability distribution over the three authored timing schedules.

## Chosen proof route

Use an explicit four-coordinate decomposition. A general finite-graph spectral
or Markov-chain theorem would enlarge the abstraction and import surface without
being needed for this result. A numerical convergence tolerance alone would not
prove the requested limit. The selected route gives an algebraic, reviewable
formula and reduces the analytic part to decay of geometric sequences.

Let `AllBroadcast x` mean every coordinate is in [1/2,1]. For the canonical
undirected path, the all-broadcast belief update is

```
P(x) = ((x0+x1)/2,
        x0/4+x1/2+x2/4,
        x1/4+x2/2+x3/4,
        (x2+x3)/2).
```

The first bridge theorem must identify this operation with belief projection of
the actual `NetworkPropagation.propagate` for the path and stated profiles.
Incoming-neighbor sets must follow the real adjacency and broadcasting rule.
Do not assume an equality between a standalone toy matrix and the runtime.
The bridge should allow arbitrary valid exposure counters: they do not affect
belief updates or broadcasting in this model.

Prove `AllBroadcast` is preserved by the update and by iteration. Convexity
keeps every value above the inclusive broadcast threshold and below one.

Define the degree-weighted mean

`mu(x) = (x0 + 2*x1 + 2*x2 + x3)/6`.

Prove `mu(P(x)) = mu(x)`. Uniform mean is not the invariant on this path.

Use the following basis and eigenvalues; verify identities directly by rational
algebra rather than invoking a general spectral theorem:

| Mode | Coordinates | Eigenvalue |
| --- | --- | --- |
| u | [1,1,1,1] | 1 |
| v | [1,1/2,-1/2,-1] | 3/4 |
| w | [1,-1/2,-1/2,1] | 1/4 |
| z | [1,-1,1,-1] | 0 |

For any rational x, coefficients are

- `mu = (x0+2*x1+2*x2+x3)/6`;
- `a = (x0+x1-x2-x3)/3`;
- `b = (x0-x1-x2+x3)/3`;
- `c = (x0-2*x1+2*x2-x3)/6`.

Prove decomposition and the n-step closed form

`P^[n](x) = mu*u + a*(3/4)^n*v + b*(1/4)^n*w + c*0^n*z`.

Keep n=0 valid: natural powers give 0^0=1. The zero mode disappears only for
n>=1. Prove pointwise real `Tendsto` to `mu` by geometric decay, then transport
that theorem through the actual propagation bridge. Locate available mathlib
limit lemmas during implementation planning; no unverified lemma names are
assumed by this specification. A quantitative error bound is optional, not an
acceptance condition.

## Connect the actual BB histories

Create canonical rational fixtures corresponding to the three Python inputs,
and prove successful execution through existing `FitnessABM.replay`/typed
execution machinery. Check actual graph, profiles, node count and beliefs.
Do not start with unexplained activation vectors and call them replay results.
Existing private fixture helpers may require small local proof adapters; do not
refactor the earlier fixture corpus or expand its trust assumptions for reuse.

Required exact checkpoints:

| History | Global activation tick | First all-broadcast belief vector | mu |
| --- | --- | --- | --- |
| BBII | 6 | [45,44,43,35]/64 | 127/192 |
| BIBI | 6 | [45,44,43,35]/64 | 127/192 |
| IIBB | 7 | [183,180,177,147]/256 | 87/128 |

Certify the relevant earlier states are not all-broadcast, including global
tick 4 and the immediate predecessor of activation. Connect the fixed-size
tail to repeated `FitnessABM.advance`, using the real post-prefix state with
its graph and profiles. Prove BBII/BIBI belief equality for all common tail
lengths; do not claim complete history equality (their clocks/ages differ).

State final limits on a common global clock: for every i : Fin 4, as k tends
to infinity, the real cast of the belief at global tick `4+k` tends to 127/192
for BBII/BIBI and 87/128 for IIBB. Activation offsets are 2 and 3; handle finite
prefixes explicitly. At common tick `7+k`, BBII has taken k+1 all-broadcast
steps while IIBB has taken k. Comparing both after k activation-relative steps
would not by itself establish the requested common-clock result.

Derive the signed IIBB-minus-BBII pointwise difference limit as **7/384** and
prove it is positive. This is a precise permanent asymptotic separation claim
for the exact model, not just a finite inequality or a universal timing effect.

## Files and proof interface

Expected implementation scope:

- `NarrativeDynamics/Core/FitnessABMPath4.lean`: canonical path, all-broadcast
  bridge, invariant, decomposition and rational iteration results.
- `NarrativeDynamics/Core/FitnessABMPath4Convergence.lean`: real-cast limits and
  common-clock convergence machinery, keeping analytic imports separate.
- `NarrativeDynamics/Tests/FitnessABMPath4.lean`: checked actual histories,
  activation states, concrete limits, equality control and positive separation.
- `tools/check_fitness_abm_path4.sh`: bounded source/dependency/build gate.
- `.github/workflows/proof.yml`: add one bounded gate without changing event
  selection, checkout behavior, old proof gates or full Python discovery.

Existing runtime source, existing fixture corpora, toolchain and dependency pins
stay unchanged. Prefer direct imports of the new modules over broadening every
old core import unless repository conventions demonstrably require it.

The public dependency audit must explicitly require named reports for the
actual-propagation bridge, region invariance, mean invariance, iteration formula,
general all-broadcast limit, each of three checked activation fixtures, each of
three common-clock history limits, all-tail equality control and positive
separation limit. Missing reports fail the gate. Exact declarations and commands
will be frozen in the implementation plan rather than guessed as compiled APIs.

## Verification and acceptance

1. Prove all required conclusions with the unchanged allowlist `propext`,
   `Classical.choice`, `Quot.sound`. No `sorry`, custom axioms, `admit`, or native
   evaluation shortcuts in the proof dependency chain.
2. Keep finite literals bounded and separate from generic symbolic proofs.
   Use the existing 240-second per-phase wall bound and default heartbeats.
   A fixture-local finite recursion allowance requires the same justification
   as existing BB fixtures; never use an unlimited option to hide a proof stall.
3. Run a negative check by mutating a claimed activation vector or limit and
   observing Lean failure; restore and rebuild. Missing-report enforcement must
   also be exercised. Preserve exact RED/GREEN evidence and restoration state.
4. Existing default build, all earlier proof gates, full Python discovery and
   applicable World Studio CI must pass on the published implementation head.
   Record actual checkout revisions and merge-tree identity where applicable.
5. Inspect new theorem dependency reports and obtain independent implementation
   review. No code is considered proved merely because this design or Python
   experiment passed checks.

Local environment inspection found no available `lake` command. Actual Lean
execution must use an available authorized toolchain or ordinary GitHub CI.
Current design preparation checked all four rational mode identities and both
activation-vector decompositions with Python `Fraction`; this is arithmetic
cross-checking only, not Lean verification.

## Delivery boundary

This commit contains only the design. Next: written-spec review, then a concrete
implementation plan pinning theorem statements, fixture construction, CI units
and RED/GREEN commands. Production proof implementation starts after that
workflow's review checkpoints. The prior user approval establishes the goal;
the written specification makes the proof scope and evidence requirements
reviewable before introducing the new modules.
