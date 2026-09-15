# Equal post-growth propagation tails

## Controlled extension

This extends [the four-round experiment](bb-birth-timing-experiment.md) after
merged PR #76, base `11f0cf15d17d9a5049d70041252a490f21ebaf29`.
The three authored schedules BBII, BIBI and IIBB retain the same seed, static
profiles, ordered births, fixed unit fitness, final path 0--1--2--3 and exact
conditional attachment mass 1/8. Each receives 64 more idle propagation ticks,
with observations after 0, 1, 2, 4, 8, 16, 32 and 64 extra rounds.

```bash
python -m examples.bb_birth_timing_tail
python -m unittest tests.test_network_abm_bb_birth_timing_tail -v
```

The example uses the existing checked runtime to replay each complete history
from its seed once, then samples its frames. It adds no resume API, sampling
kernel or production propagation rule. The original command
`python -m examples.bb_birth_timing` retains its original four-round output.

The between-history gap is the maximum absolute difference between corresponding
node beliefs over every pair of histories. Within-history spread is the largest
minus smallest node belief in one history. New exposures subtract each history's
own global-tick-four counts. Broadcasting flags are derived from the unchanged
V1 rule, belief >= the agent's broadcast threshold (0.5 here).

## Observed results

Values below are rounded for readability; JSON preserves Python float precision.
BBII and BIBI have identical belief vectors at every reported checkpoint.

| Extra rounds | Global tick | Max between-history gap | BBII/BIBI internal spread | IIBB internal spread |
| --- | --- | --- | --- | --- |
| 0 | 4 | 0.25 | 0.5 | 0.75 |
| 1 | 5 | 0.15625 | 0.28125 | 0.46875 |
| 2 | 6 | 0.078125 | 0.15625 | 0.2578125 |
| 4 | 8 | 0.0244140625 | 0.064453125 | 0.076171875 |
| 8 | 12 | 0.02005767822 | 0.02039337158 | 0.02410125732 |
| 16 | 20 | 0.01841476990 | 0.002041639877 | 0.002412847127 |
| 32 | 36 | 0.01823102690 | 0.00002046253117 | 0.00002418299138 |
| 64 | 68 | 0.01822916685 | 2.05551e-9 | 2.42924e-9 |

At global tick 68, BBII/BIBI beliefs are all approximately 0.6614583333;
IIBB beliefs are all approximately 0.6796875000. Each group is internally close,
yet the two groups remain separated at this finite horizon.

New exposure vectors at that checkpoint, in node order [0,1,2,3], are
[64,128,126,64] for BBII/BIBI and [64,128,125,64] for IIBB.
Accumulated exposures are not a consensus criterion. They can retain differences
from earlier transmissions even when belief differences shrink.

## Mechanism and a candidate mathematical explanation

At the four-round baseline, node 3 does not broadcast in any history. It crosses
the threshold after two additional rounds in BBII/BIBI and three in IIBB.
Thus the histories briefly retain different sets of active senders after growth
has stopped. By the checkpoint at four additional rounds all nodes broadcast.

Once all nodes broadcast, the exact-arithmetic counterpart of this unit-channel,
receptivity-1/2 update on the undirected path is

`x_i(next) = x_i/2 + (sum of neighbor beliefs)/(2 * degree_i)`.

Its degree-weighted sum is invariant: multiplying by degree and summing counts
each neighboring value degree-many times. The weighted mean has weights
[1,2,2,1]/6, not uniform weights. Updates are convex combinations; once all
beliefs are at least the threshold, the exact rule keeps them there.

Hand calculation at first all-broadcast frames gives:

- BBII/BIBI, extra round 2: [45,44,43,35]/64, weighted mean **127/192**.
- IIBB, extra round 3: [183,180,177,147]/256, weighted mean **87/128**.
- Difference of these rational means: **7/384**, approximately 0.01822916667.

The observed later values are consistent with these distinct means. This
provides a precise candidate for a follow-up proof: threshold activation,
invariance of the all-broadcast region and degree-weighted mean, followed by
convergence of this fixed finite averaging operator. The algebra here is a
hand derivation, **not a new Lean-checked theorem** or a proof that Python floats
exactly implement the rational model for arbitrarily many rounds.

## Scope and verification

This finite run supports the narrower claim that substantial between-history
differences remain at tick 68 while within-history spreads are small. It does
not itself prove asymptotic convergence, permanent separation, a universal
timing effect, or real-world consensus. The initial histories still have
different per-agent lifetimes and prior exposures. Equal fitness and authored
targets do not identify a distinctive heterogeneous-BB-fitness effect.

Tests exercise both CLI output and the extension helper. They check the original
prefix is preserved, equal added participation rounds, unchanged topology and
mass, hand-derived first-tail-round signals and exposure increments, threshold
activation, clocks, invalid tail lengths and the bounded final observations.
Weighted-mean expectations come from the hand-derived all-broadcast frames.

Initial RED consisted of a missing tail CLI failure and the unsupported helper
keyword error. GREEN passed both new tests and the original CLI test. CI results
for the published revision are recorded separately in the PR.
