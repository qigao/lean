# Birth timing in a fixed BB growth trace

## Question and controls

Can different growth histories yield different population states even when the
final network, static agent profiles, ordered births and total propagation rounds
are the same?

Run the actual checked runtime:

```bash
python -m examples.bb_birth_timing
python -m unittest tests.test_network_abm_bb_birth_timing -v
```

The first command emits JSON with every frame, including agent IDs, beliefs,
exposure counts, global ticks and epoch-local rounds. It also reports final
profiles, topology, exact conditional trace mass and each agent's participation
rounds. Values are produced by `replay_bb_population`, not a second simulator.

The seed is edge `0--1`, with initial beliefs `(1, 0)` and zero exposures. All
agents have receptivity `0.5`, broadcast threshold `0.5`, adoption threshold `0.5`
and role `peer`. Node 2 is born with belief zero and connects to node 1; then node
3 is born with belief zero and connects to node 2. Fitness is fixed at 1 for every
node. Channels are the existing unit bidirectional BB channels.

`B` means the next ordered birth **and one propagation round**. `I` means one
propagation round without a birth. Each schedule has two births, two idle ticks
and four total propagation rounds. The initial frame is tick zero, not a round.

| Schedule | Birth ticks for nodes 2, 3 | Final edges | Conditional BB mass |
| --- | --- | --- | --- |
| BBII | 1, 2 | (0,1), (1,2), (2,3) | 1/8 |
| BIBI | 1, 3 | (0,1), (1,2), (2,3) | 1/8 |
| IIBB | 3, 4 | (0,1), (1,2), (2,3) | 1/8 |

The exact mass is `(1/2) * (1/4)`: the first target has one of two equal seed
weights; the second target, node 2, has degree one among total degree four.
Idle propagation does not change topology or fitness. This is the probability
of the authored ordered attachment choices conditional on the fixed inputs,
not a probability distribution over the three timing schedules. Their masses
must not be summed as mutually exclusive outcomes of one timing experiment.

## Observed results at global tick four

All vectors below use numeric agent order `[0, 1, 2, 3]`.

| Schedule | Final beliefs | Final exposure counts | Participation rounds |
| --- | --- | --- | --- |
| BBII | [0.75, 0.6875, 0.625, 0.25] | [3, 5, 3, 1] | [4, 4, 4, 3] |
| BIBI | [0.75, 0.6875, 0.625, 0.25] | [3, 5, 3, 1] | [4, 4, 4, 2] |
| IIBB | [0.75, 0.75, 0.5625, 0] | [3, 4, 2, 0] | [4, 4, 2, 1] |

Exposure count measures received transmissions, not age or number of rounds.
An agent can receive multiple transmissions in one round, or none.
Participation rounds count the post-growth rounds in which the agent exists;
birth ticks count. Epoch-local round indices reset at births and cannot serve
as an agent's age.

For a hand check, the synchronous V1 rule here uses only prior-round broadcasters
(belief at least 0.5). With at least one incoming signal, a receiver updates to
half its prior belief plus half the mean incoming signal; otherwise it retains
its prior belief. A newborn starts with zero exposures.

For BBII, node 2 reaches 0.25 at tick two and 0.5 at tick three. It only transmits
that 0.5 at tick four, when node 3 reaches 0.25. Delaying node 3's birth from tick
two to tick three in BIBI removes one round with no incoming transmission, so
the final belief/exposure vectors remain identical despite different ages.

In IIBB, node 2 is born at tick three and reaches 0.375. At tick four it reaches
0.5625, but synchronous updates prevent relaying this newly acquired belief in
the same round. Node 3 therefore still has belief and exposure count zero.

## Interpretation and limits

This is a finite runtime witness that final topology, static profiles, initial
belief assignments and a global time budget do not alone determine the final
population state: the timing of growth matters in this example. The BBII/BIBI
pair is an equally important no-difference control. A timing change does not
necessarily change the outcome.

This comparison holds global rounds fixed, **not per-agent lifetime or exposure**.
Different lifetimes, intermediate networks and transmission opportunities are
mechanisms of the timing intervention; the experiment does not separate their
individual causal contributions. It does not establish persistent differences
after further propagation, asymptotic behavior or lack of eventual consensus.

All fitness values are equal and the attachment targets are authored. Therefore
this experiment neither measures heterogeneous fitness effects nor samples the
BB distribution, and cannot show that BB outperforms another growth model.
The result concerns growth/propagation composition along one admissible BB trace.

No new Lean theorem or conformance corpus is introduced. The runtime and its
existing proof gates are unchanged from merged base
`0d2d8bdd232379351c1b34bc1e40c01fbc5b8a68`. These Python example results do not
extend the existing finite conformance evidence to universal Python/Lean
equivalence, real-world social prediction, small-world bounds or consensus.

## Verification record

- Baseline: 15 existing replay tests passed before the experiment was added.
- RED: the new CLI test failed because `examples.bb_birth_timing` did not exist.
- GREEN: the CLI test passed against the implemented example and independently
  hand-derived result vectors. All 76 BB-related discovered tests passed.
- The test checks actual JSON output, common final topology/profiles/fitness,
  ordered birth targets, exact mass, global frame count, participation rounds,
  and final belief/exposure vectors, including the no-difference control.
- Local execution is recorded here; GitHub CI status belongs to the PR and must
  be checked separately for its published head.
