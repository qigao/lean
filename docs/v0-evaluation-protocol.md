# V0 evaluation protocol boundary

## Purpose

V0 separates **plumbing evidence** from **connectome-topology evidence**. The synthetic protocol is frozen so that feature encoding, deterministic training, held-out evaluation, manifests, and CLI behavior can be reproduced. It is not evidence that FlyWire topology improves behavior recognition.

## Frozen synthetic protocol

`protocols/v0-synthetic.json` fixes:

- dataset: `synthetic-v0`;
- point-sequence/feature split hash and observation-schema hash produced by the V0 encoder;
- five declared seeds: 7, 11, 19, 23, 31;
- training budget: 20 epochs, 40 maximum updates, 50,000 parameter ceiling;
- primary metric: macro F1;
- secondary metrics: balanced accuracy plus the named noise/masking robustness metrics reserved for the robustness task;
- smoke success threshold: 0.60;
- rewiring algorithm identifier: `directed-double-edge-swap-v1`;
- final-test selection flag: false.

The 0.60 threshold is only a harness smoke gate. It must not be cited as evidence for a FlyWire topology advantage.

## Real-data topology template

`protocols/v0-real-template.json` is deliberately **non-executable**. The following values are null until measured/selected and frozen before any final-test run:

- dataset ID;
- YOLO/Pose version;
- split hash;
- observation-schema hash;
- exact FlyWire release/data identity;
- explicit FlyWire selection rule;
- success threshold.

The template already declares the comparison families and the rewiring algorithm so the intended falsification structure is visible, but the CLI rejects it while any required provenance field remains null.

## Claim boundary

A topology-specific claim requires a FlyWire arm and a matched rewired arm. The matched pair must use the same observation schema, split, and training budget. The final test cannot be used for checkpoint selection, graph selection, neuron/region selection, hyperparameter selection, or success-threshold selection.

The formal Lean contract proves this structural admissibility boundary. Empirical metric thresholds remain data and are never promoted to Lean theorems.

## Required sequence for a real run

1. Freeze the real dataset and split.
2. Freeze YOLO/Pose extraction version and the observation schema.
3. Freeze the FlyWire release and explicit subgraph selection rule without inspecting final-test outcomes.
4. Generate the matched rewired control with `directed-double-edge-swap-v1`.
5. Freeze seeds, budgets, primary/secondary metrics, and the success threshold.
6. Validate the protocol before training.
7. Use validation data for model/checkpoint selection only.
8. Evaluate the sealed final test once for the frozen comparison.

Negative or null results are valid outcomes. V0 does not permit changing topology selection or thresholds after seeing final-test results and then presenting the revised run as confirmatory evidence.
