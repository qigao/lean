# Feher/Hare Two-Stage V1 — Preregistration Verification Freeze

This file records metadata-only evidence for the repository revision that will be used to prepare the external preregistration bundle. It contains no raw human rows and does not report or compute FINAL behavioral results.

## Frozen source

- Upstream repository: `carolfs/muddled_models`
- Upstream revision: `4567763780a2c596fd6510af720ec468a8214a8f`
- Source manifest hash: `sha256:3609e980af172823cfef78290e7f2337fdb337d67f1f27641e17d473aeedd10d`
- Source snapshot hash: `sha256:25bdc2e4bff4110f38b098b89e7d59aa38c9658727fcc89a38d50e107d243186`
- Verified files: 90 total = 45 scientific-evidence files + 45 matching transform-metadata files
- Practice/questionnaire files: excluded

## Real-source conformance

- Magic Carpet eligible participants: 24
- Spaceship eligible participants: 21
- Total eligible participants: 45
- Magic Carpet split: 14 TRAIN / 5 SELECTION / 5 FINAL
- Spaceship split: 13 TRAIN / 4 SELECTION / 4 FINAL
- Total split: 27 TRAIN / 9 SELECTION / 9 FINAL
- Retained trials: 10,074
- `slow == 1` excluded trials: 133
- Structural exclusions: 0

## Frozen identities from source preparation

- Transform report hash: `sha256:2fb8ab6dc796a9e4ece5653a5485d865ef44dd0f5f7dff92d94a904932d6e541`
- Participant assignment hash: `sha256:fc144c35f6713e27f75141d678872cc04ca44c7a0fd8e109e8618c72b4111ccf`
- Dataset hash: `sha256:17789130372d7eace05e1216a57bdae2ffbd519960333ffe814aee2d2d404781`
- FINAL target hash: `sha256:927e1727d37993e9a5c887f79ac8712d07a155622deacf0a3122d41f90b16ed2`

The real-source preparation was executed against implementation revision `97dd0c7a7a9acbe50a491f01b7342172b7a6271c` before the metadata-only source-manifest commit. The later manifest/freeze commits do not change study code, transform logic, model code, or the generated source manifest contents.

## Scope lock

- Claim scope remains `external_observational_predictive_only`.
- No participant-specific or task-specific fitted parameters are introduced.
- TRAIN and SELECTION remain Brier-only.
- FINAL is not an implementation/CI acceptance test.
- No real FINAL model execution has occurred at this freeze point.
- Any later change to scientific code, source manifest, transform, candidates, protocols, releases, or the preregistration bundle requires a new study revision and renewed external registration.
