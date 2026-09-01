# V17 final fix report

Date: 2026-09-01

Branch: `feature/narrative-projection-v17`

Base: `09bdd90`

Commit: this final-fix commit, `fix(abm): complete narrative projection integrity`

## Scope

This single final wave resolves every whole-branch V17 final-review finding:

1. canonical presentation order and chronological validation;
2. objective-cut private-entitlement ownership;
3. event-only selection forcing, complete candidate identity, and consistent causal-payoff weighting;
4. complete cognitive, recall, social, objective-event, and private-percept provenance;
5. optional unknown place metadata and non-grouping;
6. the deferred support-triple, duplicate multi-POV validation, and multi-POV causal non-disclosure minors.

No unrelated code was changed.

## TDD evidence

### 1. Presentation order, chronology, hashes, and private ownership

RED command:

```powershell
python -m pytest tests/test_network_abm_situated_projection_contracts.py -q
```

Observed RED:

- 4 failed, 24 passed, 9 subtests passed.
- Reversed chronological cut order was accepted.
- Top-level beat and scene catalog permutations were accepted.
- An Alice-active objective beat could cite Bob's private entitlement.
- The known-entitlement/mismatching-support-triple regression already passed because triple-level support validation existed.

GREEN command:

```powershell
python -m pytest tests/test_network_abm_situated_projection_contracts.py -q
```

Observed GREEN: 28 passed, 9 subtests passed.

Implementation:

- `cut.scene_ids` is required to equal the top-level scene order.
- Flattened `scene.beat_ids` is required to equal the top-level beat order.
- Chronological projections are checked against `(round_index, phase, sequence, active_pov-or-empty, beat_id)`.
- Any cited private entitlement must be owned by the beat's non-null active POV, including under objective authority.
- Objective entitlements remain freely citeable.
- The duplicate multi-POV cardinality check was removed while preserving its public error contract.

### 2. Selection eligibility, identity, payoff weighting, and multi-POV causality

RED command:

```powershell
python -m pytest tests/test_network_abm_situated_projection.py -q -k "internal_candidates_are_not_boundary_or_event_gap_anchors or selection_does_not_conflate_support_versions_with_same_artifact_id or causal_payoff_weight_controls_selection_kind_and_salience or multi_pov_perceived_endpoints_do_not_disclose_objective_causal_edges"
```

Observed RED: 3 failed, 1 passed, 30 deselected.

- Low-salience internal beats were boundary/gap-forced.
- Two support versions sharing one artifact ID were conflated.
- `CAUSAL_PAYOFF` policy weight did not select the payoff candidate.
- The dedicated multi-POV causal non-disclosure regression already passed through the existing private extraction boundary.

GREEN command:

```powershell
python -m pytest tests/test_network_abm_situated_projection.py -q -k "internal_candidates_are_not_boundary_or_event_gap_anchors or selection_does_not_conflate_support_versions_with_same_artifact_id or causal_payoff_weight_controls_selection_kind_and_salience or multi_pov_perceived_endpoints_do_not_disclose_objective_causal_edges"
```

Observed GREEN: 4 passed, 30 deselected.

Implementation:

- Candidates now explicitly record whether they originate from an event/percept.
- First/last and omission-gap forcing operate only on event-origin candidates in each POV lane.
- Internal candidates are selected only by declared salience.
- Candidate identity includes source, round, phase, sequence, place, POV, agents, full primary/additional support triples, entitlement hash, cause IDs, magnitude, and origin class.
- Objective candidates with authorized direct causes use `CAUSAL_PAYOFF` consistently for selection weight, emitted kind, and emitted salience.
- Private percept candidates continue to carry no objective causal edges.

Self-review found an additional beat-ID identity edge. The tightened regression produced 1 expected failure because same-support candidates at different sequence positions shared a beat ID; deriving beat IDs from the full candidate identity made the focused test pass.

### 3. Complete trajectory provenance and private decision metadata

RED commands:

```powershell
python -m pytest tests/test_network_abm_situated_projection.py -q -k "internal_places_and_metadata_cite_complete_trajectory_provenance or limited_decision_metadata_comes_only_from_exact_self_percepts or limited_decision_without_exact_self_percept_stays_event_unbound"
python -m pytest tests/test_network_abm_situated_projection.py::test_limited_decision_without_exact_self_percept_stays_event_unbound -q
```

Observed RED:

- The first run exposed the expected missing mind/percept supports plus a test-fixture nesting error; the fixture was corrected before production changes.
- The corrected adversarial test then failed as expected: the forged decision still emitted objective source event `r0006:e0002` instead of `None`.

GREEN command:

```powershell
python -m pytest tests/test_network_abm_situated_projection.py -q -k "internal_places_and_metadata_cite_complete_trajectory_provenance or limited_decision_metadata_comes_only_from_exact_self_percepts or limited_decision_without_exact_self_percept_stays_event_unbound"
```

Observed GREEN: 3 passed, 34 deselected.

Implementation:

- Belief and reversal beats cite the exact prior cognitive mind whose place they use.
- Reversal beats additionally cite the exact prior decision supplying the prior selected action.
- Recall beats cite the exact `recall.prior_mind` hash.
- Claim and relationship beats cite the exact next cognitive mind supplying their destination place.
- Objective decision beats cite the exact accepted world event used for source metadata.
- Limited/multi decision beats never resolve objective events for metadata. They bind only a unique exact same-round owner self-percept, cite it, and take source event/round from it.
- A private decision without that exact percept stays event-unbound and cites neither a world event nor a percept.
- Private decision sequences are deterministic internal phase sequences and do not reuse objective event sequence.

### 4. Optional unknown place

RED command:

```powershell
python -m pytest tests/test_network_abm_situated_projection_contracts.py::NarrativeProjectionContractTests::test_unknown_place_is_optional_content_addressed_metadata tests/test_network_abm_situated_projection.py::test_detected_percept_exposes_signal_without_actor_kind_or_outcome tests/test_network_abm_situated_projection.py::test_unknown_location_beats_each_start_their_own_scene tests/test_network_abm_situated_projection.py::test_known_world_place_named_undisclosed_preserves_normal_grouping -q
```

Observed RED: 3 failed, 1 passed.

- Contract construction rejected `None` place metadata.
- Detected percepts emitted the string sentinel `undisclosed`.
- Multiple unknown beats merged into one scene.
- A legitimate world place ID `undisclosed` already grouped normally.

GREEN command: the same four-test command.

Observed GREEN: 4 passed.

Implementation:

- `NarrativeBeat.place_id` and `NarrativeScene.place_id` are optional.
- Detected percepts with no disclosed place emit `None`.
- Any unknown-place beat starts a new scene.
- Known locations, including a real place ID `undisclosed`, retain normal grouping.
- Serialization and content hashing retain explicit JSON `null` deterministically.

## Changed files

- `narrative_dynamics/abm/situated_projection_contracts.py`
- `narrative_dynamics/abm/situated_projection.py`
- `tests/test_network_abm_situated_projection_contracts.py`
- `tests/test_network_abm_situated_projection.py`
- `README.md`
- `.superpowers/sdd/2026-09-01-narrative-projection-v17/final-fix-report.md`

## Final verification

```powershell
python -m pytest tests/test_network_abm_situated_projection_contracts.py tests/test_network_abm_situated_projection.py tests/test_network_abm_public_api.py -q
```

Result: 70 passed, 9 subtests passed in 12.05s; exit 0.

```powershell
python -m unittest discover -s tests -p "test_network_abm*.py"
```

Result: 382 tests passed in 6.089s; exit 0.

```powershell
python -m compileall -q narrative_dynamics
```

Result: exit 0, no output.

```powershell
git diff --check
```

Result: exit 0. Git printed only working-copy LF-to-CRLF conversion warnings; there were no whitespace errors.

## Self-review

- Re-read all final findings against the final diff.
- Confirmed presentation order is validated before authored/chronological policy-specific checks.
- Confirmed objective authority enforces private ownership without restricting objective entitlements.
- Confirmed selection forcing cannot count internal candidates and causal coverage does not widen private authority.
- Confirmed selection identity and beat IDs use content-aware complete candidate identity.
- Confirmed every newly used place/source metadata field has exact cited provenance.
- Confirmed limited/multi decision metadata contains no objective event support or sequence lookup.
- Confirmed unknown location cannot equal or group with any known place, including `undisclosed`.
- Confirmed public constructor compatibility is preserved for existing string places.
- Confirmed README documents the changed public order and optional-place semantics.
- Confirmed no machine-local database paths or prose generation entered hashes or output.
- Confirmed the worktree contains only in-scope V17 changes.

Concerns: none. The only command noise is Git's pre-existing Windows LF-to-CRLF working-copy warning.
