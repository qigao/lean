# Simulated Story Production Architecture Design

## Objective

Turn the V10-V14 situated multi-agent simulator into a constrained story-production
system that can eventually emit a novel chapter or screenplay without allowing an
LLM to invent world facts, leak private knowledge, or bypass agent autonomy.

The simulator remains the authority for what happens. Language models may interpret
utterances, propose admissible actions or interventions, select an authorized view
of the event history, and realize that view as prose or screenplay. Every accepted
model output becomes a validated, content-addressed artifact before it can influence
another stage.

This design covers the end-to-end architecture. Its subsystems are intentionally
delivered as separate versions because environment perception, semantic grounding,
story projection, rendering, and branch direction have independent correctness
boundaries.

## Product distinction: world, story, and telling

The system separates three products that ordinary generative-story systems often
collapse:

1. **World history (fabula):** the objective parent-linked V10-V14 trajectory of
   places, actions, events, observations, beliefs, memories, claims, and
   relationships.
2. **Story projection (plot):** an immutable selection and ordering of supported
   events into beats and scenes for one authorized narrator or point of view.
3. **Narrative realization (discourse):** prose, dialogue, scene headings, and
   screenplay action generated only from a story projection.

```text
typed environment graph
        |
        v
deterministic world + independent agents  ----> objective truth ledger
        |                                              |
        v                                              v
private percepts -> memory/belief/social revision   story projection
        |                                              |
        v                                              v
grounded language artifacts                        scene packets
                                                       |
                                                       v
                                            LLM prose/screenplay renderer
                                                       |
                                                       v
                                            continuity/fidelity validation
```

An LLM-rendered sentence never becomes a historical fact merely because it appears
in output text. Only a validated action or intervention committed through the world
transition may create a new event.

## Existing foundation

The design builds on the current layers rather than replacing them:

- V10 provides topological places/passages, embodied actions, objective events,
  private observation records, causal references, and deterministic replay.
- V11 provides private Bayesian belief and finite-horizon POMDP-like action choice.
- V12 stores agent-private immutable memories in SQLite with a rebuildable FTS5
  search index.
- V13 recalls early private events at later checkpoints and lets them alter later
  beliefs and physical decisions.
- V14 consolidates social claims, preserves revisions, learns directed source trust
  and affinity, forgets bounded unresolved claims, and trust-weights later recall.

The principal missing boundaries are richer graph-derived perception, grounded
natural language, scene/beat projection, constrained rendering, and optional
branch-level direction.

## Authority model

Every artifact belongs to exactly one authority class:

| Authority | May assert | Must not assert |
|---|---|---|
| World transition | Physical state and objective events | Hidden motives or prose-only facts |
| Perception projection | What one agent could detect | Unperceived event details |
| Cognitive/social transition | Beliefs, decisions, claims, relationships | Objective truth not in private evidence |
| Semantic grounder | Candidate meaning of observed language | Truth, physical effects, undeclared entities |
| Story projector | Which supported facts appear in a cut | New actions, dialogue, knowledge, or state |
| Narrative renderer | Surface wording of authorized facts | New canonical facts or private-information leaks |
| Director | Candidate interventions or branch choices | Silent mutation of an accepted trajectory |

Crossing an authority boundary requires a typed validator. No component receives a
larger information scope merely because it uses an LLM.

## V15: perceptual environment graph

### Overlay rather than V10 mutation

V10 places and directed passages already form a movement graph. V15 adds a separate
typed perception overlay bound to the exact V10 world-model hash. This preserves all
V10-V14 hashes and lets one physical boundary have different effects on movement,
vision, sound, and direct interaction.

The overlay contains directed edges for:

- `visibility`: cumulative non-negative visual cost;
- `auditory`: cumulative non-negative sound loss;
- `interaction`: a direct, non-transitive reach relation.

An edge may always be active or may depend on one V10 passage being open or closed.
This permits an open-door visual edge, separate open- and closed-door acoustic loss,
and interaction only through an open doorway without changing passage semantics.

Agent perception profiles declare a maximum visual cost plus detectable and clear
sound thresholds. Event signal profiles declare whether an action is visually
observable and, when applicable, its emitted sound intensity.

### Derived access

For each event, V15 uses the prior-round world snapshot:

- visibility uses deterministic shortest cumulative visual cost;
- audibility uses `received = source_intensity - shortest_sound_loss`;
- interaction is true only for co-location or one active direct interaction edge;
- the actor always receives its own outcome;
- inspection remains exact and private to the actor.

Sound at or above the clear threshold can disclose the utterance. Sound at or above
only the detectable threshold discloses that a sound occurred, not its message. A
visual path may identify an actor and action without disclosing private details.

### Sanitized percepts

The existing `SituatedPerspectiveEvent` joins an observation to the complete exact
event and is therefore unsuitable for partial hearing. V15 introduces an immutable
`SituatedPercept` that references the objective event ID/hash but contains only the
fields disclosed to that observer:

- observing agent and round;
- source event ID/hash;
- one or more access channels;
- fidelity `detected`, `identified`, or `exact`;
- optional observed actor, action kind, place, outcome, and disclosed details.

The full objective event is never embedded in a partial percept. Later cognition,
memory, semantic grounding, and narration consume the sanitized percept, while the
analyst may separately join it to the truth ledger.

V15 first ships as a query/projection layer. A follow-up integration plan migrates
V11-V14 observation admission and V12 memory records to sanitized percept views.
This split makes leakage tests independently reviewable before percepts can change
agent behavior.

## V16: grounded natural language

### LLM role

The language model is a semantic compiler, not a transition function. It receives:

- one exact or partial private percept;
- the requesting agent's relevant private memories and active claims;
- a finite allowlist of agent, place, object, topic, predicate, and value IDs;
- an output JSON schema and explicit evidence budget.

It returns a `SemanticGroundingArtifact` containing zero or more candidate
`GroundedClaim` values. Each claim records subject, predicate, value, polarity,
modality, temporal scope, speaker/source, confidence, and supporting percept or
memory IDs.

The validator rejects:

- unknown or out-of-scope entity IDs;
- evidence not privately available to the observer;
- unsupported temporal references;
- exact message claims derived from a merely detected sound;
- claims that omit provenance;
- provider output outside the schema.

The artifact says "this utterance was interpreted as X". It does not say X is true.
Accepted claims enter the existing V14 social-revision path or a later open claim
graph; physical effects still require an action.

### Prompt-injection boundary

Character dialogue and retrieved memories are quoted data, never instructions. The
provider interface exposes no tools and receives canonical JSON rather than an
unescaped prompt transcript. Output IDs must come from the supplied allowlists.
Unknown meaning is a valid empty/uncertain result and is preferred to invention.

### Replay

An online LLM call is not assumed deterministic. The first accepted response is
stored with provider ID, model ID, schema hash, prompt-template hash, private-context
hash, raw response hash, validation result, and content hash. Replay consumes that
artifact and performs no provider call. Thus the simulation becomes deterministic
at the accepted semantic-artifact boundary.

## V17: story projection and scene graph

V17 converts an accepted trajectory into a non-generative narrative structure. A
`NarrativeProjectionPolicy` declares:

- narrator authority: objective, named-agent limited POV, or declared multi-POV;
- chronological or explicitly authored temporal ordering;
- salience weights and scene-break rules;
- maximum event omission gap and required causal coverage;
- whether beliefs, memories, claim revisions, and relationship changes may appear.

### Beats

A `NarrativeBeat` references exact supporting artifact hashes and classifies one
change such as:

- physical action or environment transition;
- information discovery or propagation;
- material belief shift;
- selected-goal/action reversal;
- claim creation, supersession, confirmation, contradiction, or forgetting;
- directed trust/affinity change;
- causal payoff of an earlier event.

Salience is computed from declared deterministic features, not from LLM taste alone.
Examples include belief-distance change, goal-value change, relationship delta,
claim terminality, causal fan-out, intervention membership, and first/last occurrence.

### Scenes and cuts

Beats are grouped into `NarrativeScene` values by place, time continuity, active POV,
and causal adjacency. A `NarrativeCut` orders scenes and carries a complete
entitlement bundle: the exact facts, private knowledge, dialogue text, and internal
state the renderer may mention.

A limited Bob cut cannot include Alice's private inspection until Bob learns it.
An objective cut may reveal both, but must still preserve their different knowledge
states. Flashbacks reorder supported scenes; they do not rewrite event time.

## V18: constrained LLM rendering

The renderer consumes one immutable scene packet at a time and returns a structured
`RenderedSceneArtifact`. Two adapters share the same truth bundle:

- **Novel:** POV, person, tense, interiority permissions, prose paragraphs, and
  dialogue.
- **Screenplay:** slugline, time/place, visible action, character cues, dialogue,
  and optional transition; camera directions are disabled by default.

Every factual sentence, action line, dialogue line, and permitted interior statement
must cite one or more entitlement IDs. Stylistic connective text may have no citation
only when it asserts no entity, action, state, knowledge, or causal fact.

The continuity validator rejects output that:

- moves a character without a supporting event;
- gives a character information outside the selected POV entitlement;
- changes names, places, time, object ownership, or claim status;
- creates uncited dialogue that is treated as a historical utterance;
- converts uncertain belief into objective truth;
- contradicts a terminal or superseded claim timeline;
- refers to a later event before an authorized flash-forward.

Rejected output is never canonical. A retry is a new candidate; acceptance stores
the exact provider and validation artifacts. Rendering never feeds back into the
world unless a separately validated diegetic dialogue proposal is submitted as a
future `TELL` intent.

## V19: director and branch search

The director is optional and operates above deterministic simulation. It may propose
only typed, authorized interventions such as opening a passage, placing an object,
changing an exogenous signal, or scheduling an arrival. Each proposal is validated,
assigned an intervention ID, and run as a separate branch from an exact checkpoint.

The branch evaluator scores declared objectives such as:

- causal coherence and payoff;
- character-agency preservation;
- conflict/escalation and later resolution;
- POV information asymmetry;
- relationship or belief change;
- scene diversity and bounded duration;
- divergence from a control trajectory.

Selection records every candidate branch hash, score component, and chosen branch.
The director cannot edit an event after it happens or choose an agent's action. It
changes circumstances, runs the agents, and selects among resulting histories.

## Character-generated dialogue

Dialogue that affects the simulated world follows a different path from rendered
dialogue:

```text
private agent context
    -> LLM dialogue/action proposal
    -> schema + entity + evidence validation
    -> SituatedActionIntent(TELL)
    -> deterministic world transition
    -> graph-derived private percepts
    -> grounded claims and later cognition
```

The proposal must cite only events the speaker observed, exactly as V10 already
requires. A post-hoc screenplay paraphrase may improve wording but cannot change the
canonical utterance meaning or become new evidence.

## Artifact and storage boundaries

The objective trajectory and accepted semantic/narrative artifacts are immutable.
Suggested content-addressed artifacts are:

- `SituatedPerceptualProjection`;
- `SemanticGroundingArtifact`;
- `NarrativeProjection` containing beats, scenes, and a cut;
- `RenderedSceneArtifact` and `RenderedNarrativeArtifact`;
- `NarrativeBranchEvaluation`.

SQLite may store provider responses, projections, and renderings, but database paths
never enter content hashes. FTS5 is used only as a rebuildable retrieval index. The
authoritative row stores the exact source hashes and accepted structured payload.

## End-to-end acceptance scenario

An office graph has an open workspace, corridor, glass meeting room, records room,
and two doors with separate movement, visual, and acoustic behavior.

1. Alice privately inspects an approved restructuring memo in records.
2. She tells Carol in the corridor using an ambiguous qualified sentence.
3. Bob sees Alice through glass but, behind a closed door, detects only fragments of
   the conversation and receives no exact message detail.
4. V16 grounds different private interpretations for Carol and Bob with exact
   provenance; neither receives Alice's inspection fact directly.
5. Their V11-V14 beliefs, actions, memories, claims, and directed trust evolve over
   later rounds. Dana independently verifies the memo and resolves one claim.
6. V17 creates an objective cut and a Bob-limited cut. The limited cut withholds the
   memo truth while preserving clues Bob actually perceived.
7. V18 renders both a novel scene and screenplay scene. A candidate line revealing
   the memo to Bob is rejected by continuity validation.
8. V19 compares a control branch with an intervention that leaves the meeting-room
   door open, exposing how one environmental edge changes information flow, belief,
   relationships, actions, scenes, and ending.

## Evaluation gates

Each version must retain all earlier deterministic regression tests and add:

- **privacy:** no field unavailable in a private percept enters cognition or text;
- **grounding fidelity:** every semantic claim cites admissible private evidence;
- **causal fidelity:** every selected beat has exact support and causal ancestry;
- **continuity:** location, time, knowledge, inventory, claims, and relationships do
  not contradict the selected trajectory;
- **replay:** accepted artifact replay performs no LLM call and reproduces hashes;
- **counterfactual isolation:** branch artifacts never contaminate another branch's
  memories, claims, or renderings;
- **style independence:** different render styles share the same factual cut hash.

## Delivery sequence

1. **V15 perceptual environment graph:** typed visual/auditory/interaction overlay,
   deterministic reach, and sanitized private percept projection.
2. **V15 cognition/memory integration:** migrate private admission and memory to
   percept views without changing objective events.
3. **V16 grounded language:** provider protocol, schemas, private RAG context,
   validation, accepted-artifact replay, and V14 claim bridging.
4. **V17 narrative projection:** deterministic beats, scenes, cuts, POV entitlement,
   and salience/causal coverage.
5. **V18 constrained rendering:** novel and screenplay adapters plus continuity
   validation and content-addressed replay.
6. **V19 director/branch search:** typed interventions, checkpoint branching,
   explicit scoring, and branch selection.

Each item produces useful testable software by itself. The first implementation plan
covers only the V15 perception projection foundation; subsequent plans must preserve
the boundaries defined here.

## Deferred work

- continuous 3D geometry, ray tracing, fluid acoustics, and photorealistic rendering;
- unrestricted open-world entities or predicates created by an LLM;
- hidden model chain-of-thought as a canonical character state;
- an LLM directly mutating world, mind, memory, claim, or relationship state;
- silent retconning of accepted histories;
- autonomous publication or external distribution of generated narratives.
