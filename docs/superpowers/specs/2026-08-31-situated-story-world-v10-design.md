# Situated Story World V10 Design

## Objective

Move the evolving network ABM from abstract information exchange into a situated,
physical story world. Every agent is an independent embodied actor located in one
place, acts from a private observation history, and can affect or learn about the
world only through explicit actions. The runtime retains an objective event ledger
and causal graph so an analyst can reconstruct what happened without exposing
global truth to agents.

V10 is a world and evidence kernel. It explains environmental preconditions,
conflicts, observation access, and explicit information provenance. Inferring hidden
motives or choosing actions from beliefs and goals is deferred to V11.

## World model

The fixed model contains:

- places forming the topological world;
- directed passages between places, with open/closed state;
- embodied agents with one initial place and inventory capacity;
- objects with one initial place, portability, and immutable evidence facts.

The immutable world state contains one body state per agent, one object state per
object, and one passage state per passage. Objects are either at exactly one place
or held by exactly one agent. A held object counts against that agent's inventory
capacity. Every non-initial state is parent-linked to the exact prior state hash.

V10 deliberately models topological places rather than continuous geometry. It has
no global chat room, shared memory, or privileged agent view.

## Actions and simultaneous rounds

Each agent submits at most one `SituatedActionIntent` per round. Missing intents are
implicit waits. V10 supports:

- `WAIT`: do nothing;
- `MOVE`: traverse one open outgoing passage;
- `LOOK`: privately inspect the visible contents of the current place;
- `INSPECT`: privately reveal an object's evidence facts when co-located or held;
- `TAKE`: acquire a portable co-located object when capacity allows;
- `DROP`: place a held object in the current place;
- `TELL`: make a local audible claim, optionally citing previously observed events.

All preconditions are evaluated against the prior snapshot. Independent successful
effects commit together. Conflicting `TAKE` actions for one object are resolved by
canonical action identity; losing events explicitly cite the winning event as their
direct cause. Input tuple ordering cannot affect the result.

One deterministic event is emitted for every explicit or implicit action. Event IDs
are derived from round and canonical sequence, never random allocation order.

## Event ledger and causal graph

Every `SituatedWorldEvent` records:

- event identity, round, and canonical sequence;
- action kind, actor, place, and target;
- success or failure plus a machine-readable outcome reason;
- public event details needed to reconstruct the result;
- direct cause event IDs.

An `INSPECT` event is grounded in the inspected object's evidence. A `TELL` event may
cite only events the speaker previously observed; those references become causal
edges. Conflict failures cite the winning action. Causal references must exist and
must precede their effect, making the event graph acyclic by construction.

The event ledger is an analyst artifact. It is not placed inside any agent state.

## Observation projection

`SituatedObservation` records that one agent received one event through `self`,
`visual`, `auditory`, or `inspection` access. Projection follows these rules:

- an actor observes its own action outcome;
- `LOOK` and `INSPECT` details are private to the actor;
- `TELL` is heard only by agents co-located with the speaker in the prior snapshot;
- `TAKE` and `DROP` are visible only to agents in the action place;
- `MOVE` is visible to agents at the origin or destination, using the prior snapshot;
- `WAIT` is private to the actor.

An agent perspective contains only projected observations and their referenced event
views. It contains no other agent's memory, belief, goal, inventory contents, or the
objective world state. Event occurrence, observation, interpretation, and belief
remain separate stages.

## Story run and queries

`SituatedStory` binds the exact model, initial world state, and a parent-consistent
sequence of round results. Advancing a story validates cited evidence against the
speaker's prior private observations. Replaying the same initial state and canonical
action schedule must reproduce identical states, events, observations, causes, and
content hashes.

The query surface supports:

- the objective event timeline;
- one agent's observation-limited timeline;
- direct causes and transitive causal ancestors of an event;
- an information-source chain ending in an inspection or local telling;
- a grounded event explanation containing outcome and world-rule reason.

These are structural answers, not unconstrained natural-language invention.

## Acceptance scenario: small office

An office has a lobby, open office, manager office, and records room. Alice, Bob,
Carol, and Dana begin in different places. A portable memo in the records room
contains an official restructuring fact.

1. Alice enters the records room and inspects the memo.
2. Alice moves to the open office and tells Bob, citing the inspection event.
3. Bob tells Carol locally, citing Alice's telling event.
4. Dana, elsewhere, receives neither telling.
5. Two agents attempting to take the memo simultaneously produce one deterministic
   success and one causally explained conflict failure.

The analyst can reconstruct the complete objective story and information chain.
Alice, Bob, Carol, and Dana each receive different, valid perspective timelines.

## Deferred work

- belief updates, interpretation, goals, motives, and POMDP action selection (V11);
- continuous geometry, line-of-sight occlusion, sound attenuation, and travel time;
- object containers, locks/keys, object creation, damage, and resource consumption;
- stochastic action outcomes or noisy observation;
- natural-language narration beyond structured grounded explanations;
- population birth/death integration with the situated world.
