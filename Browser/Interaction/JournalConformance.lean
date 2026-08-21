import Browser.Interaction.Conformance
import Lean.Data.Json

namespace Browser.Interaction

open Lean

abbrev InteractionLaneId := Nat

private structure JournalHeader where
  kind : String
  version : Nat
  deriving FromJson

private structure JournalEnvelope where
  kind : String
  seq : Nat
  lane : InteractionLaneId
  deriving FromJson

structure InteractionJournalState where
  lastSeq : Nat := 0
  lanes : InteractionLaneId → InteractionTraceState := fun _ => {}

private def laneState (state : InteractionJournalState) (lane : InteractionLaneId) : InteractionTraceState :=
  state.lanes lane

private def updateLane
    (state : InteractionJournalState)
    (lane : InteractionLaneId)
    (value : InteractionTraceState) : InteractionJournalState :=
  { state with lanes := fun q => if q = lane then value else state.lanes q }

private def verifyJournalEvent
    (state : InteractionJournalState) (json : Json) : Except String InteractionJournalState := do
  let envelope : JournalEnvelope ← fromJson? json
  if envelope.kind != "interaction" then
    throw s!"unexpected journal record kind: {envelope.kind}"
  if envelope.seq ≤ state.lastSeq then
    throw s!"interaction journal sequence is not strictly increasing: {envelope.seq}"
  let event ← decodeInteractionJson json
  let current := laneState state envelope.lane
  if !interactionEventAllowed current event then
    throw s!"interaction not authorized in lane {envelope.lane}: {repr event}"
  let next := applyInteractionTraceEvent current event
  if !interactionTraceSafe next then
    throw s!"interaction invariant violated in lane {envelope.lane}: {repr event}"
  pure { (updateLane state envelope.lane next) with lastSeq := envelope.seq }

private def verifyJournalRecords :
    InteractionJournalState → List String → Except String InteractionJournalState
  | state, [] => pure state
  | state, line :: rest => do
      let json ← Json.parse line
      verifyJournalRecords (← verifyJournalEvent state json) rest

/-- Verify the Driver-facing multiplexed InteractionJournal. `seq` captures the
    total observed journal order, while every opaque `lane` has an independent
    compact protocol state. Cross-lane event interleaving is therefore allowed
    without cross-lane state mutation. -/
def verifyInteractionJournalJsonl (text : String) : Except String InteractionJournalState := do
  let lines := (text.splitOn "\n").filter (fun line => !line.isEmpty)
  match lines with
  | [] => throw "empty interaction journal"
  | first :: rest =>
      let json ← Json.parse first
      let header : JournalHeader ← fromJson? json
      if header.kind != "interaction-journal" then
        throw s!"first record must be interaction-journal header, got {header.kind}"
      if header.version != 1 then
        throw s!"unsupported interaction journal version: {header.version}"
      verifyJournalRecords {} rest

end Browser.Interaction
