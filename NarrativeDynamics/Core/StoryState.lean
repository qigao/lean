import NarrativeDynamics.Core.WorldGraph
import NarrativeDynamics.Core.ObservationAdmission

namespace NarrativeDynamics

/-- One temporal object-location change associated with an existing world event. -/
structure StoryRelocation (Event Object Location : Type*) where
  event : Event
  object : Object
  fromLocation : Option Location
  toLocation : Location

/-- Apply one relocation to a current object-location state. -/
def applyStoryRelocation
    {Event Object Location : Type*}
    [DecidableEq Object]
    (state : Object → Option Location)
    (relocation : StoryRelocation Event Object Location) :
    Object → Option Location :=
  fun object =>
    if object = relocation.object then some relocation.toLocation else state object

/-- Latest objective object location after replaying every relocation in order. -/
def objectiveLocation
    {Event Object Location : Type*}
    [DecidableEq Object]
    (history : List (StoryRelocation Event Object Location))
    (object : Object) : Option Location :=
  (history.foldl applyStoryRelocation (fun _ => none)) object

/-- Latest subjective object location after replaying only events observed by one agent. -/
noncomputable def subjectiveLocation
    {Agent Event Object Location Node : Type*}
    [DecidableEq Object]
    (world : WorldGraph Agent Event Node)
    (history : List (StoryRelocation Event Object Location))
    (agent : Agent)
    (object : Object) : Option Location := by
  classical
  exact objectiveLocation
    (history.filter (fun relocation => world.observed relocation.event agent))
    object

/-- Objective relocation history is continuous and strictly ordered by the world's event clock. -/
def storyHistoryCompatibleFrom
    {Agent Event Object Location Node : Type*}
    [DecidableEq Object]
    (world : WorldGraph Agent Event Node)
    (state : Object → Option Location)
    (previousTime : Option Nat) :
    List (StoryRelocation Event Object Location) → Prop
  | [] => True
  | relocation :: rest =>
      relocation.fromLocation = state relocation.object ∧
      (match previousTime with
       | none => True
       | some time => time < world.eventTime relocation.event) ∧
      storyHistoryCompatibleFrom world
        (applyStoryRelocation state relocation)
        (some (world.eventTime relocation.event))
        rest

/-- A valid story history starts from every object's unknown location. -/
def StoryHistoryCompatible
    {Agent Event Object Location Node : Type*}
    [DecidableEq Object]
    (world : WorldGraph Agent Event Node)
    (history : List (StoryRelocation Event Object Location)) : Prop :=
  storyHistoryCompatibleFrom world (fun _ => none) none history

/-- Appending a relocation makes its destination the latest objective location. -/
theorem objectiveLocation_append_relocation
    {Event Object Location : Type*}
    [DecidableEq Object]
    (history : List (StoryRelocation Event Object Location))
    (relocation : StoryRelocation Event Object Location) :
    objectiveLocation (history ++ [relocation]) relocation.object =
      some relocation.toLocation := by
  simp [objectiveLocation, List.foldl_append, applyStoryRelocation]

/-- An unobserved relocation cannot change this agent's subjective location. -/
theorem subjectiveLocation_append_unobserved
    {Agent Event Object Location Node : Type*}
    [DecidableEq Object]
    (world : WorldGraph Agent Event Node)
    (history : List (StoryRelocation Event Object Location))
    (agent : Agent)
    (object : Object)
    (relocation : StoryRelocation Event Object Location)
    (hnot : ¬ world.observed relocation.event agent) :
    subjectiveLocation world (history ++ [relocation]) agent object =
      subjectiveLocation world history agent object := by
  classical
  simp [subjectiveLocation, hnot]

/-- An observed relocation becomes this agent's latest subjective destination. -/
theorem subjectiveLocation_append_observed_relocation
    {Agent Event Object Location Node : Type*}
    [DecidableEq Object]
    (world : WorldGraph Agent Event Node)
    (history : List (StoryRelocation Event Object Location))
    (agent : Agent)
    (relocation : StoryRelocation Event Object Location)
    (hobs : world.observed relocation.event agent) :
    subjectiveLocation world (history ++ [relocation]) agent relocation.object =
      some relocation.toLocation := by
  classical
  simp [subjectiveLocation, hobs, objectiveLocation_append_relocation]

/-- In a well-formed world, every story event recorded as observed has an information path. -/
theorem observed_story_event_has_info_path
    {Agent Event Object Location Node : Type*}
    (world : WorldGraph Agent Event Node)
    (hworld : WorldInvariant world)
    (relocation : StoryRelocation Event Object Location)
    (agent : Agent)
    (hobs : world.observed relocation.event agent) :
    infoReachable world.info
      (world.eventNode relocation.event)
      (world.agentNode agent) := by
  exact hworld.2.1 relocation.event agent hobs

/-- Without an information path, a well-formed world cannot record the event as observed. -/
theorem no_info_path_story_event_unobserved
    {Agent Event Object Location Node : Type*}
    (world : WorldGraph Agent Event Node)
    (hworld : WorldInvariant world)
    (relocation : StoryRelocation Event Object Location)
    (agent : Agent)
    (hno : ¬ infoReachable world.info
      (world.eventNode relocation.event)
      (world.agentNode agent)) :
    ¬ world.observed relocation.event agent := by
  intro hobs
  exact hno (observed_story_event_has_info_path world hworld relocation agent hobs)

end NarrativeDynamics
