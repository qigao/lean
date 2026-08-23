import NarrativeDynamics.Core.StoryState

namespace NarrativeDynamics.Tests.StoryState

inductive Agent where
  | bob
  deriving DecidableEq, Repr

inductive Event where
  | e1
  | e2
  deriving DecidableEq, Repr

inductive Object where
  | key
  deriving DecidableEq, Repr

inductive Location where
  | drawer
  | box
  deriving DecidableEq, Repr

inductive Node where
  | event1
  | event2
  | bob
  deriving DecidableEq, Repr

open Agent Event Object Location

private def eventNode : Event → Node
  | .e1 => .event1
  | .e2 => .event2

private def agentNode : Agent → Node
  | .bob => .bob

private def eventTime : Event → Nat
  | .e1 => 1
  | .e2 => 2

private def falseBeliefWorld : WorldGraph Agent Event Node := {
  alive := fun _ => True
  canAct := fun _ => True
  info := fun _ _ => True
  eventNode := eventNode
  agentNode := agentNode
  observed := fun e a => e = .e1 ∧ a = .bob
  causal := fun _ _ => False
  eventTime := eventTime
}

private def informedWorld : WorldGraph Agent Event Node := {
  falseBeliefWorld with
  observed := fun e a => (e = .e1 ∨ e = .e2) ∧ a = .bob
}

private def history : List (StoryRelocation Event Object Location) := [
  { event := .e1, object := .key, fromLocation := none, toLocation := .drawer },
  { event := .e2, object := .key, fromLocation := some .drawer, toLocation := .box }
]

example : StoryHistoryCompatible falseBeliefWorld history := by
  simp [StoryHistoryCompatible, storyHistoryCompatibleFrom, falseBeliefWorld,
    history, eventTime, applyStoryRelocation]

example : objectiveLocation history .key = some .box := by
  rfl

example : subjectiveLocation falseBeliefWorld history .bob .key = some .drawer := by
  simp [subjectiveLocation, falseBeliefWorld, history, objectiveLocation,
    applyStoryRelocation]

example : subjectiveLocation informedWorld history .bob .key = some .box := by
  simp [subjectiveLocation, informedWorld, falseBeliefWorld, history,
    objectiveLocation, applyStoryRelocation]

private def score (known action : Location) : Nat :=
  if known = action then 1 else 0

example :
    score .drawer .box < score .drawer .drawer ∧
    score .box .drawer < score .box .box := by
  decide

#check NarrativeDynamics.no_info_path_no_admissible_observation

end NarrativeDynamics.Tests.StoryState
