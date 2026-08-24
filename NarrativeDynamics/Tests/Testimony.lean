import NarrativeDynamics.Core.Testimony

namespace NarrativeDynamics.Tests.Testimony

inductive Agent where
  | alice | bob
  deriving DecidableEq, Repr

inductive Event where
  | e1 | e2 | r1
  deriving DecidableEq, Repr

inductive Object where
  | key
  deriving DecidableEq, Repr

inductive Location where
  | drawer | box
  deriving DecidableEq, Repr

inductive Node where
  | event1 | event2 | report1 | alice | bob
  deriving DecidableEq, Repr

open Agent Event Object Location

private def eventNode : Event → Node
  | .e1 => .event1
  | .e2 => .event2
  | .r1 => .report1

private def agentNode : Agent → Node
  | .alice => .alice
  | .bob => .bob

private def eventTime : Event → Nat
  | .e1 => 1
  | .e2 => 2
  | .r1 => 3

private def testimonyWorld : WorldGraph Agent Event Node := {
  alive := fun _ => True
  canAct := fun _ => True
  info := fun _ _ => True
  eventNode := eventNode
  agentNode := agentNode
  observed := fun e a =>
    (e = .e1 ∧ a = .bob) ∨
    (e = .e2 ∧ a = .alice) ∨
    (e = .r1 ∧ a = .bob)
  causal := fun _ _ => False
  eventTime := eventTime
}

private def history : List (StoryRelocation Event Object Location) := [
  { event := .e1, object := .key, fromLocation := none, toLocation := .drawer },
  { event := .e2, object := .key, fromLocation := some .drawer, toLocation := .box }
]

private def truthfulReport : LocationReport Agent Event Object Location := {
  reportEvent := .r1
  supportEvent := .e2
  speaker := .alice
  object := .key
  location := .box
}

private def staleReport : LocationReport Agent Event Object Location := {
  reportEvent := .r1
  supportEvent := .e2
  speaker := .alice
  object := .key
  location := .drawer
}

example : ReportSupported testimonyWorld truthfulReport := by
  simp [ReportSupported, testimonyWorld, truthfulReport, eventTime]

example : ReportSupported testimonyWorld staleReport := by
  simp [ReportSupported, testimonyWorld, staleReport, eventTime]

example : objectiveLocation history .key = some .box := by
  rfl

example : subjectiveLocation testimonyWorld history .bob .key = some .drawer := by
  simp [subjectiveLocation, testimonyWorld, history, objectiveLocation,
    applyStoryRelocation]

example : reportedLocation testimonyWorld [truthfulReport] .bob .key = some .box := by
  simp [reportedLocation, reportAvailableTo, testimonyWorld, truthfulReport]

example : reportedLocation testimonyWorld [staleReport] .bob .key = some .drawer := by
  simp [reportedLocation, reportAvailableTo, testimonyWorld, staleReport]

example :
    ReportSupported testimonyWorld staleReport ∧
    reportAvailableTo testimonyWorld staleReport .bob ∧
    objectiveLocation history .key = some .box ∧
    reportedLocation testimonyWorld [staleReport] .bob .key = some .drawer := by
  simp [ReportSupported, reportAvailableTo, testimonyWorld, staleReport,
    history, eventTime, objectiveLocation, applyStoryRelocation, reportedLocation]

#check NarrativeDynamics.reportedLocation_append_unreceived
#check NarrativeDynamics.reportedLocation_append_received
#check NarrativeDynamics.received_report_has_info_path
#check NarrativeDynamics.unobserved_support_cannot_support_report

end NarrativeDynamics.Tests.Testimony
