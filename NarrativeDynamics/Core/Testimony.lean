import NarrativeDynamics.Core.StoryState

namespace NarrativeDynamics

/-- A report event carries an asserted object location and identifies the
world event the speaker claims as its canonical support. -/
structure LocationReport (Agent Event Object Location : Type*) where
  reportEvent : Event
  supportEvent : Event
  speaker : Agent
  object : Object
  location : Location

/-- Canonical report support is epistemic and temporal, not a truth predicate. -/
def ReportSupported
    {Agent Event Object Location Node : Type*}
    (world : WorldGraph Agent Event Node)
    (report : LocationReport Agent Event Object Location) : Prop :=
  world.observed report.supportEvent report.speaker ∧
  world.eventTime report.supportEvent < world.eventTime report.reportEvent

/-- A recipient has the report only when the world records observation of the
report event itself. -/
def reportAvailableTo
    {Agent Event Object Location Node : Type*}
    (world : WorldGraph Agent Event Node)
    (report : LocationReport Agent Event Object Location)
    (recipient : Agent) : Prop :=
  world.observed report.reportEvent recipient

/-- Latest asserted location among reports actually received by one agent.
Authored report order is the semantic report order; Python V2 validation binds
that order to strict logical time. -/
noncomputable def reportedLocation
    {Agent Event Object Location Node : Type*}
    [DecidableEq Object]
    (world : WorldGraph Agent Event Node)
    (reports : List (LocationReport Agent Event Object Location))
    (recipient : Agent)
    (object : Object) : Option Location := by
  classical
  exact ((reports.filter fun report =>
    report.object = object ∧ reportAvailableTo world report recipient).getLast?).map
      LocationReport.location

/-- Appending an unreceived report cannot change the recipient's report state. -/
theorem reportedLocation_append_unreceived
    {Agent Event Object Location Node : Type*}
    [DecidableEq Object]
    (world : WorldGraph Agent Event Node)
    (reports : List (LocationReport Agent Event Object Location))
    (recipient : Agent) (object : Object)
    (report : LocationReport Agent Event Object Location)
    (hnot : ¬ reportAvailableTo world report recipient) :
    reportedLocation world (reports ++ [report]) recipient object =
      reportedLocation world reports recipient object := by
  classical
  simp [reportedLocation, reportAvailableTo, hnot]

/-- A received appended report about the queried object becomes the latest
reported location. -/
theorem reportedLocation_append_received
    {Agent Event Object Location Node : Type*}
    [DecidableEq Object]
    (world : WorldGraph Agent Event Node)
    (reports : List (LocationReport Agent Event Object Location))
    (recipient : Agent) (object : Object)
    (report : LocationReport Agent Event Object Location)
    (hreceived : reportAvailableTo world report recipient)
    (hobject : report.object = object) :
    reportedLocation world (reports ++ [report]) recipient object =
      some report.location := by
  classical
  simp [reportedLocation, reportAvailableTo, hreceived, hobject]

/-- Existing WorldInvariant information reachability remains authoritative for
report reception. -/
theorem received_report_has_info_path
    {Agent Event Object Location Node : Type*}
    (world : WorldGraph Agent Event Node)
    (hworld : WorldInvariant world)
    (report : LocationReport Agent Event Object Location)
    (recipient : Agent)
    (hreceived : reportAvailableTo world report recipient) :
    infoReachable world.info
      (world.eventNode report.reportEvent)
      (world.agentNode recipient) := by
  exact hworld.2.1 report.reportEvent recipient hreceived

/-- A report cannot satisfy canonical support when the speaker did not observe
its declared support event. -/
theorem unobserved_support_cannot_support_report
    {Agent Event Object Location Node : Type*}
    (world : WorldGraph Agent Event Node)
    (report : LocationReport Agent Event Object Location)
    (hnot : ¬ world.observed report.supportEvent report.speaker) :
    ¬ ReportSupported world report := by
  intro hsupported
  exact hnot hsupported.1

end NarrativeDynamics
