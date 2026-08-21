import Browser.State

namespace Browser

/-- Reusable predicates that an action may declare as runtime dependencies. -/
inductive ActionRequirement where
  | browserAvailable
  | contextAvailable
  | pageReady
  | runtimeReady
  | automationOwned
  deriving Repr, DecidableEq, BEq

/-- A concrete action binds a page to the dependencies it requires while running. -/
structure ActionContract where
  page : PageId
  requires : List ActionRequirement
  deriving Repr

/-- Evaluate one declared dependency against the current runtime model. -/
def requirementHolds (m : Model) (p : PageId) : ActionRequirement → Bool
  | .browserAvailable => m.browserAlive
  | .contextAvailable => m.contextAvailable (m.graph.contextOf p)
  | .pageReady => (m.page p).lifecycle == .ready
  | .runtimeReady => (m.page p).runtime == .ready
  | .automationOwned => (m.page p).input == .automation

/-- Executable conjunction of every dependency declared by an action. -/
def requirementsHold (m : Model) (contract : ActionContract) : Bool :=
  contract.requires.all (requirementHolds m contract.page)

/-- Current primitive browser input actions require every availability/input
    predicate. More specialized actions can declare a different list later. -/
def primitiveContract (p : PageId) : ActionContract := {
  page := p
  requires := [
    .browserAvailable,
    .contextAvailable,
    .pageReady,
    .runtimeReady,
    .automationOwned
  ]
}

/-- If a running action loses any declared dependency, make the invalidation
    explicit in state instead of leaving `action = executing` while blocked. -/
def reconcileAction (m : Model) (contract : ActionContract) : Model :=
  let s := m.page contract.page
  if (s.action == .executing) && !(requirementsHold m contract) then
    updatePage m contract.page fun current =>
      { current with input := .idle, action := .suspended }
  else
    m

/-- Reconcile all Page actions known to the Browser supervisor. -/
def reconcileKnownActions (m : Model) : Model :=
  m.knownPages.foldl
    (fun current page => reconcileAction current (primitiveContract page))
    m

end Browser
