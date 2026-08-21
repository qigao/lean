import Browser.Command

namespace Browser

def pagesInContext (m : Model) (context : ContextId) : List PageId :=
  m.knownPages.filter (fun page => m.graph.contextOf page == context)

/-- Policy is the only layer allowed to turn a scoped event into commands for
    pages other than the event's source page. -/
def commandsFor (m : Model) : RuntimeEvent → List PageCommand
  | .contextUnavailable context => (pagesInContext m context).map pauseCommand
  | .contextAvailable context => (pagesInContext m context).map resumeCommand
  | .browserDisconnected => m.knownPages.map pauseCommand
  | .browserConnected => m.knownPages.map resumeCommand
  | .local _ => []

/-- Attach causal metadata to policy output. The semantic command list is still
    produced by `commandsFor`, so existing state proofs remain unchanged. -/
def issueCommandsFor (m : Model) (envelope : EventEnvelope) : List IssuedCommand :=
  (commandsFor m envelope.event).map fun command => {
    command := command
    cause := childCause envelope
  }

end Browser
