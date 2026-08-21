import Browser.Event
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

end Browser
