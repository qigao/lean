import Browser.Causality

namespace Browser

inductive PageCommandKind where
  | pauseAutomation
  | resumeAutomation
  deriving Repr, DecidableEq, BEq

structure PageCommand where
  page : PageId
  kind : PageCommandKind
  deriving Repr, DecidableEq, BEq

/-- A policy-issued command carries the causal identity of the reaction that
    created it without changing the command's semantic payload. -/
structure IssuedCommand where
  command : PageCommand
  cause : Cause
  deriving Repr

def pauseCommand (page : PageId) : PageCommand :=
  { page := page, kind := .pauseAutomation }

def resumeCommand (page : PageId) : PageCommand :=
  { page := page, kind := .resumeAutomation }

def applyPageCommand (kind : PageCommandKind) (s : PageState) : PageState :=
  match kind with
  | .pauseAutomation =>
      match s.action with
      | .idle => s
      | .timedOut => s
      | _ => { s with input := .idle, action := .suspended }
  | .resumeAutomation =>
      if s.action = .suspended ∧ s.lifecycle = .ready ∧ s.runtime = .ready ∧ s.input = .idle then
        { s with action := .waiting }
      else
        s

def applyCommand (m : Model) (cmd : PageCommand) : Model :=
  updatePage m cmd.page (applyPageCommand cmd.kind)

def applyCommands : Model → List PageCommand → Model
  | m, [] => m
  | m, cmd :: rest => applyCommands (applyCommand m cmd) rest

end Browser
