import Browser.Transition

namespace Browser

/-- A page-local event cannot mutate an unrelated sibling page. -/
theorem local_event_isolated
    (m : Model) (e : LocalEvent) (q : PageId) (h : q ≠ e.page) :
    (step m (.local e)).page q = m.page q := by
  simp [step, updatePage, h]

/-- Human input arriving while automation owns the page enters the explicit
    conflict state and suspends the running action. -/
theorem human_input_conflicts_with_automation
    (m : Model) (p : PageId)
    (h : (m.page p).input = .automation) :
    ((step m (.local { page := p, kind := .humanInput })).page p).input = .conflict ∧
    ((step m (.local { page := p, kind := .humanInput })).page p).action = .suspended := by
  simp [step, updatePage, applyLocal, h]

/-- Once an execution context is destroyed, no primitive action is allowed to
    keep executing against that stale context. -/
theorem destroyed_context_blocks_execution
    (m : Model) (p : PageId) :
    ¬ canExecute (step m (.local { page := p, kind := .executionContextDestroyed })) p := by
  simp [canExecute, step, updatePage, applyLocal]

/-- Closing a page immediately makes primitive execution impossible. -/
theorem closed_page_blocks_execution
    (m : Model) (p : PageId) :
    ¬ canExecute (step m (.local { page := p, kind := .pageClosed })) p := by
  simp [canExecute, step, updatePage, applyLocal]

/-- Parent browser availability constrains all descendant pages. -/
theorem browser_disconnect_blocks_all_pages
    (m : Model) (p : PageId) :
    ¬ canExecute (step m .browserDisconnected) p := by
  simp [canExecute, step]

/-- An explicit page command cannot mutate any sibling page. -/
theorem page_command_isolated
    (m : Model) (cmd : PageCommand) (q : PageId) (h : q ≠ cmd.page) :
    (applyCommand m cmd).page q = m.page q := by
  simp [applyCommand, updatePage, h]

/-- Parent-context unavailability blocks primitive execution for every page in
    that context before any recovery policy is considered. -/
theorem context_unavailable_blocks_descendants
    (m : Model) (c : ContextId) (p : PageId)
    (hctx : m.graph.contextOf p = c) :
    ¬ canExecute (step m (.contextUnavailable c)) p := by
  simp [canExecute, step, updateContextAvailability, hctx]

/-- Membership in the policy's context page set carries the context relation. -/
theorem pages_in_context_correct
    (m : Model) (c : ContextId) (p : PageId)
    (h : p ∈ pagesInContext m c) :
    m.graph.contextOf p = c := by
  simp [pagesInContext] at h
  exact h.2

/-- Every pause command generated for a context-unavailable event targets only
    a page in that exact context. -/
theorem context_policy_targets_only_context
    (m : Model) (c : ContextId) (cmd : PageCommand)
    (hcmd : cmd ∈ commandsFor m (.contextUnavailable c)) :
    m.graph.contextOf cmd.page = c := by
  have hm : cmd ∈ (pagesInContext m c).map pauseCommand := by
    simpa [commandsFor] using hcmd
  rcases List.mem_map.mp hm with ⟨p, hp, rfl⟩
  simpa [pauseCommand] using pages_in_context_correct m c p hp

/-- Applying any list of page commands leaves a page unchanged when no command
    in the list targets that page. -/
theorem apply_commands_isolated
    (m : Model) (cmds : List PageCommand) (q : PageId)
    (h : ∀ cmd ∈ cmds, q ≠ cmd.page) :
    (applyCommands m cmds).page q = m.page q := by
  induction cmds generalizing m with
  | nil => rfl
  | cons cmd rest ih =>
      calc
        (applyCommands m (cmd :: rest)).page q
            = (applyCommands (applyCommand m cmd) rest).page q := rfl
        _ = (applyCommand m cmd).page q := by
              apply ih
              intro cmd' hmem
              exact h cmd' (by simp [hmem])
        _ = m.page q := page_command_isolated m cmd q (h cmd (by simp))

/-- A context-scoped reaction cannot mutate the page state of a page belonging
    to a different context. Cross-page effects are therefore explicit and
    context-bounded. -/
theorem context_reaction_isolated
    (m : Model) (c : ContextId) (q : PageId)
    (hctx : m.graph.contextOf q ≠ c) :
    (react m (.contextUnavailable c)).page q = m.page q := by
  have htargets : ∀ cmd ∈ commandsFor m (.contextUnavailable c), q ≠ cmd.page := by
    intro cmd hcmd hEq
    apply hctx
    rw [hEq]
    exact context_policy_targets_only_context m c cmd hcmd
  have hiso := apply_commands_isolated
    (m := step m (.contextUnavailable c))
    (cmds := commandsFor m (.contextUnavailable c))
    (q := q)
    htargets
  calc
    (react m (.contextUnavailable c)).page q
        = (step m (.contextUnavailable c)).page q := by
            simpa [react] using hiso
    _ = m.page q := by
            simp [step, updateContextAvailability]

/-- Every policy-issued command stays in the triggering correlation. -/
theorem issued_command_preserves_correlation
    (m : Model) (envelope : EventEnvelope) (issued : IssuedCommand)
    (h : issued ∈ issueCommandsFor m envelope) :
    issued.cause.correlation = envelope.cause.correlation := by
  have hm : issued ∈ (commandsFor m envelope.event).map (fun command =>
      ({ command := command, cause := childCause envelope } : IssuedCommand)) := by
    simpa [issueCommandsFor] using h
  rcases List.mem_map.mp hm with ⟨command, _, rfl⟩
  rfl

/-- Every policy-issued command links directly to the triggering event id. -/
theorem issued_command_parent_is_trigger
    (m : Model) (envelope : EventEnvelope) (issued : IssuedCommand)
    (h : issued ∈ issueCommandsFor m envelope) :
    issued.cause.parent = some envelope.id := by
  have hm : issued ∈ (commandsFor m envelope.event).map (fun command =>
      ({ command := command, cause := childCause envelope } : IssuedCommand)) := by
    simpa [issueCommandsFor] using h
  rcases List.mem_map.mp hm with ⟨command, _, rfl⟩
  rfl

/-- Every policy reaction advances causal depth by exactly one. -/
theorem issued_command_depth_succ
    (m : Model) (envelope : EventEnvelope) (issued : IssuedCommand)
    (h : issued ∈ issueCommandsFor m envelope) :
    issued.cause.depth = envelope.cause.depth + 1 := by
  have hm : issued ∈ (commandsFor m envelope.event).map (fun command =>
      ({ command := command, cause := childCause envelope } : IssuedCommand)) := by
    simpa [issueCommandsFor] using h
  rcases List.mem_map.mp hm with ⟨command, _, rfl⟩
  rfl

/-- An event deeper than the finite reaction budget is rejected before state
    mutation, preventing unbounded policy self-reaction. -/
theorem over_budget_rejected
    (budget : ReactionBudget) (m : Model) (envelope : EventEnvelope)
    (h : budget.maxDepth < envelope.cause.depth) :
    reactEnvelope budget m envelope = none := by
  simp [reactEnvelope, Nat.not_le.mpr h]

end Browser
