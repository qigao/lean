import Browser.Interaction.Cdp
import Browser.Interaction.Deadline

namespace Browser.Interaction

private def cdp : CdpProtocol := {
  currentAction := some 2
  pending := some (2, 17)
}

example : receiveCdpResponse cdp 1 17 = cdp := by
  native_decide

example : receiveCdpResponse cdp 2 18 = cdp := by
  native_decide

example :
    let consumed := receiveCdpResponse cdp 2 17
    consumed.pending = none := by
  native_decide

private def deadline : DeadlineProtocol := {
  currentAction := some 2
  deadline := some (2, 100)
  timedOut := false
}

example : expireDeadline deadline 1 999 = deadline := by
  native_decide

example : expireDeadline deadline 2 99 = deadline := by
  native_decide

example : (expireDeadline deadline 2 100).timedOut = true := by
  native_decide

end Browser.Interaction
