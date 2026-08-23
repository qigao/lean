import Browser.Interaction.Epoch
import Browser.Interaction.Ownership

namespace Browser.Interaction

private def epoch : EpochProtocol := { epoch := 2, alive := true }

example : classifyEpoch epoch 1 = .stale := by native_decide
example : classifyEpoch epoch 2 = .current := by native_decide
example : classifyEpoch epoch 3 = .future := by native_decide
example : (recreateEpoch epoch).epoch = 3 := by native_decide

private def ownership : Ownership := {
  pageContext := fun page => if page = 1 then 10 else 20
  framePage := fun frame => if frame = 100 then 1 else 2
}

example : authorizedContextEffect ownership 10 1 = true := by native_decide
example : authorizedContextEffect ownership 10 2 = false := by native_decide
example : authorizedFrameEffect ownership 1 100 = true := by native_decide
example : authorizedFrameEffect ownership 2 100 = false := by native_decide

end Browser.Interaction
