import Browser.Interaction.Types

namespace Browser.Interaction

structure EpochProtocol where
  epoch : NodeEpoch := 0
  alive : Bool := false
  deriving Repr, DecidableEq, BEq

inductive EpochClass where
  | current
  | stale
  | future
  | dead
  deriving Repr, DecidableEq, BEq

/-- Epoch comparison is independent of Page/Frame state. -/
def classifyEpoch (slot : EpochProtocol) (incoming : NodeEpoch) : EpochClass :=
  if incoming < slot.epoch then
    .stale
  else if slot.epoch < incoming then
    .future
  else if slot.alive then
    .current
  else
    .dead

def epochAccepts (slot : EpochProtocol) (incoming : NodeEpoch) : Bool :=
  classifyEpoch slot incoming == .current

/-- Recreation creates the next incarnation; epochs are never reused. -/
def recreateEpoch (slot : EpochProtocol) : EpochProtocol :=
  { epoch := slot.epoch + 1, alive := true }

theorem stale_epoch_not_accepted
    (slot : EpochProtocol) (incoming : NodeEpoch)
    (h : incoming < slot.epoch) :
    epochAccepts slot incoming = false := by
  simp [epochAccepts, classifyEpoch, h]

theorem future_epoch_not_accepted
    (slot : EpochProtocol) (incoming : NodeEpoch)
    (h : slot.epoch < incoming) :
    epochAccepts slot incoming = false := by
  have hnot : ¬ incoming < slot.epoch := Nat.not_lt.mpr (Nat.le_of_lt h)
  simp [epochAccepts, classifyEpoch, h, hnot]

theorem recreate_epoch_advances (slot : EpochProtocol) :
    (recreateEpoch slot).epoch = slot.epoch + 1 ∧
    (recreateEpoch slot).alive = true := by
  rfl

end Browser.Interaction
