import Browser.Interaction.DecoderRecovery

namespace Browser.Interaction

/-- Least recovery action that is at least as strong as both inputs. Recovery
    rank is a total order, so this is the join used to merge concurrent faults. -/
def joinRecovery (a b : RecoveryAction) : RecoveryAction :=
  if recoveryRank a ≤ recoveryRank b then b else a

theorem join_recovery_idempotent (action : RecoveryAction) :
    joinRecovery action action = action := by
  cases action <;> rfl

theorem join_recovery_commutative (a b : RecoveryAction) :
    joinRecovery a b = joinRecovery b a := by
  cases a <;> cases b <;> rfl

theorem join_recovery_associative (a b c : RecoveryAction) :
    joinRecovery (joinRecovery a b) c = joinRecovery a (joinRecovery b c) := by
  cases a <;> cases b <;> cases c <;> rfl

theorem join_recovery_left_upper (a b : RecoveryAction) :
    recoveryRank a ≤ recoveryRank (joinRecovery a b) := by
  cases a <;> cases b <;> decide

theorem join_recovery_right_upper (a b : RecoveryAction) :
    recoveryRank b ≤ recoveryRank (joinRecovery a b) := by
  cases a <;> cases b <;> decide

theorem join_recovery_least_upper
    (a b candidate : RecoveryAction)
    (ha : recoveryRank a ≤ recoveryRank candidate)
    (hb : recoveryRank b ≤ recoveryRank candidate) :
    recoveryRank (joinRecovery a b) ≤ recoveryRank candidate := by
  cases a <;> cases b <;> cases candidate <;> simp_all [joinRecovery, recoveryRank]

end Browser.Interaction
