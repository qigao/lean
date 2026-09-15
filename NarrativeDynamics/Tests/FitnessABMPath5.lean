import NarrativeDynamics.Core.FitnessABMPath5

open NarrativeDynamics
open NarrativeDynamics.NetworkPropagation
open NarrativeDynamics.FitnessABMPath5

example (x : Beliefs) (e : Fin 5 → Nat) :
    project (propagate path5Adj (population x e)) =
      project (propagate path5Adj (population x (fun _ => 0))) := by
  exact propagate_independent_exposures x e

example (x : Beliefs) (hx : allBroadcast x) :
    allBroadcast (beliefStep x) := by
  exact allBroadcast_step x hx

example (x : Beliefs) (hx : allBroadcast x) (n : Nat) :
    allBroadcast (trajectory x n) := by
  exact allBroadcast_iterate x hx n

example (x : Beliefs) (hx : allBroadcast x) :
    mean (beliefStep x) = mean x := by
  exact mean_step x hx

/-!
Finite rational timing fixtures.

The issue-#81 `TailModel` implementation referenced by the plan is not present
on this branch, so these fixtures stay deliberately concrete: they compare two
supplied Path5 replay states at a shared clock using only the actual Path5
`trajectory`. They do not introduce a replacement generic tail abstraction.
-/

def earlyReplay : Beliefs := ![1, 3/4, 1/2, 3/4, 1]
def delayedReplay : Beliefs := ![7/8, 3/4, 5/8, 3/4, 7/8]
def commonClockReplay : Beliefs := ![13/16, 3/4, 11/16, 3/4, 13/16]

-- One actual propagation round takes the earlier supplied replay state to the
-- later supplied replay state.
example : trajectory earlyReplay 1 = delayedReplay := by
  decide_cbv

-- If the earlier history activates at clock 1 and the delayed history at clock
-- 2 with the supplied state above, both exact tails agree at common clock 3.
example : trajectory earlyReplay 2 = commonClockReplay := by
  decide_cbv

example : trajectory delayedReplay 1 = commonClockReplay := by
  decide_cbv

example : trajectory earlyReplay 2 = trajectory delayedReplay 1 := by
  decide_cbv

-- The stationary degree-weighted mean agrees across the finite replay states.
example : mean earlyReplay = mean delayedReplay := by
  decide_cbv

example : mean delayedReplay = mean commonClockReplay := by
  decide_cbv
