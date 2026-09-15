import NarrativeDynamics.Core.FitnessABMPath4
import Mathlib.Analysis.SpecificLimits.Basic

/-!
# Real limits of the four-node path dynamics

The rational trajectory is cast only after applying its proved closed form.
Every decaying mode is retained, including the zero mode at time zero.
-/

namespace NarrativeDynamics.FitnessABMPath4

open Filter Topology

theorem trajectory_tendsto (x : Beliefs) (hx : allBroadcast x) (i : Fin 4) :
    Tendsto (fun n : Nat => (trajectory x n i : Real))
      atTop (nhds (mean x : Real)) := by
  have hA : Tendsto (fun n : Nat => (3/4 : Real)^n) atTop (nhds 0) :=
    tendsto_pow_atTop_nhds_zero_of_lt_one (by norm_num) (by norm_num)
  have hB : Tendsto (fun n : Nat => (1/4 : Real)^n) atTop (nhds 0) :=
    tendsto_pow_atTop_nhds_zero_of_lt_one (by norm_num) (by norm_num)
  have hC : Tendsto (fun n : Nat => (0 : Real)^n) atTop (nhds 0) :=
    tendsto_pow_atTop_nhds_zero_of_lt_one (by norm_num) (by norm_num)
  have hform : Tendsto (fun n : Nat =>
      (mean x : Real) + (coeffA x : Real)*(3/4 : Real)^n*(modeV i : Real) +
        (coeffB x : Real)*(1/4 : Real)^n*(modeW i : Real) +
        (coeffC x : Real)*(0 : Real)^n*(modeZ i : Real))
      atTop (nhds (mean x : Real)) := by
    simpa only [mul_zero, zero_mul, add_zero] using
      ((tendsto_const_nhds.add ((hA.const_mul _).mul_const _)).add
        ((hB.const_mul _).mul_const _)).add ((hC.const_mul _).mul_const _)
  convert hform using 1
  funext n
  rw [iterate_closedForm x hx]
  dsimp only [closedForm]
  push_cast
  rfl

end NarrativeDynamics.FitnessABMPath4
