import NarrativeDynamics.Core.FitnessABMPathNParameterConvergence

open NarrativeDynamics
open NarrativeDynamics.FiniteConsensus
open NarrativeDynamics.FitnessABMPathNParameters
open NarrativeDynamics.FitnessABMPathNParameterConvergence
open Filter Topology

private def p34 : ResponseParameters := ⟨3/4, 1/3⟩

example : ResponseParameters.Valid p34 := by
  norm_num [p34, ResponseParameters.Valid]

example (n : Nat) (hn : 2 ≤ n) :
    StationaryWeights (pathKernel p34 n) (FitnessABMPathN.stationaryWeight n) := by
  exact path_stationary_weights p34
    (by norm_num [p34, ResponseParameters.Valid]) n hn

example (n : Nat) (hn : 2 ≤ n) (x : Beliefs n) :
    FitnessABMPathN.mean n (applyKernel (pathKernel p34 n) x) =
      FitnessABMPathN.mean n x := by
  exact mean_kernel_step p34
    (by norm_num [p34, ResponseParameters.Valid]) n hn x

-- Task 2: strict-interior receptivity provides a uniform positive local mass
-- and therefore a block common-column mass on every finite path.
example : 0 < beta p34 := by
  exact beta_pos p34 (by norm_num [p34]) (by norm_num [p34])

example (n : Nat) (i : Fin n) :
    beta p34 ≤ pathKernel p34 n i i := by
  exact pathKernel_self_lower p34
    (by norm_num [p34]) (by norm_num [p34]) n i

example (n : Nat) (hn : 2 ≤ n) (i j : Fin n)
    (h : FitnessABMPathN.pathAdj n j i) :
    beta p34 ≤ pathKernel p34 n i j := by
  exact pathKernel_adj_lower p34
    (by norm_num [p34]) (by norm_num [p34]) n hn i j h

example (n : Nat) (hn : 2 ≤ n) :
    CommonColumnMass
      ((pathKernel p34 n) ^ FitnessABMPathN.block n)
      (delta p34 n) := by
  exact path_block_common_mass p34
    (by norm_num [p34, ResponseParameters.Valid])
    (by norm_num [p34]) (by norm_num [p34]) n hn

-- Task 3: the real executable trajectory agrees with the proof kernel
-- trajectory inside the invariant all-broadcast region and converges
-- coordinatewise to the existing degree-weighted mean.
private def p23 : ResponseParameters := ⟨2/3, 1/4⟩
private def allBroadcast3 : Beliefs 3 := ![1/4, 3/4, 1]

example : ResponseParameters.Valid p23 := by
  norm_num [p23, ResponseParameters.Valid]

example : allBroadcast p23 3 allBroadcast3 := by
  intro i
  fin_cases i <;> norm_num [p23, allBroadcast3]

example (k : Nat) :
    ((beliefStep p23 3)^[k] allBroadcast3) =
      kernelTrajectory (pathKernel p23 3) allBroadcast3 k := by
  exact trajectory_eq_kernelTrajectory p23
    (by norm_num [p23, ResponseParameters.Valid]) 3 (by decide)
    allBroadcast3 (by
      intro i
      fin_cases i <;> norm_num [p23, allBroadcast3]) k

example (i : Fin 3) :
    Tendsto
      (fun k : Nat => ((((beliefStep p23 3)^[k] allBroadcast3) i : Rat) : Real))
      atTop
      (nhds (FitnessABMPathN.mean 3 allBroadcast3 : Real)) := by
  exact trajectory_tendsto p23
    (by norm_num [p23, ResponseParameters.Valid])
    (by norm_num [p23]) (by norm_num [p23])
    3 (by decide) allBroadcast3
    (by
      intro j
      fin_cases j <;> norm_num [p23, allBroadcast3]) i

-- Task 4: exact endpoint counterexamples justify the strict 0 < alpha < 1
-- hypothesis of the generic convergence theorem.
private def zeroResponse : ResponseParameters := ⟨0, 0⟩
private def oneResponse : ResponseParameters := ⟨1, 0⟩
private def path2Split : Beliefs 2 := ![1, 0]
private def path2Swap : Beliefs 2 := ![0, 1]

example : ResponseParameters.Valid zeroResponse := by
  norm_num [zeroResponse, ResponseParameters.Valid]

example : ResponseParameters.Valid oneResponse := by
  norm_num [oneResponse, ResponseParameters.Valid]

example : allBroadcast zeroResponse 2 path2Split := by
  intro i
  fin_cases i <;> norm_num [zeroResponse, path2Split]

example : allBroadcast oneResponse 2 path2Split := by
  intro i
  fin_cases i <;> norm_num [oneResponse, path2Split]

theorem zero_response_path2_step :
    beliefStep zeroResponse 2 path2Split = path2Split := by
  decide_cbv

theorem zero_response_path2_iterate (k : Nat) :
    (beliefStep zeroResponse 2)^[k] path2Split = path2Split := by
  induction k with
  | zero => simp
  | succ k ih =>
      rw [Function.iterate_succ_apply', ih]
      exact zero_response_path2_step

theorem one_response_path2_step :
    beliefStep oneResponse 2 path2Split = path2Swap := by
  decide_cbv

theorem one_response_path2_back :
    beliefStep oneResponse 2 path2Swap = path2Split := by
  decide_cbv

theorem one_response_path2_two :
    (beliefStep oneResponse 2)^[2] path2Split = path2Split := by
  simp [Function.iterate_succ_apply', one_response_path2_step,
    one_response_path2_back]

theorem one_response_path2_even (k : Nat) :
    (beliefStep oneResponse 2)^[2 * k] path2Split = path2Split := by
  induction k with
  | zero => simp
  | succ k ih =>
      rw [Nat.mul_succ]
      rw [Function.iterate_add_apply
        (beliefStep oneResponse 2) (2 * k) 2 path2Split]
      rw [one_response_path2_two, ih]

#print axioms NarrativeDynamics.FitnessABMPathNParameterConvergence.path_stationary_weights
#print axioms NarrativeDynamics.FitnessABMPathNParameterConvergence.path_block_common_mass
#print axioms NarrativeDynamics.FitnessABMPathNParameterConvergence.trajectory_tendsto
#print axioms zero_response_path2_iterate
#print axioms one_response_path2_even
