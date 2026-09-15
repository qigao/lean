import NarrativeDynamics.Core.FitnessABMPathN
import NarrativeDynamics.Core.FitnessABMPath4
import NarrativeDynamics.Core.FitnessABMPath4Convergence
import NarrativeDynamics.Core.FitnessABMPath5

open NarrativeDynamics.FitnessABMPathN
open NarrativeDynamics.NetworkPropagation
open NarrativeDynamics.FiniteConsensus
open Filter Topology

example : pathAdj 6 (0 : Fin 6) 1 := by
  decide

example : degree 6 0 = 1 := by
  decide_cbv

example : degree 6 3 = 2 := by
  decide_cbv

example : pathKernel 6 0 0 = 1/2 := by
  decide_cbv

example : pathKernel 6 0 1 = 1/2 := by
  decide_cbv

example : pathKernel 6 3 2 = 1/4 := by
  decide_cbv

example (x : Beliefs 6) (e : Fin 6 → Nat) :
    project 6 (propagate (pathAdj 6) (population 6 x e)) = beliefStep 6 x := by
  exact propagate_independent_exposures 6 x e

example (x : Beliefs 6) (hx : allBroadcast 6 x) (k : Nat) :
    trajectory 6 x k = kernelTrajectory (pathKernel 6) x k := by
  exact trajectory_eq_kernelTrajectory 6 (by omega) x hx k

example (n : Nat) (hn : 2 ≤ n) (x : Beliefs n)
    (hx : allBroadcast n x) :
    mean n (beliefStep n x) = mean n x :=
  mean_step n hn x hx

example (n : Nat) (hn : 2 ≤ n) (x : Beliefs n) (k : Nat) :
    mean n (kernelTrajectory (pathKernel n) x k) = mean n x :=
  mean_kernel_iterate n hn x k

example (n : Nat) (hn : 2 ≤ n) :
    CommonColumnMass ((pathKernel n) ^ block n) (delta n) :=
  path_block_common_mass n hn

example : block 6 = 5 := by
  decide

example : delta 6 = 1/1024 := by
  norm_num [delta, block]

example
    (n : Nat) (hn : 2 ≤ n)
    (x : Beliefs n) (hx : allBroadcast n x) (i : Fin n) :
    Tendsto
      (fun k : Nat => (trajectory n x k i : Real))
      atTop
      (nhds (mean n x : Real)) :=
  trajectory_tendsto n hn x hx i

#print axioms NarrativeDynamics.FitnessABMPathN.propagate_independent_exposures
#print axioms NarrativeDynamics.FitnessABMPathN.propagate_eq_kernel
#print axioms NarrativeDynamics.FitnessABMPathN.allBroadcast_iterate
#print axioms NarrativeDynamics.FitnessABMPathN.path_stationary_weights
#print axioms NarrativeDynamics.FitnessABMPathN.mean_step
#print axioms NarrativeDynamics.FitnessABMPathN.path_block_common_mass
#print axioms NarrativeDynamics.FitnessABMPathN.trajectory_tendsto

theorem path4_adj_compat (i j : Fin 4) :
    NarrativeDynamics.FitnessABMPathN.pathAdj 4 i j ↔
      NarrativeDynamics.FitnessABMPath4.pathAdj i j := by
  rfl

theorem path4_step_compat (x : Fin 4 → Rat) :
    NarrativeDynamics.FitnessABMPathN.beliefStep 4 x =
      NarrativeDynamics.FitnessABMPath4.beliefStep x := by
  rfl

theorem path4_trajectory_compat (x : Fin 4 → Rat) (k : Nat) :
    NarrativeDynamics.FitnessABMPathN.trajectory 4 x k =
      NarrativeDynamics.FitnessABMPath4.trajectory x k := by
  rfl

theorem path4_mean_compat (x : Fin 4 → Rat) :
    NarrativeDynamics.FitnessABMPathN.mean 4 x =
      NarrativeDynamics.FitnessABMPath4.mean x := by
  have h0 : NarrativeDynamics.FitnessABMPathN.degree 4 (0 : Fin 4) = 1 := by decide_cbv
  have h1 : NarrativeDynamics.FitnessABMPathN.degree 4 (1 : Fin 4) = 2 := by decide_cbv
  have h2 : NarrativeDynamics.FitnessABMPathN.degree 4 (2 : Fin 4) = 2 := by decide_cbv
  have h3 : NarrativeDynamics.FitnessABMPathN.degree 4 (3 : Fin 4) = 1 := by decide_cbv
  unfold NarrativeDynamics.FitnessABMPathN.mean
    NarrativeDynamics.FiniteConsensus.weightedMean
    NarrativeDynamics.FitnessABMPathN.stationaryWeight
  rw [NarrativeDynamics.FitnessABMPathN.path_degree_sum 4 (by decide)]
  rw [Fin.sum_univ_four]
  rw [h0, h1, h2, h3]
  norm_num [NarrativeDynamics.FitnessABMPath4.mean]
  ring

theorem path5_adj_compat (i j : Fin 5) :
    NarrativeDynamics.FitnessABMPathN.pathAdj 5 i j ↔
      NarrativeDynamics.FitnessABMPath5.path5Adj i j := by
  rfl

theorem path5_step_compat (x : Fin 5 → Rat) :
    NarrativeDynamics.FitnessABMPathN.beliefStep 5 x =
      NarrativeDynamics.FitnessABMPath5.beliefStep x := by
  rfl

theorem path5_trajectory_compat (x : Fin 5 → Rat) (k : Nat) :
    NarrativeDynamics.FitnessABMPathN.trajectory 5 x k =
      NarrativeDynamics.FitnessABMPath5.trajectory x k := by
  rfl

theorem path5_mean_compat (x : Fin 5 → Rat) :
    NarrativeDynamics.FitnessABMPathN.mean 5 x =
      NarrativeDynamics.FitnessABMPath5.mean x := by
  have h0 : NarrativeDynamics.FitnessABMPathN.degree 5 (0 : Fin 5) = 1 := by decide_cbv
  have h1 : NarrativeDynamics.FitnessABMPathN.degree 5 (Fin.succ 0) = 2 := by decide_cbv
  have h2 : NarrativeDynamics.FitnessABMPathN.degree 5 (Fin.succ 1) = 2 := by decide_cbv
  have h3 : NarrativeDynamics.FitnessABMPathN.degree 5 (Fin.succ 2) = 2 := by decide_cbv
  have h4 : NarrativeDynamics.FitnessABMPathN.degree 5 (Fin.succ 3) = 1 := by decide_cbv
  unfold NarrativeDynamics.FitnessABMPathN.mean
    NarrativeDynamics.FiniteConsensus.weightedMean
    NarrativeDynamics.FitnessABMPathN.stationaryWeight
  rw [NarrativeDynamics.FitnessABMPathN.path_degree_sum 5 (by decide)]
  rw [Fin.sum_univ_succ, Fin.sum_univ_four]
  rw [h0, h1, h2, h3, h4]
  norm_num [NarrativeDynamics.FitnessABMPath5.mean]
  ring

example (i j : Fin 4) :
    NarrativeDynamics.FitnessABMPathN.pathAdj 4 i j ↔
      NarrativeDynamics.FitnessABMPath4.pathAdj i j :=
  path4_adj_compat i j

example (x : Fin 4 → Rat) :
    NarrativeDynamics.FitnessABMPathN.beliefStep 4 x =
      NarrativeDynamics.FitnessABMPath4.beliefStep x :=
  path4_step_compat x

example (x : Fin 4 → Rat) (k : Nat) :
    NarrativeDynamics.FitnessABMPathN.trajectory 4 x k =
      NarrativeDynamics.FitnessABMPath4.trajectory x k :=
  path4_trajectory_compat x k

example (x : Fin 4 → Rat) :
    NarrativeDynamics.FitnessABMPathN.mean 4 x =
      NarrativeDynamics.FitnessABMPath4.mean x :=
  path4_mean_compat x

example (i j : Fin 5) :
    NarrativeDynamics.FitnessABMPathN.pathAdj 5 i j ↔
      NarrativeDynamics.FitnessABMPath5.path5Adj i j :=
  path5_adj_compat i j

example (x : Fin 5 → Rat) :
    NarrativeDynamics.FitnessABMPathN.beliefStep 5 x =
      NarrativeDynamics.FitnessABMPath5.beliefStep x :=
  path5_step_compat x

example (x : Fin 5 → Rat) (k : Nat) :
    NarrativeDynamics.FitnessABMPathN.trajectory 5 x k =
      NarrativeDynamics.FitnessABMPath5.trajectory x k :=
  path5_trajectory_compat x k

example (x : Fin 5 → Rat) :
    NarrativeDynamics.FitnessABMPathN.mean 5 x =
      NarrativeDynamics.FitnessABMPath5.mean x :=
  path5_mean_compat x

example
    (x : NarrativeDynamics.FitnessABMPath4.Beliefs)
    (hx : NarrativeDynamics.FitnessABMPath4.allBroadcast x)
    (i : Fin 4) :
    Tendsto
      (fun k : Nat => (NarrativeDynamics.FitnessABMPath4.trajectory x k i : Real))
      atTop
      (nhds (NarrativeDynamics.FitnessABMPath4.mean x : Real)) := by
  have hxN : NarrativeDynamics.FitnessABMPathN.allBroadcast 4 x := by
    simpa [NarrativeDynamics.FitnessABMPathN.allBroadcast,
      NarrativeDynamics.FitnessABMPath4.allBroadcast] using hx
  simpa only [path4_trajectory_compat, path4_mean_compat] using
    NarrativeDynamics.FitnessABMPathN.trajectory_tendsto
      4 (by decide) x hxN i

def smoke6 : NarrativeDynamics.FitnessABMPathN.Beliefs 6 :=
  ![1, 7/8, 3/4, 5/8, 3/4, 1]

theorem smoke6_allBroadcast :
    NarrativeDynamics.FitnessABMPathN.allBroadcast 6 smoke6 := by
  intro j
  fin_cases j <;>
    norm_num [smoke6, NarrativeDynamics.FitnessABMPathN.allBroadcast]

example (i : Fin 6) :
    Tendsto
      (fun k : Nat =>
        (NarrativeDynamics.FitnessABMPathN.trajectory 6 smoke6 k i : Real))
      atTop
      (nhds (NarrativeDynamics.FitnessABMPathN.mean 6 smoke6 : Real)) :=
  NarrativeDynamics.FitnessABMPathN.trajectory_tendsto
    6 (by decide) smoke6 smoke6_allBroadcast i
