"""Lean proof fragment for the slow-zero no-common-limit certificate.

This module contains no executable model semantics. It only emits a closed proof
fragment composed from the production Path2 mean invariant and the production
slow-zero theorem.
"""

from __future__ import annotations


def no_common_limit_lines(
    *, params: str, state: str, params_eq: str, state_eq: str, theorem_tail: str
) -> list[str]:
    return [
        "    ¬ ∃ c : Real, ∀ i : Fin 2,",
        f"      Tendsto (fun k => (beliefs ((step {params} 2)^[k] {state}) i : Real))",
        "        atTop (nhds c) := by",
        "  rintro ⟨c, hc⟩",
        f"  have hvalid : {params}.Valid := by",
        f"    rw [{params_eq}]",
        "    exact slowZeroSchedule_valid",
        f"  have he : ({state} 0).exposure = ({state} 1).exposure := by",
        f"    norm_num [{state}]",
        f"  have hb : allBroadcast {params} {state} := by",
        "    intro i",
        f"    fin_cases i <;> norm_num [allBroadcast, {params}, {state}]",
        "  let m : Nat → Real := fun k =>",
        f"    ((beliefs ((step {params} 2)^[k] {state}) 0 : Real) +",
        f"      (beliefs ((step {params} 2)^[k] {state}) 1 : Real)) / 2",
        "  have hm_to_c : Tendsto m atTop (nhds c) := by",
        "    have hsum := (hc (0 : Fin 2)).add (hc (1 : Fin 2))",
        "    have hscaled := hsum.mul_const (1/2 : Real)",
        "    convert hscaled using 1 <;> dsimp [m] <;> ring",
        "  have hm_eq (k : Nat) : m k = (1/2 : Real) := by",
        "    dsimp [m]",
        f"    have hrat := path2_mean_iterate {params} hvalid {state} he hb k",
        "    have hreal := congrArg (fun q : Rat => (q : Real)) hrat",
        f"    simpa [{state}] using hreal",
        "  have hm_to_half : Tendsto m atTop (nhds (1/2 : Real)) := by",
        "    have hconst : Tendsto (fun _ : Nat => (1/2 : Real)) atTop (nhds (1/2 : Real)) :=",
        "      tendsto_const_nhds",
        "    exact hconst.congr' (Filter.Eventually.of_forall fun k => (hm_eq k).symm)",
        "  have hc_eq : c = (1/2 : Real) := tendsto_nhds_unique hm_to_c hm_to_half",
        "  have hhalf : ∀ i : Fin 2,",
        f"      Tendsto (fun k => (beliefs ((step {params} 2)^[k] {state}) i : Real))",
        "        atTop (nhds (1/2 : Real)) := by",
        "    intro i",
        "    simpa [hc_eq] using hc i",
        f"  rw [{params_eq}, {state_eq}] at hhalf",
        f"  exact {theorem_tail} hhalf",
    ]
