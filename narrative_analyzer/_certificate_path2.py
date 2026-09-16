"""Closed Path2 certificate templates."""

from __future__ import annotations

from .slow_zero import no_common_limit_lines


def named_path2_body(
    *,
    schedule_id: str,
    params: str,
    state: str,
    params_eq: str,
    state_eq: str,
    lemma: str,
    theorem_tail: str,
) -> list[str]:
    if schedule_id == "harmonicSchedule":
        return [
            f"private theorem {lemma} :",
            "    ∀ i : Fin 2,",
            f"      Tendsto (fun k => (beliefs ((step {params} 2)^[k] {state}) i : Real))",
            "        atTop (nhds (1/2 : Real)) := by",
            f"  rw [{params_eq}, {state_eq}]",
            f"  exact {theorem_tail}",
        ]
    if schedule_id == "slowZeroSchedule":
        return [f"private theorem {lemma} :", *no_common_limit_lines(
            params=params,
            state=state,
            params_eq=params_eq,
            state_eq=state_eq,
            theorem_tail=theorem_tail,
        )]
    if schedule_id == "nearOneSchedule":
        return [
            f"private theorem {lemma} :",
            "    ¬ ∃ c : Real, ∀ i : Fin 2,",
            f"      Tendsto (fun k => (beliefs ((step {params} 2)^[k] {state}) i : Real))",
            "        atTop (nhds c) := by",
            f"  rw [{params_eq}, {state_eq}]",
            f"  exact {theorem_tail}",
        ]
    raise ValueError(f"unsupported fixed Path2 schedule: {schedule_id}")
