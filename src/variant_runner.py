from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set

from src.a_star import a_star_search
from src.greedy import greedy_search
from src.llm_interface import evaluate_trajectory, interpret_objective, suggest_next_course
from src.problem_formalization import PlanningInstance, load_instance_from_file

logger = logging.getLogger(__name__)

VariantName = str
AlgorithmName = str
VariantResult = Dict[str, Any]


def select_algorithm(algorithm_name: str) -> Callable[[PlanningInstance], Dict[str, Any]]:
    if algorithm_name == "a_star":
        return a_star_search
    return greedy_search


def run_algorithm(
    instance: PlanningInstance,
    objective: Set[str],
    algorithm_name: AlgorithmName = "greedy",
) -> Dict[str, Any]:
    algorithm = select_algorithm(algorithm_name)
    if objective != instance.target_skills:
        instance = PlanningInstance(
            skills=set(instance.skills),
            courses=instance.courses,
            initial_skills=set(instance.initial_skills),
            target_skills=set(objective),
        )
    return algorithm(instance)


def _build_common_result(
    trajectory: Optional[list],
    total_cost: int,
    elapsed_time: float,
    num_courses: int,
    success: bool,
    llm_calls: int,
    llm_evaluation: Optional[Dict[str, Any]] = None,
    llm_suggestion: Optional[Dict[str, Any]] = None,
    llm_step_log: Optional[list] = None,
    interpreted_objective: Optional[Set[str]] = None,
) -> VariantResult:
    result: VariantResult = {
        "trajectory": trajectory,
        "total_cost": total_cost,
        "elapsed_time": elapsed_time,
        "num_courses": num_courses,
        "success": success,
        "llm_calls": llm_calls,
    }
    if llm_evaluation is not None:
        result["llm_evaluation"] = llm_evaluation
    if llm_suggestion is not None:
        result["llm_suggestion"] = llm_suggestion
    if llm_step_log is not None:
        result["llm_step_log"] = llm_step_log
    if interpreted_objective is not None:
        result["interpreted_objective"] = sorted(interpreted_objective)
    return result


def run_base_variant(
    instance: PlanningInstance,
    objective: Set[str],
    algorithm_name: AlgorithmName = "greedy",
) -> VariantResult:
    start_time = time.perf_counter()
    search_result = run_algorithm(instance, objective, algorithm_name)
    elapsed_time = time.perf_counter() - start_time
    return _build_common_result(
        trajectory=search_result.get("trajectory"),
        total_cost=search_result.get("total_cost", 0),
        elapsed_time=elapsed_time,
        num_courses=search_result.get("num_courses", 0),
        success=search_result.get("success", False),
        llm_calls=0,
    )


def run_interpret_variant(
    instance: PlanningInstance,
    objective_text: str,
    algorithm_name: AlgorithmName = "greedy",
    use_ollama: bool = True,
) -> VariantResult:
    if not use_ollama:
        raise ValueError("interpret variant requires Ollama enabled.")
    interpreted_objective = interpret_objective(objective_text)
    base_result = run_base_variant(instance, interpreted_objective, algorithm_name)
    base_result["llm_calls"] = 1
    base_result["interpreted_objective"] = sorted(interpreted_objective)
    return base_result


def run_evaluate_variant(
    instance: PlanningInstance,
    objective: Set[str],
    algorithm_name: AlgorithmName = "greedy",
    use_ollama: bool = True,
) -> VariantResult:
    if not use_ollama:
        raise ValueError("evaluate variant requires Ollama enabled.")
    base_result = run_base_variant(instance, objective, algorithm_name)
    llm_eval = evaluate_trajectory(base_result["trajectory"] or [], instance, objective)
    base_result["llm_calls"] = 1
    base_result["llm_evaluation"] = llm_eval
    return base_result


def run_guided_variant(
    instance: PlanningInstance,
    objective: Set[str],
    algorithm_name: AlgorithmName = "greedy",
    use_ollama: bool = True,
) -> VariantResult:
    if not use_ollama:
        raise ValueError("guided variant requires Ollama enabled.")
    raise NotImplementedError("Guided variant is implemented in the next phase.")


def run_variant(
    variant: VariantName,
    instance: PlanningInstance,
    objective: Any,
    algorithm_name: AlgorithmName = "greedy",
    use_ollama: bool = False,
) -> VariantResult:
    if variant == "A" or variant == "base":
        if not isinstance(objective, set):
            raise ValueError("Variant A requires a set of objective skills.")
        return run_base_variant(instance, objective, algorithm_name)
    if variant == "B" or variant == "interpret":
        if not isinstance(objective, str):
            raise ValueError("Variant B requires a natural language objective string.")
        return run_interpret_variant(instance, objective, algorithm_name, use_ollama)
    if variant == "C" or variant == "evaluate":
        if not isinstance(objective, set):
            raise ValueError("Variant C requires a set of objective skills.")
        return run_evaluate_variant(instance, objective, algorithm_name, use_ollama)
    if variant == "D" or variant == "guided":
        if not isinstance(objective, set):
            raise ValueError("Variant D requires a set of objective skills.")
        return run_guided_variant(instance, objective, algorithm_name, use_ollama)
    raise ValueError(f"Unknown variant '{variant}'. Choose A, B, C or D.")


def load_variant_instance(instance_path: Path) -> PlanningInstance:
    return load_instance_from_file(instance_path)
