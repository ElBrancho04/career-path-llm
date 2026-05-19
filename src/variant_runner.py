from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set

from src.a_star import a_star_search
from src.greedy import greedy_search
from src.llm_interface import evaluate_trajectory, interpret_objective, suggest_next_course
from src.problem_formalization import PlanningInstance, load_instance_from_file
from src.problem_formalization import available_courses

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

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
    instance_name: Optional[str] = None,
) -> VariantResult:
    logger.info("Running base variant: algorithm=%s instance=%s", algorithm_name, instance_name or "unknown")
    start_time = time.perf_counter()
    search_result = run_algorithm(instance, objective, algorithm_name)
    elapsed_time = time.perf_counter() - start_time
    logger.info(
        "Base variant completed: instance=%s algo=%s success=%s num_courses=%s llm_calls=0 elapsed_time=%.4f",
        instance_name or "unknown",
        algorithm_name,
        search_result.get("success", False),
        search_result.get("num_courses", 0),
        elapsed_time,
    )
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
    instance_name: Optional[str] = None,
) -> VariantResult:
    if not use_ollama:
        raise ValueError("interpret variant requires Ollama enabled.")
    logger.info("Running interpret variant: instance=%s algorithm=%s", instance_name or "unknown", algorithm_name)
    interpreted_objective = interpret_objective(objective_text)
    base_result = run_base_variant(instance, interpreted_objective, algorithm_name)
    base_result["llm_calls"] = 1
    base_result["interpreted_objective"] = sorted(interpreted_objective)
    logger.info(
        "Interpret variant completed: instance=%s algo=%s success=%s num_courses=%s llm_calls=1 elapsed_time=%.4f",
        instance_name or "unknown",
        algorithm_name,
        base_result.get("success", False),
        base_result.get("num_courses", 0),
        base_result.get("elapsed_time", 0.0),
    )
    return base_result


def run_evaluate_variant(
    instance: PlanningInstance,
    objective: Set[str],
    algorithm_name: AlgorithmName = "greedy",
    use_ollama: bool = True,
    instance_name: Optional[str] = None,
) -> VariantResult:
    if not use_ollama:
        raise ValueError("evaluate variant requires Ollama enabled.")
    logger.info("Running evaluate variant: instance=%s algorithm=%s", instance_name or "unknown", algorithm_name)
    base_result = run_base_variant(instance, objective, algorithm_name)
    try:
        llm_eval = evaluate_trajectory(base_result["trajectory"] or [], instance, objective)
    except Exception as exc:
        logger.warning("run_evaluate_variant: evaluation failed: %s", exc)
        llm_eval = {
            "score": None,
            "nota": None,
            "justification": f"Evaluation failed: {exc}",
            "qualitative_comment": "LLM evaluation unavailable due to error.",
            "llm_response": None,
            "valid": base_result.get("success", False),
            "coverage": None,
        }
    base_result["llm_calls"] = 1
    base_result["llm_evaluation"] = llm_eval
    logger.info(
        "Evaluate variant completed: instance=%s algo=%s success=%s num_courses=%s llm_calls=1 elapsed_time=%.4f",
        instance_name or "unknown",
        algorithm_name,
        base_result.get("success", False),
        base_result.get("num_courses", 0),
        base_result.get("elapsed_time", 0.0),
    )
    return base_result


def run_guided_variant(
    instance: PlanningInstance,
    objective: Set[str],
    algorithm_name: AlgorithmName = "greedy",
    use_ollama: bool = True,
    instance_name: Optional[str] = None,
) -> VariantResult:
    if not use_ollama:
        raise ValueError("guided variant requires Ollama enabled.")
    logger.info("Running guided variant: instance=%s algorithm=%s", instance_name or "unknown", algorithm_name)
    start_time = time.perf_counter()
    acquired_skills = set(instance.initial_skills)
    completed_courses: Set[str] = set()
    trajectory: list = []
    llm_step_log: list = []
    llm_calls = 0
    last_suggestion: Optional[Dict[str, Any]] = None

    while not objective.issubset(acquired_skills):
        candidates = available_courses(instance, acquired_skills, completed_courses)
        if not candidates:
            break

        llm_calls += 1
        suggestion = suggest_next_course(trajectory, instance, objective)
        chosen_course_id = suggestion.get("course_id")
        source = "llm"
        candidate_ids = {course.id for course in candidates}
        if chosen_course_id not in candidate_ids:
            chosen_course_id = next(iter(candidate_ids), None)
            source = "fallback"
            suggestion["justification"] = suggestion.get("justification") or "LLM failed; using fallback."

        if not chosen_course_id or chosen_course_id in completed_courses:
            break

        chosen_course = next((course for course in candidates if course.id == chosen_course_id), None)
        if chosen_course is None:
            break

        acquired_skills.update(chosen_course.skills_granted)
        completed_courses.add(chosen_course_id)
        trajectory.append(chosen_course_id)
        suggestion_match = source == "llm" and suggestion.get("course_id") == chosen_course_id
        llm_step_log.append(
            {
                "chosen_course": chosen_course_id,
                "source": source,
                "suggestion_match": suggestion_match,
                "available_courses": [f"{course.id} ({course.name})" for course in candidates],
                "justification": suggestion.get("justification"),
                "step_index": len(llm_step_log) + 1,
            }
        )
        last_suggestion = suggestion

    total_cost = 0
    num_courses = len(trajectory)
    success = objective.issubset(acquired_skills)
    if trajectory:
        total_cost = sum(instance.courses[course_id].credits for course_id in trajectory if course_id in instance.courses)
    elapsed_time = time.perf_counter() - start_time
    logger.info(
        "Guided variant completed: instance=%s algo=%s success=%s num_courses=%s llm_calls=%s elapsed_time=%.4f",
        instance_name or "unknown",
        algorithm_name,
        success,
        num_courses,
        llm_calls,
        elapsed_time,
    )

    return _build_common_result(
        trajectory=trajectory if success else None,
        total_cost=total_cost,
        elapsed_time=elapsed_time,
        num_courses=num_courses,
        success=success,
        llm_calls=llm_calls,
        llm_suggestion=last_suggestion,
        llm_step_log=llm_step_log,
    )


def run_variant(
    variant: VariantName,
    instance: PlanningInstance,
    objective: Any,
    algorithm_name: AlgorithmName = "greedy",
    use_ollama: bool = False,
    instance_name: Optional[str] = None,
) -> VariantResult:
    if variant == "A" or variant == "base":
        if not isinstance(objective, set):
            raise ValueError("Variant A requires a set of objective skills.")
        return run_base_variant(instance, objective, algorithm_name, instance_name=instance_name)
    if variant == "B" or variant == "interpret":
        if not isinstance(objective, str):
            raise ValueError("Variant B requires a natural language objective string.")
        return run_interpret_variant(instance, objective, algorithm_name, use_ollama, instance_name=instance_name)
    if variant == "C" or variant == "evaluate":
        if not isinstance(objective, set):
            raise ValueError("Variant C requires a set of objective skills.")
        return run_evaluate_variant(instance, objective, algorithm_name, use_ollama, instance_name=instance_name)
    if variant == "D" or variant == "guided":
        if not isinstance(objective, set):
            raise ValueError("Variant D requires a set of objective skills.")
        return run_guided_variant(instance, objective, algorithm_name, use_ollama, instance_name=instance_name)
    raise ValueError(f"Unknown variant '{variant}'. Choose A, B, C or D.")


def load_variant_instance(instance_path: Path) -> PlanningInstance:
    return load_instance_from_file(instance_path)
