from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set

from src.llm_wrapper import call_ollama
from src.problem_formalization import (
    PlanningInstance,
    compute_trajectory_cost,
    is_valid_trajectory,
    skills_after_sequence,
    available_courses,
)

logger = logging.getLogger(__name__)


def _build_objective_prompt(text: str) -> str:
    return (
        'Extrae las habilidades profesionales deseadas de la siguiente frase. '
        'Responde únicamente un JSON: {"habilidades": [..]} sin texto adicional. '
        f'Frase: {text}'
    )


def _build_objective_prompt_strict(text: str) -> str:
    return (
        'Extrae las habilidades profesionales deseadas de la siguiente frase. '
        'Responde únicamente un JSON válido con la clave "habilidades". '
        'No incluyas texto adicional. '
        f'Frase: {text}'
    )


def _extract_text_from_response(response: Dict[str, Any]) -> str:
    if not isinstance(response, dict):
        return str(response)

    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            if "message" in first and isinstance(first["message"], dict):
                return str(first["message"].get("content", ""))
            if "content" in first:
                return str(first["content"])
            if "text" in first:
                return str(first["text"])
    if "text" in response:
        return str(response["text"])
    if "response" in response:
        return str(response["response"])
    if "output" in response:
        output = response["output"]
        if isinstance(output, list) and output:
            return str(output[0])
        return str(output)
    return json.dumps(response)


def _parse_json_snippet(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract first JSON object substring; use a greedy search for braces content.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Fallback: try to locate a JSON object by finding the first '{' and last '}'
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass

    raise ValueError("No valid JSON found in response text.")


def _build_evaluate_prompt(objective: Set[str], trajectory: List[str], instance: PlanningInstance) -> str:
    course_list = [f"{course_id} ({instance.courses[course_id].name})" for course_id in trajectory]
    objective_str = ", ".join(sorted(objective))
    return (
        f"Evalúa la siguiente secuencia de cursos para alcanzar el objetivo {objective_str}. "
        f"Cursos: {course_list}. "
        'Devuelve únicamente un JSON: {"nota": float, "justificacion": "..."}.'
    )


def _parse_evaluate_response(response: Dict[str, Any]) -> Dict[str, Any]:
    raw_text = _extract_text_from_response(response)
    parsed = _parse_json_snippet(raw_text)
    nota = parsed.get("nota")
    justificacion = parsed.get("justificacion")
    if nota is None or justificacion is None:
        raise ValueError("LLM response missing nota or justificacion.")
    return {
        "nota": float(nota),
        "justificacion": str(justificacion),
        "raw_text": raw_text,
    }


def _heuristic_score(
    trajectory: List[str],
    instance: PlanningInstance,
    objective: Set[str],
) -> float:
    if not objective:
        return 10.0

    acquired_skills = set(instance.initial_skills)
    for course_id in trajectory:
        if course_id in instance.courses:
            acquired_skills.update(instance.courses[course_id].skills_granted)

    coverage = len(acquired_skills & objective) / len(objective)
    validity = 1.0 if is_valid_trajectory(trajectory, instance) else 0.0
    course_penalty = max(0.0, 1.0 - (len(trajectory) / max(len(instance.courses), 1)))
    score = 10.0 * (0.5 * coverage + 0.3 * validity + 0.2 * course_penalty)
    return max(0.0, min(10.0, score))


def evaluate_trajectory(
    trajectory: List[str],
    instance: PlanningInstance,
    objective: Set[str],
) -> Dict[str, Any]:
    prompt = _build_evaluate_prompt(objective, trajectory, instance)
    llm_response: Optional[Dict[str, Any]] = None
    llm_result: Dict[str, Any] = {}
    try:
        llm_response = call_ollama(prompt)
        llm_result = _parse_evaluate_response(llm_response)
    except Exception as exc:
        logger.warning("evaluate_trajectory: LLM evaluation failed: %s", exc)
        llm_result = {"nota": None, "justificacion": "LLM evaluation not available."}

    heuristic = _heuristic_score(trajectory, instance, objective)
    valid = is_valid_trajectory(trajectory, instance)
    acquired_skills = skills_after_sequence(trajectory, instance, instance.initial_skills)
    coverage = len(acquired_skills & objective) / len(objective) if objective else 1.0
    qualitative_comment = (
        f"La trayectoria {'es válida' if valid else 'no es válida'}. "
        f"Cobertura del objetivo: {coverage:.2f}. Cursos: {len(trajectory)}."
    )

    return {
        "score": heuristic,
        "nota": llm_result.get("nota"),
        "justification": llm_result.get("justificacion"),
        "qualitative_comment": qualitative_comment,
        "llm_response": llm_response,
        "valid": valid,
        "coverage": coverage,
    }


def _select_greedy_next_course(
    available: List[Any],
    acquired_skills: Set[str],
    objective: Set[str],
) -> Optional[str]:
    best_course_id: Optional[str] = None
    best_score = (-1, -1, float("inf"), float("inf"))
    missing_target = objective - acquired_skills
    for course in available:
        new_skills = course.skills_granted - acquired_skills
        primary = len(new_skills & missing_target)
        secondary = len(new_skills)
        score = (primary, secondary, -course.credits, -course.difficulty)
        if score > best_score:
            best_score = score
            best_course_id = course.id
    return best_course_id


def _build_suggestion_prompt(objective: Set[str], available_courses_list: List[str]) -> str:
    objective_str = ", ".join(sorted(objective))
    course_list_str = ", ".join(available_courses_list)
    return (
        f"¿Qué curso recomendarías a continuación para alcanzar {objective_str}? "
        f"Cursos disponibles: {course_list_str}. "
        'Responde únicamente un JSON con {"course_id": "...", "justificacion": "..."}.'
    )


def _parse_suggestion_response(response: Dict[str, Any]) -> Dict[str, str]:
    raw_text = _extract_text_from_response(response)
    parsed = _parse_json_snippet(raw_text)
    course_id = parsed.get("course_id")
    justification = parsed.get("justificacion")
    if not course_id or not justification:
        raise ValueError("LLM suggestion missing course_id or justificacion.")
    return {"course_id": str(course_id).strip(), "justification": str(justification).strip()}


def suggest_next_course(
    partial_trajectory: List[str],
    instance: PlanningInstance,
    objective: Set[str],
) -> Dict[str, Any]:
    acquired_skills = set(instance.initial_skills)
    try:
        acquired_skills = skills_after_sequence(partial_trajectory, instance, instance.initial_skills)
    except Exception as exc:
        logger.warning("suggest_next_course: invalid partial trajectory: %s", exc)

    completed_courses = set(partial_trajectory)
    available = available_courses(instance, acquired_skills, completed_courses)
    available_ids = [course.id for course in available]
    best_greedy = _select_greedy_next_course(available, acquired_skills, objective)

    if not available_ids:
        result = {
            "course_id": None,
            "justification": "No available courses remain.",
            "model_course_id": None,
            "base_course_id": best_greedy,
            "match": best_greedy is None,
            "available_courses": available_ids,
            "llm_response": None,
        }
        logger.info("suggest_next_course result: %s", result)
        return result

    prompt = _build_suggestion_prompt(objective, [f"{c.id} ({c.name})" for c in available])
    llm_response = None
    suggestion: Dict[str, Optional[str]] = {"course_id": None, "justification": ""}
    try:
        llm_response = call_ollama(prompt)
        parsed = _parse_suggestion_response(llm_response)
        suggestion["course_id"] = parsed["course_id"]
        suggestion["justification"] = parsed["justification"]
    except Exception as exc:
        logger.warning("suggest_next_course: LLM suggestion failed: %s", exc)
        suggestion["justification"] = "LLM suggestion unavailable."

    match = suggestion["course_id"] == best_greedy if suggestion["course_id"] is not None else False
    result = {
        "course_id": suggestion["course_id"],
        "justification": suggestion["justification"],
        "model_course_id": suggestion["course_id"],
        "base_course_id": best_greedy,
        "match": match,
        "available_courses": available_ids,
        "llm_response": llm_response,
    }
    logger.info("suggest_next_course comparison: %s", result)
    return result


def interpret_objective(text: str) -> Set[str]:
    prompt = _build_objective_prompt(text)
    try:
        response = call_ollama(prompt)
        raw_text = _extract_text_from_response(response)
        parsed = _parse_json_snippet(raw_text)
        habilidades = parsed.get("habilidades")
        if isinstance(habilidades, list):
            return {str(item).strip() for item in habilidades if item is not None}
        logger.warning("interpret_objective: JSON parsed but 'habilidades' missing or invalid. Response: %s", raw_text)
    except Exception as exc:
        logger.warning("interpret_objective first attempt failed: %s", exc)

    # Second attempt with stricter prompt
    prompt = _build_objective_prompt_strict(text)
    try:
        response = call_ollama(prompt)
        raw_text = _extract_text_from_response(response)
        parsed = _parse_json_snippet(raw_text)
        habilidades = parsed.get("habilidades")
        if isinstance(habilidades, list):
            return {str(item).strip() for item in habilidades if item is not None}
        logger.warning("interpret_objective second attempt: JSON parsed but 'habilidades' missing or invalid. Response: %s", raw_text)
    except Exception as exc:
        logger.error("interpret_objective second attempt failed: %s", exc)

    return set()
