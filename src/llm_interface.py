from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Set

from src.llm_wrapper import call_ollama

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
