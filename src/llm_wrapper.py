from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "llm_config.json"
CACHE_PATH = PROJECT_ROOT / "data" / "llm_cache.json"
LOG_PATH = PROJECT_ROOT / "data" / "llm_wrapper.log"

LOGGING_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOGGING_FORMAT,
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def load_llm_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"LLM configuration not found: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _ensure_cache_file() -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CACHE_PATH.exists():
        with CACHE_PATH.open("w", encoding="utf-8") as stream:
            json.dump({}, stream, indent=2)


def load_cache() -> Dict[str, Any]:
    _ensure_cache_file()
    with CACHE_PATH.open("r", encoding="utf-8") as stream:
        try:
            data = json.load(stream)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            logger.warning("Cache file was invalid JSON; resetting cache.")
    return {}


def save_cache(cache: Dict[str, Any]) -> None:
    _ensure_cache_file()
    with CACHE_PATH.open("w", encoding="utf-8") as stream:
        json.dump(cache, stream, indent=2, ensure_ascii=False)


def _make_cache_key(prompt: str, stop: Optional[List[str]], config: Dict[str, Any]) -> str:
    payload = {
        "prompt": prompt,
        "stop": stop,
        "model": config.get("model"),
        "temperature": config.get("temperature"),
        "max_tokens": config.get("max_tokens"),
        "endpoint_local": config.get("endpoint_local"),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _build_payload(prompt: str, stop: Optional[List[str]], config: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": config["model"],
        "temperature": config.get("temperature", 0.0),
        "max_tokens": config.get("max_tokens", 256),
        "prompt": prompt,
    }
    if stop is not None:
        payload["stop"] = stop
    return payload


def _send_request(endpoint: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    url = endpoint.rstrip("/") + "/v1/completions"
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _coerce_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        coerced = float(value)
        if coerced <= 0:
            return default
        return coerced
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int, *, minimum: int = 1) -> int:
    try:
        if value is None:
            return default
        coerced = int(value)
        if coerced < minimum:
            return default
        return coerced
    except (TypeError, ValueError):
        return default


def check_ollama_available(timeout: float = 4.0) -> bool:
    """Verifica rápidamente si Ollama está corriendo y responde.

    Hace un GET al endpoint raíz. Si responde en menos de `timeout` segundos,
    Ollama está disponible. Usar antes de lanzar el experimento completo para
    fallar rápido en lugar de repetir errores durante horas.
    """
    try:
        config = load_llm_config()
        endpoint = config.get("endpoint_local", "http://localhost:11434")
        response = requests.get(endpoint.rstrip("/") + "/api/tags", timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            model_name = config.get("model", "")
            available_models = [m.get("name", "") for m in data.get("models", [])]
            if not any(model_name in m for m in available_models):
                logger.warning(
                    "Ollama está corriendo pero el modelo '%s' no está descargado. "
                    "Ejecuta: ollama pull %s",
                    model_name, model_name,
                )
                return False
            return True
        return False
    except Exception as exc:
        logger.warning("Ollama no disponible: %s", exc)
        return False


def call_ollama(
    prompt: str,
    stop: Optional[List[str]] = None,
    max_retries: Optional[int] = None,
    backoff_base: Optional[float] = None,
) -> Dict[str, Any]:
    config = load_llm_config()
    endpoint = config.get("endpoint_local")
    if not endpoint:
        raise ValueError("LLM config must specify endpoint_local")

    request_timeout = _coerce_float(config.get("request_timeout"), 8.0)
    resolved_max_retries = _coerce_int(max_retries if max_retries is not None else config.get("max_retries"), 2)
    resolved_backoff_base = _coerce_float(
        backoff_base if backoff_base is not None else config.get("backoff_base"),
        0.5,
    )

    cache = load_cache()
    cache_key = _make_cache_key(prompt, stop, config)
    if cache_key in cache:
        logger.info("Cache hit for prompt.")
        return cache[cache_key]

    payload = _build_payload(prompt, stop, config)
    last_exception: Optional[Exception] = None
    for attempt in range(1, resolved_max_retries + 1):
        try:
            logger.info("Calling Ollama (attempt %d)", attempt)
            response = _send_request(endpoint, payload, timeout=request_timeout)
            cache[cache_key] = response
            save_cache(cache)
            logger.info("Ollama response cached successfully.")
            return response
        except requests.RequestException as exc:
            last_exception = exc
            wait = resolved_backoff_base * (2 ** (attempt - 1))
            logger.warning("Ollama request failed on attempt %d: %s", attempt, exc)
            if attempt < resolved_max_retries:
                logger.info("Retrying after %.1f seconds...", wait)
                time.sleep(wait)
        except ValueError as exc:
            last_exception = exc
            logger.error("Invalid response from Ollama: %s", exc)
            break

    error_message = f"Failed to call Ollama after {resolved_max_retries} attempts."
    logger.error(error_message)
    if last_exception:
        logger.exception(last_exception)
    raise RuntimeError(error_message)
