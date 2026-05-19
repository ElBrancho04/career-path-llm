from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import requests


def load_config() -> Dict[str, Any]:
    config_path = Path(__file__).resolve().parents[1] / "config" / "llm_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"LLM config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def check_service(endpoint: str, timeout: float = 5.0) -> bool:
    info_url = endpoint.rstrip("/") + "/v1/models"
    try:
        response = requests.get(info_url, timeout=timeout)
        response.raise_for_status()
        return True
    except requests.RequestException:
        return False


def model_is_available(endpoint: str, model_name: str, timeout: float = 5.0) -> bool:
    info_url = endpoint.rstrip("/") + "/v1/models"
    try:
        response = requests.get(info_url, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            models = data.get("models")
            if isinstance(models, list):
                return any(model_name == item.get("id") or model_name == item.get("name") for item in models if isinstance(item, dict))
        elif isinstance(data, list):
            return any(model_name == item.get("id") or model_name == item.get("name") for item in data if isinstance(item, dict))
        return False
    except requests.RequestException:
        return False
    except ValueError:
        return False


def run_simple_completion(endpoint: str, model_name: str, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
    completion_url = endpoint.rstrip("/") + "/v1/completions"
    payloads = [
        {
            "model": model_name,
            "messages": [
                {"role": "user", "content": "Hello!"}
            ],
            "temperature": 0.0,
            "max_tokens": 5,
        },
        {
            "model": model_name,
            "prompt": "Hello!",
            "temperature": 0.0,
            "max_tokens": 5,
        },
    ]
    for payload in payloads:
        try:
            response = requests.post(completion_url, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            continue
        except ValueError:
            continue
    return None


def main() -> int:
    try:
        config = load_config()
    except Exception as exc:
        print(f"ERROR: could not load config - {exc}")
        return 1

    endpoint_local = config.get("endpoint_local")
    model_name = config.get("model")

    if not endpoint_local or not model_name:
        print("ERROR: config file must contain 'endpoint_local' and 'model'.")
        return 1

    print(f"Checking Ollama endpoint: {endpoint_local}")
    service_ok = check_service(endpoint_local)
    if not service_ok:
        print("FAIL: Ollama service did not respond at the configured endpoint.")
        print("Make sure Ollama is running with `ollama serve` and the endpoint is correct.")
        return 1

    print("OK: Ollama service is reachable.")

    print(f"Checking model availability: {model_name}")
    model_ok = model_is_available(endpoint_local, model_name)
    if model_ok:
        print(f"OK: Model '{model_name}' appears available on Ollama.")
    else:
        print(f"WARNING: Model '{model_name}' was not listed by the endpoint.")
        print("Attempting a small completion request to verify the model directly.")
        completion = run_simple_completion(endpoint_local, model_name)
        if completion is None:
            print(f"FAIL: Could not complete request with model '{model_name}'.")
            return 1
        print(f"OK: Completion request succeeded with model '{model_name}'.")
        return 0

    print("Running a simple completion check...")
    completion = run_simple_completion(endpoint_local, model_name)
    if completion is None:
        print("FAIL: Completion request failed despite service and model availability.")
        return 1

    print("OK: Completion request succeeded.")
    print("Ollama is configured correctly for this project.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
