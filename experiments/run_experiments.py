from __future__ import annotations

import argparse
import inspect
import sys
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.variant_runner import load_instance_from_file, run_variant

INSTANCES_DIR = ROOT_DIR / "data" / "instances"
RESULTS_DIR = ROOT_DIR / "results"

VARIANTS = ["A", "B", "C", "D"]
ALGORITHMS = ["greedy", "a_star"]
SEEDS = [1, 2, 3, 4, 5]

SMALL_INSTANCES = [f"synthetic_10_courses_{i:02d}.json" for i in range(1, 11)]
MEDIUM_INSTANCES = [f"synthetic_30_courses_{i:02d}.json" for i in range(1, 11)]
LARGE_INSTANCES = [f"synthetic_100_courses_{i:02d}.json" for i in range(1, 6)]
MANUAL_INSTANCES = [
    "manual_01_impossible_cycle.json",
    "manual_02_goal_already_achieved.json",
    "manual_03_unreachable_target.json",
    "manual_04_no_prereqs.json",
    "manual_05_full_chain.json",
]

SIZE_LABELS = {
    "synthetic_10_courses_": "small",
    "synthetic_30_courses_": "medium",
    "synthetic_100_courses_": "large",
    "manual_": "manual",
}


def get_instance_size(instance_path: Path) -> str:
    """Return the experiment size label for a given instance path."""
    name = instance_path.name
    if name.startswith("synthetic_10_courses_"):
        return "small"
    if name.startswith("synthetic_30_courses_"):
        return "medium"
    if name.startswith("synthetic_100_courses_"):
        return "large"
    if name.startswith("manual_"):
        return "manual"
    raise ValueError(f"Unexpected instance file name for phase 6 selection: {name}")


def build_instance_catalog() -> List[Dict[str, str]]:
    """Return the fixed instance catalog for phase 6 with size labels.

    Only the required files listed for phase 6 are included.
    """
    instance_files: List[Dict[str, str]] = []
    for filename in SMALL_INSTANCES + MEDIUM_INSTANCES + LARGE_INSTANCES + MANUAL_INSTANCES:
        path = INSTANCES_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Required instance file not found: {path}")
        instance_files.append({"path": str(path), "size": get_instance_size(path)})
    return instance_files


def build_smoke_instance_catalog() -> List[Dict[str, str]]:
    """Return a tiny 4-instance catalog (1 per size) for fast validation.

    This supports Phase 6 (Bloque 8): run a reduced subset before launching the
    full experiment matrix.
    """
    candidates = [
        INSTANCES_DIR / SMALL_INSTANCES[0],
        INSTANCES_DIR / MEDIUM_INSTANCES[0],
        INSTANCES_DIR / LARGE_INSTANCES[0],
        INSTANCES_DIR / MANUAL_INSTANCES[0],
    ]
    catalog: List[Dict[str, str]] = []
    for path in candidates:
        if not path.exists():
            raise FileNotFoundError(f"Smoke test instance not found: {path}")
        catalog.append({"path": str(path), "size": get_instance_size(path)})
    return catalog


def normalize_objective(variant: str, objective_text: str) -> Any:
    """Normalize the objective based on the variant."""
    if variant == "B":
        return objective_text
    return set(map(str.strip, objective_text.split(",")))


def get_objective_for_variant(instance: Any, variant: str) -> Any:
    if variant == "B":
        return "Learn skills: " + ", ".join(sorted(instance.target_skills))
    return set(instance.target_skills)


def _run_variant_with_optional_seed(
    variant: str,
    instance: Any,
    objective_value: Any,
    algorithm: str,
    use_ollama: bool,
    instance_name: str,
    seed: int,
) -> Dict[str, Any]:
    signature = inspect.signature(run_variant)
    kwargs = {
        "variant": variant,
        "instance": instance,
        "objective": objective_value,
        "algorithm_name": algorithm,
        "use_ollama": use_ollama,
        "instance_name": instance_name,
    }
    if "seed" in signature.parameters:
        kwargs["seed"] = seed
    return run_variant(**kwargs)


def run_single_execution(
    instance_path: str,
    variant: str,
    algorithm: str,
    objective_value: Any,
    seed: int,
) -> Dict[str, Any]:
    """Run a single execution of the variant on the given instance."""
    instance = load_instance_from_file(Path(instance_path))
    use_ollama = variant in {"B", "C", "D"}
    result = _run_variant_with_optional_seed(
        variant,
        instance,
        objective_value,
        algorithm,
        use_ollama,
        Path(instance_path).name,
        seed,
    )
    return {
        "instance_path": instance_path,
        "variant": variant,
        "algorithm": algorithm,
        "seed": seed,
        "objective_value": objective_value,
        "result": result,
    }


def verify_instance_selection() -> None:
    """Verify the fixed phase 6 instance selection exists and is complete."""
    expected_files = set(SMALL_INSTANCES + MEDIUM_INSTANCES + LARGE_INSTANCES + MANUAL_INSTANCES)
    missing = [filename for filename in expected_files if not (INSTANCES_DIR / filename).exists()]
    if missing:
        missing.sort()
        raise FileNotFoundError("Missing required phase 6 instance files: " + ", ".join(missing))

    counts = {"small": 0, "medium": 0, "large": 0, "manual": 0}
    for filename in expected_files:
        path = INSTANCES_DIR / filename
        counts[get_instance_size(path)] += 1
    expected_counts = {"small": 10, "medium": 10, "large": 5, "manual": 5}
    if counts != expected_counts:
        raise ValueError(f"Phase 6 instance selection counts mismatch: expected {expected_counts}, found {counts}")

    actual_files = {path.name for path in INSTANCES_DIR.iterdir() if path.is_file()}
    extra_files = sorted(actual_files - expected_files)
    if extra_files:
        print(
            "Warning: data/instances contains extra files not used in phase 6 selection:",
            ", ".join(extra_files),
        )


def get_variant_type(variant: str) -> str:
    if variant == "A":
        return "oracle"
    if variant == "B":
        return "auto"
    if variant in {"C", "D"}:
        return "guided"
    return "unknown"


def build_error_row(
    instance_path: str,
    variant: str,
    algorithm: str,
    seed: int,
    instance_size: str,
    run_number: int,
    error: Exception,
) -> Dict[str, Any]:
    return {
        "instance": Path(instance_path).name,
        "instance_size": instance_size,
        "variant": variant,
        "algorithm": algorithm,
        "seed": seed,
        "run": run_number,
        "success": 0,
        "total_cost": None,
        "num_courses": None,
        "elapsed_time": None,
        "llm_calls": 0,
        "llm_evaluation_score": None,
        "llm_evaluation_nota": None,
        "trajectory_found": 0,
        "variant_type": get_variant_type(variant),
        "error_message": str(error),
        "failed": 1,
    }


def build_result_row(
    execution: Dict[str, Any],
    instance_size: str,
    run_number: int,
) -> Dict[str, Any]:
    result = execution["result"]
    trajectory = result.get("trajectory")
    llm_evaluation = result.get("llm_evaluation", {}) or {}
    return {
        "instance": Path(execution["instance_path"]).name,
        "instance_size": instance_size,
        "variant": execution["variant"],
        "algorithm": execution["algorithm"],
        "seed": execution["seed"],
        "run": run_number,
        "success": 1 if result.get("success") else 0,
        "total_cost": result.get("total_cost"),
        "num_courses": result.get("num_courses"),
        "elapsed_time": result.get("elapsed_time"),
        "llm_calls": result.get("llm_calls", 0),
        "llm_evaluation_score": llm_evaluation.get("score"),
        "llm_evaluation_nota": llm_evaluation.get("nota"),
        "trajectory_found": 0 if trajectory is None else 1,
        "variant_type": get_variant_type(execution["variant"]),
        "error_message": None,
        "failed": 0,
    }


def run_experiment_repetitions() -> List[Dict[str, Any]]:
    catalog = build_instance_catalog()
    return run_experiment_repetitions_for_catalog(catalog, VARIANTS, ALGORITHMS, SEEDS)


def run_experiment_repetitions_for_catalog(
    catalog: List[Dict[str, str]],
    variants: List[str],
    algorithms: List[str],
    seeds: List[int],
) -> List[Dict[str, Any]]:
    """Run the repetition loop for a provided catalog.

    Used by --smoke to keep the same row schema and error handling.
    """
    rows: List[Dict[str, Any]] = []
    for entry in catalog:
        instance_path = entry["path"]
        instance_size = entry["size"]
        instance = load_instance_from_file(Path(instance_path))
        for variant in variants:
            objective_value = get_objective_for_variant(instance, variant)
            for algorithm in algorithms:
                for run_number, seed in enumerate(seeds, start=1):
                    try:
                        execution = run_single_execution(
                            instance_path,
                            variant,
                            algorithm,
                            objective_value,
                            seed,
                        )
                        rows.append(build_result_row(execution, instance_size, run_number))
                    except Exception as exc:
                        rows.append(
                            build_error_row(
                                instance_path,
                                variant,
                                algorithm,
                                seed,
                                instance_size,
                                run_number,
                                exc,
                            )
                        )
    return rows


def save_experiment_results(rows: List[Dict[str, Any]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(CSV_OUTPUT_PATH, index=False)
    dataframe.to_json(JSON_OUTPUT_PATH, orient="records", indent=2)


CSV_OUTPUT_PATH = RESULTS_DIR / "experiment_results.csv"
JSON_OUTPUT_PATH = RESULTS_DIR / "experiment_results.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate experiment rows comparing variants A/B/C/D over the fixed Phase 6 instances. "
            "Use --smoke to validate the pipeline quickly (recommended)."
        )
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a reduced subset (1 instance per size, 1 seed) to validate outputs.",
    )
    args = parser.parse_args()

    verify_instance_selection()

    if args.smoke:
        catalog = build_smoke_instance_catalog()
        seeds = [SEEDS[0]]
        print("Running SMOKE subset (Phase 6 Bloque 8):")
        print(f"  instances={len(catalog)} (small/medium/large/manual)")
        print(f"  variants={VARIANTS}")
        print(f"  algorithms={ALGORITHMS}")
        print(f"  seeds={seeds}")
        rows = run_experiment_repetitions_for_catalog(catalog, VARIANTS, ALGORITHMS, seeds)
    else:
        catalog = build_instance_catalog()
        print(f"Verified {len(catalog)} fixed phase 6 instances.")
        print("First instance entry:", catalog[0])
        print("Building experimental rows for reproducibility...")
        rows = run_experiment_repetitions_for_catalog(catalog, VARIANTS, ALGORITHMS, SEEDS)

    print(f"Prepared {len(rows)} experiment rows.")
    save_experiment_results(rows)
    print(f"Saved results to {CSV_OUTPUT_PATH} and {JSON_OUTPUT_PATH}.")
