from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Set

from src.variant_runner import load_variant_instance, run_variant


def parse_objective_text(objective: str) -> Set[str]:
    return {item.strip() for item in objective.split(",") if item.strip()}


def save_result(result: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)


def summarize_result(result: Dict[str, Any]) -> None:
    print("Result summary:")
    print(f"  success: {result.get('success')}")
    print(f"  total_cost: {result.get('total_cost')}")
    print(f"  num_courses: {result.get('num_courses')}")
    print(f"  elapsed_time: {result.get('elapsed_time'):.4f} sec")
    print(f"  llm_calls: {result.get('llm_calls')}")
    if "llm_evaluation" in result:
        evaluation = result["llm_evaluation"]
        print(f"  llm_evaluation.score: {evaluation.get('score')}")
        print(f"  llm_evaluation.nota: {evaluation.get('nota')}")
    if "llm_suggestion" in result:
        print(f"  llm_suggestion: {result.get('llm_suggestion')}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run career path planner variants.")
    parser.add_argument("--instance", required=True, help="Path to the instance JSON file.")
    parser.add_argument(
        "--variant",
        required=True,
        choices=["A", "B", "C", "D"],
        help="Variant to execute: A, B, C or D.",
    )
    parser.add_argument(
        "--algorithm",
        default="greedy",
        choices=["greedy", "a_star"],
        help="Base algorithm to use for variants A, B, C and D.",
    )
    parser.add_argument(
        "--objective",
        required=True,
        help="Objective text for B, or comma-separated skills for A/C/D.",
    )
    parser.add_argument(
        "--use_ollama",
        action="store_true",
        help="Enable Ollama calls for variants that require the LLM.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save the result JSON (e.g. results/trajectories/output.json).",
    )

    args = parser.parse_args()
    instance_path = Path(args.instance)
    if not instance_path.exists():
        parser.error(f"Instance file does not exist: {instance_path}")

    if args.variant in {"B", "C", "D"} and not args.use_ollama:
        parser.error("Variants B, C and D require --use_ollama to be enabled.")

    instance = load_variant_instance(instance_path)
    objective_value: Any
    if args.variant == "B":
        objective_value = args.objective
    else:
        objective_value = parse_objective_text(args.objective)

    result = run_variant(
        variant=args.variant,
        instance=instance,
        objective=objective_value,
        algorithm_name=args.algorithm,
        use_ollama=args.use_ollama,
    )

    summarize_result(result)

    if args.output:
        output_path = Path(args.output)
        save_result(result, output_path)
        print(f"Result saved to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
