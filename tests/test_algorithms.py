from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.problem_formalization import (
    load_instance_from_file,
    is_valid_trajectory,
    compute_trajectory_cost,
)
from src.exact_small import exact_small_search
from src.a_star import a_star_search
from src.greedy import greedy_search


def run_test_for_instance(instance_path: Path) -> None:
    instance = load_instance_from_file(instance_path)
    print(f"\nTesting instance: {instance_path.name}")
    results: List[Dict[str, object]] = []
    algorithms: List[tuple[str, Callable]] = [
        ("Exact", exact_small_search),
        ("A*", a_star_search),
        ("Greedy", greedy_search),
    ]

    for name, algorithm in algorithms:
        print(f"Running {name}...")
        result = algorithm(instance)
        trajectory = result.get("trajectory")
        success = result.get("success", trajectory is not None)
        total_cost = result.get("total_cost", 0)
        num_courses = result.get("num_courses", 0)
        elapsed_time = result.get("elapsed_time", 0.0)

        valid_trajectory = None
        if trajectory is not None:
            valid_trajectory = is_valid_trajectory(trajectory, instance)
            computed_cost = compute_trajectory_cost(trajectory, instance)
            if computed_cost != total_cost:
                raise AssertionError(
                    f"Cost mismatch for {name} on {instance_path.name}: reported {total_cost}, computed {computed_cost}"
                )
            if not valid_trajectory:
                raise AssertionError(
                    f"Invalid trajectory returned by {name} for {instance_path.name}."
                )
        elif success:
            raise AssertionError(
                f"Algorithm {name} claimed success but returned no trajectory for {instance_path.name}."
            )

        results.append(
            {
                "name": name,
                "trajectory": trajectory,
                "success": success,
                "total_cost": total_cost,
                "num_courses": num_courses,
                "elapsed_time": elapsed_time,
                "valid_trajectory": valid_trajectory,
            }
        )

        print(
            f"  {name}: success={success}, cost={total_cost}, courses={num_courses}, time={elapsed_time:.4f}s"
        )

    exact_result = next(r for r in results if r["name"] == "Exact")
    astar_result = next(r for r in results if r["name"] == "A*")
    greedy_result = next(r for r in results if r["name"] == "Greedy")

    if exact_result["success"] and astar_result["success"]:
        if astar_result["total_cost"] > exact_result["total_cost"]:
            print(
                f"  A* found a suboptimal solution relative to Exact by {astar_result['total_cost'] - exact_result['total_cost']} credits."
            )
    if exact_result["success"] and greedy_result["success"]:
        if greedy_result["total_cost"] > exact_result["total_cost"]:
            print(
                f"  Greedy found a suboptimal solution relative to Exact by {greedy_result['total_cost'] - exact_result['total_cost']} credits."
            )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    instances_folder = root / "data" / "instances"
    test_files = [
        instances_folder / "synthetic_10_courses_01.json",
        instances_folder / "manual_01_impossible_cycle.json",
        instances_folder / "manual_02_goal_already_achieved.json",
        instances_folder / "manual_03_unreachable_target.json",
        instances_folder / "manual_04_no_prereqs.json",
        instances_folder / "manual_05_full_chain.json",
    ]

    for test_file in test_files:
        if not test_file.exists():
            raise FileNotFoundError(f"Test instance not found: {test_file}")
        run_test_for_instance(test_file)

    print("\nAll selected tests completed successfully.")


if __name__ == "__main__":
    main()
