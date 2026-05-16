from __future__ import annotations

import time
from typing import Dict, List, Optional, Set

from src.problem_formalization import (
    PlanningInstance,
    available_courses,
    compute_trajectory_cost,
)


def greedy_search(
    instance: PlanningInstance,
) -> Dict[str, Optional[object]]:
    """Greedy baseline solver for professional trajectory planning.

    This baseline selects at each step the available course that provides the
    largest number of new target skills still missing. In case of ties it prefers
    lower credit courses and lower difficulty.

    It is intended as a fast heuristic baseline, not as an optimal planner.
    """
    start_time = time.perf_counter()
    acquired_skills: Set[str] = set(instance.initial_skills)
    completed_courses: Set[str] = set()
    trajectory: List[str] = []

    while not instance.target_skills.issubset(acquired_skills):
        candidates = available_courses(instance, acquired_skills, completed_courses)
        if not candidates:
            break

        missing_target = instance.target_skills - acquired_skills
        best_course = None
        best_score = (-1, -1, float("inf"), float("inf"))

        for course in candidates:
            new_skills = course.skills_granted - acquired_skills
            primary = len(new_skills & missing_target)
            secondary = len(new_skills)
            score = (primary, secondary, -course.credits, -course.difficulty)
            if score > best_score:
                best_score = score
                best_course = course

        if best_course is None:
            break

        # If the best course does not contribute any new skills, stop.
        if best_score[0] == 0 and best_score[1] == 0:
            break

        acquired_skills.update(best_course.skills_granted)
        completed_courses.add(best_course.id)
        trajectory.append(best_course.id)

    total_cost = compute_trajectory_cost(trajectory, instance)
    elapsed_time = time.perf_counter() - start_time
    success = instance.target_skills.issubset(acquired_skills)

    return {
        "trajectory": trajectory if success else None,
        "total_cost": total_cost,
        "elapsed_time": elapsed_time,
        "num_courses": len(trajectory),
        "success": success,
    }
