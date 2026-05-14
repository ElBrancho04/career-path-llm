from __future__ import annotations

import heapq
import time
from typing import Dict, List, Optional, Set, Tuple

from src.problem_formalization import (
    PlanningInstance,
    Course,
    compute_trajectory_cost,
    is_course_available,
    skills_after_sequence,
)


def _state_key(completed_courses: Set[str]) -> frozenset[str]:
    return frozenset(completed_courses)


def _available_courses(
    instance: PlanningInstance,
    acquired_skills: Set[str],
    completed_courses: Set[str],
) -> List[Course]:
    return [
        course
        for course in instance.courses.values()
        if course.id not in completed_courses and is_course_available(course, acquired_skills, completed_courses)
    ]


def exact_small_search(
    instance: PlanningInstance,
    max_iterations: int = 100000,
) -> Dict[str, Optional[object]]:
    """Find the minimum-cost trajectory for a small planning instance.

    This algorithm is intended for small instances where an exact search over
    the state space is feasible. It uses Dijkstra-like expansion over states
    represented by the set of completed courses.

    Returns a dictionary with keys:
    - trajectory: list of course ids or None
    - total_cost: integer credit cost of the trajectory
    - elapsed_time: seconds spent searching
    - num_courses: number of courses in the trajectory
    - success: boolean indicator
    """
    start_time = time.perf_counter()
    initial_skills = set(instance.initial_skills)
    start_state: frozenset[str] = _state_key(set())
    initial_acquired = set(initial_skills)

    if instance.target_skills.issubset(initial_acquired):
        return {
            "trajectory": [],
            "total_cost": 0,
            "elapsed_time": time.perf_counter() - start_time,
            "num_courses": 0,
            "success": True,
        }

    frontier: List[Tuple[int, frozenset[str], List[str]]] = [
        (0, start_state, [])
    ]
    best_cost: Dict[frozenset[str], int] = {start_state: 0}
    iterations = 0

    while frontier and iterations < max_iterations:
        iterations += 1
        cost, state, trajectory = heapq.heappop(frontier)
        if best_cost.get(state, float("inf")) < cost:
            continue

        completed_courses = set(state)
        acquired_skills = set(initial_skills)
        for course_id in trajectory:
            course = instance.courses[course_id]
            acquired_skills.update(course.skills_granted)

        if instance.target_skills.issubset(acquired_skills):
            return {
                "trajectory": trajectory,
                "total_cost": cost,
                "elapsed_time": time.perf_counter() - start_time,
                "num_courses": len(trajectory),
                "success": True,
            }

        for course in _available_courses(instance, acquired_skills, completed_courses):
            next_completed = set(completed_courses)
            next_completed.add(course.id)
            next_state = _state_key(next_completed)
            next_cost = cost + course.credits
            if next_cost >= best_cost.get(next_state, float("inf")):
                continue

            next_trajectory = trajectory + [course.id]
            best_cost[next_state] = next_cost
            heapq.heappush(frontier, (next_cost, next_state, next_trajectory))

    elapsed_time = time.perf_counter() - start_time
    return {
        "trajectory": None,
        "total_cost": 0,
        "elapsed_time": elapsed_time,
        "num_courses": 0,
        "success": False,
    }


def verify_trajectory_cost(
    trajectory: List[str],
    instance: PlanningInstance,
) -> int:
    """Compute and verify the cost of a trajectory for the given instance."""
    return compute_trajectory_cost(trajectory, instance)
