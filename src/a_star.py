from __future__ import annotations

import heapq
import itertools
import random
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


def _heuristic_missing_courses(
    instance: PlanningInstance,
    acquired_skills: Set[str],
    completed_courses: Set[str],
) -> int:
    missing_skills = set(instance.target_skills) - acquired_skills
    if not missing_skills:
        return 0

    candidate_courses = [
        course
        for course in instance.courses.values()
        if course.id not in completed_courses
    ]

    min_credits = min((c.credits for c in instance.courses.values()), default=1)

    remaining = set(missing_skills)
    steps = 0
    while remaining:
        best_course = max(
            candidate_courses,
            key=lambda course: len(course.skills_granted & remaining),
            default=None,
        )
        if best_course is None or not (best_course.skills_granted & remaining):
            return len(remaining) * min_credits
        remaining -= best_course.skills_granted
        steps += 1
    return steps * min_credits


def a_star_search(
    instance: PlanningInstance,
    max_iterations: int = 200000,
    rng: Optional[random.Random] = None,
) -> Dict[str, Optional[object]]:
    """A* search for a trajectory that reaches the target skills.

    This solver models states as the set of completed courses and expands
    successors by adding one valid available course at a time.
    It is intended as a heuristic method for moderate-sized instances.
    """
    start_time = time.perf_counter()
    initial_skills = set(instance.initial_skills)
    start_state = _state_key(set())

    if instance.target_skills.issubset(initial_skills):
        return {
            "trajectory": [],
            "total_cost": 0,
            "elapsed_time": time.perf_counter() - start_time,
            "num_courses": 0,
            "success": True,
        }

    start_h = _heuristic_missing_courses(instance, initial_skills, set())
    # C3: Tie-break without perturbing f-score (preserves A* admissibility).
    # We use an insertion counter as second key and optionally shuffle successor
    # generation order, which affects exploration but not costs.
    counter = itertools.count()
    frontier: List[Tuple[int, int, int, frozenset[str], List[str]]] = [
        (start_h, next(counter), 0, start_state, [])
    ]
    best_cost: Dict[frozenset[str], int] = {start_state: 0}
    iterations = 0

    while frontier and iterations < max_iterations:
        iterations += 1
        f, _, g, state, trajectory = heapq.heappop(frontier)
        if best_cost.get(state, float("inf")) < g:
            continue

        completed_courses = set(state)
        acquired_skills = skills_after_sequence(trajectory, instance, initial_skills)

        if instance.target_skills.issubset(acquired_skills):
            return {
                "trajectory": trajectory,
                "total_cost": g,
                "elapsed_time": time.perf_counter() - start_time,
                "num_courses": len(trajectory),
                "success": True,
            }

        successors = _available_courses(instance, acquired_skills, completed_courses)
        if rng is not None:
            rng.shuffle(successors)
        for course in successors:
            next_completed = set(completed_courses)
            next_completed.add(course.id)
            next_state = _state_key(next_completed)
            next_g = g + course.credits
            if next_g >= best_cost.get(next_state, float("inf")):
                continue

            next_acquired_skills = set(acquired_skills)
            next_acquired_skills.update(course.skills_granted)
            next_h = _heuristic_missing_courses(instance, next_acquired_skills, next_completed)
            next_f = next_g + next_h
            next_trajectory = trajectory + [course.id]

            best_cost[next_state] = next_g
            heapq.heappush(frontier, (next_f, next(counter), next_g, next_state, next_trajectory))

    elapsed_time = time.perf_counter() - start_time
    return {
        "trajectory": None,
        "total_cost": 0,
        "elapsed_time": elapsed_time,
        "num_courses": 0,
        "success": False,
    }
