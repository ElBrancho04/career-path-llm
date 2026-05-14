from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional

Skill = str
CourseId = str


@dataclass
class Course:
    id: CourseId
    name: str
    skills_granted: Set[Skill]
    skills_required: Set[Skill] = field(default_factory=set)
    prerequisites: Set[CourseId] = field(default_factory=set)
    credits: int = 1
    difficulty: float = 1.0
    description: Optional[str] = None


@dataclass
class PlanningInstance:
    skills: Set[Skill]
    courses: Dict[CourseId, Course]
    initial_skills: Set[Skill]
    target_skills: Set[Skill]


@dataclass
class SolutionTrajectory:
    course_sequence: List[CourseId]
    total_credits: int
    total_courses: int
    acquired_skills: Set[Skill]


def is_course_available(course: Course, acquired_skills: Set[Skill], completed_courses: Set[CourseId]) -> bool:
    """Check if a course can be taken given current skills and completed courses."""
    if not course.skills_required.issubset(acquired_skills):
        return False
    if not course.prerequisites.issubset(completed_courses):
        return False
    return True


def apply_course(course: Course, acquired_skills: Set[Skill], completed_courses: Set[CourseId]) -> None:
    """Update skill and course sets after taking a course."""
    acquired_skills.update(course.skills_granted)
    completed_courses.add(course.id)


def is_valid_trajectory(
    trajectory: List[CourseId],
    instance: PlanningInstance,
    initial_skills: Optional[Set[Skill]] = None,
) -> bool:
    """Verify that a sequence of courses is valid for the given instance."""
    if initial_skills is None:
        initial_skills = set(instance.initial_skills)
    acquired_skills = set(initial_skills)
    completed_courses: Set[CourseId] = set()

    for course_id in trajectory:
        if course_id not in instance.courses:
            return False
        course = instance.courses[course_id]
        if not is_course_available(course, acquired_skills, completed_courses):
            return False
        apply_course(course, acquired_skills, completed_courses)

    return instance.target_skills.issubset(acquired_skills)


def compute_trajectory_cost(trajectory: List[CourseId], instance: PlanningInstance) -> int:
    """Compute total credit cost for a trajectory."""
    return sum(instance.courses[course_id].credits for course_id in trajectory if course_id in instance.courses)


def build_solution(
    trajectory: List[CourseId],
    instance: PlanningInstance,
    initial_skills: Optional[Set[Skill]] = None,
) -> SolutionTrajectory:
    """Build a solution object with aggregated cost and acquired skills."""
    if initial_skills is None:
        initial_skills = set(instance.initial_skills)
    acquired_skills = set(initial_skills)
    completed_courses: Set[CourseId] = set()
    for course_id in trajectory:
        course = instance.courses[course_id]
        apply_course(course, acquired_skills, completed_courses)

    return SolutionTrajectory(
        course_sequence=trajectory,
        total_credits=compute_trajectory_cost(trajectory, instance),
        total_courses=len(trajectory),
        acquired_skills=acquired_skills,
    )


def example_instance() -> PlanningInstance:
    """Return a small example of the planning problem."""
    courses = {
        "C1": Course(
            id="C1",
            name="Introducción a Python",
            skills_granted={"Python"},
            skills_required=set(),
            prerequisites=set(),
            credits=3,
            difficulty=1.0,
        ),
        "C2": Course(
            id="C2",
            name="Fundamentos de SQL",
            skills_granted={"SQL"},
            skills_required={"Python"},
            prerequisites={"C1"},
            credits=3,
            difficulty=1.0,
        ),
        "C3": Course(
            id="C3",
            name="Machine Learning Básico",
            skills_granted={"Machine Learning"},
            skills_required={"Python", "SQL"},
            prerequisites={"C1", "C2"},
            credits=4,
            difficulty=2.0,
        ),
    }

    return PlanningInstance(
        skills={"Python", "SQL", "Machine Learning"},
        courses=courses,
        initial_skills=set(),
        target_skills={"Machine Learning"},
    )
