from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


@dataclass
class GeneratorParameters:
    num_skills: int
    num_courses: int
    prereq_density: float
    target_skill_size: int
    initial_skill_size: int = 0
    max_credits: int = 5
    max_difficulty: float = 5.0


DEFAULT_SKILL_NAMES = [
    "Python",
    "SQL",
    "Statistics",
    "Data Analysis",
    "Machine Learning",
    "Deep Learning",
    "Cloud Computing",
    "Data Engineering",
    "Algorithms",
    "Software Design",
    "Natural Language Processing",
    "Computer Vision",
    "Databases",
    "Distributed Systems",
    "Cybersecurity",
    "Project Management",
    "Communication",
    "DevOps",
    "Big Data",
    "AI Ethics",
]


def _build_skill_pool(num_skills: int) -> List[str]:
    if num_skills <= len(DEFAULT_SKILL_NAMES):
        return DEFAULT_SKILL_NAMES[:num_skills]
    return DEFAULT_SKILL_NAMES + [f"Skill {i}" for i in range(len(DEFAULT_SKILL_NAMES) + 1, num_skills + 1)]


def _make_course_name(index: int, granted_skills: Set[str]) -> str:
    skills_label = ", ".join(sorted(list(granted_skills)))
    return f"Course {index:02d} ({skills_label})"


def generate_instance(seed: int, parameters: GeneratorParameters) -> Dict[str, Any]:
    rng = random.Random(seed)
    skill_pool = _build_skill_pool(parameters.num_skills)
    course_ids = [f"C{index:02d}" for index in range(1, parameters.num_courses + 1)]

    skill_to_courses: Dict[str, List[str]] = {skill: [] for skill in skill_pool}
    courses: List[Dict[str, Any]] = []
    available_skills: Set[str] = set()

    remaining_skills = list(skill_pool)
    rng.shuffle(remaining_skills)

    for idx, course_id in enumerate(course_ids, start=1):
        course_required_skills: Set[str] = set()
        course_prerequisites: Set[str] = set()

        # Ensure progressive coverage of skills across courses
        grant_count = rng.choice([1, 1, 2])
        skills_granted: Set[str] = set()
        while len(skills_granted) < grant_count and remaining_skills:
            skills_granted.add(remaining_skills.pop())
        while len(skills_granted) < grant_count:
            skills_granted.add(rng.choice(skill_pool))

        # Determine skills required from already available skills
        if available_skills:
            required_count = rng.choices([0, 1, 2], weights=[0.4, 0.4, 0.2], k=1)[0]
            course_required_skills = set(rng.sample(sorted(available_skills), min(required_count, len(available_skills))))

        # Build prerequisites from skill requirements
        for skill in sorted(course_required_skills):
            candidate_courses = skill_to_courses.get(skill, [])
            if candidate_courses:
                course_prerequisites.add(rng.choice(candidate_courses))


        if idx > 1:
            previous_courses = course_ids[: idx - 1]
            for prev_course_id in previous_courses:
                if rng.random() < parameters.prereq_density:
                    prev_course_entry = next(
                        (c for c in courses if c["id"] == prev_course_id), None
                    )
                    if prev_course_entry:
                        course_prerequisites.add(prev_course_id)
                        prev_skills = set(prev_course_entry.get("skills_granted", []))
                        addable = prev_skills - course_required_skills
                        if addable:
                            course_required_skills.add(rng.choice(sorted(addable)))

        credits = rng.randint(2, parameters.max_credits)
        difficulty = round(rng.uniform(1.0, parameters.max_difficulty), 1)

        courses.append(
            {
                "id": course_id,
                "name": _make_course_name(idx, skills_granted),
                "skills_granted": sorted(skills_granted),
                "skills_required": sorted(course_required_skills),
                "prerequisites": sorted(course_prerequisites),
                "credits": credits,
                "difficulty": difficulty,
            }
        )

        available_skills.update(skills_granted)
        for skill in skills_granted:
            skill_to_courses[skill].append(course_id)

    # Choose target skills from the active skill pool
    target_skills = sorted(rng.sample(skill_pool, min(parameters.target_skill_size, len(skill_pool))))
    initial_skills = sorted(rng.sample(skill_pool, min(parameters.initial_skill_size, len(skill_pool))))

    return {
        "skills": sorted(skill_pool),
        "courses": courses,
        "initial_skills": initial_skills,
        "target_skills": target_skills,
    }


def save_instance(instance: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(instance, stream, indent=2, ensure_ascii=False)


def generate_synthetic_instances(output_folder: Path) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)
    size_profiles = [
        (10, [0.05, 0.08, 0.12], [3, 4, 5]),
        (30, [0.04, 0.06, 0.08], [3, 4, 5]),
        (100, [0.02, 0.04, 0.06], [3, 4, 5]),
    ]
    index = 1
    for num_courses, densities, target_options in size_profiles:
        for repeat in range(1, 11):
            density = densities[(repeat - 1) % len(densities)]
            target_size = target_options[(repeat - 1) % len(target_options)]
            params = GeneratorParameters(
                # CAMBIO 4: corregido num_courses // 1 (no-op) por num_courses // 5.
                # Antes: para 100 cursos se generaban 100 habilidades, produciendo
                # instancias extremadamente dispersas sin estructura real.
                # Ahora: para 100 cursos se generan 20 habilidades, consistente con
                # DEFAULT_SKILL_NAMES y con los perfiles de tamaño pequeño y mediano.
                num_skills=max(num_courses // 5, 10),
                num_courses=num_courses,
                prereq_density=density,
                target_skill_size=target_size,
                initial_skill_size=0,
            )
            instance = generate_instance(seed=index, parameters=params)
            file_name = f"synthetic_{num_courses}_courses_{repeat:02d}.json"
            save_instance(instance, output_folder / file_name)
            index += 1

    # Produce extra variations with some initial skills
    for extra in range(1, 4):
        params = GeneratorParameters(
            num_skills=20,
            num_courses=30,
            prereq_density=0.08,
            target_skill_size=4,
            initial_skill_size=2,
        )
        instance = generate_instance(seed=100 + extra, parameters=params)
        file_name = f"synthetic_30_courses_initial_skills_{extra:02d}.json"
        save_instance(instance, output_folder / file_name)


def generate_manual_instances(output_folder: Path) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)

    manual_cases = [
        {
            "file": "manual_01_impossible_cycle.json",
            "instance": {
                "skills": ["Python", "SQL", "Machine Learning"],
                "courses": [
                    {
                        "id": "C1",
                        "name": "Intro to Python",
                        "skills_granted": ["Python"],
                        "skills_required": [],
                        "prerequisites": ["C3"],
                        "credits": 3,
                        "difficulty": 1.0,
                    },
                    {
                        "id": "C2",
                        "name": "SQL Basics",
                        "skills_granted": ["SQL"],
                        "skills_required": ["Python"],
                        "prerequisites": ["C1"],
                        "credits": 3,
                        "difficulty": 1.2,
                    },
                    {
                        "id": "C3",
                        "name": "Machine Learning Overview",
                        "skills_granted": ["Machine Learning"],
                        "skills_required": ["SQL"],
                        "prerequisites": ["C2"],
                        "credits": 4,
                        "difficulty": 2.0,
                    },
                ],
                "initial_skills": [],
                "target_skills": ["Machine Learning"],
            },
        },
        {
            "file": "manual_02_goal_already_achieved.json",
            "instance": {
                "skills": ["Python", "SQL", "Data Analysis"],
                "courses": [
                    {
                        "id": "C1",
                        "name": "Intro to Python",
                        "skills_granted": ["Python"],
                        "skills_required": [],
                        "prerequisites": [],
                        "credits": 2,
                        "difficulty": 1.0,
                    },
                    {
                        "id": "C2",
                        "name": "Intro to SQL",
                        "skills_granted": ["SQL"],
                        "skills_required": [],
                        "prerequisites": [],
                        "credits": 2,
                        "difficulty": 1.1,
                    },
                ],
                "initial_skills": ["Python", "Data Analysis"],
                "target_skills": ["Data Analysis"],
            },
        },
        {
            "file": "manual_03_unreachable_target.json",
            "instance": {
                "skills": ["Python", "SQL", "Machine Learning", "Deep Learning"],
                "courses": [
                    {
                        "id": "C1",
                        "name": "Intro to Python",
                        "skills_granted": ["Python"],
                        "skills_required": [],
                        "prerequisites": [],
                        "credits": 3,
                        "difficulty": 1.0,
                    },
                    {
                        "id": "C2",
                        "name": "SQL Basics",
                        "skills_granted": ["SQL"],
                        "skills_required": ["Python"],
                        "prerequisites": ["C1"],
                        "credits": 3,
                        "difficulty": 1.5,
                    },
                ],
                "initial_skills": [],
                "target_skills": ["Deep Learning"],
            },
        },
        {
            "file": "manual_04_no_prereqs.json",
            "instance": {
                "skills": ["Python", "SQL", "Statistics", "Machine Learning"],
                "courses": [
                    {
                        "id": "C1",
                        "name": "Intro to Python",
                        "skills_granted": ["Python"],
                        "skills_required": [],
                        "prerequisites": [],
                        "credits": 2,
                        "difficulty": 1.0,
                    },
                    {
                        "id": "C2",
                        "name": "Statistics Fundamentals",
                        "skills_granted": ["Statistics"],
                        "skills_required": [],
                        "prerequisites": [],
                        "credits": 2,
                        "difficulty": 1.3,
                    },
                    {
                        "id": "C3",
                        "name": "Machine Learning Basics",
                        "skills_granted": ["Machine Learning"],
                        "skills_required": [],
                        "prerequisites": [],
                        "credits": 4,
                        "difficulty": 2.0,
                    },
                ],
                "initial_skills": [],
                "target_skills": ["Machine Learning"],
            },
        },
        {
            "file": "manual_05_full_chain.json",
            "instance": {
                "skills": [
                    "Programming Basics",
                    "Python",
                    "Data Structures",
                    "Algorithms",
                    "Machine Learning",
                ],
                "courses": [
                    {
                        "id": "C1",
                        "name": "Programming Basics",
                        "skills_granted": ["Programming Basics"],
                        "skills_required": [],
                        "prerequisites": [],
                        "credits": 2,
                        "difficulty": 1.0,
                    },
                    {
                        "id": "C2",
                        "name": "Python Programming",
                        "skills_granted": ["Python"],
                        "skills_required": ["Programming Basics"],
                        "prerequisites": ["C1"],
                        "credits": 3,
                        "difficulty": 1.5,
                    },
                    {
                        "id": "C3",
                        "name": "Data Structures",
                        "skills_granted": ["Data Structures"],
                        "skills_required": ["Python"],
                        "prerequisites": ["C2"],
                        "credits": 3,
                        "difficulty": 2.0,
                    },
                    {
                        "id": "C4",
                        "name": "Algorithms",
                        "skills_granted": ["Algorithms"],
                        "skills_required": ["Data Structures"],
                        "prerequisites": ["C3"],
                        "credits": 4,
                        "difficulty": 2.5,
                    },
                    {
                        "id": "C5",
                        "name": "Applied Machine Learning",
                        "skills_granted": ["Machine Learning"],
                        "skills_required": ["Algorithms"],
                        "prerequisites": ["C4"],
                        "credits": 4,
                        "difficulty": 3.0,
                    },
                ],
                "initial_skills": [],
                "target_skills": ["Machine Learning"],
            },
        },
    ]

    for case in manual_cases:
        save_instance(case["instance"], output_folder / case["file"])


def create_data_readme(output_folder: Path) -> None:
    content = """# Instance format

Each JSON instance describes a professional planning problem.

Fields:
- `skills`: list of all skills available in the instance.
- `courses`: list of course objects.
- `initial_skills`: skills already acquired before planning.
- `target_skills`: skills required to meet the professional objective.

Each course object contains:
- `id`: unique course identifier.
- `name`: readable course name.
- `skills_granted`: skills acquired after completing the course.
- `skills_required`: skills that must already be available before taking the course.
- `prerequisites`: list of course ids that must be completed before the course.
- `credits`: integer credit cost.
- `difficulty`: floating difficulty score.

The planning problem is to find an ordered sequence of courses that:
1. respects all course prerequisites,
2. satisfies each course's skill requirements at the time it is taken,
3. and accumulates at least the target skills by the end.
"""
    output_folder.mkdir(parents=True, exist_ok=True)
    with (output_folder / "README.md").open("w", encoding="utf-8") as stream:
        stream.write(content)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    instances_folder = project_root / "data" / "instances"
    generate_synthetic_instances(instances_folder)
    generate_manual_instances(instances_folder)
    create_data_readme(project_root / "data")


if __name__ == "__main__":
    main()
