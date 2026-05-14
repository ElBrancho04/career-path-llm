# Instance format

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

1.  respects all course prerequisites,
2.  satisfies each course's skill requirements at the time it is taken,
3.  and accumulates at least the target skills by the end.

## Regenerating instances

Use the generation script at `src/instance_generator.py` to recreate synthetic and manual instances:

```bash
python src/instance_generator.py
```
