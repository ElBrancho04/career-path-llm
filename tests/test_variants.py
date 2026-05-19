from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.problem_formalization import Course, PlanningInstance
from src.variant_runner import run_variant


class TestVariantRunner(unittest.TestCase):
    def setUp(self) -> None:
        self.instance = PlanningInstance(
            skills={"Python", "SQL"},
            courses={
                "C1": Course(
                    id="C1",
                    name="Intro Python",
                    skills_granted={"Python"},
                    skills_required=set(),
                    prerequisites=set(),
                    credits=3,
                    difficulty=1.0,
                ),
                "C2": Course(
                    id="C2",
                    name="SQL Basics",
                    skills_granted={"SQL"},
                    skills_required={"Python"},
                    prerequisites={"C1"},
                    credits=3,
                    difficulty=1.0,
                ),
            },
            initial_skills=set(),
            target_skills={"SQL"},
        )

    def test_variant_a_base_greedy(self):
        result = run_variant(
            variant="A",
            instance=self.instance,
            objective={"SQL"},
            algorithm_name="greedy",
            use_ollama=False,
            instance_name="test_instance.json",
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["llm_calls"], 0)
        self.assertEqual(result["num_courses"], 2)
        self.assertEqual(result["trajectory"], ["C1", "C2"])

    @patch("src.variant_runner.interpret_objective")
    def test_variant_b_interpret(self, mock_interpret):
        mock_interpret.return_value = {"SQL"}
        result = run_variant(
            variant="B",
            instance=self.instance,
            objective="I want to learn SQL.",
            algorithm_name="greedy",
            use_ollama=True,
            instance_name="test_instance.json",
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["llm_calls"], 1)
        self.assertEqual(result["interpreted_objective"], ["SQL"])

    @patch("src.variant_runner.evaluate_trajectory")
    def test_variant_c_evaluate(self, mock_evaluate):
        mock_evaluate.return_value = {
            "score": 9.0,
            "nota": 9.0,
            "justification": "Good path.",
            "qualitative_comment": "Valid.",
        }
        result = run_variant(
            variant="C",
            instance=self.instance,
            objective={"SQL"},
            algorithm_name="greedy",
            use_ollama=True,
            instance_name="test_instance.json",
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["llm_calls"], 1)
        self.assertIn("llm_evaluation", result)
        self.assertEqual(result["llm_evaluation"]["nota"], 9.0)

    @patch("src.variant_runner.suggest_next_course")
    def test_variant_d_guided(self, mock_suggest):
        mock_suggest.return_value = {
            "course_id": "C1",
            "justification": "Start with Python.",
        }
        result = run_variant(
            variant="D", 
            instance=self.instance,
            objective={"SQL"},
            algorithm_name="greedy",
            use_ollama=True,
            instance_name="test_instance.json",
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["llm_calls"], 2)
        self.assertEqual(result["trajectory"], ["C1", "C2"])
        self.assertIn("llm_step_log", result)
        self.assertEqual(result["llm_step_log"][0]["source"], "llm")


if __name__ == "__main__":
    unittest.main()
