from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm_interface import evaluate_trajectory, interpret_objective, suggest_next_course
from src.problem_formalization import Course, PlanningInstance


class TestLLMInterface(unittest.TestCase):
    @patch("src.llm_interface.call_ollama")
    def test_interpret_objective_valid_response(self, mock_call):
        mock_call.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"habilidades": ["Data Analysis", "Machine Learning"]}'
                    }
                }
            ]
        }
        result = interpret_objective("Necesito avanzar en análisis de datos y machine learning.")
        self.assertEqual(result, {"Data Analysis", "Machine Learning"})

    @patch("src.llm_interface.call_ollama")
    def test_interpret_objective_invalid_json_fallback(self, mock_call):
        mock_call.side_effect = [
            {"choices": [{"message": {"content": 'No puedo responder eso.'}}]},
            {"choices": [{"message": {"content": '{"habilidades": ["Python"]}'}}]},
        ]
        result = interpret_objective("Quiero aprender Python.")
        self.assertEqual(result, {"Python"})

    @patch("src.llm_interface.call_ollama")
    def test_interpret_objective_no_habilidades(self, mock_call):
        mock_call.return_value = {"choices": [{"message": {"content": '{"skills": ["Python"]}'}}]}
        result = interpret_objective("Necesito habilidades técnicas.")
        self.assertEqual(result, set())

    @patch("src.llm_interface.call_ollama")
    def test_interpret_objective_response_text_with_json(self, mock_call):
        mock_call.return_value = {
            "choices": [
                {
                    "message": {
                        "content": 'Aquí está tu respuesta: {"habilidades": ["SQL", "Cloud Computing"]}'
                    }
                }
            ]
        }
        result = interpret_objective("Quiero destacar en SQL y cloud.")
        self.assertEqual(result, {"SQL", "Cloud Computing"})

    @patch("src.llm_interface.call_ollama")
    def test_evaluate_trajectory_valid_llm_response(self, mock_call):
        mock_call.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"nota": 8.5, "justificacion": "Buena trayectoria."}'
                    }
                }
            ]
        }
        instance = PlanningInstance(
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
        result = evaluate_trajectory(["C1", "C2"], instance, {"SQL"})
        self.assertEqual(result["nota"], 8.5)
        self.assertEqual(result["justification"], "Buena trayectoria.")
        self.assertTrue(0.0 <= result["score"] <= 10.0)
        self.assertTrue(result["valid"])
        self.assertEqual(result["coverage"], 1.0)

    @patch("src.llm_interface.call_ollama")
    def test_evaluate_trajectory_invalid_llm_response(self, mock_call):
        mock_call.return_value = {"choices": [{"message": {"content": 'Error inesperado'}}]}
        instance = PlanningInstance(
            skills={"Python"},
            courses={
                "C1": Course(
                    id="C1",
                    name="Intro Python",
                    skills_granted={"Python"},
                    skills_required=set(),
                    prerequisites=set(),
                    credits=3,
                    difficulty=1.0,
                )
            },
            initial_skills=set(),
            target_skills={"Python"},
        )
        result = evaluate_trajectory(["C1"], instance, {"Python"})
        self.assertIsNone(result["nota"])
        self.assertIn("LLM evaluation not available", result["justification"])
        self.assertTrue(0.0 <= result["score"] <= 10.0)
        self.assertTrue(result["valid"])
        self.assertEqual(result["coverage"], 1.0)

    @patch("src.llm_interface.call_ollama")
    def test_suggest_next_course_matches_base(self, mock_call):
        mock_call.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"course_id": "C2", "justificacion": "Sigue la ruta necesaria."}'
                    }
                }
            ]
        }
        instance = PlanningInstance(
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
        result = suggest_next_course(["C1"], instance, {"SQL"})
        self.assertEqual(result["course_id"], "C2")
        self.assertEqual(result["base_course_id"], "C2")
        self.assertTrue(result["match"])
        self.assertIn("C2", result["available_courses"])

    @patch("src.llm_interface.call_ollama")
    def test_suggest_next_course_invalid_response(self, mock_call):
        mock_call.return_value = {"choices": [{"message": {"content": 'No entiendo.'}}]}
        instance = PlanningInstance(
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
        result = suggest_next_course(["C1"], instance, {"SQL"})
        self.assertIsNone(result["course_id"])
        self.assertFalse(result["match"])
        self.assertIn("C2", result["available_courses"])


if __name__ == "__main__":
    unittest.main()
