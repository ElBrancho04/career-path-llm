from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm_interface import interpret_objective


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


if __name__ == "__main__":
    unittest.main()
