import json
import os
import tempfile
import unittest

from evaluation.eval_pma_mem0 import evaluate_memory_system
from src.memory_backends import InMemoryBackend
from src.privacy_abstraction import load_abstraction_policy
from src.privacy_auditor import AdversarialMemoryAuditor


class MemoryEvaluationIntegrationTest(unittest.TestCase):
    def test_real_write_retrieve_answer_attack_loop(self):
        user = {
            "uuid": "u1",
            "metadata": {"user_name": "Test"},
            "dialogues": [
                {
                    "role": "user",
                    "content": "15 Rue de Rivoli",
                    "privacy_info": [
                        {
                            "original_text": "15 Rue de Rivoli",
                            "privacy_type": "Detailed Address",
                            "privacy_level": "PL2",
                        }
                    ],
                }
            ],
            "questions": [
                {
                    "question": "Which address was stated?",
                    "answer": "A",
                    "all_options": ["A. 15 Rue de Rivoli", "B. Unknown"],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            dataset = os.path.join(directory, "users.jsonl")
            with open(dataset, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(user) + "\n")

            def answer(memory, _question):
                return "A" if "15 Rue de Rivoli" in memory else "B"

            result = evaluate_memory_system(
                input_path=dataset,
                users=None,
                max_turns=None,
                methods=["raw", "complete", "type_specific"],
                task_family="general",
                policy=load_abstraction_policy(),
                utility_threshold=0.85,
                mask_levels={"PL2", "PL3", "PL4"},
                turns_per_chunk=1,
                memory_backend_factory=lambda _user, _method: InMemoryBackend(),
                answer_fn=answer,
                judge_fn=None,
                auditor=AdversarialMemoryAuditor(),
                memory_system_name="in_memory_ci",
                paper_evidence=False,
            )
        self.assertEqual(result["methods"]["raw"]["utility"]["mcq_accuracy"], 1.0)
        self.assertEqual(
            result["methods"]["complete"]["utility"]["mcq_accuracy"],
            0.0,
        )
        self.assertEqual(
            result["methods"]["raw"]["privacy"]["exact_reconstruction_rate"],
            1.0,
        )
        self.assertEqual(
            result["methods"]["complete"]["privacy"]["exact_reconstruction_rate"],
            0.0,
        )
        self.assertIn(
            "Detailed Address",
            result["methods"]["raw"]["privacy"]["per_type"],
        )
        self.assertFalse(result["run"]["paper_evidence"])


if __name__ == "__main__":
    unittest.main()
