import json
import unittest

from src.privacy_abstraction import PrivacyMemoryAbstractor


class PMABackendTest(unittest.TestCase):
    def test_oracle_calls_model_and_keeps_safe_fallbacks(self):
        calls = []

        def complete(prompt):
            calls.append(prompt)
            return json.dumps(
                {
                    "candidates": [
                        {
                            "level": "L2",
                            "public_memory": (
                                "The user benefits from low-allergen environments."
                            ),
                            "private_residue": [],
                            "abstraction_trace": [
                                {
                                    "raw": "pollen allergy",
                                    "public_abstraction": "low-allergen environments",
                                    "reason": "Preserves a functional constraint.",
                                }
                            ],
                        },
                        {
                            "level": "L3",
                            "public_memory": "The user prefers low-allergen plans.",
                            "private_residue": [],
                            "abstraction_trace": [
                                {
                                    "raw": "pollen allergy",
                                    "public_abstraction": "low-allergen plans",
                                    "reason": "Preserves a task preference.",
                                }
                            ],
                        },
                    ]
                }
            )

        abstractor = PrivacyMemoryAbstractor(
            backend="oracle_prompt",
            completion_fn=complete,
            fallback_on_error=False,
        )
        candidates = abstractor.generate_candidates(
            "My pollen allergy is active.",
            [
                {
                    "original_text": "pollen allergy",
                    "privacy_type": "Medical Health",
                    "privacy_level": "PL3",
                }
            ],
            "recommendation",
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            [candidate.level for candidate in candidates],
            ["L2", "L3", "L4", "L5"],
        )
        self.assertTrue(
            all(
                "pollen allergy" not in candidate.public_memory
                for candidate in candidates
            )
        )

    def test_pl4_is_redacted_even_in_l4_fallback(self):
        abstractor = PrivacyMemoryAbstractor(backend="heuristic")
        candidates = abstractor.generate_candidates(
            "Use code 829417.",
            [
                {
                    "original_text": "829417",
                    "privacy_type": "Verification Code",
                    "privacy_level": "PL4",
                }
            ],
        )
        l4 = next(candidate for candidate in candidates if candidate.level == "L4")
        self.assertIn("***", l4.public_memory)
        self.assertNotIn("829417", l4.public_memory)
        self.assertEqual(l4.private_residue[0].retention, "no_retention")

    def test_backend_failure_only_emits_safe_fallbacks(self):
        def fail(_prompt):
            raise RuntimeError("offline")

        abstractor = PrivacyMemoryAbstractor(
            backend="oracle_prompt",
            completion_fn=fail,
            fallback_on_error=True,
        )
        candidates = abstractor.generate_candidates(
            "Email me at user@example.com.",
            [
                {
                    "original_text": "user@example.com",
                    "privacy_type": "Email",
                    "privacy_level": "PL2",
                }
            ],
        )
        self.assertEqual([candidate.level for candidate in candidates], ["L4", "L5"])
        self.assertIn("RuntimeError", abstractor.last_backend_error)


if __name__ == "__main__":
    unittest.main()
