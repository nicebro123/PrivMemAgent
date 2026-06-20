import json
import unittest

from src.privacy_auditor import AdversarialMemoryAuditor


class AuditorIsolationTest(unittest.TestCase):
    def test_llm_attack_prompt_never_contains_raw_value(self):
        prompts = []

        def complete(prompt):
            prompts.append(prompt)
            return json.dumps(
                {"guess": "", "confidence": 0.1, "rationale": "Insufficient data."}
            )

        auditor = AdversarialMemoryAuditor(
            backend="llm",
            completion_fn=complete,
        )
        result = auditor.attack_exact_reconstruction(
            "The user's verification code was removed.",
            {
                "original_text": "829417",
                "privacy_type": "Verification Code",
                "privacy_level": "PL4",
            },
        )
        self.assertFalse(result.success)
        self.assertEqual(len(prompts), 1)
        self.assertNotIn("829417", prompts[0])

    def test_public_raw_value_is_visible_to_attacker(self):
        auditor = AdversarialMemoryAuditor(
            backend="llm",
            completion_fn=lambda _prompt: json.dumps(
                {"guess": "829417", "confidence": 1.0, "rationale": "Visible."}
            ),
        )
        result = auditor.attack_exact_reconstruction(
            "The code is 829417.",
            {
                "original_text": "829417",
                "privacy_type": "Verification Code",
                "privacy_level": "PL4",
            },
        )
        self.assertTrue(result.success)


if __name__ == "__main__":
    unittest.main()
