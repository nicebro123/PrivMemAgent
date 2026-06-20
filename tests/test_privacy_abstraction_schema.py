import unittest

from src.privacy_schema import (
    AbstractionCandidate,
    AbstractionTrace,
    PolicyValidationError,
    PrivacyItem,
    PrivateResidue,
    contains_raw_value,
    validate_candidate_against_policy,
)


class PrivacyAbstractionSchemaTest(unittest.TestCase):
    def test_valid_candidate_passes(self):
        candidate = AbstractionCandidate(
            candidate_id="c1",
            source_id="s1",
            level="L2",
            public_memory="The user prefers nearby indoor activities.",
            private_residue=[
                PrivateResidue(
                    raw="15 Rue de Rivoli",
                    privacy_type="Detailed Address",
                    privacy_level="PL2",
                    retention="local_only",
                )
            ],
            abstraction_trace=[
                AbstractionTrace(
                    raw="15 Rue de Rivoli",
                    public_abstraction="nearby indoor activities",
                    reason="Hides exact address.",
                )
            ],
        )
        self.assertEqual(candidate.to_dict()["level"], "L2")

    def test_missing_public_memory_fails_for_non_redaction(self):
        with self.assertRaises(ValueError):
            AbstractionCandidate.from_dict(
                {
                    "candidate_id": "c1",
                    "source_id": "s1",
                    "level": "L2",
                    "public_memory": "",
                    "private_residue": [],
                    "abstraction_trace": [],
                }
            )

    def test_invalid_level_fails(self):
        with self.assertRaises(ValueError):
            AbstractionCandidate.from_dict(
                {
                    "candidate_id": "c1",
                    "source_id": "s1",
                    "level": "L9",
                    "public_memory": "memory",
                    "private_residue": [],
                    "abstraction_trace": [],
                }
            )

    def test_contains_raw_value_detects_public_leak(self):
        items = [
            PrivacyItem(
                original_text="829417",
                privacy_type="Verification Code",
                privacy_level="PL4",
            )
        ]
        self.assertTrue(contains_raw_value("The code is 829417.", items))
        self.assertFalse(contains_raw_value("The code is hidden.", items))

    def test_direct_construction_is_validated(self):
        with self.assertRaises(ValueError):
            PrivacyItem("", "Email", "PL2")

    def test_policy_rejects_raw_private_value(self):
        item = PrivacyItem("829417", "Verification Code", "PL4")
        candidate = AbstractionCandidate(
            candidate_id="c1",
            source_id="s1",
            level="L5",
            public_memory="The code is 829417.",
            private_residue=[
                PrivateResidue(
                    "829417",
                    "Verification Code",
                    "PL4",
                    "no_retention",
                )
            ],
            abstraction_trace=[AbstractionTrace("829417", "***", "Remove the secret.")],
            metadata={
                "effective_levels": [
                    {
                        "raw": "829417",
                        "privacy_type": "Verification Code",
                        "level": "L5",
                    }
                ]
            },
        )
        policy = {
            "allowed_levels": {"PL4": ["L5"]},
            "type_overrides": {
                "Verification Code": {
                    "allowed_levels": ["L5"],
                    "retention": "no_retention",
                }
            },
        }
        with self.assertRaises(PolicyValidationError):
            validate_candidate_against_policy(candidate, [item], policy)

    def test_policy_requires_local_residue(self):
        item = PrivacyItem("pollen allergy", "Medical Health", "PL3")
        candidate = AbstractionCandidate(
            candidate_id="c1",
            source_id="s1",
            level="L2",
            public_memory="The user needs low-allergen environments.",
            abstraction_trace=[
                AbstractionTrace(
                    "pollen allergy",
                    "low-allergen environments",
                    "Preserves a functional constraint.",
                )
            ],
            metadata={
                "effective_levels": [
                    {
                        "raw": "pollen allergy",
                        "privacy_type": "Medical Health",
                        "level": "L2",
                    }
                ]
            },
        )
        policy = {
            "allowed_levels": {"PL3": ["L2", "L3", "L4", "L5"]},
            "type_overrides": {},
        }
        with self.assertRaises(PolicyValidationError):
            validate_candidate_against_policy(candidate, [item], policy)


if __name__ == "__main__":
    unittest.main()
