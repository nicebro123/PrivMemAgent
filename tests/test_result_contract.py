import unittest

from evaluation.compare_pma_results import validate_result_contract


class ResultContractTest(unittest.TestCase):
    def test_rejects_proxy_or_incomparable_results(self):
        result = {
            "run": {"memory_system": "in_memory_ci", "paper_evidence": False},
            "methods": {
                "raw": {
                    "utility": {"total_num": 1, "proxy": True},
                    "privacy": {"num_attacks": 1, "proxy": True, "per_type": {}},
                }
            },
            "records": [],
        }
        errors = validate_result_contract(
            result,
            {"raw", "type_specific"},
            require_paper_evidence=True,
        )
        self.assertTrue(any("missing required methods" in error for error in errors))
        self.assertTrue(any("non-paper evidence" in error for error in errors))
        self.assertTrue(any("proxy" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
