import unittest

from src.privacy_auditor import is_successful_reconstruction


class AMAMatchingTest(unittest.TestCase):
    def test_exact_otp_match_succeeds(self):
        success, score = is_successful_reconstruction(
            "829417", "829417", "Verification Code"
        )
        self.assertTrue(success)
        self.assertEqual(score, 1.0)

    def test_different_otp_fails(self):
        success, score = is_successful_reconstruction(
            "000000", "829417", "Verification Code"
        )
        self.assertFalse(success)
        self.assertEqual(score, 0.0)

    def test_address_substring_succeeds(self):
        success, score = is_successful_reconstruction(
            "15 Rue de Rivoli",
            "15 Rue de Rivoli, Paris",
            "Detailed Address",
        )
        self.assertTrue(success)
        self.assertGreaterEqual(score, 0.8)

    def test_non_strict_health_overlap_succeeds(self):
        success, score = is_successful_reconstruction(
            "seasonal pollen allergy",
            "pollen allergy",
            "Medical Health",
        )
        self.assertTrue(success)
        self.assertGreaterEqual(score, 0.5)


if __name__ == "__main__":
    unittest.main()
