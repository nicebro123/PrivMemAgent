import unittest

from evaluation.build_pma_train_data import build_preference_records


class TrainingDataBuilderTest(unittest.TestCase):
    def test_skips_identical_and_duplicate_rejected_outputs(self):
        candidate = {
            "candidate_id": "c1",
            "source_id": "s1",
            "public_memory": "***",
            "private_residue": [],
            "abstraction_trace": [],
        }
        records = build_preference_records(
            [
                {
                    "example_id": "s1",
                    "input": {"dialogue": "secret"},
                    "chosen": candidate,
                    "scores": {},
                    "rejected": [
                        {
                            "candidate": {
                                **candidate,
                                "candidate_id": "c2",
                            },
                            "score": {},
                        },
                        {
                            "candidate": {
                                **candidate,
                                "candidate_id": "c3",
                                "public_memory": "<Secret_1>",
                            },
                            "score": {},
                        },
                        {
                            "candidate": {
                                **candidate,
                                "candidate_id": "c4",
                                "public_memory": "<Secret_1>",
                            },
                            "score": {},
                        },
                    ],
                }
            ]
        )
        self.assertEqual(len(records), 1)
        self.assertIn("<Secret_1>", records[0]["rejected"])


if __name__ == "__main__":
    unittest.main()
