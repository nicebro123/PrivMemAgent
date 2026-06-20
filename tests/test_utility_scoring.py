import unittest

from src.privacy_critic import QuestionAnswerUtilityEvaluator


class UtilityScoringTest(unittest.TestCase):
    def test_mcq_utility_comes_from_answer_outputs(self):
        def answer(memory, _question):
            return "A" if "central Paris" in memory else "B"

        evaluator = QuestionAnswerUtilityEvaluator(answer)
        questions = [
            {
                "question": "Where is the user based?",
                "answer": "A",
                "all_options": ["A. central Paris", "B. London"],
            }
        ]
        good = evaluator.evaluate("The user is based in central Paris.", questions)
        bad = evaluator.evaluate("The location was removed.", questions)
        self.assertEqual(good.mcq_accuracy, 1.0)
        self.assertEqual(bad.mcq_accuracy, 0.0)
        self.assertFalse(good.is_proxy)

    def test_open_qa_uses_judge(self):
        evaluator = QuestionAnswerUtilityEvaluator(
            lambda _memory, _question: "partly right",
            lambda _question, _prediction, _reference: (0.5, True),
        )
        score = evaluator.evaluate(
            "memory",
            [{"question": "Q", "answer": "reference"}],
        )
        self.assertEqual(score.answer_consistency, 0.5)
        self.assertEqual(score.num_valid, 1)


if __name__ == "__main__":
    unittest.main()
