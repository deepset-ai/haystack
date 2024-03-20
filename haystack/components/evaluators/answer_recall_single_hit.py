import itertools
from typing import Any, Dict, List

from haystack.core.component import component


@component
class AnswerRecallSingleHitEvaluator:
    """
    Evaluator that calculates the Recall single-hit score for a list of questions.
    Returns both a list of scores for each question and the average.
    Each question can have multiple ground truth answers and multiple predicted answers.

    Usage example:
    ```python
    from haystack.components.evaluators import AnswerRecallSingleHitEvaluator

    evaluator = AnswerRecallSingleHitEvaluator()
    result = evaluator.run(
        questions=["What is the capital of Germany?", "What is the capital of France?"],
        ground_truth_answers=[["Berlin"], ["Paris"]],
        predicted_answers=[["Paris"], ["London"]],
    )

    print(result["scores"])
    # [0.0, 0.0]
    print(result["average"])
    # 0.0
    ```
    """

    @component.output_types(scores=List[float], average=float)
    def run(
        self, questions: List[str], ground_truth_answers: List[List[str]], predicted_answers: List[List[str]]
    ) -> Dict[str, Any]:
        """
        Run the AnswerRecallSingleHitEvaluator on the given inputs.
        All lists must have the same length.

        :param questions:
            A list of questions.
        :param ground_truth_answers:
            A list of expected answers for each question.
        :param predicted_answers:
            A list of predicted answers for each question.

        A dictionary with the following outputs:
        - `scores` - A list of numbers from 0.0 to 1.0 that represents the proportion of matching answers retrieved.
        - `score` - The average of calculated scores.

        """
        if not len(questions) == len(ground_truth_answers) == len(predicted_answers):
            msg = "The length of questions, ground_truth_answers, and predicted_answers must be the same."
            raise ValueError(msg)

        scores = []
        for ground_truth, predicted in zip(ground_truth_answers, predicted_answers):
            retrieved_ground_truths = {g for g, p in itertools.product(ground_truth, predicted) if g in p}
            score = len(retrieved_ground_truths) / len(ground_truth)
            scores.append(score)

        return {"scores": scores, "average": sum(scores) / len(questions)}
