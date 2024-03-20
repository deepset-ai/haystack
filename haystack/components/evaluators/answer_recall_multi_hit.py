import itertools
from typing import Any, Dict, List

from haystack.core.component import component


@component
class AnswerRecallMultiHitEvaluator:
    """
    Evaluator that calculates the Recall multi-hit score for a list of questions.
    The result is number from 0.0 to 1.0 that represents the proportion of matching answers retrieved across
    all questions divided by the total number of relevant items in all answers.
    Each question can have multiple ground truth answers and multiple predicted answers.

    Usage example:
    ```python
    from haystack.components.evaluators import AnswerRecallMultiHitEvaluator

    evaluator = AnswerRecallMultiHitEvaluator()
    result = evaluator.run(
        questions=["What is the capital of Germany?", "What is the capital of France?"],
        ground_truth_answers=[["Berlin"], ["Paris"]],
        predicted_answers=[["Berlin"], ["London"]],
    )
    print(result["score"])
    # 0.5
    ```
    """

    @component.output_types(score=float)
    def run(
        self, questions: List[str], ground_truth_answers: List[List[str]], predicted_answers: List[List[str]]
    ) -> Dict[str, Any]:
        """
        Run the AnswerRecallMultiHitEvaluator on the given inputs.
        All lists must have the same length.

        :param questions:
            A list of questions.
        :param ground_truth_answers:
            A list of expected answers for each question.
        :param predicted_answers:
            A list of predicted answers for each question.

        :returns:
            A dictionary with the following outputs:
            - `score` - A number from 0.0 to 1.0 that represents the proportion of matching answers retrieved across
                        all questions divided by the total number of relevant items in all answers.
        """
        if not len(questions) == len(ground_truth_answers) == len(predicted_answers):
            msg = "The length of questions, ground_truth_answers, and predicted_answers must be the same."
            raise ValueError(msg)

        correct_retrievals = set()
        for ground_truth, predicted in zip(ground_truth_answers, predicted_answers):
            retrieved_ground_truths = {g for g, p in itertools.product(ground_truth, predicted) if g in p}
            correct_retrievals.update(retrieved_ground_truths)

        score = len(correct_retrievals) / sum(len(answers) for answers in ground_truth_answers)

        return {"score": score}
