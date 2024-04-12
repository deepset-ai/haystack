from typing import Any, Dict, List

from haystack.core.component import component


@component
class AnswerExactMatchEvaluator:
    """
    Evaluator that checks if predicted answers exactly match ground truth answers.

    Each predicted answer is compared to one ground truth answer.
    The final score is a number ranging from 0.0 to 1.0.
    It represents the proportion of predicted answers that match their corresponding ground truth answer.

    Usage example:
    ```python
    from haystack.components.evaluators import AnswerExactMatchEvaluator

    evaluator = AnswerExactMatchEvaluator()
    result = evaluator.run(
        ground_truth_answers=["Berlin", "Paris"],
        predicted_answers=["Berlin", "Lyon"],
    )

    print(result["individual_scores"])
    # [1, 0]
    print(result["score"])
    # 0.5
    ```
    """

    @component.output_types(individual_scores=List[int], score=float)
    def run(self, ground_truth_answers: List[str], predicted_answers: List[str]) -> Dict[str, Any]:
        """
        Run the AnswerExactMatchEvaluator on the given inputs.
        `ground_truth_answers` and `retrieved_answers` must have the same length.

        :param ground_truth_answers:
            A list of expected answers.
        :param predicted_answers:
            A list of predicted answers.
        :returns:
            A dictionary with the following outputs:
            - `individual_scores` - A list of 0s and 1s, where 1 means that the predicted answer matched one of the ground truth.
            - `score` - A number from 0.0 to 1.0 that represents the proportion of questions where any predicted
                         answer matched one of the ground truth answers.
        """
        if not len(ground_truth_answers) == len(predicted_answers):
            raise ValueError("The length of ground_truth_answers and predicted_answers must be the same.")

        matches = []
        for truth, extracted in zip(ground_truth_answers, predicted_answers):
            if truth == extracted:
                matches.append(1)
            else:
                matches.append(0)

        # The proportion of questions where any predicted answer matched one of the ground truth answers
        average = sum(matches) / len(predicted_answers)

        return {"individual_scores": matches, "score": average}
