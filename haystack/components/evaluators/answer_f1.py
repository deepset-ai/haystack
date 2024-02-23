from typing import Any, Dict, List

from haystack import default_from_dict, default_to_dict
from haystack.core.component import component


@component
class AnswerF1Evaluator:
    """
    Evaluator that calculates the F1 score of the matches between the predicted and the ground truth answers.
    We first calculate the F1 score for each question, sum all the scores and divide by the number of questions.
    The result is a number from 0.0 to 1.0.

    Each question can have multiple ground truth answers and multiple predicted answers.
    """

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnswerF1Evaluator":
        return default_from_dict(cls, data)

    @component.output_types(result=float)
    def run(
        self, questions: List[str], ground_truth_answers: List[List[str]], predicted_answers: List[List[str]]
    ) -> Dict[str, float]:
        """
        Run the AnswerF1Evaluator on the given inputs.
        All lists must have the same length.

        :param questions: A list of questions.
        :param ground_truth_answers: A list of expected answers for each question.
        :param predicted_answers: A list of predicted answers for each question.
        :returns: A dictionary with the following outputs:
                * `result` - A number from 0.0 to 1.0 that represents the average F1 score of the predicted
                answer matched with the ground truth answers.
        """
        if not len(questions) == len(ground_truth_answers) == len(predicted_answers):
            raise ValueError("The length of questions, ground_truth_answers, and predicted_answers must be the same.")

        scores = []
        for truths, predicted in zip(ground_truth_answers, predicted_answers):
            if len(truths) == 0 and len(predicted) == 0:
                scores.append(1.0)
                continue

            matches = len(set(truths) & set(predicted))
            precision = matches / len(predicted)
            recall = matches / len(truths)
            if (tp := precision + recall) > 0:
                f1 = 2 * (precision * recall) / tp
            else:
                f1 = 0.0
            scores.append(f1)

        result = sum(scores) / len(questions)

        return {"result": result}
