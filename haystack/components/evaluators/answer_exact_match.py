from typing import Any, Dict, List

from haystack import default_from_dict, default_to_dict
from haystack.core.component import component


@component
class AnswerExactMatchEvaluator:
    """
    Evaluator that checks if the predicted answers matches any of the ground truth answers exactly.
    """

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnswerExactMatchEvaluator":
        return default_from_dict(cls, data)

    @component.output_types(result=float)
    def run(
        self, questions: List[str], ground_truth_answers: List[List[str]], predicted_answers: List[List[str]]
    ) -> Dict[str, float]:
        """
        Run the AnswerExactMatchEvaluator on the given inputs.
        All lists must have the same length.

        :param questions: A list of questions.
        :param ground_truth_answers: A list of expected answers for each question.
        :param predicted_answers: A list of predicted answers for each question.
        :returns: A dictionary with the following outputs:
                * `result` - A number from 0.0 to 1.0, it represents the proportion of questions where any predicted
                answer matched one of the ground truth answers.
        """
        if not len(questions) == len(ground_truth_answers) == len(predicted_answers):
            raise ValueError("The length of questions, ground_truth_answers, and predicted_answers must be the same.")

        matches = 0
        for truths, extracted in zip(ground_truth_answers, predicted_answers):
            if set(truths) & set(extracted):
                matches += 1

        # The proportion of questions where any predicted answer matched one of the ground truth answers
        result = matches / len(questions)

        return {"result": result}
