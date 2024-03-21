from typing import Any, Dict, List

from haystack.core.component import component


@component
class AnswerExactMatchEvaluator:
    """
    Evaluator that checks if the predicted answers matches any of the ground truth answers exactly.
    The result is a number from 0.0 to 1.0, it represents the proportion of questions where any predicted answer
    matched one of the ground truth answers.
    Each question can have multiple ground truth answers and multiple predicted answers.

    Usage example:
    ```python
    from haystack.components.evaluators import AnswerExactMatchEvaluator

    evaluator = AnswerExactMatchEvaluator()
    result = evaluator.run(
        questions=["What is the capital of Germany?", "What is the capital of France?"],
        ground_truth_answers=[["Berlin"], ["Paris"]],
        predicted_answers=[["Berlin"], ["Lyon"]],
    )

    print(result["individual_scores"])
    # [1, 0]
    print(result["score"])
    # 0.5
    ```
    """

    @component.output_types(individual_scores=List[int], score=float)
    def run(
        self, questions: List[str], ground_truth_answers: List[List[str]], predicted_answers: List[List[str]]
    ) -> Dict[str, Any]:
        """
        Run the AnswerExactMatchEvaluator on the given inputs.
        All lists must have the same length.

        :param questions:
            A list of questions.
        :param ground_truth_answers:
            A list of expected answers for each question.
        :param predicted_answers:
            A list of predicted answers for each question.
        :returns:
            A dictionary with the following outputs:
            - `individual_scores` - A list of 0s and 1s, where 1 means that the predicted answer matched one of the ground truth.
            - `score` - A number from 0.0 to 1.0 that represents the proportion of questions where any predicted
                         answer matched one of the ground truth answers.
        """
        if not len(questions) == len(ground_truth_answers) == len(predicted_answers):
            raise ValueError("The length of questions, ground_truth_answers, and predicted_answers must be the same.")

        matches = []
        for truths, extracted in zip(ground_truth_answers, predicted_answers):
            if set(truths) & set(extracted):
                matches.append(1)
            else:
                matches.append(0)

        # The proportion of questions where any predicted answer matched one of the ground truth answers
        average = sum(matches) / len(questions)

        return {"individual_scores": matches, "score": average}
