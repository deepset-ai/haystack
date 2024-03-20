from typing import Any, Dict, List

from haystack.core.component import component


@component
class AnswerF1Evaluator:
    """
    Evaluator that calculates the average F1 score of the matches between the predicted and the ground truth answers.
    We first calculate the F1 score for each question, sum all the scores and divide by the number of questions.
    The result is a number from 0.0 to 1.0.

    Each question can have multiple ground truth answers and multiple predicted answers.

    Usage example:
    ```python
    from haystack.components.evaluators import AnswerF1Evaluator

    evaluator = AnswerF1Evaluator()
    result = evaluator.run(
        questions=["What is the capital of Germany?", "What is the capital of France?"],
        ground_truth_answers=[["Berlin"], ["Paris"]],
        predicted_answers=[["Berlin"], ["London"]],
    )

    print(result["scores"])
    # [1.0, 0.0]

    print(result["average"])
    # 0.5
    ```
    """

    @component.output_types(scores=List[float], average=float)
    def run(
        self, questions: List[str], ground_truth_answers: List[List[str]], predicted_answers: List[List[str]]
    ) -> Dict[str, Any]:
        """
        Run the AnswerF1Evaluator on the given inputs.
        All lists must have the same length.

        :param questions:
            A list of questions.
        :param ground_truth_answers:
            A list of expected answers for each question.
        :param predicted_answers:
            A list of predicted answers for each question.
        :returns:
            A dictionary with the following outputs:
            - `scores`: A list of numbers from 0.0 to 1.0 that represents the F1 score for each question.
            - `average`: A number from 0.0 to 1.0 that represents the average F1 score of the predicted
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

        return {"scores": scores, "average": sum(scores) / len(questions)}
