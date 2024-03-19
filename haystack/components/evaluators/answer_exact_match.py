from typing import Dict, List

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
        predicted_answers=[["Berlin"], ["Paris"]],
    )
    print(result["result"])
    # 1.0
    ```
    """

    @component.output_types(result=float)
    def run(
        self, questions: List[str], ground_truth_answers: List[List[str]], predicted_answers: List[List[str]]
    ) -> Dict[str, float]:
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
            - `result` - A number from 0.0 to 1.0 that represents the proportion of questions where any predicted
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
