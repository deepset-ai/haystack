# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any

from numpy import mean as np_mean

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.components.evaluators.llm_evaluator import LLMEvaluator
from haystack.components.generators.chat.types import ChatGenerator
from haystack.core.serialization import component_to_dict
from haystack.utils import deserialize_chatgenerator_inplace

logger = logging.getLogger(__name__)

# Default examples to include in the prompt if the user does not provide any examples
_DEFAULT_EXAMPLES = [
    {
        "inputs": {
            "questions": "What is the capital of Germany and when was it founded?",
            "contexts": ["Berlin is the capital of Germany and was founded in 1244."],
            "predicted_answers": "The capital of Germany, Berlin, was founded in the 13th century.",
        },
        "outputs": {
            "statements": ["Berlin is the capital of Germany.", "Berlin was founded in 1244."],
            "statement_scores": [1, 1],
        },
    },
    {
        "inputs": {
            "questions": "What is the capital of France?",
            "contexts": ["Berlin is the capital of Germany."],
            "predicted_answers": "Paris",
        },
        "outputs": {"statements": ["Paris is the capital of France."], "statement_scores": [0]},
    },
    {
        "inputs": {
            "questions": "What is the capital of Italy?",
            "contexts": ["Rome is the capital of Italy."],
            "predicted_answers": "Rome is the capital of Italy with more than 4 million inhabitants.",
        },
        "outputs": {
            "statements": ["Rome is the capital of Italy.", "Rome has more than 4 million inhabitants."],
            "statement_scores": [1, 0],
        },
    },
]


@component
class FaithfulnessEvaluator(LLMEvaluator):
    """
    Evaluator that checks if a generated answer can be inferred from the provided contexts.

    An LLM separates the answer into multiple statements and checks whether the statement can be inferred from the
    context or not. The final score for the full answer is a number from 0.0 to 1.0. It represents the proportion of
    statements that can be inferred from the provided contexts.

    Usage example:
    ```python
    from haystack.components.evaluators import FaithfulnessEvaluator

    questions = ["Who created the Python language?"]
    contexts = [
        [(
            "Python, created by Guido van Rossum in the late 1980s, is a high-level general-purpose programming "
            "language. Its design philosophy emphasizes code readability, and its language constructs aim to help "
            "programmers write clear, logical code for both small and large-scale software projects."
        )],
    ]
    predicted_answers = [
        "Python is a high-level general-purpose programming language that was created by George Lucas."
    ]
    evaluator = FaithfulnessEvaluator()
    result = evaluator.run(questions=questions, contexts=contexts, predicted_answers=predicted_answers)

    print(result["individual_scores"])
    # [0.5]
    print(result["score"])
    # 0.5
    print(result["results"])
    # [{'statements': ['Python is a high-level general-purpose programming language.',
    # 'Python was created by George Lucas.'], 'statement_scores': [1, 0], 'score': 0.5}]
    ```
    """

    def __init__(
        self,
        examples: list[dict[str, Any]] | None = None,
        progress_bar: bool = True,
        raise_on_failure: bool = True,
        chat_generator: ChatGenerator | None = None,
    ) -> None:
        """
        Creates an instance of FaithfulnessEvaluator.

        If no LLM is specified using the `chat_generator` parameter, the component will use OpenAI in JSON mode.

        :param examples:
            Optional few-shot examples conforming to the expected input and output format of FaithfulnessEvaluator.
            Default examples will be used if none are provided.
            Each example must be a dictionary with keys "inputs" and "outputs".
            "inputs" must be a dictionary with keys "questions", "contexts", and "predicted_answers".
            "outputs" must be a dictionary with "statements" and "statement_scores".
            Expected format:
            ```python
            [{
                "inputs": {
                    "questions": "What is the capital of Italy?", "contexts": ["Rome is the capital of Italy."],
                    "predicted_answers": "Rome is the capital of Italy with more than 4 million inhabitants.",
                },
                "outputs": {
                    "statements": ["Rome is the capital of Italy.", "Rome has more than 4 million inhabitants."],
                    "statement_scores": [1, 0],
                },
            }]
            ```
        :param progress_bar:
            Whether to show a progress bar during the evaluation.
        :param raise_on_failure:
            Whether to raise an exception if the API call fails.
        :param chat_generator:
            a ChatGenerator instance which represents the LLM.
            In order for the component to work, the LLM should be configured to return a JSON object. For example,
            when using the OpenAIChatGenerator, you should pass `{"response_format": {"type": "json_object"}}` in the
            `generation_kwargs`.
        """
        self.instructions = (
            "Your task is to judge the faithfulness or groundedness of statements based "
            "on context information. First, please extract statements from a provided "
            "predicted answer to a question. Second, calculate a faithfulness score for each "
            "statement made in the predicted answer. The score is 1 if the statement can be "
            "inferred from the provided context or 0 if it cannot be inferred."
        )
        self.inputs = [("questions", list[str]), ("contexts", list[list[str]]), ("predicted_answers", list[str])]
        self.outputs = ["statements", "statement_scores"]
        self.examples = examples or _DEFAULT_EXAMPLES

        super(FaithfulnessEvaluator, self).__init__(  # noqa: UP008
            instructions=self.instructions,
            inputs=self.inputs,
            outputs=self.outputs,
            examples=self.examples,
            chat_generator=chat_generator,
            raise_on_failure=raise_on_failure,
            progress_bar=progress_bar,
        )

    @component.output_types(individual_scores=list[float], score=float, results=list[dict[str, Any]])
    def run(self, **inputs: Any) -> dict[str, Any]:
        """
        Run the LLM evaluator.

        :param questions:
            A list of questions.
        :param contexts:
            A nested list of contexts that correspond to the questions.
        :param predicted_answers:
            A list of predicted answers.
        :returns:
            A dictionary with the following outputs:
                - `score`: Mean faithfulness score over all the provided input answers.
                - `individual_scores`: A list of faithfulness scores for each input answer.
                - `results`: A list of dictionaries with `statements` and `statement_scores` for each input answer.
        """
        result = super(FaithfulnessEvaluator, self).run(**inputs)  # noqa: UP008
        # Post-process the raw results to calculate relevance metrics and scores
        return self._postprocess_results(result)

    @component.output_types(individual_scores=list[float], score=float, results=list[dict[str, Any]])
    async def run_async(self, **inputs: Any) -> dict[str, Any]:
        """
        Run the LLM evaluator asynchronously.

        :param questions:
            A list of questions.
        :param contexts:
            A nested list of contexts that correspond to the questions.
        :param predicted_answers:
            A list of predicted answers.
        :returns:
            A dictionary with the following outputs:
                - `score`: Mean faithfulness score over all the provided input answers.
                - `individual_scores`: A list of faithfulness scores for each input answer.
                - `results`: A list of dictionaries with `statements` and `statement_scores` for each input answer.
        """
        result = await super(FaithfulnessEvaluator, self).run_async(**inputs)  # noqa: UP008
        # Post-process the raw results to calculate relevance metrics and scores
        return self._postprocess_results(result)

    def _postprocess_results(self, result: dict[str, Any]) -> dict[str, Any]:
        """
        Post-processes raw LLM evaluator outputs to compute faithfulness scores.

        Calculates statement-level score averages, computes the overall mean faithfulness
        score across successful queries, and updates the result payload.

        :param result:
            The raw evaluation dictionary from the base LLM evaluator.
        :returns:
            The updated dictionary containing final scores and tracking metrics.
        """

        # calculate average statement faithfulness score per query
        for idx, res in enumerate(result["results"]):
            if res is None:
                result["results"][idx] = {"statements": [], "statement_scores": [], "score": float("nan")}
                continue
            if not res["statements"]:
                res["score"] = 0
            else:
                res["score"] = np_mean(res["statement_scores"])

        # calculate average answer faithfulness score over all queries
        scores = [res["score"] for res in result["results"]]
        valid_scores = [s for s in scores if not math.isnan(s)]
        skipped = len(scores) - len(valid_scores)
        if skipped:
            logger.warning("{skipped} query(s) failed and were excluded from the score.", skipped=skipped)
        result["score"] = np_mean(valid_scores) if valid_scores else float("nan")
        result["individual_scores"] = scores

        return result

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            A dictionary with serialized data.
        """
        return default_to_dict(
            self,
            chat_generator=component_to_dict(obj=self._chat_generator, name="chat_generator"),
            examples=self.examples,
            progress_bar=self.progress_bar,
            raise_on_failure=self.raise_on_failure,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FaithfulnessEvaluator":
        """
        Deserialize this component from a dictionary.

        :param data:
            The dictionary representation of this component.
        :returns:
            The deserialized component instance.
        """
        if data["init_parameters"].get("chat_generator"):
            deserialize_chatgenerator_inplace(data["init_parameters"], key="chat_generator")
        return default_from_dict(cls, data)
