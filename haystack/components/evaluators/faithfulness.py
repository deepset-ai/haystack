# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

from numpy import mean as np_mean

from haystack import component, default_from_dict, default_to_dict
from haystack.components.evaluators.llm_evaluator import LLMEvaluator
from haystack.utils import Secret, deserialize_secrets_inplace

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
    'Python was created by George Lucas.'], 'statement_scores': [1, 0], 'score': 0.5}]
    ```
    """

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        examples: Optional[List[Dict[str, Any]]] = None,
        progress_bar: bool = True,
        api: str = "openai",
        api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
        api_params: Optional[Dict[str, Any]] = None,
        raise_on_failure: bool = True,
    ):
        """
        Creates an instance of FaithfulnessEvaluator.

        :param examples:
            Optional few-shot examples conforming to the expected input and output format of FaithfulnessEvaluator.
            Default examples will be used if none are provided.
            Each example must be a dictionary with keys "inputs" and "outputs".
            "inputs" must be a dictionary with keys "questions", "contexts", and "predicted_answers".
            "outputs" must be a dictionary with "statements" and "statement_scores".
            Expected format:
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
        :param progress_bar:
            Whether to show a progress bar during the evaluation.
        :param api:
            The API to use for calling an LLM through a Generator.
            Supported APIs: "openai".
        :param api_key:
            The API key.
        :param api_params:
            Parameters for an OpenAI API compatible completions call.
        :param raise_on_failure:
            Whether to raise an exception if the API call fails.

        """
        self.instructions = (
            "Your task is to judge the faithfulness or groundedness of statements based "
            "on context information. First, please extract statements from a provided "
            "predicted answer to a question. Second, calculate a faithfulness score for each "
            "statement made in the predicted answer. The score is 1 if the statement can be "
            "inferred from the provided context or 0 if it cannot be inferred."
        )
        self.inputs = [("questions", List[str]), ("contexts", List[List[str]]), ("predicted_answers", List[str])]
        self.outputs = ["statements", "statement_scores"]
        self.examples = examples or _DEFAULT_EXAMPLES
        self.api = api
        self.api_key = api_key
        self.api_params = api_params or {}

        super(FaithfulnessEvaluator, self).__init__(
            instructions=self.instructions,
            inputs=self.inputs,
            outputs=self.outputs,
            examples=self.examples,
            api=self.api,
            api_key=self.api_key,
            api_params=self.api_params,
            raise_on_failure=raise_on_failure,
            progress_bar=progress_bar,
        )

    @component.output_types(individual_scores=List[int], score=float, results=List[Dict[str, Any]])
    def run(self, **inputs) -> Dict[str, Any]:
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
        result = super(FaithfulnessEvaluator, self).run(**inputs)

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
        result["score"] = np_mean([res["score"] for res in result["results"]])
        result["individual_scores"] = [res["score"] for res in result["results"]]

        return result

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            A dictionary with serialized data.
        """
        return default_to_dict(
            self,
            api=self.api,
            api_key=self.api_key.to_dict() if self.api_key else None,
            api_params=self.api_params,
            examples=self.examples,
            progress_bar=self.progress_bar,
            raise_on_failure=self.raise_on_failure,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FaithfulnessEvaluator":
        """
        Deserialize this component from a dictionary.

        :param data:
            The dictionary representation of this component.
        :returns:
            The deserialized component instance.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)
