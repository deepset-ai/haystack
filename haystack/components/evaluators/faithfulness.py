from typing import Any, Dict, List, Optional, Tuple, Type

from haystack import default_from_dict
from haystack.components.evaluators import LLMEvaluator
from haystack.core.component import component
from haystack.utils import Secret, deserialize_secrets_inplace


@component
class FaithfulnessEvaluator(LLMEvaluator):
    """
    Evaluator that checks if the predicted answers can be inferred from the retrieved documents.
    For each sample, the result is binary, either 1 if the predicted answer can be inferred from the retrieved
    documents, or 0 otherwise.

    Usage example:
    ```python
    from haystack.components.evaluators import FaithfulnessEvaluator

    evaluator = FaithfulnessEvaluator()
    result = evaluator.run(
        questions=["What is the capital of Germany?", "What is the capital of France?"],
        contexts=[["Berlin is the capital of Germany."], ["Paris is the capital of France."]],
        responses=["Berlin", "Paris"],
    )
    print(result["result"])
    # 1.0
    ```
    """

    def __init__(
        self,
        instructions: str = "Can the answer be inferred from the context? Answer with 1 if yes, otherwise 0.",
        inputs: Optional[List[Tuple[str, Type[List]]]] = None,
        outputs: Optional[List[str]] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
        api: str = "openai",
        api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
    ):
        """
        Creates an instance of LLMEvaluator.

        :param instructions:
            The prompt instructions to use for evaluation.
            Should be a question about the inputs that can be answered with yes or no.
        :param inputs:
            The inputs that the component expects as incoming connections and that it evaluates.
            Each input is a tuple of an input name and input type. Input types must be lists.
        :param outputs:
            Output names of the evaluation results. They correspond to keys in the output dictionary.
            The default is a single key "score".
        :param examples:
            Few-shot examples conforming to the expected input and output format as defined in the `inputs` and
             `outputs` parameters.
            Each example is a dictionary with keys "inputs" and "outputs"
            They contain the input and output as dictionaries respectively.
        :param api:
            The API to use for calling an LLM through a Generator.
            Supported APIs: "openai".
        :param api_key:
            The API key.

        """
        self.instructions = instructions
        self.inputs = inputs or [("questions", List[str]), ("contexts", List[List[str]]), ("responses", List[str])]
        self.outputs = outputs or ["score"]
        self.examples = examples or [
            {
                "inputs": {
                    "questions": "What is the capital of Germany?",
                    "contexts": ["Berlin is the capital of Germany."],
                    "responses": "Berlin",
                },
                "outputs": {"score": 1},
            },
            {
                "inputs": {
                    "questions": "What is the capital of France?",
                    "contexts": ["Berlin is the capital of Germany."],
                    "responses": "Paris",
                },
                "outputs": {"score": 0},
            },
        ]
        self.api = api
        self.api_key = api_key

        super().__init__(
            instructions=instructions, inputs=inputs, outputs=outputs, examples=examples, api=api, api_key=api_key
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
