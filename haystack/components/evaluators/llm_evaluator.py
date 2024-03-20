from typing import Any, Dict, List, Tuple, Type

from haystack import component, default_from_dict, default_to_dict
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret, deserialize_secrets_inplace


@component
class LLMEvaluator:
    """
    A component that uses an LLM to evaluate inputs against a specific metric.

    Most of them require an OpenAI API key to be provided as an environment variable "OPENAI_API_KEY".
    The inputs of the component are metric-dependent.
    The output is a nested list of evaluation results where each inner list contains the results for a single input.
    """

    def __init__(
        self,
        instruction: str,
        inputs: List[Tuple[str, Type]],
        outputs: List[str],
        *,
        api: str = "openai",
        api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
    ):
        """
        Creates an instance of LLMEvaluator.

        :param api:
            The API to use for evaluation.

            Supported APIs: "openai".
        :param api_key:
            The API key to use.
        """

        self.instruction = instruction
        self.inputs = inputs
        self.outputs = outputs
        self.api = api
        self.api_key = api_key
        expected_inputs = dict(inputs)
        if api == "openai":
            self.generator = OpenAIGenerator(api_key=api_key)

        component.set_input_types(self, **expected_inputs)

    @component.output_types(results=List[List[Dict[str, Any]]])
    def run(self, **inputs) -> Dict[str, Any]:
        """
        Run the LLM evaluator.

        Example:
        ```python
        p = Pipeline()
        evaluator = LLMEvaluator(
            api = "openai",
        )
        p.add_component("evaluator", evaluator)

        results = p.run({"evaluator": {"questions": QUESTIONS, "contexts": CONTEXTS, "ground_truths": GROUND_TRUTHS}})
        ```

        :param inputs:
            The inputs to evaluate. These are determined by the
            metric being calculated. See :class:`RagasMetric` for more
            information.
        :returns:
            A nested list of metric results. Each input can have one or more
            results, depending on the metric. Each result is a dictionary
            containing the following keys and values:
                * `name` - The name of the metric.
                * `score` - The score of the metric.
        """
        # TODO: validate input parameters
        # InputConverters.validate_input_parameters(self.metric, self.descriptor.input_parameters, inputs)

        results = []
        for input_socket in self.inputs:
            self.instruction += f"{input_socket[0]}: {{{{ input_socket[1] }}}}"  # fix: do not hardcode
        builder = PromptBuilder(template=self.instruction)
        for response in inputs["responses"]:  # fix: do not hardcode
            prompt = builder.run(response=response)
            result = self.generator.run(prompt=prompt["prompt"])
            results.append(result["replies"])
        # todo: convert result list
        return {"results": [{"name": "llm", "score": 1.0}]}

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"api": self.api}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        return default_to_dict(self, instruction=self.instruction, api=self.api, api_key=self.api_key.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMEvaluator":
        """
        Deserialize this component from a dictionary.

        :param data:
            The dictionary representation of this component.
        :returns:
            The deserialized component instance.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)
