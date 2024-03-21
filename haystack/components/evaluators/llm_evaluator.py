import json
from typing import Any, Dict, List, Optional, Tuple, Type

from haystack import component, default_from_dict, default_to_dict
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret, deserialize_secrets_inplace


@component
class LLMEvaluator:
    """
    Uses an LLM to evaluate inputs based on provided instructions and examples.

    The default api requires an OpenAI API key to be provided as an environment variable "OPENAI_API_KEY".
    The inputs of the component are metric-dependent.
    The output is a dictionary with a key `results` containing a list of evaluation results.

    Usage example:
    ```python
    from haystack.components.evaluators import LLMEvaluator
    evaluator = LLMEvaluator(
        instructions="Is this answer problematic for children?",
        inputs=[("responses", List[str])],
        outputs=["score"],
    )
    RESPONSES = [
        "Football is the most popular sport with around 4 billion followers worldwide",
        "Python language was created by Guido van Rossum.",
    ]
    results = evaluator.run(responses=RESPONSES)
    ```
    """

    def __init__(
        self,
        instructions: str,
        inputs: List[Tuple[str, Type]],
        outputs: List[str],
        *,
        api: str = "openai",
        api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
        examples: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Creates an instance of LLMEvaluator.

        :param instructions:
            The prompt instructions to use for evaluation.
        :param inputs:
            The inputs to use for evaluation. Each input is a tuple containing
            the name of the input and the type of the input.
        :param outputs:
            The output names of the evaluation results.
        :param api:
            The API to use for evaluation.
            Supported APIs: "openai".
        :param api_key:
            The API key to use.
        :param examples:
            Few-shot examples conforming to the input and output format.
        """

        self.instructions = instructions
        self.inputs = inputs
        self.outputs = outputs
        self.api = api
        self.api_key = api_key
        self.examples = examples
        expected_inputs = dict(inputs)
        if api == "openai":
            self.generator = OpenAIGenerator(api_key=api_key)
        else:
            raise ValueError(f"Unsupported API: {api}")

        component.set_input_types(self, **expected_inputs)

    @component.output_types(results=List[List[Dict[str, Any]]])
    def run(self, **inputs) -> Dict[str, Any]:
        """
        Run the LLM evaluator.

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
        self.validate_input_parameters(dict(self.inputs), inputs)
        self.validate_lengths(*inputs.values())

        results = []
        template = self.prepare_template()
        builder = PromptBuilder(template=template)

        # inputs is a dictionary with keys being input names and values being a list of input values
        # We need to iterate through the lists in parallel for all keys of the dictionary
        input_names, values = inputs.keys(), list(zip(*inputs.values()))
        list_of_input_names_to_values = [dict(zip(input_names, v)) for v in values]

        for input_names_to_values in list_of_input_names_to_values:
            prompt = builder.run(**input_names_to_values)
            # TODO rendered prompt should contain " instead of ' for filled in values such as responses.
            #  and for strings it should contain " instead of currently no delimiters
            #  json.dumps() instead of str() should be used
            result = self.generator.run(prompt=prompt["prompt"])

            self.validate_outputs(expected=self.outputs, received=result["replies"][0])
            parsed_result = json.loads(result["replies"][0])
            parsed_result["name"] = "llm"
            results.append(parsed_result)

        return {"results": results}

    def prepare_template(self) -> str:
        """
        Combine instructions, inputs, outputs, and examples into one prompt template.
        """
        inputs_section = (
            "{" + ",".join([f'"{input_socket[0]}": {{{{ {input_socket[0]} }}}}' for input_socket in self.inputs]) + "}"
        )
        examples_section = ""
        if self.examples:
            for example in self.examples:
                examples_section += (
                    "Inputs:\n" + json.dumps(example["inputs"]) + "\nOutputs:\n" + json.dumps(example["outputs"]) + "\n"
                )
        return f"Respond only in JSON format with a key {json.dumps(self.outputs)} and a value of either 0 for FALSE or 1 for TRUE.\n{self.instructions}\n{examples_section}Inputs:\n{inputs_section}\nOutputs:\n"

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
        return default_to_dict(
            self,
            instructions=self.instructions,
            inputs=self.inputs,
            outputs=self.outputs,
            api=self.api,
            api_key=self.api_key.to_dict(),
            examples=self.examples,
        )

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

    @staticmethod
    def validate_lengths(*lists):
        """
        Validate that all input lists have the same length.

        :param lists:
            The lists to validate.
        """
        length = len(lists[0])
        if all(len(lst) == length for lst in lists[1:]):
            return True
        else:
            msg = f"LLM evaluator expects all input lists to have the same length but received {lists} with lengths {[len(lst) for lst in lists]}."
            raise ValueError(msg)

    @staticmethod
    def validate_input_parameters(expected: Dict[str, Any], received: Dict[str, Any]) -> None:
        """
        Validate the input parameters.

        :param expected:
            The expected input parameters.
        :param received:
            The received input parameters.

        :raises ValueError:
            If not all expected inputs are present in the received inputs
        """
        for param in expected.keys():
            if param not in received:
                msg = f"LLM evaluator expected input parameter '{param}' but received only {received.keys()}."
                raise ValueError(msg)

    @staticmethod
    def validate_outputs(expected: List[str], received: str) -> None:
        """
        Validate the output.

        :param expected:
            Names of expected outputs
        :param received:
            Names of received outputs

        :raises ValueError:
            If not all expected outputs are present in the received outputs
        """
        parsed_output = json.loads(received)
        if not all(output in parsed_output for output in expected):
            msg = f"Expected response from LLM evaluator to be JSON with keys {expected}, got {received}."
            raise ValueError(msg)
