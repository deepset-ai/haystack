import json
from typing import Any, Dict, List, Tuple, Type

from haystack import component, default_from_dict, default_to_dict
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret, deserialize_secrets_inplace


@component
class LLMEvaluator:
    """
    Uses an LLM to evaluate inputs based on a prompt containing instructions and examples.

    The default API requires an OpenAI API key to be provided as an environment variable "OPENAI_API_KEY".
    The inputs are lists that are user-defined depending on the desired metric.
    The output is a dictionary with a key `results` containing a list of evaluation results.
    Each result is a dictionary with user-defined keys and values of either 0 for FALSE or 1 for TRUE respectively.

    Usage example:
    ```python
    from typing import List
    from haystack.components.evaluators import LLMEvaluator
    evaluator = LLMEvaluator(
        instructions="Is this answer problematic for children?",
        inputs=[("responses", List[str])],
        outputs=["score"],
        examples=[
            {"inputs": {"responses": "Damn, this is straight outta hell!!!"}, "outputs": {"score": 1}},
            {"inputs": {"responses": "Football is the most popular sport."}, "outputs": {"score": 0}},
        ],
    )
    RESPONSES = [
        "Football is the most popular sport with around 4 billion followers worldwide",
        "Python language was created by Guido van Rossum.",
    ]
    results = evaluator.run(responses=RESPONSES)
    print(results)
    # {'results': [{'score': 0}, {'score': 0}]}
    ```
    """

    def __init__(
        self,
        instructions: str,
        inputs: List[Tuple[str, Type[List]]],
        outputs: List[str],
        examples: List[Dict[str, Any]],
        *,
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
        self.validate_init_parameters(inputs, outputs, examples)

        self.instructions = instructions
        self.inputs = inputs
        self.outputs = outputs
        self.examples = examples
        self.api = api
        self.api_key = api_key

        if api == "openai":
            self.generator = OpenAIGenerator(api_key=api_key)
        else:
            raise ValueError(f"Unsupported API: {api}")

        template = self.prepare_template()
        self.builder = PromptBuilder(template=template)

        component.set_input_types(self, **dict(inputs))

    def validate_init_parameters(
        self, inputs: List[Tuple[str, Type[List]]], outputs: List[str], examples: List[Dict[str, Any]]
    ):
        """
        Validate the init parameters.

        :param inputs:
            The inputs to validate.
        :param outputs:
            The outputs to validate.
        :param examples:
            The examples to validate.

        :raises ValueError:
            If the inputs are not a list of tuples with a string and a type of list.
            If the outputs are not a list of strings.
            If the examples are not a list of dictionaries.
            If any example does not have keys "inputs" and "outputs" with values that are dictionaries with string keys.
        """
        # Validate inputs
        if (
            not isinstance(inputs, list)
            or not all(isinstance(_input, tuple) for _input in inputs)
            or not all(isinstance(_input[0], str) and _input[1] is not list and len(_input) == 2 for _input in inputs)
        ):
            msg = (
                f"LLM evaluator expects inputs to be a list of tuples. Each tuple must contain an input name and "
                f"type of list but received {inputs}."
            )
            raise ValueError(msg)

        # Validate outputs
        if not isinstance(outputs, list) or not all(isinstance(output, str) for output in outputs):
            msg = f"LLM evaluator expects outputs to be a list of str but received {outputs}."
            raise ValueError(msg)

        # Validate examples are lists of dicts
        if not isinstance(examples, list) or not all(isinstance(example, dict) for example in examples):
            msg = f"LLM evaluator expects examples to be a list of dictionaries but received {examples}."
            raise ValueError(msg)

        # Validate each example
        for example in examples:
            if (
                {"inputs", "outputs"} != example.keys()
                or not all(isinstance(example[param], dict) for param in ["inputs", "outputs"])
                or not all(isinstance(key, str) for param in ["inputs", "outputs"] for key in example[param])
            ):
                msg = (
                    f"LLM evaluator expects each example to have keys `inputs` and `outputs` with values that are "
                    f"dictionaries with str keys but received {example}."
                )
                raise ValueError(msg)

    @component.output_types(results=List[Dict[str, Any]])
    def run(self, **inputs) -> Dict[str, Any]:
        """
        Run the LLM evaluator.

        :param inputs:
            The input values to evaluate. The keys are the input names and the values are lists of input values.
        :returns:
            A dictionary with a single `results` entry that contains a list of results.
            Each result is a dictionary containing the keys as defined in the `outputs` parameter of the LLMEvaluator
            and the evaluation results as the values.
        """
        self.validate_input_parameters(dict(self.inputs), inputs)

        # inputs is a dictionary with keys being input names and values being a list of input values
        # We need to iterate through the lists in parallel for all keys of the dictionary
        input_names, values = inputs.keys(), list(zip(*inputs.values()))
        list_of_input_names_to_values = [dict(zip(input_names, v)) for v in values]

        results = []
        for input_names_to_values in list_of_input_names_to_values:
            prompt = self.builder.run(**input_names_to_values)
            result = self.generator.run(prompt=prompt["prompt"])

            self.validate_outputs(expected=self.outputs, received=result["replies"][0])
            parsed_result = json.loads(result["replies"][0])
            results.append(parsed_result)

        return {"results": results}

    def prepare_template(self) -> str:
        """
        Prepare the prompt template.

        Combine instructions, inputs, outputs, and examples into one prompt template with the following format:
        Instructions:
        <instructions>

        Generate the response in JSON format with the following keys:
        <list of output keys>
        Consider the instructions and the examples below to determine those values.

        Examples:
        <examples>

        Inputs:
        <inputs>
        Outputs:

        :returns:
            The prompt template.
        """
        inputs_section = (
            "{" + ",".join([f'"{input_socket[0]}": {{{{ {input_socket[0]} }}}}' for input_socket in self.inputs]) + "}"
        )

        examples_section = "\n".join(
            [
                "Inputs:\n" + json.dumps(example["inputs"]) + "\nOutputs:\n" + json.dumps(example["outputs"])
                for example in self.examples
            ]
        )
        return (
            f"Instructions:\n"
            f"{self.instructions}\n\n"
            f"Generate the response in JSON format with the following keys:\n"
            f"{json.dumps(self.outputs)}\n"
            f"Consider the instructions and the examples below to determine those values.\n\n"
            f"Examples:\n"
            f"{examples_section}\n\n"
            f"Inputs:\n"
            f"{inputs_section}\n"
            f"Outputs:\n"
        )

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
            examples=self.examples,
            api=self.api,
            api_key=self.api_key.to_dict(),
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
    def validate_input_parameters(expected: Dict[str, Any], received: Dict[str, Any]) -> None:
        """
        Validate the input parameters.

        :param expected:
            The expected input parameters.
        :param received:
            The received input parameters.

        :raises ValueError:
            If not all expected inputs are present in the received inputs
            If the received inputs are not lists or have different lengths
        """
        # Validate that all expected inputs are present in the received inputs
        for param in expected.keys():
            if param not in received:
                msg = f"LLM evaluator expected input parameter '{param}' but received only {received.keys()}."
                raise ValueError(msg)

        # Validate that all received inputs are lists
        if not all(isinstance(_input, list) for _input in received.values()):
            msg = f"LLM evaluator expects all input values to be lists but received {[type(_input) for _input in received.values()]}."
            raise ValueError(msg)

        # Validate that all received inputs are of the same length
        inputs = received.values()
        length = len(next(iter(inputs)))
        if not all(len(_input) == length for _input in inputs):
            msg = (
                f"LLM evaluator expects all input lists to have the same length but received {inputs} with lengths "
                f"{[len(_input) for _input in inputs]}."
            )
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
