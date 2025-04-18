# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, Dict, List, Optional, Tuple, Type

from tqdm import tqdm

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.components.builders import PromptBuilder
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.components.generators.chat.types import ChatGenerator
from haystack.core.serialization import component_to_dict
from haystack.dataclasses.chat_message import ChatMessage
from haystack.utils import deserialize_chatgenerator_inplace, deserialize_type, serialize_type

logger = logging.getLogger(__name__)


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
        inputs=[("predicted_answers", List[str])],
        outputs=["score"],
        examples=[
            {"inputs": {"predicted_answers": "Damn, this is straight outta hell!!!"}, "outputs": {"score": 1}},
            {"inputs": {"predicted_answers": "Football is the most popular sport."}, "outputs": {"score": 0}},
        ],
    )
    predicted_answers = [
        "Football is the most popular sport with around 4 billion followers worldwide",
        "Python language was created by Guido van Rossum.",
    ]
    results = evaluator.run(predicted_answers=predicted_answers)
    print(results)
    # {'results': [{'score': 0}, {'score': 0}]}
    ```
    """

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        instructions: str,
        inputs: List[Tuple[str, Type[List]]],
        outputs: List[str],
        examples: List[Dict[str, Any]],
        progress_bar: bool = True,
        *,
        raise_on_failure: bool = True,
        chat_generator: Optional[ChatGenerator] = None,
    ):
        """
        Creates an instance of LLMEvaluator.

        If no LLM is specified using the `chat_generator` parameter, the component will use OpenAI in JSON mode.

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
        :param raise_on_failure:
            If True, the component will raise an exception on an unsuccessful API call.
        :param progress_bar:
            Whether to show a progress bar during the evaluation.
        :param chat_generator:
            a ChatGenerator instance which represents the LLM.
            In order for the component to work, the LLM should be configured to return a JSON object. For example,
            when using the OpenAIChatGenerator, you should pass `{"response_format": {"type": "json_object"}}` in the
            `generation_kwargs`.
        """
        self.validate_init_parameters(inputs, outputs, examples)
        component.set_input_types(self, **dict(inputs))

        self.raise_on_failure = raise_on_failure
        self.instructions = instructions
        self.inputs = inputs
        self.outputs = outputs
        self.examples = examples
        self.progress_bar = progress_bar

        template = self.prepare_template()
        self.builder = PromptBuilder(template=template)

        if chat_generator is not None:
            self._chat_generator = chat_generator
        else:
            generation_kwargs = {"response_format": {"type": "json_object"}, "seed": 42}
            self._chat_generator = OpenAIChatGenerator(generation_kwargs=generation_kwargs)

    @staticmethod
    def validate_init_parameters(
        inputs: List[Tuple[str, Type[List]]], outputs: List[str], examples: List[Dict[str, Any]]
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
            A dictionary with a `results` entry that contains a list of results.
            Each result is a dictionary containing the keys as defined in the `outputs` parameter of the LLMEvaluator
            and the evaluation results as the values. If an exception occurs for a particular input value, the result
            will be `None` for that entry.
            If the API is "openai" and the response contains a "meta" key, the metadata from OpenAI will be included
            in the output dictionary, under the key "meta".
        :raises ValueError:
            Only in the case that  `raise_on_failure` is set to True and the received inputs are not lists or have
            different lengths, or if the output is not a valid JSON or doesn't contain the expected keys.
        """
        self.validate_input_parameters(dict(self.inputs), inputs)

        # inputs is a dictionary with keys being input names and values being a list of input values
        # We need to iterate through the lists in parallel for all keys of the dictionary
        input_names, values = inputs.keys(), list(zip(*inputs.values()))
        list_of_input_names_to_values = [dict(zip(input_names, v)) for v in values]

        results: List[Optional[Dict[str, Any]]] = []
        metadata = []
        errors = 0
        for input_names_to_values in tqdm(list_of_input_names_to_values, disable=not self.progress_bar):
            prompt = self.builder.run(**input_names_to_values)
            messages = [ChatMessage.from_user(prompt["prompt"])]
            try:
                result = self._chat_generator.run(messages=messages)
            except Exception as e:
                if self.raise_on_failure:
                    raise ValueError(f"Error while generating response for prompt: {prompt}. Error: {e}")
                logger.warning("Error while generating response for prompt: {prompt}. Error: {e}", prompt=prompt, e=e)
                results.append(None)
                errors += 1
                continue

            if self.is_valid_json_and_has_expected_keys(expected=self.outputs, received=result["replies"][0].text):
                parsed_result = json.loads(result["replies"][0].text)
                results.append(parsed_result)
            else:
                results.append(None)
                errors += 1

            if result["replies"][0].meta:
                metadata.append(result["replies"][0].meta)

        if errors > 0:
            logger.warning(
                "LLM evaluator failed for {errors} out of {len(list_of_input_names_to_values)} inputs.",
                errors=errors,
                len=len(list_of_input_names_to_values),
            )

        return {"results": results, "meta": metadata or None}

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
            "{" + ", ".join([f'"{input_socket[0]}": {{{{ {input_socket[0]} }}}}' for input_socket in self.inputs]) + "}"
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
        # Since we cannot currently serialize tuples, convert the inputs to a list.
        inputs = [[name, serialize_type(type_)] for name, type_ in self.inputs]
        return default_to_dict(
            self,
            instructions=self.instructions,
            inputs=inputs,
            outputs=self.outputs,
            examples=self.examples,
            chat_generator=component_to_dict(obj=self._chat_generator, name="chat_generator"),
            progress_bar=self.progress_bar,
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
        data["init_parameters"]["inputs"] = [
            (name, deserialize_type(type_)) for name, type_ in data["init_parameters"]["inputs"]
        ]

        if data["init_parameters"].get("chat_generator"):
            deserialize_chatgenerator_inplace(data["init_parameters"], key="chat_generator")

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
            msg = (
                "LLM evaluator expects all input values to be lists but received "
                f"{[type(_input) for _input in received.values()]}."
            )
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

    def is_valid_json_and_has_expected_keys(self, expected: List[str], received: str) -> bool:
        """
        Output must be a valid JSON with the expected keys.

        :param expected:
            Names of expected outputs
        :param received:
            Names of received outputs

        :raises ValueError:
            If the output is not a valid JSON with the expected keys:
            - with `raise_on_failure` set to True a ValueError is raised.
            - with `raise_on_failure` set to False a warning is issued and False is returned.

        :returns:
            True if the received output is a valid JSON with the expected keys, False otherwise.
        """
        try:
            parsed_output = json.loads(received)
        except json.JSONDecodeError:
            msg = "Response from LLM evaluator is not a valid JSON."
            if self.raise_on_failure:
                raise ValueError(msg)
            logger.warning(msg)
            return False

        if not all(output in parsed_output for output in expected):
            if self.raise_on_failure:
                raise ValueError(
                    f"Expected response from LLM evaluator to be JSON with keys {expected}, got {received}."
                )
            logger.warning(
                "Expected response from LLM evaluator to be JSON with keys {expected}, got {received}.",
                expected=expected,
                received=received,
            )
            return False

        return True
