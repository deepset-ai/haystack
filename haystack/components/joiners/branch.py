# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Type

from haystack import component, default_from_dict, default_to_dict
from haystack.core.component.types import GreedyVariadic
from haystack.utils import deserialize_type, serialize_type


@component
class BranchJoiner:
    """
    A component that merges multiple input branches of a pipeline into a single output stream.

    `BranchJoiner` receives multiple inputs of the same data type and forwards the first received value
    to its output. This is useful for scenarios where multiple branches need to converge before proceeding.

    ### Common Use Cases:
    - **Loop Handling:** `BranchJoiner` helps close loops in pipelines. For example, if a pipeline component validates
      or modifies incoming data and produces an error-handling branch, `BranchJoiner` can merge both branches and send
      (or resend in the case of a loop) the data to the component that evaluates errors. See "Usage example" below.

    - **Decision-Based Merging:** `BranchJoiner` reconciles branches coming from Router components (such as
      `ConditionalRouter`, `TextLanguageRouter`). Suppose a `TextLanguageRouter` directs user queries to different
      Retrievers based on the detected language. Each Retriever processes its assigned query and passes the results
      to `BranchJoiner`, which consolidates them into a single output before passing them to the next component, such
      as a `PromptBuilder`.

    ### Example Usage:
    ```python
    import json
    from typing import List

    from haystack import Pipeline
    from haystack.components.converters import OutputAdapter
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.components.joiners import BranchJoiner
    from haystack.components.validators import JsonSchemaValidator
    from haystack.dataclasses import ChatMessage

    # Define a schema for validation
    person_schema = {
        "type": "object",
        "properties": {
            "first_name": {"type": "string", "pattern": "^[A-Z][a-z]+$"},
            "last_name": {"type": "string", "pattern": "^[A-Z][a-z]+$"},
            "nationality": {"type": "string", "enum": ["Italian", "Portuguese", "American"]},
        },
        "required": ["first_name", "last_name", "nationality"]
    }

    # Initialize a pipeline
    pipe = Pipeline()

    # Add components to the pipeline
    pipe.add_component('joiner', BranchJoiner(List[ChatMessage]))
    pipe.add_component('generator', OpenAIChatGenerator(model="gpt-4o-mini"))
    pipe.add_component('validator', JsonSchemaValidator(json_schema=person_schema))
    pipe.add_component('adapter', OutputAdapter("{{chat_message}}", List[ChatMessage], unsafe=True))

    # And connect them
    pipe.connect("adapter", "joiner")
    pipe.connect("joiner", "generator")
    pipe.connect("generator.replies", "validator.messages")
    pipe.connect("validator.validation_error", "joiner")

    result = pipe.run(
        data={
        "generator": {"generation_kwargs": {"response_format": {"type": "json_object"}}},
        "adapter": {"chat_message": [ChatMessage.from_user("Create json from Peter Parker")]}}
    )

    print(json.loads(result["validator"]["validated"][0].text))


    >> {'first_name': 'Peter', 'last_name': 'Parker', 'nationality': 'American', 'name': 'Spider-Man', 'occupation':
    >> 'Superhero', 'age': 23, 'location': 'New York City'}
    ```

    Note that `BranchJoiner` can manage only one data type at a time. In this case, `BranchJoiner` is created for
    passing `List[ChatMessage]`. This determines the type of data that `BranchJoiner` will receive from the upstream
    connected components and also the type of data that `BranchJoiner` will send through its output.

    In the code example, `BranchJoiner` receives a looped back `List[ChatMessage]` from the `JsonSchemaValidator` and
    sends it down to the `OpenAIChatGenerator` for re-generation. We can have multiple loopback connections in the
    pipeline. In this instance, the downstream component is only one (the `OpenAIChatGenerator`), but the pipeline could
    have more than one downstream component.
    """

    def __init__(self, type_: Type):
        """
        Creates a `BranchJoiner` component.

        :param type_: The expected data type of inputs and outputs.
        """
        self.type_ = type_
        component.set_input_types(self, value=GreedyVariadic[type_])  # type: ignore
        component.set_output_types(self, value=type_)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component into a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(self, type_=serialize_type(self.type_))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BranchJoiner":
        """
        Deserializes a `BranchJoiner` instance from a dictionary.

        :param data: The dictionary containing serialized component data.
        :returns:
            A deserialized `BranchJoiner` instance.
        """
        data["init_parameters"]["type_"] = deserialize_type(data["init_parameters"]["type_"])
        return default_from_dict(cls, data)

    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Executes the `BranchJoiner`, selecting the first available input value and passing it downstream.

        :param **kwargs: The input data. Must be of the type declared by `type_` during initialization.
        :returns:
            A dictionary with a single key `value`, containing the first input received.
        """
        if (inputs_count := len(kwargs["value"])) != 1:
            raise ValueError(f"BranchJoiner expects only one input, but {inputs_count} were received.")
        return {"value": kwargs["value"][0]}
