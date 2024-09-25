# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Type

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.core.component.types import GreedyVariadic
from haystack.utils import deserialize_type, serialize_type

logger = logging.getLogger(__name__)


@component()
class BranchJoiner:
    """
    A component to join different branches of a pipeline into one single output.

    `BranchJoiner` receives multiple data connections of the same type from other components and passes the first
    value coming to its single output, possibly distributing it to various other components.

    `BranchJoiner` is fundamental to close loops in a pipeline, where the two branches it joins are the ones
    coming from the previous component and one coming back from a loop. For example, `BranchJoiner` could be used
    to send data to a component evaluating errors. `BranchJoiner` would receive two connections, one to get the
    original data and another one to get modified data in case there was an error. In both cases, `BranchJoiner`
    would send (or re-send in case of a loop) data to the component evaluating errors. See "Usage example" below.

    Another use case with a need for `BranchJoiner` is to reconcile multiple branches coming out of a decision
    or Classifier component. For example, in a RAG pipeline, there might be a "query language classifier" component
    sending the query to different retrievers, selecting one specifically according to the detected language. After the
    retrieval step the pipeline would ideally continue with a `PromptBuilder`, and since we don't know in advance the
    language of the query, all the retrievers should be ideally connected to the single `PromptBuilder`. Since the
    `PromptBuilder` won't accept more than one connection in input, we would connect all the retrievers to a
    `BranchJoiner` component and reconcile them in a single output that can be connected to the `PromptBuilder`
    downstream.

    Usage example:

    ```python
    import json
    from typing import List

    from haystack import Pipeline
    from haystack.components.converters import OutputAdapter
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.components.joiners import BranchJoiner
    from haystack.components.validators import JsonSchemaValidator
    from haystack.dataclasses import ChatMessage

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
    pipe.add_component('fc_llm', OpenAIChatGenerator(model="gpt-4o-mini"))
    pipe.add_component('validator', JsonSchemaValidator(json_schema=person_schema))
    pipe.add_component('adapter', OutputAdapter("{{chat_message}}", List[ChatMessage])),
    # And connect them
    pipe.connect("adapter", "joiner")
    pipe.connect("joiner", "fc_llm")
    pipe.connect("fc_llm.replies", "validator.messages")
    pipe.connect("validator.validation_error", "joiner")

    result = pipe.run(data={"fc_llm": {"generation_kwargs": {"response_format": {"type": "json_object"}}},
                            "adapter": {"chat_message": [ChatMessage.from_user("Create json from Peter Parker")]}})

    print(json.loads(result["validator"]["validated"][0].content))


    >> {'first_name': 'Peter', 'last_name': 'Parker', 'nationality': 'American', 'name': 'Spider-Man', 'occupation':
    >> 'Superhero', 'age': 23, 'location': 'New York City'}
    ```

    Note that `BranchJoiner` can manage only one data type at a time. In this case, `BranchJoiner` is created for
    passing `List[ChatMessage]`. This determines the type of data that `BranchJoiner` will receive from the upstream
    connected components and also the type of data that `BranchJoiner` will send through its output.

    In the code example, `BranchJoiner` receives a looped back `List[ChatMessage]` from the `JsonSchemaValidator` and
    sends it down to the `OpenAIChatGenerator` for re-generation. We can have multiple loopback connections in the
    pipeline. In this instance, the downstream component is only one (the `OpenAIChatGenerator`), but the pipeline might
    have more than one downstream component.
    """

    def __init__(self, type_: Type):
        """
        Create a `BranchJoiner` component.

        :param type_: The type of data that the `BranchJoiner` will receive from the upstream connected components and
                        distribute to the downstream connected components.
        """
        self.type_ = type_
        # type_'s type can't be determined statically
        component.set_input_types(self, value=GreedyVariadic[type_])  # type: ignore
        component.set_output_types(self, value=type_)

    def to_dict(self):
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(self, type_=serialize_type(self.type_))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BranchJoiner":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
              Deserialized component.
        """
        data["init_parameters"]["type_"] = deserialize_type(data["init_parameters"]["type_"])
        return default_from_dict(cls, data)

    def run(self, **kwargs):
        """
        The run method of the `BranchJoiner` component.

        Multiplexes the input data from the upstream connected components and distributes it to the downstream connected
        components.

        :param **kwargs: The input data. Must be of the type declared in `__init__`.
        :return: A dictionary with the following keys:
            - `value`: The input data.
        """
        if (inputs_count := len(kwargs["value"])) != 1:
            raise ValueError(f"BranchJoiner expects only one input, but {inputs_count} were received.")
        return {"value": kwargs["value"][0]}
