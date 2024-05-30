# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
import warnings
from typing import Any, Dict

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.core.component.types import Variadic
from haystack.utils import deserialize_type, serialize_type

if sys.version_info < (3, 10):
    from typing_extensions import TypeAlias
else:
    from typing import TypeAlias


logger = logging.getLogger(__name__)


@component(is_greedy=True)
class Multiplexer:
    """
    A component which receives data connections from multiple components and distributes them to multiple components.

    `Multiplexer` offers the ability to both receive data connections from multiple other
    components and to distribute it to various other components, enhancing the functionality of complex data
    processing pipelines.

    `Multiplexer` is important for spreading outputs from a single source like a Large Language Model (LLM) across
    different branches of a pipeline. It is especially valuable in error correction loops by rerouting data for
    reevaluation if errors are detected. For instance, in an example pipeline below, `Multiplexer` helps create
    a schema valid JSON object (given a person's data) with the help of an `OpenAIChatGenerator` and a `JsonSchemaValidator`.
    In case the generated JSON object fails schema validation, `JsonSchemaValidator` starts a correction loop, sending
    the data back through the `Multiplexer` to the `OpenAIChatGenerator` until it passes schema validation. If we didn't
    have `Multiplexer`, we wouldn't be able to loop back the data to `OpenAIChatGenerator` for re-generation, as components
    accept only one input connection for the declared run method parameters.

    Usage example:

    ```python
    import json
    from typing import List

    from haystack import Pipeline
    from haystack.components.converters import OutputAdapter
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.components.others import Multiplexer
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
    pipe.add_component('mx', Multiplexer(List[ChatMessage]))
    pipe.add_component('fc_llm', OpenAIChatGenerator(model="gpt-3.5-turbo-0125"))
    pipe.add_component('validator', JsonSchemaValidator(json_schema=person_schema))
    pipe.add_component('adapter', OutputAdapter("{{chat_message}}", List[ChatMessage])),
    # And connect them
    pipe.connect("adapter", "mx")
    pipe.connect("mx", "fc_llm")
    pipe.connect("fc_llm.replies", "validator.messages")
    pipe.connect("validator.validation_error", "mx")

    result = pipe.run(data={"fc_llm": {"generation_kwargs": {"response_format": {"type": "json_object"}}},
                            "adapter": {"chat_message": [ChatMessage.from_user("Create json object from Peter Parker")]}})

    print(json.loads(result["validator"]["validated"][0].content))


    >> {'first_name': 'Peter', 'last_name': 'Parker', 'nationality': 'American', 'name': 'Spider-Man', 'occupation':
    >> 'Superhero', 'age': 23, 'location': 'New York City'}
    ```

    Note that `Multiplexer` is created with a single type parameter. This determines the
    type of data that `Multiplexer` will receive from the upstream connected components and also the
    type of data that `Multiplexer` will distribute to the downstream connected components. In the example
    above, the `Multiplexer` is created with the type `List[ChatMessage]`. This means `Multiplexer` will receive
    a list of `ChatMessage` objects from the upstream connected components and also distribute a list of `ChatMessage`
    objects to the downstream connected components.

    In the code example, `Multiplexer` receives a looped back `List[ChatMessage]` from the `JsonSchemaValidator` and
    sends it down to the `OpenAIChatGenerator` for re-generation. We can have multiple loop back connections in the
    pipeline. In this instance, the downstream component is only one – the `OpenAIChatGenerator` – but the pipeline can have more
    than one downstream component.
    """

    def __init__(self, type_: TypeAlias):
        """
        Create a `Multiplexer` component.

        :param type_: The type of data that the `Multiplexer` will receive from the upstream connected components and
                        distribute to the downstream connected components.
        """
        warnings.warn(
            "`Multiplexer` is deprecated and will be removed in Haystack 2.4.0. Use `joiners.BranchJoiner` instead.",
            DeprecationWarning,
        )
        self.type_ = type_
        component.set_input_types(self, value=Variadic[type_])
        component.set_output_types(self, value=type_)

    def to_dict(self):
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(self, type_=serialize_type(self.type_))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Multiplexer":
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
        The run method of the `Multiplexer` component.

        Multiplexes the input data from the upstream connected components and distributes it to the downstream connected
        components.

        :param **kwargs: The input data. Must be of the type declared in `__init__`.
        :return: A dictionary with the following keys:
            - `value`: The input data.
        """
        if (inputs_count := len(kwargs["value"])) != 1:
            raise ValueError(f"Multiplexer expects only one input, but {inputs_count} were received.")
        return {"value": kwargs["value"][0]}
