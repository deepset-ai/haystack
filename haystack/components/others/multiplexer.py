import sys
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
    The Multiplexer is a pivotal component in Haystack's framework enabled with the dual capability to receive data from
    multiple upstream components and to distribute data to numerous downstream components. This multiplexing capability
    is particularly advantageous in complex data processing pipelines.

    It enables outputs from components, like those from a Large Language Model (LLM), to be efficiently disseminated
    across various processing pipeline branches. Additionally, the Multiplexer is particularly useful in error
    correction loops: it facilitates the redirection of data back into the processing cycle for re-evaluation whenever
    a discrepancy or error is identified in the processing pipeline.

    In this example below, a pipeline is set up with a `Multiplexer` to generate a JSON object based on a person's data.
    The `OpenAIChatGenerator` is used as an LLM component to generate the JSON object. The `JsonSchemaValidator`
    ensures that the generated JSON adheres to the specified schema. If the generated JSON does not conform to the
    schema, a correction loop is initiated which sends the data back to the LLM via Multiplexer for re-generation until
    the output validates against the schema.

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
            "nationality": {"type": "string", "enum": ["Italian", "Portuguese", "Hungarian"]},
        },
        "required": ["first_name", "last_name", "nationality"]
    }

    # Initialize a pipeline
    pipe = Pipeline()

    # Add components to the pipeline
    pipe.add_component('mx', Multiplexer(List[ChatMessage]))
    pipe.add_component('llm', OpenAIChatGenerator(model="gpt-3.5-turbo-0125"))
    pipe.add_component('validator', JsonSchemaValidator(json_schema=person_schema))
    pipe.add_component('a1', OutputAdapter("{{name | create_cm}}",
                                               List[ChatMessage],
                                               {"create_cm": lambda s: [ChatMessage.from_user("Create json from " + s)]}))

    pipe.connect("a1", "mx")
    pipe.connect("mx", "llm")
    pipe.connect("llm.replies", "validator.messages")
    pipe.connect("validator.validation_error", "mx")

    result = pipe.run(data={"a1": {"name": "Sara Zanzottera"},
                                "llm":  {"generation_kwargs": {"response_format": { "type": "json_object" }}}})

    print(json.loads(result["validator"]["validated"][0].content))

    >> {'first_name': 'Sara', 'last_name': 'Zanzottera', 'nationality': 'Italian', 'occupation': 'Software Engineer',
    >> 'skills': ['JavaScript', 'Python', 'Java', 'HTML', 'CSS']}
    ```

    The most important thing to note is that the Multiplexer is created with a single type parameter. The type
    determines the type of the data that the Multiplexer will receive from the upstream connected components and also
    the type of the data that the Multiplexer will distribute to the downstream connected components. In the example
    above, the Multiplexer is created with the type `List[ChatMessage]`. This means that the Multiplexer will receive
    a list of `ChatMessage` objects from the upstream connected components. And the Multiplexer will then distribute
    a list of `ChatMessage` objects to the downstream connected components.

    In the example above the Multiplexer receives a looped back `List[ChatMessage]` from the `JsonSchemaValidator` and
    sends it back to the `OpenAIChatGenerator` for re-generation. We can have multiple loop back connections in the
    pipeline. In this example, the downstream component is only one, the `OpenAIChatGenerator`, but it can have more
    than one downstream component.
    """

    def __init__(self, type_: TypeAlias):
        """
        Create a Multiplexer component.

        :param type_: The type of data that the Multiplexer will receive from the upstream connected components and
                        distribute to the downstream connected components.
        """
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
        Multiplexes the input data from the upstream connected components and distributes it to the downstream connected
        components.

        :param kwargs: The input data which has to be a type of declared in the init.
        :return: A dictionary with the following keys:
            - `value`: The input data.
        """
        if (inputs_count := len(kwargs["value"])) != 1:
            raise ValueError(f"Multiplexer expects only one input, but {inputs_count} were received.")
        return {"value": kwargs["value"][0]}
