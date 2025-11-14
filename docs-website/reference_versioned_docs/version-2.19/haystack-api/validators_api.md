---
title: Validators
id: validators-api
description: Validators validate LLM outputs
slug: "/validators-api"
---

<a id="json_schema"></a>

# Module json\_schema

<a id="json_schema.is_valid_json"></a>

#### is\_valid\_json

```python
def is_valid_json(s: str) -> bool
```

Check if the provided string is a valid JSON.

**Arguments**:

- `s`: The string to be checked.

**Returns**:

`True` if the string is a valid JSON; otherwise, `False`.

<a id="json_schema.JsonSchemaValidator"></a>

## JsonSchemaValidator

Validates JSON content of `ChatMessage` against a specified [JSON Schema](https://json-schema.org/).

If JSON content of a message conforms to the provided schema, the message is passed along the "validated" output.
If the JSON content does not conform to the schema, the message is passed along the "validation_error" output.
In the latter case, the error message is constructed using the provided `error_template` or a default template.
These error ChatMessages can be used by LLMs in Haystack 2.x recovery loops.

Usage example:

```python
from haystack import Pipeline
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.joiners import BranchJoiner
from haystack.components.validators import JsonSchemaValidator
from haystack import component
from haystack.dataclasses import ChatMessage


@component
class MessageProducer:

    @component.output_types(messages=list[ChatMessage])
    def run(self, messages: list[ChatMessage]) -> dict:
        return {"messages": messages}


p = Pipeline()
p.add_component("llm", OpenAIChatGenerator(model="gpt-4-1106-preview",
                                           generation_kwargs={"response_format": {"type": "json_object"}}))
p.add_component("schema_validator", JsonSchemaValidator())
p.add_component("joiner_for_llm", BranchJoiner(list[ChatMessage]))
p.add_component("message_producer", MessageProducer())

p.connect("message_producer.messages", "joiner_for_llm")
p.connect("joiner_for_llm", "llm")
p.connect("llm.replies", "schema_validator.messages")
p.connect("schema_validator.validation_error", "joiner_for_llm")

result = p.run(data={
    "message_producer": {
        "messages":[ChatMessage.from_user("Generate JSON for person with name 'John' and age 30")]},
        "schema_validator": {
            "json_schema": {
                "type": "object",
                "properties": {"name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
    }
})
print(result)
>> {'schema_validator': {'validated': [ChatMessage(_role=<ChatRole.ASSISTANT: 'assistant'>,
_content=[TextContent(text="\n{\n  "name": "John",\n  "age": 30\n}")],
_name=None, _meta={'model': 'gpt-4-1106-preview', 'index': 0,
'finish_reason': 'stop', 'usage': {'completion_tokens': 17, 'prompt_tokens': 20, 'total_tokens': 37}})]}}
```

<a id="json_schema.JsonSchemaValidator.__init__"></a>

#### JsonSchemaValidator.\_\_init\_\_

```python
def __init__(json_schema: Optional[dict[str, Any]] = None,
             error_template: Optional[str] = None)
```

Initialize the JsonSchemaValidator component.

**Arguments**:

- `json_schema`: A dictionary representing the [JSON schema](https://json-schema.org/) against which
the messages' content is validated.
- `error_template`: A custom template string for formatting the error message in case of validation failure.

<a id="json_schema.JsonSchemaValidator.run"></a>

#### JsonSchemaValidator.run

```python
@component.output_types(validated=list[ChatMessage],
                        validation_error=list[ChatMessage])
def run(messages: list[ChatMessage],
        json_schema: Optional[dict[str, Any]] = None,
        error_template: Optional[str] = None) -> dict[str, list[ChatMessage]]
```

Validates the last of the provided messages against the specified json schema.

If it does, the message is passed along the "validated" output. If it does not, the message is passed along
the "validation_error" output.

**Arguments**:

- `messages`: A list of ChatMessage instances to be validated. The last message in this list is the one
that is validated.
- `json_schema`: A dictionary representing the [JSON schema](https://json-schema.org/)
against which the messages' content is validated. If not provided, the schema from the component init
is used.
- `error_template`: A custom template string for formatting the error message in case of validation. If not
provided, the `error_template` from the component init is used.

**Raises**:

- `ValueError`: If no JSON schema is provided or if the message content is not a dictionary or a list of
dictionaries.

**Returns**:

A dictionary with the following keys:
- "validated": A list of messages if the last message is valid.
- "validation_error": A list of messages if the last message is invalid.
