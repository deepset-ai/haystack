import json
from typing import Any, Dict, List

from haystack.dataclasses import ChatMessage, ChatRole

# pylint: disable=pointless-string-statement
"""
## Custom Filters for OutputAdapter

This module defines a collection of custom filters for use with the `OutputAdapter` in processing and transforming
data within a pipeline. These filters allow for dynamic data manipulation based on the specific needs of the
pipeline components.

### Defined Filters:

- `change_role`: Adjusts the role of the last message in a list of chat messages.
- `prepare_fc_params`: Prepares function call parameters for OpenAI functions.
- `tojson`: Converts a string in JSON format to a Python dictionary.

### Usage:

To utilize these filters, import the `ALL_FILTERS` dictionary and pass it to the `OutputAdapter` as the
`custom_filters` argument. You can either specify a subset of these filters or use all of them.

### Examples:

#### a) Using a Subset of Custom Filters:

```python
from haystack.components.converters import OutputAdapter
from haystack.components.converters.output_adapter_filters import ALL_FILTERS

# Use only the 'prepare_fc_params' filter
custom_filters_subset = {"prepare_fc_params": ALL_FILTERS["prepare_fc_params"]}

# Example pipeline component setup with one custom filter
pipe.add_component("a2", OutputAdapter(template="{{fc_kwargs | prepare_fc_params}}",
                                       output_type=Dict[str, Any],
                                       custom_filters=custom_filters_subset))
```

#### b) Using All Filters:

```python
from haystack.components.converters import OutputAdapter
from haystack.components.converters.output_adapter_filters import ALL_FILTERS

# Example pipeline setup using all available custom filters
pipe = Pipeline()
pipe.add_component("fc_llm", OpenAIChatGenerator(model="gpt-3.5-turbo-0125"))
pipe.add_component("spec_to_functions", OpenAPIServiceToFunctions())
pipe.add_component("openapi_container", OpenAPIServiceConnector())
pipe.add_component("a1", OutputAdapter(template="{{documents[0].content | tojson}}",
                                       output_type=Dict[str, Any],
                                       custom_filters=ALL_FILTERS))
pipe.add_component("a2", OutputAdapter(template="{{fc_kwargs | prepare_fc_params}}",
                                       output_type=Dict[str, Any],
                                       custom_filters=ALL_FILTERS))
# Additional components utilizing different filters from ALL_FILTERS
```

This approach enables flexible and dynamic data manipulation within the Haystack pipeline, providing the capability
to adjust data formats and values between different components seamlessly.

### Note:

When use above below listed custom filters or your own developed filters, ensure that the data passed through the
pipelines, OutputAdapters, and filters is properly sanitized and validated, especially if it originates from untrusted
sources, to mitigate security risks.
"""


def change_role(messages: List[ChatMessage], role: ChatRole) -> List[ChatMessage]:
    """
    Change the role of the last message in a list of chat messages.

    :param messages: List of chat messages
    :param role: New role for the last message
    :return: List of chat messages with the last message's role changed to the specified role
    """
    messages[-1].role = role
    return messages


def prepare_fc_params(openai_functions_schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare function call parameters for OpenAI functions given the OpenAI functions schema.

    :param openai_functions_schema: OpenAI functions schema
    :return: Prepared function call parameters
    """
    if openai_functions_schema:
        return {
            "tools": [{"type": "function", "function": openai_functions_schema}],
            "tool_choice": {"type": "function", "function": {"name": openai_functions_schema["name"]}},
        }
    else:
        return {}


def tojson(string_like_json: Any) -> Dict[str, Any]:
    """
    Convert a string in JSON format to a Python dictionary.

    :param string_like_json: String in JSON format
    :return: Python dictionary representation of the JSON string
    """
    return json.loads(str(string_like_json))


ALL_FILTERS = {"change_role": change_role, "prepare_fc_params": prepare_fc_params, "tojson": tojson}
