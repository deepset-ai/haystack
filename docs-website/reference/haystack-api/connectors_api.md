---
title: "Connectors"
id: connectors-api
description: "Various connectors to integrate with external services."
slug: "/connectors-api"
---


## `haystack.components.connectors.openapi`

### `haystack.components.connectors.openapi.OpenAPIConnector`

OpenAPIConnector enables direct invocation of REST endpoints defined in an OpenAPI specification.

The OpenAPIConnector serves as a bridge between Haystack pipelines and any REST API that follows
the OpenAPI(formerly Swagger) specification. It dynamically interprets the API specification and
provides an interface for executing API operations. It is usually invoked by passing input
arguments to it from a Haystack pipeline run method or by other components in a pipeline that
pass input arguments to this component.

Example:

```python
from haystack.utils import Secret
from haystack.components.connectors.openapi import OpenAPIConnector

connector = OpenAPIConnector(
    openapi_spec="https://bit.ly/serperdev_openapi",
    credentials=Secret.from_env_var("SERPERDEV_API_KEY"),
    service_kwargs={"config_factory": my_custom_config_factory}
)
response = connector.run(
    operation_id="search",
    arguments={"q": "Who was Nikola Tesla?"}
)
```

Note:

- The `parameters` argument is required for this component.
- The `service_kwargs` argument is optional, it can be used to pass additional options to the OpenAPIClient.

#### `__init__`

```python
__init__(
    openapi_spec: str,
    credentials: Secret | None = None,
    service_kwargs: dict[str, Any] | None = None,
)
```

Initialize the OpenAPIConnector with a specification and optional credentials.

**Parameters:**

- **openapi_spec** (<code>str</code>) – URL, file path, or raw string of the OpenAPI specification
- **credentials** (<code>Secret | None</code>) – Optional API key or credentials for the service wrapped in a Secret
- **service_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments passed to OpenAPIClient.from_spec()
  For example, you can pass a custom config_factory or other configuration options.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serialize this component to a dictionary.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> OpenAPIConnector
```

Deserialize this component from a dictionary.

#### `run`

```python
run(
    operation_id: str, arguments: dict[str, Any] | None = None
) -> dict[str, Any]
```

Invokes a REST endpoint specified in the OpenAPI specification.

**Parameters:**

- **operation_id** (<code>str</code>) – The operationId from the OpenAPI spec to invoke
- **arguments** (<code>dict\[str, Any\] | None</code>) – Optional parameters for the endpoint (query, path, or body parameters)

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary containing the service response

## `haystack.components.connectors.openapi_service`

### `haystack.components.connectors.openapi_service.patch_request`

```python
patch_request(
    self,
    base_url: str,
    *,
    data: Any | None = None,
    parameters: dict[str, Any] | None = None,
    raw_response: bool = False,
    security: dict[str, str] | None = None,
    session: Any | None = None,
    verify: bool | str = True
) -> Any | None
```

Sends an HTTP request as described by this path.

**Parameters:**

- **base_url** (<code>str</code>) – The URL to append this operation's path to when making
  the call.
- **data** (<code>Any | None</code>) – The request body to send.
- **parameters** (<code>dict\[str, Any\] | None</code>) – The parameters used to create the path.
- **raw_response** (<code>bool</code>) – If true, return the raw response instead of validating
  and exterpolating it.
- **security** (<code>dict\[str, str\] | None</code>) – The security scheme to use, and the values it needs to
  process successfully.
- **session** (<code>Any | None</code>) – A persistent request session.
- **verify** (<code>bool | str</code>) – If we should do an ssl verification on the request or not.
  In case str was provided, will use that as the CA.

**Returns:**

- <code>Any | None</code> – The response data, either raw or processed depending on raw_response flag.

### `haystack.components.connectors.openapi_service.OpenAPIServiceConnector`

A component which connects the Haystack framework to OpenAPI services.

The `OpenAPIServiceConnector` component connects the Haystack framework to OpenAPI services, enabling it to call
operations as defined in the OpenAPI specification of the service.

It integrates with `ChatMessage` dataclass, where the payload in messages is used to determine the method to be
called and the parameters to be passed. The message payload should be an OpenAI JSON formatted function calling
string consisting of the method name and the parameters to be passed to the method. The method name and parameters
are then used to invoke the method on the OpenAPI service. The response from the service is returned as a
`ChatMessage`.

Before using this component, users usually resolve service endpoint parameters with a help of
`OpenAPIServiceToFunctions` component.

The example below demonstrates how to use the `OpenAPIServiceConnector` to invoke a method on a https://serper.dev/
service specified via OpenAPI specification.

Note, however, that `OpenAPIServiceConnector` is usually not meant to be used directly, but rather as part of a
pipeline that includes the `OpenAPIServiceToFunctions` component and an `OpenAIChatGenerator` component using LLM
with the function calling capabilities. In the example below we use the function calling payload directly, but in a
real-world scenario, the function calling payload would usually be generated by the `OpenAIChatGenerator` component.

Usage example:

```python
import json
import requests

from haystack.components.connectors import OpenAPIServiceConnector
from haystack.dataclasses import ChatMessage


fc_payload = [{'function': {'arguments': '{"q": "Why was Sam Altman ousted from OpenAI?"}', 'name': 'search'},
               'id': 'call_PmEBYvZ7mGrQP5PUASA5m9wO', 'type': 'function'}]

serper_token = <your_serper_dev_token>
serperdev_openapi_spec = json.loads(requests.get("https://bit.ly/serper_dev_spec").text)
service_connector = OpenAPIServiceConnector()
result = service_connector.run(messages=[ChatMessage.from_assistant(json.dumps(fc_payload))],
                               service_openapi_spec=serperdev_openapi_spec, service_credentials=serper_token)
print(result)

>> {'service_response': ChatMessage(_role=<ChatRole.ASSISTANT: 'assistant'>, _content=[TextContent(text=
>> '{"searchParameters": {"q": "Why was Sam Altman ousted from OpenAI?",
>> "type": "search", "engine": "google"}, "answerBox": {"snippet": "Concerns over AI safety and OpenAI's role
>> in protecting were at the center of Altman's brief ouster from the company."...
```

#### `__init__`

```python
__init__(ssl_verify: bool | str | None = None)
```

Initializes the OpenAPIServiceConnector instance

**Parameters:**

- **ssl_verify** (<code>[bool | str | None</code>) – Decide if to use SSL verification to the requests or not,
  in case a string is passed, will be used as the CA.

#### `run`

```python
run(
    messages: list[ChatMessage],
    service_openapi_spec: dict[str, Any],
    service_credentials: dict | str | None = None,
) -> dict[str, list[ChatMessage]]
```

Processes a list of chat messages to invoke a method on an OpenAPI service.

It parses the last message in the list, expecting it to contain tool calls.

**Parameters:**

- **messages** (<code>list\[ChatMessage\]</code>) – A list of `ChatMessage` objects containing the messages to be processed. The last message
  should contain the tool calls.
- **service_openapi_spec** (<code>dict\[str, Any\]</code>) – The OpenAPI JSON specification object of the service to be invoked. All the refs
  should already be resolved.
- **service_credentials** (<code>dict | str | None</code>) – The credentials to be used for authentication with the service.
  Currently, only the http and apiKey OpenAPI security schemes are supported.

**Returns:**

- <code>dict\[str, list\[ChatMessage\]\]</code> – A dictionary with the following keys:
- `service_response`: a list of `ChatMessage` objects, each containing the response from the service. The
  response is in JSON format, and the `content` attribute of the `ChatMessage` contains
  the JSON string.

**Raises:**

- <code>ValueError</code> – If the last message is not from the assistant or if it does not contain tool calls.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> OpenAPIServiceConnector
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to deserialize from.

**Returns:**

- <code>OpenAPIServiceConnector</code> – The deserialized component.
