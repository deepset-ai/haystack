---
title: "Utils"
id: utils-api
description: "Utility functions and classes used across the library."
slug: "/utils-api"
---


## `haystack.utils.asynchronous`

### `haystack.utils.asynchronous.is_callable_async_compatible`

```python
is_callable_async_compatible(func: Callable) -> bool
```

Returns if the given callable is usable inside a component's `run_async` method.

**Parameters:**

- **func** (<code>Callable</code>) – The callable to check.

**Returns:**

- <code>bool</code> – True if the callable is compatible, False otherwise.

## `haystack.utils.auth`

### `haystack.utils.auth.SecretType`

Bases: <code>Enum</code>

#### `from_str`

```python
from_str(string: str) -> SecretType
```

Convert a string to a SecretType.

**Parameters:**

- **string** (<code>str</code>) – The string to convert.

### `haystack.utils.auth.Secret`

Bases: <code>ABC</code>

Encapsulates a secret used for authentication.

Usage example:

```python
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret

generator = OpenAIGenerator(api_key=Secret.from_token("<here_goes_your_token>"))
```

#### `from_token`

```python
from_token(token: str) -> Secret
```

Create a token-based secret. Cannot be serialized.

**Parameters:**

- **token** (<code>str</code>) – The token to use for authentication.

#### `from_env_var`

```python
from_env_var(env_vars: str | list[str], *, strict: bool = True) -> Secret
```

Create an environment variable-based secret. Accepts one or more environment variables.

Upon resolution, it returns a string token from the first environment variable that is set.

**Parameters:**

- **env_vars** (<code>str | list\[str\]</code>) – A single environment variable or an ordered list of
  candidate environment variables.
- **strict** (<code>bool</code>) – Whether to raise an exception if none of the environment
  variables are set.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Convert the secret to a JSON-serializable dictionary.

Some secrets may not be serializable.

**Returns:**

- <code>dict\[str, Any\]</code> – The serialized policy.

#### `from_dict`

```python
from_dict(dict: dict[str, Any]) -> Secret
```

Create a secret from a JSON-serializable dictionary.

**Parameters:**

- **dict** (<code>dict\[str, Any\]</code>) – The dictionary with the serialized data.

**Returns:**

- <code>Secret</code> – The deserialized secret.

#### `resolve_value`

```python
resolve_value() -> Any | None
```

Resolve the secret to an atomic value. The semantics of the value is secret-dependent.

**Returns:**

- <code>Any | None</code> – The value of the secret, if any.

#### `type`

```python
type: SecretType
```

The type of the secret.

### `haystack.utils.auth.TokenSecret`

Bases: <code>Secret</code>

A secret that uses a string token/API key.

Cannot be serialized.

#### `from_token`

```python
from_token(token: str) -> Secret
```

Create a token-based secret. Cannot be serialized.

**Parameters:**

- **token** (<code>str</code>) – The token to use for authentication.

#### `from_env_var`

```python
from_env_var(env_vars: str | list[str], *, strict: bool = True) -> Secret
```

Create an environment variable-based secret. Accepts one or more environment variables.

Upon resolution, it returns a string token from the first environment variable that is set.

**Parameters:**

- **env_vars** (<code>str | list\[str\]</code>) – A single environment variable or an ordered list of
  candidate environment variables.
- **strict** (<code>bool</code>) – Whether to raise an exception if none of the environment
  variables are set.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Convert the secret to a JSON-serializable dictionary.

Some secrets may not be serializable.

**Returns:**

- <code>dict\[str, Any\]</code> – The serialized policy.

#### `from_dict`

```python
from_dict(dict: dict[str, Any]) -> Secret
```

Create a secret from a JSON-serializable dictionary.

**Parameters:**

- **dict** (<code>dict\[str, Any\]</code>) – The dictionary with the serialized data.

**Returns:**

- <code>Secret</code> – The deserialized secret.

#### `resolve_value`

```python
resolve_value() -> Any | None
```

Return the token.

#### `type`

```python
type: SecretType
```

The type of the secret.

### `haystack.utils.auth.EnvVarSecret`

Bases: <code>Secret</code>

A secret that accepts one or more environment variables.

Upon resolution, it returns a string token from the first environment variable that is set. Can be serialized.

#### `from_token`

```python
from_token(token: str) -> Secret
```

Create a token-based secret. Cannot be serialized.

**Parameters:**

- **token** (<code>str</code>) – The token to use for authentication.

#### `from_env_var`

```python
from_env_var(env_vars: str | list[str], *, strict: bool = True) -> Secret
```

Create an environment variable-based secret. Accepts one or more environment variables.

Upon resolution, it returns a string token from the first environment variable that is set.

**Parameters:**

- **env_vars** (<code>str | list\[str\]</code>) – A single environment variable or an ordered list of
  candidate environment variables.
- **strict** (<code>bool</code>) – Whether to raise an exception if none of the environment
  variables are set.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Convert the secret to a JSON-serializable dictionary.

Some secrets may not be serializable.

**Returns:**

- <code>dict\[str, Any\]</code> – The serialized policy.

#### `from_dict`

```python
from_dict(dict: dict[str, Any]) -> Secret
```

Create a secret from a JSON-serializable dictionary.

**Parameters:**

- **dict** (<code>dict\[str, Any\]</code>) – The dictionary with the serialized data.

**Returns:**

- <code>Secret</code> – The deserialized secret.

#### `resolve_value`

```python
resolve_value() -> Any | None
```

Resolve the secret to an atomic value. The semantics of the value is secret-dependent.

#### `type`

```python
type: SecretType
```

The type of the secret.

### `haystack.utils.auth.deserialize_secrets_inplace`

```python
deserialize_secrets_inplace(
    data: dict[str, Any], keys: Iterable[str], *, recursive: bool = False
) -> None
```

Deserialize secrets in a dictionary inplace.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary with the serialized data.
- **keys** (<code>Iterable\[str\]</code>) – The keys of the secrets to deserialize.
- **recursive** (<code>bool</code>) – Whether to recursively deserialize nested dictionaries.

## `haystack.utils.azure`

### `haystack.utils.azure.default_azure_ad_token_provider`

```python
default_azure_ad_token_provider() -> str
```

Get a Azure AD token using the DefaultAzureCredential and the "https://cognitiveservices.azure.com/.default" scope.

## `haystack.utils.base_serialization`

### `haystack.utils.base_serialization.serialize_class_instance`

```python
serialize_class_instance(obj: Any) -> dict[str, Any]
```

Serializes an object that has a `to_dict` method into a dictionary.

**Parameters:**

- **obj** (<code>Any</code>) – The object to be serialized.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary representation of the object.

**Raises:**

- <code>SerializationError</code> – If the object does not have a `to_dict` method.

### `haystack.utils.base_serialization.deserialize_class_instance`

```python
deserialize_class_instance(data: dict[str, Any]) -> Any
```

Deserializes an object from a dictionary representation generated by `auto_serialize_class_instance`.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to deserialize from.

**Returns:**

- <code>Any</code> – The deserialized object.

**Raises:**

- <code>DeserializationError</code> – If the serialization data is malformed, the class type cannot be imported, or the
  class does not have a `from_dict` method.

## `haystack.utils.callable_serialization`

### `haystack.utils.callable_serialization.serialize_callable`

```python
serialize_callable(callable_handle: Callable) -> str
```

Serializes a callable to its full path.

**Parameters:**

- **callable_handle** (<code>Callable</code>) – The callable to serialize

**Returns:**

- <code>str</code> – The full path of the callable

### `haystack.utils.callable_serialization.deserialize_callable`

```python
deserialize_callable(callable_handle: str) -> Callable
```

Deserializes a callable given its full import path as a string.

**Parameters:**

- **callable_handle** (<code>str</code>) – The full path of the callable_handle

**Returns:**

- <code>Callable</code> – The callable

**Raises:**

- <code>DeserializationError</code> – If the callable cannot be found

## `haystack.utils.deserialization`

### `haystack.utils.deserialization.deserialize_chatgenerator_inplace`

```python
deserialize_chatgenerator_inplace(
    data: dict[str, Any], key: str = "chat_generator"
) -> None
```

Deserialize a ChatGenerator in a dictionary inplace.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary with the serialized data.
- **key** (<code>str</code>) – The key in the dictionary where the ChatGenerator is stored.

**Raises:**

- <code>DeserializationError</code> – If the key is missing in the serialized data, the value is not a dictionary,
  the type key is missing, the class cannot be imported, or the class lacks a 'from_dict' method.

### `haystack.utils.deserialization.deserialize_component_inplace`

```python
deserialize_component_inplace(
    data: dict[str, Any], key: str = "chat_generator"
) -> None
```

Deserialize a Component in a dictionary inplace.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary with the serialized data.
- **key** (<code>str</code>) – The key in the dictionary where the Component is stored. Default is "chat_generator".

**Raises:**

- <code>DeserializationError</code> – If the key is missing in the serialized data, the value is not a dictionary,
  the type key is missing, the class cannot be imported, or the class lacks a 'from_dict' method.

## `haystack.utils.device`

### `haystack.utils.device.DeviceType`

Bases: <code>Enum</code>

Represents device types supported by Haystack.

This also includes devices that are not directly used by models - for example, the disk device is exclusively used
in device maps for frameworks that support offloading model weights to disk.

#### `from_str`

```python
from_str(string: str) -> DeviceType
```

Create a device type from a string.

**Parameters:**

- **string** (<code>str</code>) – The string to convert.

**Returns:**

- <code>DeviceType</code> – The device type.

### `haystack.utils.device.Device`

A generic representation of a device.

**Parameters:**

- **type** (<code>DeviceType</code>) – The device type.
- **id** (<code>int | None</code>) – The optional device id.

#### `__init__`

```python
__init__(type: DeviceType, id: int | None = None)
```

Create a generic device.

**Parameters:**

- **type** (<code>DeviceType</code>) – The device type.
- **id** (<code>int | None</code>) – The device id.

#### `cpu`

```python
cpu() -> Device
```

Create a generic CPU device.

**Returns:**

- <code>Device</code> – The CPU device.

#### `gpu`

```python
gpu(id: int = 0) -> Device
```

Create a generic GPU device.

**Parameters:**

- **id** (<code>int</code>) – The GPU id.

**Returns:**

- <code>Device</code> – The GPU device.

#### `disk`

```python
disk() -> Device
```

Create a generic disk device.

**Returns:**

- <code>Device</code> – The disk device.

#### `mps`

```python
mps() -> Device
```

Create a generic Apple Metal Performance Shader device.

**Returns:**

- <code>Device</code> – The MPS device.

#### `xpu`

```python
xpu() -> Device
```

Create a generic Intel GPU Optimization device.

**Returns:**

- <code>Device</code> – The XPU device.

#### `from_str`

```python
from_str(string: str) -> Device
```

Create a generic device from a string.

**Returns:**

- <code>Device</code> – The device.

### `haystack.utils.device.DeviceMap`

A generic mapping from strings to devices.

The semantics of the strings are dependent on target framework. Primarily used to deploy HuggingFace models to
multiple devices.

**Parameters:**

- **mapping** (<code>dict\[str, Device\]</code>) – Dictionary mapping strings to devices.

#### `to_dict`

```python
to_dict() -> dict[str, str]
```

Serialize the mapping to a JSON-serializable dictionary.

**Returns:**

- <code>dict\[str, str\]</code> – The serialized mapping.

#### `first_device`

```python
first_device: Device | None
```

Return the first device in the mapping, if any.

**Returns:**

- <code>Device | None</code> – The first device.

#### `from_dict`

```python
from_dict(dict: dict[str, str]) -> DeviceMap
```

Create a generic device map from a JSON-serialized dictionary.

**Parameters:**

- **dict** (<code>dict\[str, str\]</code>) – The serialized mapping.

**Returns:**

- <code>DeviceMap</code> – The generic device map.

#### `from_hf`

```python
from_hf(hf_device_map: dict[str, Union[int, str, torch.device]]) -> DeviceMap
```

Create a generic device map from a HuggingFace device map.

**Parameters:**

- **hf_device_map** (<code>dict\[str, Union\[int, str, device\]\]</code>) – The HuggingFace device map.

**Returns:**

- <code>DeviceMap</code> – The deserialized device map.

### `haystack.utils.device.ComponentDevice`

A representation of a device for a component.

This can be either a single device or a device map.

#### `from_str`

```python
from_str(device_str: str) -> ComponentDevice
```

Create a component device representation from a device string.

The device string can only represent a single device.

**Parameters:**

- **device_str** (<code>str</code>) – The device string.

**Returns:**

- <code>ComponentDevice</code> – The component device representation.

#### `from_single`

```python
from_single(device: Device) -> ComponentDevice
```

Create a component device representation from a single device.

Disks cannot be used as single devices.

**Parameters:**

- **device** (<code>Device</code>) – The device.

**Returns:**

- <code>ComponentDevice</code> – The component device representation.

#### `from_multiple`

```python
from_multiple(device_map: DeviceMap) -> ComponentDevice
```

Create a component device representation from a device map.

**Parameters:**

- **device_map** (<code>DeviceMap</code>) – The device map.

**Returns:**

- <code>ComponentDevice</code> – The component device representation.

#### `to_torch`

```python
to_torch() -> torch.device
```

Convert the component device representation to PyTorch format.

Device maps are not supported.

**Returns:**

- <code>device</code> – The PyTorch device representation.

#### `to_torch_str`

```python
to_torch_str() -> str
```

Convert the component device representation to PyTorch string format.

Device maps are not supported.

**Returns:**

- <code>str</code> – The PyTorch device string representation.

#### `to_spacy`

```python
to_spacy() -> int
```

Convert the component device representation to spaCy format.

Device maps are not supported.

**Returns:**

- <code>int</code> – The spaCy device representation.

#### `to_hf`

```python
to_hf() -> int | str | dict[str, int | str]
```

Convert the component device representation to HuggingFace format.

**Returns:**

- <code>int | str | dict\[str, int | str\]</code> – The HuggingFace device representation.

#### `update_hf_kwargs`

```python
update_hf_kwargs(
    hf_kwargs: dict[str, Any], *, overwrite: bool
) -> dict[str, Any]
```

Convert the component device representation to HuggingFace format.

Add them as canonical keyword arguments to the keyword arguments dictionary.

**Parameters:**

- **hf_kwargs** (<code>dict\[str, Any\]</code>) – The HuggingFace keyword arguments dictionary.
- **overwrite** (<code>bool</code>) – Whether to overwrite existing device arguments.

**Returns:**

- <code>dict\[str, Any\]</code> – The HuggingFace keyword arguments dictionary.

#### `has_multiple_devices`

```python
has_multiple_devices: bool
```

Whether this component device representation contains multiple devices.

#### `first_device`

```python
first_device: Optional[ComponentDevice]
```

Return either the single device or the first device in the device map, if any.

**Returns:**

- <code>Optional\[ComponentDevice\]</code> – The first device.

#### `resolve_device`

```python
resolve_device(device: Optional[ComponentDevice] = None) -> ComponentDevice
```

Select a device for a component. If a device is specified, it's used. Otherwise, the default device is used.

**Parameters:**

- **device** (<code>Optional\[ComponentDevice\]</code>) – The provided device, if any.

**Returns:**

- <code>ComponentDevice</code> – The resolved device.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Convert the component device representation to a JSON-serializable dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – The dictionary representation.

#### `from_dict`

```python
from_dict(dict: dict[str, Any]) -> ComponentDevice
```

Create a component device representation from a JSON-serialized dictionary.

**Parameters:**

- **dict** (<code>dict\[str, Any\]</code>) – The serialized representation.

**Returns:**

- <code>ComponentDevice</code> – The deserialized component device.

## `haystack.utils.filters`

### `haystack.utils.filters.raise_on_invalid_filter_syntax`

```python
raise_on_invalid_filter_syntax(filters: dict[str, Any] | None = None) -> None
```

Raise an error if the filter syntax is invalid.

### `haystack.utils.filters.document_matches_filter`

```python
document_matches_filter(
    filters: dict[str, Any], document: Document | ByteStream
) -> bool
```

Return whether `filters` match the Document or the ByteStream.

For a detailed specification of the filters, refer to the
`DocumentStore.filter_documents()` protocol documentation.

## `haystack.utils.http_client`

### `haystack.utils.http_client.init_http_client`

```python
init_http_client(
    http_client_kwargs: dict[str, Any] | None = None, async_client: bool = False
) -> httpx.Client | httpx.AsyncClient | None
```

Initialize an httpx client based on the http_client_kwargs.

**Parameters:**

- **http_client_kwargs** (<code>dict\[str, Any\] | None</code>) – The kwargs to pass to the httpx client.
- **async_client** (<code>bool</code>) – Whether to initialize an async client.

**Returns:**

- <code>Client | AsyncClient | None</code> – A httpx client or an async httpx client.

## `haystack.utils.jinja2_chat_extension`

### `haystack.utils.jinja2_chat_extension.ChatMessageExtension`

Bases: <code>Extension</code>

A Jinja2 extension for creating structured chat messages with mixed content types.

This extension provides a custom `{% message %}` tag that allows creating chat messages
with different attributes (role, name, meta) and mixed content types (text, images, etc.).

Inspired by [Banks](https://github.com/masci/banks).

Example:

```
{% message role="system" %}
You are a helpful assistant. You like to talk with {{user_name}}.
{% endmessage %}

{% message role="user" %}
Hello! I am {{user_name}}. Please describe the images.
{% for image in images %}
{{ image | templatize_part }}
{% endfor %}
{% endmessage %}
```

### How it works

1. The `{% message %}` tag is used to define a chat message.
1. The message can contain text and other structured content parts.
1. To include a structured content part in the message, the `| templatize_part` filter is used.
   The filter serializes the content part into a JSON string and wraps it in a `<haystack_content_part>` tag.
1. The `_build_chat_message_json` method of the extension parses the message content parts,
   converts them into a ChatMessage object and serializes it to a JSON string.
1. The obtained JSON string is usable in the ChatPromptBuilder component, where templates are rendered to actual
   ChatMessage objects.

#### `parse`

```python
parse(parser: Any) -> nodes.Node | list[nodes.Node]
```

Parse the message tag and its attributes in the Jinja2 template.

This method handles the parsing of role (mandatory), name (optional), meta (optional) and message body content.

**Parameters:**

- **parser** (<code>Any</code>) – The Jinja2 parser instance

**Returns:**

- <code>Node | list\[Node\]</code> – A CallBlock node containing the parsed message configuration

**Raises:**

- <code>TemplateSyntaxError</code> – If an invalid role is provided

### `haystack.utils.jinja2_chat_extension.templatize_part`

```python
templatize_part(value: ChatMessageContentT) -> str
```

Jinja filter to convert an ChatMessageContentT object into JSON string wrapped in special XML content tags.

**Parameters:**

- **value** (<code>ChatMessageContentT</code>) – The ChatMessageContentT object to convert

**Returns:**

- <code>str</code> – A JSON string wrapped in special XML content tags

**Raises:**

- <code>ValueError</code> – If the value is not an instance of ChatMessageContentT

## `haystack.utils.jinja2_extensions`

### `haystack.utils.jinja2_extensions.Jinja2TimeExtension`

Bases: <code>Extension</code>

#### `__init__`

```python
__init__(environment: Environment)
```

Initializes the JinjaTimeExtension object.

**Parameters:**

- **environment** (<code>Environment</code>) – The Jinja2 environment to initialize the extension with.
  It provides the context where the extension will operate.

#### `parse`

```python
parse(parser: Any) -> nodes.Node | list[nodes.Node]
```

Parse the template expression to determine how to handle the datetime formatting.

**Parameters:**

- **parser** (<code>Any</code>) – The parser object that processes the template expressions and manages the syntax tree.
  It's used to interpret the template's structure.

## `haystack.utils.jupyter`

### `haystack.utils.jupyter.is_in_jupyter`

```python
is_in_jupyter() -> bool
```

Returns `True` if in Jupyter or Google Colab, `False` otherwise.

## `haystack.utils.misc`

### `haystack.utils.misc.expand_page_range`

```python
expand_page_range(page_range: list[str | int]) -> list[int]
```

Takes a list of page numbers and ranges and expands them into a list of page numbers.

For example, given a page_range=['1-3', '5', '8', '10-12'] the function will return [1, 2, 3, 5, 8, 10, 11, 12]

**Parameters:**

- **page_range** (<code>list\[str | int\]</code>) – List of page numbers and ranges

**Returns:**

- <code>list\[int\]</code> – An expanded list of page integers

### `haystack.utils.misc.expit`

```python
expit(x: float | ndarray[Any, Any]) -> float | ndarray[Any, Any]
```

Compute logistic sigmoid function. Maps input values to a range between 0 and 1

**Parameters:**

- **x** (<code>float | ndarray\[Any, Any\]</code>) – input value. Can be a scalar or a numpy array.

## `haystack.utils.requests_utils`

### `haystack.utils.requests_utils.request_with_retry`

```python
request_with_retry(
    attempts: int = 3,
    status_codes_to_retry: list[int] | None = None,
    **kwargs: Any
) -> requests.Response
```

Executes an HTTP request with a configurable exponential backoff retry on failures.

Usage example:

```python
from haystack.utils import request_with_retry

# Sending an HTTP request with default retry configs
res = request_with_retry(method="GET", url="https://example.com")

# Sending an HTTP request with custom number of attempts
res = request_with_retry(method="GET", url="https://example.com", attempts=10)

# Sending an HTTP request with custom HTTP codes to retry
res = request_with_retry(method="GET", url="https://example.com", status_codes_to_retry=[408, 503])

# Sending an HTTP request with custom timeout in seconds
res = request_with_retry(method="GET", url="https://example.com", timeout=5)

# Sending an HTTP request with custom authorization handling
class CustomAuth(requests.auth.AuthBase):
    def __call__(self, r):
        r.headers["authorization"] = "Basic <my_token_here>"
        return r

res = request_with_retry(method="GET", url="https://example.com", auth=CustomAuth())

# All of the above combined
res = request_with_retry(
    method="GET",
    url="https://example.com",
    auth=CustomAuth(),
    attempts=10,
    status_codes_to_retry=[408, 503],
    timeout=5
)

# Sending a POST request
res = request_with_retry(method="POST", url="https://example.com", data={"key": "value"}, attempts=10)

# Retry all 5xx status codes
res = request_with_retry(method="GET", url="https://example.com", status_codes_to_retry=list(range(500, 600)))
```

**Parameters:**

- **attempts** (<code>int</code>) – Maximum number of attempts to retry the request.
- **status_codes_to_retry** (<code>list\[int\] | None</code>) – List of HTTP status codes that will trigger a retry.
  When param is `None`, HTTP 408, 418, 429 and 503 will be retried.
- **kwargs** (<code>Any</code>) – Optional arguments that `request` accepts.

**Returns:**

- <code>Response</code> – The `Response` object.

### `haystack.utils.requests_utils.async_request_with_retry`

```python
async_request_with_retry(
    attempts: int = 3,
    status_codes_to_retry: list[int] | None = None,
    **kwargs: Any
) -> httpx.Response
```

Executes an asynchronous HTTP request with a configurable exponential backoff retry on failures.

Usage example:

```python
import asyncio
from haystack.utils import async_request_with_retry

# Sending an async HTTP request with default retry configs
async def example():
    res = await async_request_with_retry(method="GET", url="https://example.com")
    return res

# Sending an async HTTP request with custom number of attempts
async def example_with_attempts():
    res = await async_request_with_retry(method="GET", url="https://example.com", attempts=10)
    return res

# Sending an async HTTP request with custom HTTP codes to retry
async def example_with_status_codes():
    res = await async_request_with_retry(method="GET", url="https://example.com", status_codes_to_retry=[408, 503])
    return res

# Sending an async HTTP request with custom timeout in seconds
async def example_with_timeout():
    res = await async_request_with_retry(method="GET", url="https://example.com", timeout=5)
    return res

# Sending an async HTTP request with custom headers
async def example_with_headers():
    headers = {"Authorization": "Bearer <my_token_here>"}
    res = await async_request_with_retry(method="GET", url="https://example.com", headers=headers)
    return res

# All of the above combined
async def example_combined():
    headers = {"Authorization": "Bearer <my_token_here>"}
    res = await async_request_with_retry(
        method="GET",
        url="https://example.com",
        headers=headers,
        attempts=10,
        status_codes_to_retry=[408, 503],
        timeout=5
    )
    return res

# Sending an async POST request
async def example_post():
    res = await async_request_with_retry(
        method="POST",
        url="https://example.com",
        json={"key": "value"},
        attempts=10
    )
    return res

# Retry all 5xx status codes
async def example_5xx():
    res = await async_request_with_retry(
        method="GET",
        url="https://example.com",
        status_codes_to_retry=list(range(500, 600))
    )
    return res
```

**Parameters:**

- **attempts** (<code>int</code>) – Maximum number of attempts to retry the request.
- **status_codes_to_retry** (<code>list\[int\] | None</code>) – List of HTTP status codes that will trigger a retry.
  When param is `None`, HTTP 408, 418, 429 and 503 will be retried.
- **kwargs** (<code>Any</code>) – Optional arguments that `httpx.AsyncClient.request` accepts.

**Returns:**

- <code>Response</code> – The `httpx.Response` object.

## `haystack.utils.type_serialization`

### `haystack.utils.type_serialization.serialize_type`

```python
serialize_type(target: Any) -> str
```

Serializes a type or an instance to its string representation, including the module name.

This function handles types, instances of types, and special typing objects.
It assumes that non-typing objects will have a '__name__' attribute.

**Parameters:**

- **target** (<code>Any</code>) – The object to serialize, can be an instance or a type.

**Returns:**

- <code>str</code> – The string representation of the type.

### `haystack.utils.type_serialization.deserialize_type`

```python
deserialize_type(type_str: str) -> Any
```

Deserializes a type given its full import path as a string, including nested generic types.

This function will dynamically import the module if it's not already imported
and then retrieve the type object from it. It also handles nested generic types like
`list[dict[int, str]]`.

**Parameters:**

- **type_str** (<code>str</code>) – The string representation of the type's full import path.

**Returns:**

- <code>Any</code> – The deserialized type object.

**Raises:**

- <code>DeserializationError</code> – If the type cannot be deserialized due to missing module or type.

### `haystack.utils.type_serialization.thread_safe_import`

```python
thread_safe_import(module_name: str) -> ModuleType
```

Import a module in a thread-safe manner.

Importing modules in a multi-threaded environment can lead to race conditions.
This function ensures that the module is imported in a thread-safe manner without having impact
on the performance of the import for single-threaded environments.

**Parameters:**

- **module_name** (<code>str</code>) – the module to import

## `haystack.utils.url_validation`

### `haystack.utils.url_validation.is_valid_http_url`

```python
is_valid_http_url(url: str) -> bool
```

Check if a URL is a valid HTTP/HTTPS URL.
