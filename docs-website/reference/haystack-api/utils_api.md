---
title: "Utils"
id: utils-api
description: "Utility functions and classes used across the library."
slug: "/utils-api"
---

<a id="azure"></a>

# Module azure

<a id="azure.default_azure_ad_token_provider"></a>

#### default\_azure\_ad\_token\_provider

```python
def default_azure_ad_token_provider() -> str
```

Get a Azure AD token using the DefaultAzureCredential and the "https://cognitiveservices.azure.com/.default" scope.

<a id="jupyter"></a>

# Module jupyter

<a id="jupyter.is_in_jupyter"></a>

#### is\_in\_jupyter

```python
def is_in_jupyter() -> bool
```

Returns `True` if in Jupyter or Google Colab, `False` otherwise.

<a id="url_validation"></a>

# Module url\_validation

<a id="url_validation.is_valid_http_url"></a>

#### is\_valid\_http\_url

```python
def is_valid_http_url(url: str) -> bool
```

Check if a URL is a valid HTTP/HTTPS URL.

<a id="auth"></a>

# Module auth

<a id="auth.SecretType"></a>

## SecretType

<a id="auth.SecretType.from_str"></a>

#### SecretType.from\_str

```python
@staticmethod
def from_str(string: str) -> "SecretType"
```

Convert a string to a SecretType.

**Arguments**:

- `string`: The string to convert.

<a id="auth.Secret"></a>

## Secret

Encapsulates a secret used for authentication.

Usage example:
```python
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret

generator = OpenAIGenerator(api_key=Secret.from_token("<here_goes_your_token>"))
```

<a id="auth.Secret.from_token"></a>

#### Secret.from\_token

```python
@staticmethod
def from_token(token: str) -> "Secret"
```

Create a token-based secret. Cannot be serialized.

**Arguments**:

- `token`: The token to use for authentication.

<a id="auth.Secret.from_env_var"></a>

#### Secret.from\_env\_var

```python
@staticmethod
def from_env_var(env_vars: Union[str, list[str]],
                 *,
                 strict: bool = True) -> "Secret"
```

Create an environment variable-based secret. Accepts one or more environment variables.

Upon resolution, it returns a string token from the first environment variable that is set.

**Arguments**:

- `env_vars`: A single environment variable or an ordered list of
candidate environment variables.
- `strict`: Whether to raise an exception if none of the environment
variables are set.

<a id="auth.Secret.to_dict"></a>

#### Secret.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Convert the secret to a JSON-serializable dictionary.

Some secrets may not be serializable.

**Returns**:

The serialized policy.

<a id="auth.Secret.from_dict"></a>

#### Secret.from\_dict

```python
@staticmethod
def from_dict(dict: dict[str, Any]) -> "Secret"
```

Create a secret from a JSON-serializable dictionary.

**Arguments**:

- `dict`: The dictionary with the serialized data.

**Returns**:

The deserialized secret.

<a id="auth.Secret.resolve_value"></a>

#### Secret.resolve\_value

```python
@abstractmethod
def resolve_value() -> Optional[Any]
```

Resolve the secret to an atomic value. The semantics of the value is secret-dependent.

**Returns**:

The value of the secret, if any.

<a id="auth.Secret.type"></a>

#### Secret.type

```python
@property
@abstractmethod
def type() -> SecretType
```

The type of the secret.

<a id="auth.deserialize_secrets_inplace"></a>

#### deserialize\_secrets\_inplace

```python
def deserialize_secrets_inplace(data: dict[str, Any],
                                keys: Iterable[str],
                                *,
                                recursive: bool = False) -> None
```

Deserialize secrets in a dictionary inplace.

**Arguments**:

- `data`: The dictionary with the serialized data.
- `keys`: The keys of the secrets to deserialize.
- `recursive`: Whether to recursively deserialize nested dictionaries.

<a id="callable_serialization"></a>

# Module callable\_serialization

<a id="callable_serialization.serialize_callable"></a>

#### serialize\_callable

```python
def serialize_callable(callable_handle: Callable) -> str
```

Serializes a callable to its full path.

**Arguments**:

- `callable_handle`: The callable to serialize

**Returns**:

The full path of the callable

<a id="callable_serialization.deserialize_callable"></a>

#### deserialize\_callable

```python
def deserialize_callable(callable_handle: str) -> Callable
```

Deserializes a callable given its full import path as a string.

**Arguments**:

- `callable_handle`: The full path of the callable_handle

**Raises**:

- `DeserializationError`: If the callable cannot be found

**Returns**:

The callable

<a id="asynchronous"></a>

# Module asynchronous

<a id="asynchronous.is_callable_async_compatible"></a>

#### is\_callable\_async\_compatible

```python
def is_callable_async_compatible(func: Callable) -> bool
```

Returns if the given callable is usable inside a component's `run_async` method.

**Arguments**:

- `callable`: The callable to check.

**Returns**:

True if the callable is compatible, False otherwise.

<a id="requests_utils"></a>

# Module requests\_utils

<a id="requests_utils.request_with_retry"></a>

#### request\_with\_retry

```python
def request_with_retry(attempts: int = 3,
                       status_codes_to_retry: Optional[list[int]] = None,
                       **kwargs: Any) -> requests.Response
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

**Arguments**:

- `attempts`: Maximum number of attempts to retry the request.
- `status_codes_to_retry`: List of HTTP status codes that will trigger a retry.
When param is `None`, HTTP 408, 418, 429 and 503 will be retried.
- `kwargs`: Optional arguments that `request` accepts.

**Returns**:

The `Response` object.

<a id="requests_utils.async_request_with_retry"></a>

#### async\_request\_with\_retry

```python
async def async_request_with_retry(attempts: int = 3,
                                   status_codes_to_retry: Optional[
                                       list[int]] = None,
                                   **kwargs: Any) -> httpx.Response
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

**Arguments**:

- `attempts`: Maximum number of attempts to retry the request.
- `status_codes_to_retry`: List of HTTP status codes that will trigger a retry.
When param is `None`, HTTP 408, 418, 429 and 503 will be retried.
- `kwargs`: Optional arguments that `httpx.AsyncClient.request` accepts.

**Returns**:

The `httpx.Response` object.

<a id="filters"></a>

# Module filters

<a id="filters.raise_on_invalid_filter_syntax"></a>

#### raise\_on\_invalid\_filter\_syntax

```python
def raise_on_invalid_filter_syntax(
        filters: Optional[dict[str, Any]] = None) -> None
```

Raise an error if the filter syntax is invalid.

<a id="filters.document_matches_filter"></a>

#### document\_matches\_filter

```python
def document_matches_filter(filters: dict[str, Any],
                            document: Union[Document, ByteStream]) -> bool
```

Return whether `filters` match the Document or the ByteStream.

For a detailed specification of the filters, refer to the
`DocumentStore.filter_documents()` protocol documentation.

<a id="misc"></a>

# Module misc

<a id="misc.expand_page_range"></a>

#### expand\_page\_range

```python
def expand_page_range(page_range: list[Union[str, int]]) -> list[int]
```

Takes a list of page numbers and ranges and expands them into a list of page numbers.

For example, given a page_range=['1-3', '5', '8', '10-12'] the function will return [1, 2, 3, 5, 8, 10, 11, 12]

**Arguments**:

- `page_range`: List of page numbers and ranges

**Returns**:

An expanded list of page integers

<a id="misc.expit"></a>

#### expit

```python
def expit(
        x: Union[float, ndarray[Any, Any]]) -> Union[float, ndarray[Any, Any]]
```

Compute logistic sigmoid function. Maps input values to a range between 0 and 1

**Arguments**:

- `x`: input value. Can be a scalar or a numpy array.

<a id="device"></a>

# Module device

<a id="device.DeviceType"></a>

## DeviceType

Represents device types supported by Haystack.

This also includes devices that are not directly used by models - for example, the disk device is exclusively used
in device maps for frameworks that support offloading model weights to disk.

<a id="device.DeviceType.from_str"></a>

#### DeviceType.from\_str

```python
@staticmethod
def from_str(string: str) -> "DeviceType"
```

Create a device type from a string.

**Arguments**:

- `string`: The string to convert.

**Returns**:

The device type.

<a id="device.Device"></a>

## Device

A generic representation of a device.

**Arguments**:

- `type`: The device type.
- `id`: The optional device id.

<a id="device.Device.__init__"></a>

#### Device.\_\_init\_\_

```python
def __init__(type: DeviceType, id: Optional[int] = None)
```

Create a generic device.

**Arguments**:

- `type`: The device type.
- `id`: The device id.

<a id="device.Device.cpu"></a>

#### Device.cpu

```python
@staticmethod
def cpu() -> "Device"
```

Create a generic CPU device.

**Returns**:

The CPU device.

<a id="device.Device.gpu"></a>

#### Device.gpu

```python
@staticmethod
def gpu(id: int = 0) -> "Device"
```

Create a generic GPU device.

**Arguments**:

- `id`: The GPU id.

**Returns**:

The GPU device.

<a id="device.Device.disk"></a>

#### Device.disk

```python
@staticmethod
def disk() -> "Device"
```

Create a generic disk device.

**Returns**:

The disk device.

<a id="device.Device.mps"></a>

#### Device.mps

```python
@staticmethod
def mps() -> "Device"
```

Create a generic Apple Metal Performance Shader device.

**Returns**:

The MPS device.

<a id="device.Device.xpu"></a>

#### Device.xpu

```python
@staticmethod
def xpu() -> "Device"
```

Create a generic Intel GPU Optimization device.

**Returns**:

The XPU device.

<a id="device.Device.from_str"></a>

#### Device.from\_str

```python
@staticmethod
def from_str(string: str) -> "Device"
```

Create a generic device from a string.

**Returns**:

The device.

<a id="device.DeviceMap"></a>

## DeviceMap

A generic mapping from strings to devices.

The semantics of the strings are dependent on target framework. Primarily used to deploy HuggingFace models to
multiple devices.

**Arguments**:

- `mapping`: Dictionary mapping strings to devices.

<a id="device.DeviceMap.to_dict"></a>

#### DeviceMap.to\_dict

```python
def to_dict() -> dict[str, str]
```

Serialize the mapping to a JSON-serializable dictionary.

**Returns**:

The serialized mapping.

<a id="device.DeviceMap.first_device"></a>

#### DeviceMap.first\_device

```python
@property
def first_device() -> Optional[Device]
```

Return the first device in the mapping, if any.

**Returns**:

The first device.

<a id="device.DeviceMap.from_dict"></a>

#### DeviceMap.from\_dict

```python
@staticmethod
def from_dict(dict: dict[str, str]) -> "DeviceMap"
```

Create a generic device map from a JSON-serialized dictionary.

**Arguments**:

- `dict`: The serialized mapping.

**Returns**:

The generic device map.

<a id="device.DeviceMap.from_hf"></a>

#### DeviceMap.from\_hf

```python
@staticmethod
def from_hf(
        hf_device_map: dict[str, Union[int, str,
                                       "torch.device"]]) -> "DeviceMap"
```

Create a generic device map from a HuggingFace device map.

**Arguments**:

- `hf_device_map`: The HuggingFace device map.

**Returns**:

The deserialized device map.

<a id="device.ComponentDevice"></a>

## ComponentDevice

A representation of a device for a component.

This can be either a single device or a device map.

<a id="device.ComponentDevice.from_str"></a>

#### ComponentDevice.from\_str

```python
@classmethod
def from_str(cls, device_str: str) -> "ComponentDevice"
```

Create a component device representation from a device string.

The device string can only represent a single device.

**Arguments**:

- `device_str`: The device string.

**Returns**:

The component device representation.

<a id="device.ComponentDevice.from_single"></a>

#### ComponentDevice.from\_single

```python
@classmethod
def from_single(cls, device: Device) -> "ComponentDevice"
```

Create a component device representation from a single device.

Disks cannot be used as single devices.

**Arguments**:

- `device`: The device.

**Returns**:

The component device representation.

<a id="device.ComponentDevice.from_multiple"></a>

#### ComponentDevice.from\_multiple

```python
@classmethod
def from_multiple(cls, device_map: DeviceMap) -> "ComponentDevice"
```

Create a component device representation from a device map.

**Arguments**:

- `device_map`: The device map.

**Returns**:

The component device representation.

<a id="device.ComponentDevice.to_torch"></a>

#### ComponentDevice.to\_torch

```python
def to_torch() -> "torch.device"
```

Convert the component device representation to PyTorch format.

Device maps are not supported.

**Returns**:

The PyTorch device representation.

<a id="device.ComponentDevice.to_torch_str"></a>

#### ComponentDevice.to\_torch\_str

```python
def to_torch_str() -> str
```

Convert the component device representation to PyTorch string format.

Device maps are not supported.

**Returns**:

The PyTorch device string representation.

<a id="device.ComponentDevice.to_spacy"></a>

#### ComponentDevice.to\_spacy

```python
def to_spacy() -> int
```

Convert the component device representation to spaCy format.

Device maps are not supported.

**Returns**:

The spaCy device representation.

<a id="device.ComponentDevice.to_hf"></a>

#### ComponentDevice.to\_hf

```python
def to_hf() -> Union[Union[int, str], dict[str, Union[int, str]]]
```

Convert the component device representation to HuggingFace format.

**Returns**:

The HuggingFace device representation.

<a id="device.ComponentDevice.update_hf_kwargs"></a>

#### ComponentDevice.update\_hf\_kwargs

```python
def update_hf_kwargs(hf_kwargs: dict[str, Any], *,
                     overwrite: bool) -> dict[str, Any]
```

Convert the component device representation to HuggingFace format.

Add them as canonical keyword arguments to the keyword arguments dictionary.

**Arguments**:

- `hf_kwargs`: The HuggingFace keyword arguments dictionary.
- `overwrite`: Whether to overwrite existing device arguments.

**Returns**:

The HuggingFace keyword arguments dictionary.

<a id="device.ComponentDevice.has_multiple_devices"></a>

#### ComponentDevice.has\_multiple\_devices

```python
@property
def has_multiple_devices() -> bool
```

Whether this component device representation contains multiple devices.

<a id="device.ComponentDevice.first_device"></a>

#### ComponentDevice.first\_device

```python
@property
def first_device() -> Optional["ComponentDevice"]
```

Return either the single device or the first device in the device map, if any.

**Returns**:

The first device.

<a id="device.ComponentDevice.resolve_device"></a>

#### ComponentDevice.resolve\_device

```python
@staticmethod
def resolve_device(
        device: Optional["ComponentDevice"] = None) -> "ComponentDevice"
```

Select a device for a component. If a device is specified, it's used. Otherwise, the default device is used.

**Arguments**:

- `device`: The provided device, if any.

**Returns**:

The resolved device.

<a id="device.ComponentDevice.to_dict"></a>

#### ComponentDevice.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Convert the component device representation to a JSON-serializable dictionary.

**Returns**:

The dictionary representation.

<a id="device.ComponentDevice.from_dict"></a>

#### ComponentDevice.from\_dict

```python
@classmethod
def from_dict(cls, dict: dict[str, Any]) -> "ComponentDevice"
```

Create a component device representation from a JSON-serialized dictionary.

**Arguments**:

- `dict`: The serialized representation.

**Returns**:

The deserialized component device.

<a id="http_client"></a>

# Module http\_client

<a id="http_client.init_http_client"></a>

#### init\_http\_client

```python
def init_http_client(
    http_client_kwargs: Optional[dict[str, Any]] = None,
    async_client: bool = False
) -> Union[httpx.Client, httpx.AsyncClient, None]
```

Initialize an httpx client based on the http_client_kwargs.

**Arguments**:

- `http_client_kwargs`: The kwargs to pass to the httpx client.
- `async_client`: Whether to initialize an async client.

**Returns**:

A httpx client or an async httpx client.

<a id="type_serialization"></a>

# Module type\_serialization

<a id="type_serialization.serialize_type"></a>

#### serialize\_type

```python
def serialize_type(target: Any) -> str
```

Serializes a type or an instance to its string representation, including the module name.

This function handles types, instances of types, and special typing objects.
It assumes that non-typing objects will have a '__name__' attribute.

**Arguments**:

- `target`: The object to serialize, can be an instance or a type.

**Returns**:

The string representation of the type.

<a id="type_serialization.deserialize_type"></a>

#### deserialize\_type

```python
def deserialize_type(type_str: str) -> Any
```

Deserializes a type given its full import path as a string, including nested generic types.

This function will dynamically import the module if it's not already imported
and then retrieve the type object from it. It also handles nested generic types like
`list[dict[int, str]]`.

**Arguments**:

- `type_str`: The string representation of the type's full import path.

**Raises**:

- `DeserializationError`: If the type cannot be deserialized due to missing module or type.

**Returns**:

The deserialized type object.

<a id="type_serialization.thread_safe_import"></a>

#### thread\_safe\_import

```python
def thread_safe_import(module_name: str) -> ModuleType
```

Import a module in a thread-safe manner.

Importing modules in a multi-threaded environment can lead to race conditions.
This function ensures that the module is imported in a thread-safe manner without having impact
on the performance of the import for single-threaded environments.

**Arguments**:

- `module_name`: the module to import

<a id="jinja2_chat_extension"></a>

# Module jinja2\_chat\_extension

<a id="jinja2_chat_extension.ChatMessageExtension"></a>

## ChatMessageExtension

A Jinja2 extension for creating structured chat messages with mixed content types.

This extension provides a custom `{% message %}` tag that allows creating chat messages
with different attributes (role, name, meta) and mixed content types (text, images, etc.).

Inspired by [Banks](https://github.com/masci/banks).

**Example**:

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
  2. The message can contain text and other structured content parts.
  3. To include a structured content part in the message, the `| templatize_part` filter is used.
  The filter serializes the content part into a JSON string and wraps it in a `<haystack_content_part>` tag.
  4. The `_build_chat_message_json` method of the extension parses the message content parts,
  converts them into a ChatMessage object and serializes it to a JSON string.
  5. The obtained JSON string is usable in the ChatPromptBuilder component, where templates are rendered to actual
  ChatMessage objects.

<a id="jinja2_chat_extension.ChatMessageExtension.parse"></a>

#### ChatMessageExtension.parse

```python
def parse(parser: Any) -> Union[nodes.Node, list[nodes.Node]]
```

Parse the message tag and its attributes in the Jinja2 template.

This method handles the parsing of role (mandatory), name (optional), meta (optional) and message body content.

**Arguments**:

- `parser`: The Jinja2 parser instance

**Raises**:

- `TemplateSyntaxError`: If an invalid role is provided

**Returns**:

A CallBlock node containing the parsed message configuration

<a id="jinja2_chat_extension.templatize_part"></a>

#### templatize\_part

```python
def templatize_part(value: ChatMessageContentT) -> str
```

Jinja filter to convert an ChatMessageContentT object into JSON string wrapped in special XML content tags.

**Arguments**:

- `value`: The ChatMessageContentT object to convert

**Raises**:

- `ValueError`: If the value is not an instance of ChatMessageContentT

**Returns**:

A JSON string wrapped in special XML content tags

<a id="jinja2_extensions"></a>

# Module jinja2\_extensions

<a id="jinja2_extensions.Jinja2TimeExtension"></a>

## Jinja2TimeExtension

<a id="jinja2_extensions.Jinja2TimeExtension.__init__"></a>

#### Jinja2TimeExtension.\_\_init\_\_

```python
def __init__(environment: Environment)
```

Initializes the JinjaTimeExtension object.

**Arguments**:

- `environment`: The Jinja2 environment to initialize the extension with.
It provides the context where the extension will operate.

<a id="jinja2_extensions.Jinja2TimeExtension.parse"></a>

#### Jinja2TimeExtension.parse

```python
def parse(parser: Any) -> Union[nodes.Node, list[nodes.Node]]
```

Parse the template expression to determine how to handle the datetime formatting.

**Arguments**:

- `parser`: The parser object that processes the template expressions and manages the syntax tree.
It's used to interpret the template's structure.

<a id="deserialization"></a>

# Module deserialization

<a id="deserialization.deserialize_document_store_in_init_params_inplace"></a>

#### deserialize\_document\_store\_in\_init\_params\_inplace

```python
def deserialize_document_store_in_init_params_inplace(
        data: dict[str, Any], key: str = "document_store") -> None
```

Deserializes a generic document store from the init_parameters of a serialized component in place.

**Arguments**:

- `data`: The dictionary to deserialize from.
- `key`: The key in the `data["init_parameters"]` dictionary where the document store is specified.

**Raises**:

- `DeserializationError`: If the document store is not properly specified in the serialization data or its type cannot be imported.

**Returns**:

The dictionary, with the document store deserialized.

<a id="deserialization.deserialize_chatgenerator_inplace"></a>

#### deserialize\_chatgenerator\_inplace

```python
def deserialize_chatgenerator_inplace(data: dict[str, Any],
                                      key: str = "chat_generator") -> None
```

Deserialize a ChatGenerator in a dictionary inplace.

**Arguments**:

- `data`: The dictionary with the serialized data.
- `key`: The key in the dictionary where the ChatGenerator is stored.

**Raises**:

- `DeserializationError`: If the key is missing in the serialized data, the value is not a dictionary,
the type key is missing, the class cannot be imported, or the class lacks a 'from_dict' method.

<a id="deserialization.deserialize_component_inplace"></a>

#### deserialize\_component\_inplace

```python
def deserialize_component_inplace(data: dict[str, Any],
                                  key: str = "chat_generator") -> None
```

Deserialize a Component in a dictionary inplace.

**Arguments**:

- `data`: The dictionary with the serialized data.
- `key`: The key in the dictionary where the Component is stored. Default is "chat_generator".

**Raises**:

- `DeserializationError`: If the key is missing in the serialized data, the value is not a dictionary,
the type key is missing, the class cannot be imported, or the class lacks a 'from_dict' method.

<a id="base_serialization"></a>

# Module base\_serialization

<a id="base_serialization.serialize_class_instance"></a>

#### serialize\_class\_instance

```python
def serialize_class_instance(obj: Any) -> dict[str, Any]
```

Serializes an object that has a `to_dict` method into a dictionary.

**Arguments**:

- `obj`: The object to be serialized.

**Raises**:

- `SerializationError`: If the object does not have a `to_dict` method.

**Returns**:

A dictionary representation of the object.

<a id="base_serialization.deserialize_class_instance"></a>

#### deserialize\_class\_instance

```python
def deserialize_class_instance(data: dict[str, Any]) -> Any
```

Deserializes an object from a dictionary representation generated by `auto_serialize_class_instance`.

**Arguments**:

- `data`: The dictionary to deserialize from.

**Raises**:

- `DeserializationError`: If the serialization data is malformed, the class type cannot be imported, or the
class does not have a `from_dict` method.

**Returns**:

The deserialized object.

