---
title: "STACKIT"
id: integrations-stackit
description: "STACKIT integration for Haystack"
slug: "/integrations-stackit"
---

<a id="haystack_integrations.components.generators.stackit.chat.chat_generator"></a>

# Module haystack\_integrations.components.generators.stackit.chat.chat\_generator

<a id="haystack_integrations.components.generators.stackit.chat.chat_generator.STACKITChatGenerator"></a>

## STACKITChatGenerator

Enables text generation using STACKIT generative models through their model serving service.

Users can pass any text generation parameters valid for the STACKIT Chat Completion API
directly to this component using the `generation_kwargs` parameter in `__init__` or the `generation_kwargs`
parameter in `run` method.

This component uses the ChatMessage format for structuring both input and output,
ensuring coherent and contextually relevant responses in chat-based text generation scenarios.
Details on the ChatMessage format can be found in the
[Haystack docs](https://docs.haystack.deepset.ai/docs/chatmessage)

### Usage example
```python
from haystack_integrations.components.generators.stackit import STACKITChatGenerator
from haystack.dataclasses import ChatMessage

generator = STACKITChatGenerator(model="neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8")

result = generator.run([ChatMessage.from_user("Tell me a joke.")])
print(result)
```

<a id="haystack_integrations.components.generators.stackit.chat.chat_generator.STACKITChatGenerator.__init__"></a>

#### STACKITChatGenerator.\_\_init\_\_

```python
def __init__(
        model: str,
        api_key: Secret = Secret.from_env_var("STACKIT_API_KEY"),
        streaming_callback: Optional[StreamingCallbackT] = None,
        api_base_url:
    Optional[
        str] = "https://api.openai-compat.model-serving.eu01.onstackit.cloud/v1",
        generation_kwargs: Optional[Dict[str, Any]] = None,
        *,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        http_client_kwargs: Optional[Dict[str, Any]] = None)
```

Creates an instance of STACKITChatGenerator class.

**Arguments**:

- `model`: The name of the chat completion model to use.
- `api_key`: The STACKIT API key.
- `streaming_callback`: A callback function that is called when a new token is received from the stream.
The callback function accepts StreamingChunk as an argument.
- `api_base_url`: The STACKIT API Base url.
- `generation_kwargs`: Other parameters to use for the model. These parameters are all sent directly to
the STACKIT endpoint.
Some of the supported parameters:
- `max_tokens`: The maximum number of tokens the output text can have.
- `temperature`: What sampling temperature to use. Higher values mean the model will take more risks.
    Try 0.9 for more creative applications and 0 (argmax sampling) for ones with a well-defined answer.
- `top_p`: An alternative to sampling with temperature, called nucleus sampling, where the model
    considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens
    comprising the top 10% probability mass are considered.
- `stream`: Whether to stream back partial progress. If set, tokens will be sent as data-only server-sent
    events as they become available, with the stream terminated by a data: [DONE] message.
- `safe_prompt`: Whether to inject a safety prompt before all conversations.
- `random_seed`: The seed to use for random sampling.
- `timeout`: Timeout for STACKIT client calls. If not set, it defaults to either the `OPENAI_TIMEOUT` environment
variable, or 30 seconds.
- `max_retries`: Maximum number of retries to contact STACKIT after an internal error.
If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
- `http_client_kwargs`: A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/`client`).

<a id="haystack_integrations.components.generators.stackit.chat.chat_generator.STACKITChatGenerator.to_dict"></a>

#### STACKITChatGenerator.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serialize this component to a dictionary.

**Returns**:

The serialized component as a dictionary.

<a id="haystack_integrations.components.generators.stackit.chat.chat_generator.STACKITChatGenerator.from_dict"></a>

#### STACKITChatGenerator.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "OpenAIChatGenerator"
```

Deserialize this component from a dictionary.

**Arguments**:

- `data`: The dictionary representation of this component.

**Returns**:

The deserialized component instance.

<a id="haystack_integrations.components.generators.stackit.chat.chat_generator.STACKITChatGenerator.run"></a>

#### STACKITChatGenerator.run

```python
@component.output_types(replies=list[ChatMessage])
def run(messages: list[ChatMessage],
        streaming_callback: Optional[StreamingCallbackT] = None,
        generation_kwargs: Optional[dict[str, Any]] = None,
        *,
        tools: Optional[ToolsType] = None,
        tools_strict: Optional[bool] = None)
```

Invokes chat completion based on the provided messages and generation parameters.

**Arguments**:

- `messages`: A list of ChatMessage instances representing the input messages.
- `streaming_callback`: A callback function that is called when a new token is received from the stream.
- `generation_kwargs`: Additional keyword arguments for text generation. These parameters will
override the parameters passed during component initialization.
For details on OpenAI API parameters, see [OpenAI documentation](https://platform.openai.com/docs/api-reference/chat/create).
- `tools`: A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
If set, it will override the `tools` parameter provided during initialization.
- `tools_strict`: Whether to enable strict schema adherence for tool calls. If set to `True`, the model will follow exactly
the schema provided in the `parameters` field of the tool definition, but this may increase latency.
If set, it will override the `tools_strict` parameter set during component initialization.

**Returns**:

A dictionary with the following key:
- `replies`: A list containing the generated responses as ChatMessage instances.

<a id="haystack_integrations.components.generators.stackit.chat.chat_generator.STACKITChatGenerator.run_async"></a>

#### STACKITChatGenerator.run\_async

```python
@component.output_types(replies=list[ChatMessage])
async def run_async(messages: list[ChatMessage],
                    streaming_callback: Optional[StreamingCallbackT] = None,
                    generation_kwargs: Optional[dict[str, Any]] = None,
                    *,
                    tools: Optional[ToolsType] = None,
                    tools_strict: Optional[bool] = None)
```

Asynchronously invokes chat completion based on the provided messages and generation parameters.

This is the asynchronous version of the `run` method. It has the same parameters and return values
but can be used with `await` in async code.

**Arguments**:

- `messages`: A list of ChatMessage instances representing the input messages.
- `streaming_callback`: A callback function that is called when a new token is received from the stream.
Must be a coroutine.
- `generation_kwargs`: Additional keyword arguments for text generation. These parameters will
override the parameters passed during component initialization.
For details on OpenAI API parameters, see [OpenAI documentation](https://platform.openai.com/docs/api-reference/chat/create).
- `tools`: A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
If set, it will override the `tools` parameter provided during initialization.
- `tools_strict`: Whether to enable strict schema adherence for tool calls. If set to `True`, the model will follow exactly
the schema provided in the `parameters` field of the tool definition, but this may increase latency.
If set, it will override the `tools_strict` parameter set during component initialization.

**Returns**:

A dictionary with the following key:
- `replies`: A list containing the generated responses as ChatMessage instances.

<a id="haystack_integrations.components.embedders.stackit.document_embedder"></a>

# Module haystack\_integrations.components.embedders.stackit.document\_embedder

<a id="haystack_integrations.components.embedders.stackit.document_embedder.STACKITDocumentEmbedder"></a>

## STACKITDocumentEmbedder

A component for computing Document embeddings using STACKIT as model provider.
The embedding of each Document is stored in the `embedding` field of the Document.

Usage example:
```python
from haystack import Document
from haystack_integrations.components.embedders.stackit import STACKITDocumentEmbedder

doc = Document(content="I love pizza!")

document_embedder = STACKITDocumentEmbedder()

result = document_embedder.run([doc])
print(result['documents'][0].embedding)

# [0.017020374536514282, -0.023255806416273117, ...]
```

<a id="haystack_integrations.components.embedders.stackit.document_embedder.STACKITDocumentEmbedder.__init__"></a>

#### STACKITDocumentEmbedder.\_\_init\_\_

```python
def __init__(
        model: str,
        api_key: Secret = Secret.from_env_var("STACKIT_API_KEY"),
        api_base_url:
    Optional[
        str] = "https://api.openai-compat.model-serving.eu01.onstackit.cloud/v1",
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 32,
        progress_bar: bool = True,
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
        *,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        http_client_kwargs: Optional[Dict[str, Any]] = None)
```

Creates a STACKITDocumentEmbedder component.

**Arguments**:

- `api_key`: The STACKIT API key.
- `model`: The name of the model to use.
- `api_base_url`: The STACKIT API Base url.
For more details, see STACKIT [docs](https://docs.stackit.cloud/stackit/en/basic-concepts-stackit-model-serving-319914567.html).
- `prefix`: A string to add to the beginning of each text.
- `suffix`: A string to add to the end of each text.
- `batch_size`: Number of Documents to encode at once.
- `progress_bar`: Whether to show a progress bar or not. Can be helpful to disable in production deployments to keep
the logs clean.
- `meta_fields_to_embed`: List of meta fields that should be embedded along with the Document text.
- `embedding_separator`: Separator used to concatenate the meta fields to the Document text.
- `timeout`: Timeout for STACKIT client calls. If not set, it defaults to either the `OPENAI_TIMEOUT` environment
variable, or 30 seconds.
- `max_retries`: Maximum number of retries to contact STACKIT after an internal error.
If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
- `http_client_kwargs`: A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/`client`).

<a id="haystack_integrations.components.embedders.stackit.document_embedder.STACKITDocumentEmbedder.to_dict"></a>

#### STACKITDocumentEmbedder.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.embedders.stackit.document_embedder.STACKITDocumentEmbedder.from_dict"></a>

#### STACKITDocumentEmbedder.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "OpenAIDocumentEmbedder"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.embedders.stackit.document_embedder.STACKITDocumentEmbedder.run"></a>

#### STACKITDocumentEmbedder.run

```python
@component.output_types(documents=list[Document], meta=dict[str, Any])
def run(documents: list[Document])
```

Embeds a list of documents.

**Arguments**:

- `documents`: A list of documents to embed.

**Returns**:

A dictionary with the following keys:
- `documents`: A list of documents with embeddings.
- `meta`: Information about the usage of the model.

<a id="haystack_integrations.components.embedders.stackit.document_embedder.STACKITDocumentEmbedder.run_async"></a>

#### STACKITDocumentEmbedder.run\_async

```python
@component.output_types(documents=list[Document], meta=dict[str, Any])
async def run_async(documents: list[Document])
```

Embeds a list of documents asynchronously.

**Arguments**:

- `documents`: A list of documents to embed.

**Returns**:

A dictionary with the following keys:
- `documents`: A list of documents with embeddings.
- `meta`: Information about the usage of the model.

<a id="haystack_integrations.components.embedders.stackit.text_embedder"></a>

# Module haystack\_integrations.components.embedders.stackit.text\_embedder

<a id="haystack_integrations.components.embedders.stackit.text_embedder.STACKITTextEmbedder"></a>

## STACKITTextEmbedder

A component for embedding strings using STACKIT as model provider.

Usage example:
 ```python
from haystack_integrations.components.embedders.stackit import STACKITTextEmbedder

text_to_embed = "I love pizza!"
text_embedder = STACKITTextEmbedder()
print(text_embedder.run(text_to_embed))
```

<a id="haystack_integrations.components.embedders.stackit.text_embedder.STACKITTextEmbedder.__init__"></a>

#### STACKITTextEmbedder.\_\_init\_\_

```python
def __init__(
        model: str,
        api_key: Secret = Secret.from_env_var("STACKIT_API_KEY"),
        api_base_url:
    Optional[
        str] = "https://api.openai-compat.model-serving.eu01.onstackit.cloud/v1",
        prefix: str = "",
        suffix: str = "",
        *,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        http_client_kwargs: Optional[Dict[str, Any]] = None)
```

Creates a STACKITTextEmbedder component.

**Arguments**:

- `api_key`: The STACKIT API key.
- `model`: The name of the STACKIT embedding model to be used.
- `api_base_url`: The STACKIT API Base url.
For more details, see STACKIT [docs](https://docs.stackit.cloud/stackit/en/basic-concepts-stackit-model-serving-319914567.html).
- `prefix`: A string to add to the beginning of each text.
- `suffix`: A string to add to the end of each text.
- `timeout`: Timeout for STACKIT client calls. If not set, it defaults to either the `OPENAI_TIMEOUT` environment
variable, or 30 seconds.
- `max_retries`: Maximum number of retries to contact STACKIT after an internal error.
If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
- `http_client_kwargs`: A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/`client`).

<a id="haystack_integrations.components.embedders.stackit.text_embedder.STACKITTextEmbedder.to_dict"></a>

#### STACKITTextEmbedder.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.embedders.stackit.text_embedder.STACKITTextEmbedder.from_dict"></a>

#### STACKITTextEmbedder.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "OpenAITextEmbedder"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.embedders.stackit.text_embedder.STACKITTextEmbedder.run"></a>

#### STACKITTextEmbedder.run

```python
@component.output_types(embedding=list[float], meta=dict[str, Any])
def run(text: str)
```

Embeds a single string.

**Arguments**:

- `text`: Text to embed.

**Returns**:

A dictionary with the following keys:
- `embedding`: The embedding of the input text.
- `meta`: Information about the usage of the model.

<a id="haystack_integrations.components.embedders.stackit.text_embedder.STACKITTextEmbedder.run_async"></a>

#### STACKITTextEmbedder.run\_async

```python
@component.output_types(embedding=list[float], meta=dict[str, Any])
async def run_async(text: str)
```

Asynchronously embed a single string.

This is the asynchronous version of the `run` method. It has the same parameters and return values
but can be used with `await` in async code.

**Arguments**:

- `text`: Text to embed.

**Returns**:

A dictionary with the following keys:
- `embedding`: The embedding of the input text.
- `meta`: Information about the usage of the model.
