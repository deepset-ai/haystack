---
title: "Mistral"
id: integrations-mistral
description: "Mistral integration for Haystack"
slug: "/integrations-mistral"
---

<a id="haystack_integrations.components.embedders.mistral.document_embedder"></a>

## Module haystack\_integrations.components.embedders.mistral.document\_embedder

<a id="haystack_integrations.components.embedders.mistral.document_embedder.MistralDocumentEmbedder"></a>

### MistralDocumentEmbedder

A component for computing Document embeddings using Mistral models.
The embedding of each Document is stored in the `embedding` field of the Document.

Usage example:
```python
from haystack import Document
from haystack_integrations.components.embedders.mistral import MistralDocumentEmbedder

doc = Document(content="I love pizza!")

document_embedder = MistralDocumentEmbedder()

result = document_embedder.run([doc])
print(result['documents'][0].embedding)

# [0.017020374536514282, -0.023255806416273117, ...]
```

<a id="haystack_integrations.components.embedders.mistral.document_embedder.MistralDocumentEmbedder.__init__"></a>

#### MistralDocumentEmbedder.\_\_init\_\_

```python
def __init__(api_key: Secret = Secret.from_env_var("MISTRAL_API_KEY"),
             model: str = "mistral-embed",
             api_base_url: Optional[str] = "https://api.mistral.ai/v1",
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

Creates a MistralDocumentEmbedder component.

**Arguments**:

- `api_key`: The Mistral API key.
- `model`: The name of the model to use.
- `api_base_url`: The Mistral API Base url. For more details, see Mistral [docs](https://docs.mistral.ai/api/).
- `prefix`: A string to add to the beginning of each text.
- `suffix`: A string to add to the end of each text.
- `batch_size`: Number of Documents to encode at once.
- `progress_bar`: Whether to show a progress bar or not. Can be helpful to disable in production deployments to keep
the logs clean.
- `meta_fields_to_embed`: List of meta fields that should be embedded along with the Document text.
- `embedding_separator`: Separator used to concatenate the meta fields to the Document text.
- `timeout`: Timeout for Mistral client calls. If not set, it defaults to either the `OPENAI_TIMEOUT` environment
variable, or 30 seconds.
- `max_retries`: Maximum number of retries to contact Mistral after an internal error.
If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
- `http_client_kwargs`: A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/`client`).

<a id="haystack_integrations.components.embedders.mistral.document_embedder.MistralDocumentEmbedder.to_dict"></a>

#### MistralDocumentEmbedder.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.embedders.mistral.document_embedder.MistralDocumentEmbedder.from_dict"></a>

#### MistralDocumentEmbedder.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "OpenAIDocumentEmbedder"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.embedders.mistral.document_embedder.MistralDocumentEmbedder.run"></a>

#### MistralDocumentEmbedder.run

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

<a id="haystack_integrations.components.embedders.mistral.document_embedder.MistralDocumentEmbedder.run_async"></a>

#### MistralDocumentEmbedder.run\_async

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

<a id="haystack_integrations.components.embedders.mistral.text_embedder"></a>

## Module haystack\_integrations.components.embedders.mistral.text\_embedder

<a id="haystack_integrations.components.embedders.mistral.text_embedder.MistralTextEmbedder"></a>

### MistralTextEmbedder

A component for embedding strings using Mistral models.

Usage example:
 ```python
from haystack_integrations.components.embedders.mistral.text_embedder import MistralTextEmbedder

text_to_embed = "I love pizza!"
text_embedder = MistralTextEmbedder()
print(text_embedder.run(text_to_embed))

__output:__

__{'embedding': [0.017020374536514282, -0.023255806416273117, ...],__

__'meta': {'model': 'mistral-embed',__

__         'usage': {'prompt_tokens': 4, 'total_tokens': 4}}}__

```

<a id="haystack_integrations.components.embedders.mistral.text_embedder.MistralTextEmbedder.__init__"></a>

#### MistralTextEmbedder.\_\_init\_\_

```python
def __init__(api_key: Secret = Secret.from_env_var("MISTRAL_API_KEY"),
             model: str = "mistral-embed",
             api_base_url: Optional[str] = "https://api.mistral.ai/v1",
             prefix: str = "",
             suffix: str = "",
             *,
             timeout: Optional[float] = None,
             max_retries: Optional[int] = None,
             http_client_kwargs: Optional[Dict[str, Any]] = None)
```

Creates an MistralTextEmbedder component.

**Arguments**:

- `api_key`: The Mistral API key.
- `model`: The name of the Mistral embedding model to be used.
- `api_base_url`: The Mistral API Base url.
For more details, see Mistral [docs](https://docs.mistral.ai/api/).
- `prefix`: A string to add to the beginning of each text.
- `suffix`: A string to add to the end of each text.
- `timeout`: Timeout for Mistral client calls. If not set, it defaults to either the `OPENAI_TIMEOUT` environment
variable, or 30 seconds.
- `max_retries`: Maximum number of retries to contact Mistral after an internal error.
If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
- `http_client_kwargs`: A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/`client`).

<a id="haystack_integrations.components.embedders.mistral.text_embedder.MistralTextEmbedder.to_dict"></a>

#### MistralTextEmbedder.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.embedders.mistral.text_embedder.MistralTextEmbedder.from_dict"></a>

#### MistralTextEmbedder.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "OpenAITextEmbedder"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.embedders.mistral.text_embedder.MistralTextEmbedder.run"></a>

#### MistralTextEmbedder.run

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

<a id="haystack_integrations.components.embedders.mistral.text_embedder.MistralTextEmbedder.run_async"></a>

#### MistralTextEmbedder.run\_async

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

<a id="haystack_integrations.components.generators.mistral.chat.chat_generator"></a>

## Module haystack\_integrations.components.generators.mistral.chat.chat\_generator

<a id="haystack_integrations.components.generators.mistral.chat.chat_generator.MistralChatGenerator"></a>

### MistralChatGenerator

Enables text generation using Mistral AI generative models.
For supported models, see [Mistral AI docs](https://docs.mistral.ai/platform/endpoints/`operation`/listModels).

Users can pass any text generation parameters valid for the Mistral Chat Completion API
directly to this component via the `generation_kwargs` parameter in `__init__` or the `generation_kwargs`
parameter in `run` method.

Key Features and Compatibility:
- **Primary Compatibility**: Designed to work seamlessly with the Mistral API Chat Completion endpoint.
- **Streaming Support**: Supports streaming responses from the Mistral API Chat Completion endpoint.
- **Customizability**: Supports all parameters supported by the Mistral API Chat Completion endpoint.

This component uses the ChatMessage format for structuring both input and output,
ensuring coherent and contextually relevant responses in chat-based text generation scenarios.
Details on the ChatMessage format can be found in the
[Haystack docs](https://docs.haystack.deepset.ai/v2.0/docs/data-classes#chatmessage)

For more details on the parameters supported by the Mistral API, refer to the
[Mistral API Docs](https://docs.mistral.ai/api/).

Usage example:
```python
from haystack_integrations.components.generators.mistral import MistralChatGenerator
from haystack.dataclasses import ChatMessage

messages = [ChatMessage.from_user("What's Natural Language Processing?")]

client = MistralChatGenerator()
response = client.run(messages)
print(response)

>>{'replies': [ChatMessage(_role=<ChatRole.ASSISTANT: 'assistant'>, _content=[TextContent(text=
>> "Natural Language Processing (NLP) is a branch of artificial intelligence
>> that focuses on enabling computers to understand, interpret, and generate human language in a way that is
>> meaningful and useful.")], _name=None,
>> _meta={'model': 'mistral-small-latest', 'index': 0, 'finish_reason': 'stop',
>> 'usage': {'prompt_tokens': 15, 'completion_tokens': 36, 'total_tokens': 51}})]}
```

<a id="haystack_integrations.components.generators.mistral.chat.chat_generator.MistralChatGenerator.__init__"></a>

#### MistralChatGenerator.\_\_init\_\_

```python
def __init__(api_key: Secret = Secret.from_env_var("MISTRAL_API_KEY"),
             model: str = "mistral-small-latest",
             streaming_callback: Optional[StreamingCallbackT] = None,
             api_base_url: Optional[str] = "https://api.mistral.ai/v1",
             generation_kwargs: Optional[Dict[str, Any]] = None,
             tools: Optional[ToolsType] = None,
             *,
             timeout: Optional[float] = None,
             max_retries: Optional[int] = None,
             http_client_kwargs: Optional[Dict[str, Any]] = None)
```

Creates an instance of MistralChatGenerator. Unless specified otherwise in the `model`, this is for Mistral's

`mistral-small-latest` model.

**Arguments**:

- `api_key`: The Mistral API key.
- `model`: The name of the Mistral chat completion model to use.
- `streaming_callback`: A callback function that is called when a new token is received from the stream.
The callback function accepts StreamingChunk as an argument.
- `api_base_url`: The Mistral API Base url.
For more details, see Mistral [docs](https://docs.mistral.ai/api/).
- `generation_kwargs`: Other parameters to use for the model. These parameters are all sent directly to
the Mistral endpoint. See [Mistral API docs](https://docs.mistral.ai/api/) for more details.
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
 - `response_format`: A JSON schema or a Pydantic model that enforces the structure of the model's response.
    If provided, the output will always be validated against this
    format (unless the model returns a tool call).
    For details, see the [OpenAI Structured Outputs documentation](https://platform.openai.com/docs/guides/structured-outputs).
    Notes:
    - For structured outputs with streaming,
      the `response_format` must be a JSON schema and not a Pydantic model.
- `tools`: A list of tools or a Toolset for which the model can prepare calls. This parameter can accept either a
list of `Tool` objects or a `Toolset` instance.
- `timeout`: The timeout for the Mistral API call. If not set, it defaults to either the `OPENAI_TIMEOUT`
environment variable, or 30 seconds.
- `max_retries`: Maximum number of retries to contact OpenAI after an internal error.
If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
- `http_client_kwargs`: A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/`client`).

<a id="haystack_integrations.components.generators.mistral.chat.chat_generator.MistralChatGenerator.to_dict"></a>

#### MistralChatGenerator.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serialize this component to a dictionary.

**Returns**:

The serialized component as a dictionary.

<a id="haystack_integrations.components.generators.mistral.chat.chat_generator.MistralChatGenerator.from_dict"></a>

#### MistralChatGenerator.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "OpenAIChatGenerator"
```

Deserialize this component from a dictionary.

**Arguments**:

- `data`: The dictionary representation of this component.

**Returns**:

The deserialized component instance.

<a id="haystack_integrations.components.generators.mistral.chat.chat_generator.MistralChatGenerator.run"></a>

#### MistralChatGenerator.run

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

<a id="haystack_integrations.components.generators.mistral.chat.chat_generator.MistralChatGenerator.run_async"></a>

#### MistralChatGenerator.run\_async

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
