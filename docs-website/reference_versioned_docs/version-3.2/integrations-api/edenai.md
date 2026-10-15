---
title: "Eden AI"
id: integrations-edenai
description: "Eden AI integration for Haystack"
slug: "/integrations-edenai"
---


## haystack_integrations.components.embedders.edenai.document_embedder

### EdenAIDocumentEmbedder

Bases: <code>OpenAIDocumentEmbedder</code>

A component for computing Document embeddings using Eden AI's OpenAI-compatible API.

The embedding of each Document is stored in the `embedding` field of the Document.

Eden AI routes embedding requests to many providers (OpenAI, Mistral, Cohere, Google, Jina, and
more) through a single API key, with EU data residency. Models use Eden AI's `provider/model`
naming convention, for example `"openai/text-embedding-3-small"` or `"mistral/mistral-embed"`.

Usage example:

```python
from haystack import Document
from haystack_integrations.components.embedders.edenai import EdenAIDocumentEmbedder

doc = Document(content="I love pizza!")

document_embedder = EdenAIDocumentEmbedder(model="mistral/mistral-embed")

result = document_embedder.run([doc])
print(result["documents"][0].embedding)

# [0.017020374536514282, -0.023255806416273117, ...]
```

#### SUPPORTED_MODELS

```python
SUPPORTED_MODELS: list[str] = [
    "openai/text-embedding-3-small",
    "openai/text-embedding-3-large",
    "mistral/mistral-embed",
    "cohere/embed-english-v3.0",
    "google/text-embedding-004",
]

```

A non-exhaustive list of embedding models supported by this component.
See the [Eden AI models catalog](https://www.edenai.co/models) for the full list.

#### __init__

```python
__init__(
    *,
    model: str = "openai/text-embedding-3-small",
    api_key: Secret = Secret.from_env_var("EDENAI_API_KEY"),
    api_base_url: str | None = "https://api.edenai.run/v3",
    prefix: str = "",
    suffix: str = "",
    batch_size: int = 32,
    progress_bar: bool = True,
    meta_fields_to_embed: list[str] | None = None,
    embedding_separator: str = "\n",
    timeout: float | None = None,
    max_retries: int | None = None,
    http_client_kwargs: dict[str, Any] | None = None
) -> None
```

Creates an `EdenAIDocumentEmbedder` component.

**Parameters:**

- **model** (<code>str</code>) – The name of the Eden AI embedding model to use, in `provider/model` format.
- **api_key** (<code>Secret</code>) – The Eden AI API key. Defaults to the `EDENAI_API_KEY` environment variable.
- **api_base_url** (<code>str | None</code>) – The Eden AI API base URL.
- **prefix** (<code>str</code>) – A string to add to the beginning of each text.
- **suffix** (<code>str</code>) – A string to add to the end of each text.
- **batch_size** (<code>int</code>) – Number of Documents to encode at once.
- **progress_bar** (<code>bool</code>) – Whether to show a progress bar or not. Can be helpful to disable in production deployments to keep
  the logs clean.
- **meta_fields_to_embed** (<code>list\[str\] | None</code>) – List of meta fields that should be embedded along with the Document text.
- **embedding_separator** (<code>str</code>) – Separator used to concatenate the meta fields to the Document text.
- **timeout** (<code>float | None</code>) – Timeout for the API call. If not set, it defaults to either the `OPENAI_TIMEOUT` environment
  variable, or 30 seconds.
- **max_retries** (<code>int | None</code>) – Maximum number of retries to contact Eden AI after an internal error.
  If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
- **http_client_kwargs** (<code>dict\[str, Any\] | None</code>) – A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
  For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

## haystack_integrations.components.embedders.edenai.text_embedder

### EdenAITextEmbedder

Bases: <code>OpenAITextEmbedder</code>

A component for embedding strings using Eden AI's OpenAI-compatible API.

Eden AI routes embedding requests to many providers (OpenAI, Mistral, Cohere, Google, Jina, and
more) through a single API key, with EU data residency. Models use Eden AI's `provider/model`
naming convention, for example `"openai/text-embedding-3-small"` or `"mistral/mistral-embed"`.

Usage example:

```python
from haystack_integrations.components.embedders.edenai import EdenAITextEmbedder

text_embedder = EdenAITextEmbedder(model="mistral/mistral-embed")
print(text_embedder.run("I love pizza!"))
```

#### SUPPORTED_MODELS

```python
SUPPORTED_MODELS: list[str] = [
    "openai/text-embedding-3-small",
    "openai/text-embedding-3-large",
    "mistral/mistral-embed",
    "cohere/embed-english-v3.0",
    "google/text-embedding-004",
]

```

A non-exhaustive list of embedding models supported by this component.
See the [Eden AI models catalog](https://www.edenai.co/models) for the full list.

#### __init__

```python
__init__(
    *,
    model: str = "openai/text-embedding-3-small",
    api_key: Secret = Secret.from_env_var("EDENAI_API_KEY"),
    api_base_url: str | None = "https://api.edenai.run/v3",
    prefix: str = "",
    suffix: str = "",
    timeout: float | None = None,
    max_retries: int | None = None,
    http_client_kwargs: dict[str, Any] | None = None
) -> None
```

Creates an `EdenAITextEmbedder` component.

**Parameters:**

- **model** (<code>str</code>) – The name of the Eden AI embedding model to use, in `provider/model` format.
- **api_key** (<code>Secret</code>) – The Eden AI API key. Defaults to the `EDENAI_API_KEY` environment variable.
- **api_base_url** (<code>str | None</code>) – The Eden AI API base URL.
- **prefix** (<code>str</code>) – A string to add to the beginning of each text.
- **suffix** (<code>str</code>) – A string to add to the end of each text.
- **timeout** (<code>float | None</code>) – Timeout for the API call. If not set, it defaults to either the `OPENAI_TIMEOUT` environment
  variable, or 30 seconds.
- **max_retries** (<code>int | None</code>) – Maximum number of retries to contact Eden AI after an internal error.
  If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
- **http_client_kwargs** (<code>dict\[str, Any\] | None</code>) – A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
  For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

## haystack_integrations.components.generators.edenai.chat.chat_generator

### EdenAIChatGenerator

Bases: <code>OpenAIChatGenerator</code>

A chat generator that uses Eden AI's OpenAI-compatible API to generate chat responses.

Eden AI is a unified API that gives access to 500+ AI models from many providers (OpenAI,
Anthropic, Mistral, Google, Cohere, and more) through a single API key, with built-in
provider fallback and EU data residency. This makes it a convenient, sovereignty-friendly
gateway for building LLM and RAG applications with Haystack.

This class extends Haystack's `OpenAIChatGenerator` to talk to Eden AI. It sets the
`api_base_url` to Eden AI's OpenAI-compatible endpoint and keeps all the standard
configurations available in the `OpenAIChatGenerator`.

Models are selected using Eden AI's `provider/model` naming convention, for example
`"openai/gpt-4o-mini"`, `"anthropic/claude-sonnet-4-5"`, or `"mistral/mistral-large-latest"`.
See the [Eden AI models catalog](https://www.edenai.co/models) for the full list.

Usage example:

```python
from haystack_integrations.components.generators.edenai import EdenAIChatGenerator
from haystack.dataclasses import ChatMessage

messages = [ChatMessage.from_user("What's Natural Language Processing?")]

client = EdenAIChatGenerator(model="mistral/mistral-large-latest")
response = client.run(messages)
print(response["replies"][0].text)
```

#### __init__

```python
__init__(
    *,
    api_key: Secret = Secret.from_env_var("EDENAI_API_KEY"),
    model: str = "openai/gpt-4o-mini",
    streaming_callback: StreamingCallbackT | None = None,
    generation_kwargs: dict[str, Any] | None = None,
    timeout: int | None = None,
    max_retries: int | None = None,
    tools: ToolsType | None = None,
    tools_strict: bool = False,
    http_client_kwargs: dict[str, Any] | None = None
) -> None
```

Creates an `EdenAIChatGenerator` instance.

**Parameters:**

- **api_key** (<code>Secret</code>) – The Eden AI API key. Defaults to the `EDENAI_API_KEY` environment variable.
- **model** (<code>str</code>) – The model to use, in Eden AI's `provider/model` format
  (e.g. `"openai/gpt-4o-mini"`, `"anthropic/claude-sonnet-4-5"`, `"mistral/mistral-large-latest"`).
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – An optional callable invoked with each chunk of a streaming response.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Optional keyword arguments passed to the underlying generation API call,
  such as `max_tokens`, `temperature`, or `top_p`. Eden AI-specific parameters (for example a
  fallback model) are forwarded as-is to the Eden AI endpoint.
- **timeout** (<code>int | None</code>) – The maximum time in seconds to wait for a response from the API.
- **max_retries** (<code>int | None</code>) – The maximum number of times to retry a failed API request.
- **tools** (<code>ToolsType | None</code>) – An optional list of tools or a Toolset the model can use for function calling.
- **tools_strict** (<code>bool</code>) – If `True`, enable strict schema adherence for tool calls.
- **http_client_kwargs** (<code>dict\[str, Any\] | None</code>) – Optional keyword arguments passed to the underlying HTTP client.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.
