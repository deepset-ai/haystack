---
title: "Perplexity"
id: integrations-perplexity
description: "Perplexity integration for Haystack"
slug: "/integrations-perplexity"
---


## haystack_integrations.components.embedders.perplexity.document_embedder

### PerplexityDocumentEmbedder

Bases: <code>OpenAIDocumentEmbedder</code>

A component for computing Document embeddings using Perplexity models.

The embedding of each Document is stored in the `embedding` field of the Document.
For supported models, see the
[Perplexity Embeddings API reference](https://docs.perplexity.ai/api-reference/embeddings-post).

Usage example:

```python
from haystack import Document
from haystack_integrations.components.embedders.perplexity import PerplexityDocumentEmbedder

doc = Document(content="I love pizza!")

document_embedder = PerplexityDocumentEmbedder()

result = document_embedder.run([doc])
print(result['documents'][0].embedding)
```

#### SUPPORTED_MODELS

```python
SUPPORTED_MODELS: list[str] = ['pplx-embed-v1-0.6b', 'pplx-embed-v1-4b']
```

A list of models supported by the Perplexity Embeddings API.
See https://docs.perplexity.ai/api-reference/embeddings-post for the current list of model IDs.

#### __init__

```python
__init__(
    *,
    api_key: Secret = Secret.from_env_var("PERPLEXITY_API_KEY"),
    model: str = "pplx-embed-v1-0.6b",
    api_base_url: str | None = "https://api.perplexity.ai/v1",
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

Creates a PerplexityDocumentEmbedder component.

**Parameters:**

- **api_key** (<code>Secret</code>) – The Perplexity API key.
- **model** (<code>str</code>) – The name of the model to use.
- **api_base_url** (<code>str | None</code>) – The Perplexity API base URL.
- **prefix** (<code>str</code>) – A string to add to the beginning of each text.
- **suffix** (<code>str</code>) – A string to add to the end of each text.
- **batch_size** (<code>int</code>) – Number of Documents to encode at once.
- **progress_bar** (<code>bool</code>) – Whether to show a progress bar or not. Can be helpful to disable in production deployments to keep
  the logs clean.
- **meta_fields_to_embed** (<code>list\[str\] | None</code>) – List of meta fields that should be embedded along with the Document text.
- **embedding_separator** (<code>str</code>) – Separator used to concatenate the meta fields to the Document text.
- **timeout** (<code>float | None</code>) – Timeout for Perplexity client calls. If not set, it defaults to either the `OPENAI_TIMEOUT` environment
  variable, or 30 seconds.
- **max_retries** (<code>int | None</code>) – Maximum number of retries to contact Perplexity after an internal error.
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

#### from_dict

```python
from_dict(data: dict[str, Any]) -> PerplexityDocumentEmbedder
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>PerplexityDocumentEmbedder</code> – Deserialized component.

## haystack_integrations.components.embedders.perplexity.text_embedder

### PerplexityTextEmbedder

Bases: <code>OpenAITextEmbedder</code>

A component for embedding strings using Perplexity models.

For supported models, see the
[Perplexity Embeddings API reference](https://docs.perplexity.ai/api-reference/embeddings-post).

Usage example:

```python
from haystack_integrations.components.embedders.perplexity.text_embedder import PerplexityTextEmbedder

text_to_embed = "I love pizza!"
text_embedder = PerplexityTextEmbedder()
print(text_embedder.run(text_to_embed))
```

#### SUPPORTED_MODELS

```python
SUPPORTED_MODELS: list[str] = ['pplx-embed-v1-0.6b', 'pplx-embed-v1-4b']
```

A list of models supported by the Perplexity Embeddings API.
See https://docs.perplexity.ai/api-reference/embeddings-post for the current list of model IDs.

#### __init__

```python
__init__(
    *,
    api_key: Secret = Secret.from_env_var("PERPLEXITY_API_KEY"),
    model: str = "pplx-embed-v1-0.6b",
    api_base_url: str | None = "https://api.perplexity.ai/v1",
    prefix: str = "",
    suffix: str = "",
    timeout: float | None = None,
    max_retries: int | None = None,
    http_client_kwargs: dict[str, Any] | None = None
) -> None
```

Creates a PerplexityTextEmbedder component.

**Parameters:**

- **api_key** (<code>Secret</code>) – The Perplexity API key.
- **model** (<code>str</code>) – The name of the Perplexity embedding model to be used.
- **api_base_url** (<code>str | None</code>) – The Perplexity API base URL.
- **prefix** (<code>str</code>) – A string to add to the beginning of each text.
- **suffix** (<code>str</code>) – A string to add to the end of each text.
- **timeout** (<code>float | None</code>) – Timeout for Perplexity client calls. If not set, it defaults to either the `OPENAI_TIMEOUT` environment
  variable, or 30 seconds.
- **max_retries** (<code>int | None</code>) – Maximum number of retries to contact Perplexity after an internal error.
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

#### from_dict

```python
from_dict(data: dict[str, Any]) -> PerplexityTextEmbedder
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>PerplexityTextEmbedder</code> – Deserialized component.

## haystack_integrations.components.generators.perplexity.chat.chat_generator

### PerplexityChatGenerator

Bases: <code>OpenAIResponsesChatGenerator</code>

Completes chats using Perplexity models.

Powered by the Perplexity Agent API (`POST /v1/agent`, OpenAI Responses-compatible).
See the [Perplexity Agent API quickstart](https://docs.perplexity.ai/docs/agent-api/quickstart)
for details.

It uses the [ChatMessage](https://docs.haystack.deepset.ai/docs/chatmessage) format in input and output.
You can customize generation by passing Perplexity Agent API parameters through `generation_kwargs`.

### Usage example

```python
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.perplexity import PerplexityChatGenerator

messages = [ChatMessage.from_user("What's Natural Language Processing?")]

client = PerplexityChatGenerator()
response = client.run(messages)
print(response)
```

#### SUPPORTED_MODELS

```python
SUPPORTED_MODELS: list[str] = [
    "openai/gpt-5.5",
    "openai/gpt-5.4",
    "openai/gpt-4o",
    "anthropic/claude-sonnet-4-6",
    "xai/grok-4-1",
    "google/gemini-3-flash-preview",
]

```

A non-exhaustive list of Agent API models supported by this component.
See https://docs.perplexity.ai/docs/agent-api/models for the full and current list.

#### __init__

```python
__init__(
    *,
    api_key: Secret = Secret.from_env_var("PERPLEXITY_API_KEY"),
    model: str = "openai/gpt-5.4",
    api_base_url: str | None = "https://api.perplexity.ai/v1",
    streaming_callback: StreamingCallbackT | None = None,
    organization: str | None = None,
    generation_kwargs: dict[str, Any] | None = None,
    tools: ToolsType | list[dict[str, Any]] | None = None,
    tools_strict: bool = False,
    timeout: float | None = None,
    extra_headers: dict[str, Any] | None = None,
    max_retries: int | None = None,
    http_client_kwargs: dict[str, Any] | None = None
) -> None
```

Initialize the PerplexityChatGenerator component.

**Parameters:**

- **api_key** (<code>Secret</code>) – The Perplexity API key.
- **model** (<code>str</code>) – The Perplexity Agent API model to use.
- **api_base_url** (<code>str | None</code>) – The Perplexity API base URL.
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – A callback function called when a new token is received from the stream.
- **organization** (<code>str | None</code>) – Organization ID forwarded to the OpenAI-compatible client.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional parameters sent directly to the Perplexity Agent API.
- **tools** (<code>ToolsType | list\[dict\[str, Any\]\] | None</code>) – A list of Haystack tools, a Toolset, or OpenAI-compatible tool definitions.
- **tools_strict** (<code>bool</code>) – Whether to enable strict schema adherence for Haystack tool calls.
- **timeout** (<code>float | None</code>) – Timeout for Perplexity API calls.
- **extra_headers** (<code>dict\[str, Any\] | None</code>) – Additional HTTP headers to include in requests to the Perplexity API.
- **max_retries** (<code>int | None</code>) – Maximum number of retries to contact Perplexity after an internal error.
- **http_client_kwargs** (<code>dict\[str, Any\] | None</code>) – A dictionary of keyword arguments to configure a custom `httpx.Client` or `httpx.AsyncClient`.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize this component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – The serialized component as a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> PerplexityChatGenerator
```

Deserialize this component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary representation of this component.

**Returns:**

- <code>PerplexityChatGenerator</code> – The deserialized component instance.

## haystack_integrations.components.websearch.perplexity.perplexity_websearch

### PerplexityWebSearch

A component that uses Perplexity to search the web and return results as Haystack Documents.

This component wraps the Perplexity Search API, enabling web search queries that return
structured documents with content and links.

You need a Perplexity API key from [perplexity.ai](https://www.perplexity.ai/).

### Usage example

```python
from haystack_integrations.components.websearch.perplexity import PerplexityWebSearch
from haystack.utils import Secret

websearch = PerplexityWebSearch(
    api_key=Secret.from_env_var("PERPLEXITY_API_KEY"),
    top_k=5,
)
result = websearch.run(query="What is Haystack by deepset?")
documents = result["documents"]
links = result["links"]
```

#### __init__

```python
__init__(
    *,
    api_key: Secret = Secret.from_env_var("PERPLEXITY_API_KEY"),
    top_k: int | None = 10,
    search_params: dict[str, Any] | None = None,
    timeout: float = 30.0
) -> None
```

Initialize the PerplexityWebSearch component.

**Parameters:**

- **api_key** (<code>Secret</code>) – API key for Perplexity. Defaults to the `PERPLEXITY_API_KEY` environment variable.
- **top_k** (<code>int | None</code>) – Maximum number of results to return. Maps to the `max_results` API parameter (1-20).
- **search_params** (<code>dict\[str, Any\] | None</code>) – Additional parameters passed to the Perplexity Search API.
  See the [Perplexity Search API reference](https://docs.perplexity.ai/api-reference/search-post)
  for available options. Supported keys include: `max_tokens_per_page`, `country`,
  `search_recency_filter`, `search_domain_filter`, `search_language_filter`,
  `last_updated_after_filter`, `last_updated_before_filter`,
  `search_after_date_filter`, `search_before_date_filter`.
- **timeout** (<code>float</code>) – Request timeout in seconds.

#### warm_up

```python
warm_up() -> None
```

Initialize the sync and async HTTP clients.

Called automatically on first use. Can be called explicitly to avoid cold-start latency.

#### run

```python
run(
    query: str, search_params: dict[str, Any] | None = None
) -> dict[str, list[Document] | list[str]]
```

Search the web using Perplexity and return results as Documents.

**Parameters:**

- **query** (<code>str</code>) – Search query string.
- **search_params** (<code>dict\[str, Any\] | None</code>) – Optional per-run override of search parameters.
  If provided, fully replaces the init-time `search_params`.

**Returns:**

- <code>dict\[str, list\[Document\] | list\[str\]\]</code> – A dictionary with:
- `documents`: List of Documents containing search result content.
- `links`: List of URLs from the search results.

#### run_async

```python
run_async(
    query: str, search_params: dict[str, Any] | None = None
) -> dict[str, list[Document] | list[str]]
```

Asynchronously search the web using Perplexity and return results as Documents.

**Parameters:**

- **query** (<code>str</code>) – Search query string.
- **search_params** (<code>dict\[str, Any\] | None</code>) – Optional per-run override of search parameters.
  If provided, fully replaces the init-time `search_params`.

**Returns:**

- <code>dict\[str, list\[Document\] | list\[str\]\]</code> – A dictionary with:
- `documents`: List of Documents containing search result content.
- `links`: List of URLs from the search results.
