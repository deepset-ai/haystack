---
title: "Otari"
id: integrations-otari
description: "Otari integration for Haystack"
slug: "/integrations-otari"
---


## haystack_integrations.components.embedders.otari.document_embedder

### OtariDocumentEmbedder

Bases: <code>OpenAIDocumentEmbedder</code>

Computes Document embeddings using models served through an [Otari](https://github.com/mozilla-ai/otari) gateway.

The embedding of each Document is stored in the `embedding` field of the Document.

By default, this component calls the otari.ai gateway. For an account in otari.ai's EU region, whose API keys
start with `otk_v1_eu_`, set `api_base_url` to `"https://eu.api.otari.ai/api/v1"`. To call a gateway you run
yourself, set `api_base_url`, for example to `"http://localhost:8000/api/v1"`.

Name models with a selector the gateway accepts, such as `"openai:text-embedding-3-small"`.

When the gateway prices a request, `meta["usage"]` contains its cost in US dollars as `cost_usd`, and
`pricing_source`. If the Documents are sent in more than one batch, `cost_usd` is the sum over all batches.

### Usage example

```python
from haystack import Document
from haystack_integrations.components.embedders.otari import OtariDocumentEmbedder

# Reads the API key from the OTARI_API_KEY environment variable
document_embedder = OtariDocumentEmbedder()
result = document_embedder.run([Document(content="I love pizza!")])
print(result["documents"][0].embedding)
# >> [0.017020374536514282, -0.023255806416273117, ...]
```

#### __init__

```python
__init__(
    *,
    api_key: Secret = Secret.from_env_var("OTARI_API_KEY"),
    model: str = "openai:text-embedding-3-small",
    dimensions: int | None = None,
    api_base_url: str = "https://api.otari.ai/api/v1",
    prefix: str = "",
    suffix: str = "",
    batch_size: int = 32,
    progress_bar: bool = True,
    meta_fields_to_embed: list[str] | None = None,
    embedding_separator: str = "\n",
    timeout: float | None = None,
    max_retries: int | None = None,
    http_client_kwargs: dict[str, Any] | None = None,
    raise_on_failure: bool = False
) -> None
```

Creates an OtariDocumentEmbedder component.

**Parameters:**

- **api_key** (<code>Secret</code>) – The Otari API key: a key issued by your gateway, or an otari.ai API key.
  Defaults to the `OTARI_API_KEY` environment variable.
- **model** (<code>str</code>) – The embedding model selector, for example `"openai:text-embedding-3-small"`.
  The gateway must have the provider configured.
- **dimensions** (<code>int | None</code>) – The number of dimensions of the resulting embeddings. Only some models support this parameter.
- **api_base_url** (<code>str</code>) – The OpenAI-compatible API root of the Otari gateway, including the `/api/v1` path.
  Defaults to the otari.ai gateway. For an account in otari.ai's EU region, use
  `"https://eu.api.otari.ai/api/v1"`.
- **prefix** (<code>str</code>) – A string to add to the beginning of each text.
- **suffix** (<code>str</code>) – A string to add to the end of each text.
- **batch_size** (<code>int</code>) – Number of Documents to encode at once.
- **progress_bar** (<code>bool</code>) – Whether to show a progress bar or not. Can be helpful to disable in production deployments to keep
  the logs clean.
- **meta_fields_to_embed** (<code>list\[str\] | None</code>) – List of meta fields that should be embedded along with the Document text.
- **embedding_separator** (<code>str</code>) – Separator used to concatenate the meta fields to the Document text.
- **timeout** (<code>float | None</code>) – Timeout for the API call. If not set, it defaults to either the `OPENAI_TIMEOUT` environment
  variable, or 30 seconds.
- **max_retries** (<code>int | None</code>) – Maximum number of retries to contact Otari after an internal error.
  If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
- **http_client_kwargs** (<code>dict\[str, Any\] | None</code>) – A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
  For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).
- **raise_on_failure** (<code>bool</code>) – Whether to raise an exception if the embedding request fails. If `False`, the component logs the error
  and continues processing the remaining documents. If `True`, it raises an exception on failure.

#### warm_up

```python
warm_up() -> None
```

Initialize the synchronous OpenAI client.

Logs a warning if the API key belongs to otari.ai's EU region but `api_base_url` is left at its default.

#### warm_up_async

```python
warm_up_async() -> None
```

Initialize the asynchronous OpenAI client.

Logs a warning if the API key belongs to otari.ai's EU region but `api_base_url` is left at its default.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

## haystack_integrations.components.embedders.otari.text_embedder

### OtariTextEmbedder

Bases: <code>OpenAITextEmbedder</code>

Embeds strings using embedding models served through an [Otari](https://github.com/mozilla-ai/otari) gateway.

By default, this component calls the otari.ai gateway. For an account in otari.ai's EU region, whose API keys
start with `otk_v1_eu_`, set `api_base_url` to `"https://eu.api.otari.ai/api/v1"`. To call a gateway you run
yourself, set `api_base_url`, for example to `"http://localhost:8000/api/v1"`.

Name models with a selector the gateway accepts, such as `"openai:text-embedding-3-small"`.

### Usage example

```python
from haystack_integrations.components.embedders.otari import OtariTextEmbedder

# Reads the API key from the OTARI_API_KEY environment variable
text_embedder = OtariTextEmbedder()
print(text_embedder.run("I love pizza!"))
# >> {'embedding': [0.017020374536514282, -0.023255806416273117, ...],
# >>  'meta': {'model': 'text-embedding-3-small', 'usage': {'prompt_tokens': 4, 'total_tokens': 4}}}
```

#### __init__

```python
__init__(
    *,
    api_key: Secret = Secret.from_env_var("OTARI_API_KEY"),
    model: str = "openai:text-embedding-3-small",
    dimensions: int | None = None,
    api_base_url: str = "https://api.otari.ai/api/v1",
    prefix: str = "",
    suffix: str = "",
    timeout: float | None = None,
    max_retries: int | None = None,
    http_client_kwargs: dict[str, Any] | None = None
) -> None
```

Creates an OtariTextEmbedder component.

**Parameters:**

- **api_key** (<code>Secret</code>) – The Otari API key: a key issued by your gateway, or an otari.ai API key.
  Defaults to the `OTARI_API_KEY` environment variable.
- **model** (<code>str</code>) – The embedding model selector, for example `"openai:text-embedding-3-small"`.
  The gateway must have the provider configured.
- **dimensions** (<code>int | None</code>) – The number of dimensions of the resulting embeddings. Only some models support this parameter.
- **api_base_url** (<code>str</code>) – The OpenAI-compatible API root of the Otari gateway, including the `/api/v1` path.
  Defaults to the otari.ai gateway. For an account in otari.ai's EU region, use
  `"https://eu.api.otari.ai/api/v1"`.
- **prefix** (<code>str</code>) – A string to add to the beginning of each text.
- **suffix** (<code>str</code>) – A string to add to the end of each text.
- **timeout** (<code>float | None</code>) – Timeout for the API call. If not set, it defaults to either the `OPENAI_TIMEOUT` environment
  variable, or 30 seconds.
- **max_retries** (<code>int | None</code>) – Maximum number of retries to contact Otari after an internal error.
  If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
- **http_client_kwargs** (<code>dict\[str, Any\] | None</code>) – A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
  For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).

#### warm_up

```python
warm_up() -> None
```

Initialize the synchronous OpenAI client.

Logs a warning if the API key belongs to otari.ai's EU region but `api_base_url` is left at its default.

#### warm_up_async

```python
warm_up_async() -> None
```

Initialize the asynchronous OpenAI client.

Logs a warning if the API key belongs to otari.ai's EU region but `api_base_url` is left at its default.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

## haystack_integrations.components.generators.otari.chat.chat_generator

### OtariChatGenerator

Bases: <code>OpenAIChatGenerator</code>

Completes chats using models served through an [Otari](https://github.com/mozilla-ai/otari) gateway.

Otari is an OpenAI-compatible LLM gateway that routes requests to many providers and applies API keys, budgets,
and usage tracking. By default, this component calls the otari.ai gateway. For an account in otari.ai's EU
region, whose API keys start with `otk_v1_eu_`, set `api_base_url` to `"https://eu.api.otari.ai/api/v1"`.
To call a gateway you run yourself, set `api_base_url`, for example to `"http://localhost:8000/api/v1"`.

Name models with a selector the gateway accepts, such as `provider:model` (`"openai:gpt-5-mini"`,
`"anthropic:claude-sonnet-4-6"`), a model alias, or a routing policy name. For details, see the
[Otari model docs](https://github.com/mozilla-ai/otari/blob/main/docs/models.md).

You can pass any parameter of the Otari chat completions endpoint through `generation_kwargs`, at initialization
or in the `run` method. Request headers, such as Otari's `Idempotency-Key`, go into
`generation_kwargs["extra_headers"]`.

When the gateway prices a request, `meta["usage"]` of the reply contains its cost in US dollars as `cost_usd`,
and `pricing_source`. When streaming, pass `generation_kwargs={"stream_options": {"include_usage": True}}` to
receive usage.

When the model reasons, for example with `generation_kwargs={"reasoning_effort": "low"}`, the reasoning is in the
`reasoning` of the reply and, when streaming, of the streamed chunks. If the gateway adds provider data that the
next request needs, such as Anthropic's thinking signature, the component sends the reasoning back with the reply
in later requests. Anthropic models need this to keep reasoning across tool calls.

### Usage example

```python
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.otari import OtariChatGenerator

# Reads the API key from the OTARI_API_KEY environment variable
generator = OtariChatGenerator()
response = generator.run([ChatMessage.from_user("What's Natural Language Processing? Be brief.")])
print(response["replies"][0].text)
# >> Natural Language Processing (NLP) is a field of artificial intelligence that enables computers to ...
print(response["replies"][0].meta["usage"]["cost_usd"])
# >> 0.000279
```

#### __init__

```python
__init__(
    *,
    api_key: Secret = Secret.from_env_var("OTARI_API_KEY"),
    model: str = "openai:gpt-5-mini",
    streaming_callback: StreamingCallbackT | None = None,
    api_base_url: str = "https://api.otari.ai/api/v1",
    generation_kwargs: dict[str, Any] | None = None,
    tools: ToolsType | None = None,
    tools_strict: bool = False,
    timeout: float | None = None,
    max_retries: int | None = None,
    http_client_kwargs: dict[str, Any] | None = None
) -> None
```

Creates an instance of OtariChatGenerator.

**Parameters:**

- **api_key** (<code>Secret</code>) – The Otari API key: a key issued by your gateway, or an otari.ai API key.
  Defaults to the `OTARI_API_KEY` environment variable.
- **model** (<code>str</code>) – The model selector, for example `"openai:gpt-5-mini"`. The gateway must have the provider configured.
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – A callback function that is called when a new token is received from the stream.
  The callback function accepts StreamingChunk as an argument.
- **api_base_url** (<code>str</code>) – The OpenAI-compatible API root of the Otari gateway, including the `/api/v1` path.
  Defaults to the otari.ai gateway. For an account in otari.ai's EU region, use
  `"https://eu.api.otari.ai/api/v1"`.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Other parameters to use for the model. These parameters are sent directly to the Otari endpoint.
  Some of the supported parameters:
- `max_tokens`: The maximum number of tokens the output text can have.
- `temperature`: What sampling temperature to use. Higher values mean the model will take more risks.
  Try 0.9 for more creative applications and 0 (argmax sampling) for ones with a well-defined answer.
- `top_p`: An alternative to sampling with temperature, called nucleus sampling, where the model
  considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens
  comprising the top 10% probability mass are considered.
- `stream_options`: Pass `{"include_usage": True}` to receive usage, including cost, when streaming.
- `extra_headers`: Headers to send with the request, for example Otari's `Idempotency-Key`.
- `response_format`: A JSON schema or a Pydantic model that enforces the structure of the model's response.
  If provided, the output will always be validated against this
  format (unless the model returns a tool call).
  For details, see the [OpenAI Structured Outputs documentation](https://platform.openai.com/docs/guides/structured-outputs).
  Notes:
  - For structured outputs with streaming,
    the `response_format` must be a JSON schema and not a Pydantic model.
- **tools** (<code>ToolsType | None</code>) – A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
  Each tool should have a unique name.
- **tools_strict** (<code>bool</code>) – Whether to enable strict schema adherence for tool calls. If set to `True`, the model will follow exactly
  the schema provided in the `parameters` field of the tool definition, but this may increase latency.
- **timeout** (<code>float | None</code>) – The timeout for the Otari API call.
- **max_retries** (<code>int | None</code>) – Maximum number of retries to contact Otari after an internal error.
  If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
- **http_client_kwargs** (<code>dict\[str, Any\] | None</code>) – A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
  For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).

#### warm_up

```python
warm_up() -> None
```

Warm up the tools and initialize the synchronous OpenAI client.

Logs a warning if the API key belongs to otari.ai's EU region but `api_base_url` is left at its default.

#### warm_up_async

```python
warm_up_async() -> None
```

Warm up the tools and initialize the asynchronous OpenAI client.

Logs a warning if the API key belongs to otari.ai's EU region but `api_base_url` is left at its default.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize this component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – The serialized component as a dictionary.

#### run

```python
run(
    messages: list[ChatMessage] | str,
    streaming_callback: StreamingCallbackT | None = None,
    generation_kwargs: dict[str, Any] | None = None,
    *,
    tools: ToolsType | None = None,
    tools_strict: bool | None = None
) -> dict[str, list[ChatMessage]]
```

Invokes chat completion based on the provided messages and generation parameters.

**Parameters:**

- **messages** (<code>list\[ChatMessage\] | str</code>) – A list of ChatMessage instances representing the input messages. If a string is provided, it is converted
  to a list containing a ChatMessage with user role.
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – A callback function that is called when a new token is received from the stream.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments for text generation. These are merged per key with the
  `generation_kwargs` passed at initialization: keys provided here take precedence, keys set
  only at initialization are kept.
- **tools** (<code>ToolsType | None</code>) – A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
  If set, it will override the `tools` parameter provided during initialization.
- **tools_strict** (<code>bool | None</code>) – Whether to enable strict schema adherence for tool calls. If set to `True`, the model will follow exactly
  the schema provided in the `parameters` field of the tool definition, but this may increase latency.
  If set, it will override the `tools_strict` parameter set during component initialization.

**Returns:**

- <code>dict\[str, list\[ChatMessage\]\]</code> – A dictionary with the following key:
- `replies`: A list containing the generated responses as ChatMessage instances, including the
  model's reasoning when it returns any.

#### run_async

```python
run_async(
    messages: list[ChatMessage] | str,
    streaming_callback: StreamingCallbackT | None = None,
    generation_kwargs: dict[str, Any] | None = None,
    *,
    tools: ToolsType | None = None,
    tools_strict: bool | None = None
) -> dict[str, list[ChatMessage]]
```

Asynchronously invokes chat completion based on the provided messages and generation parameters.

This is the asynchronous version of the `run` method. It has the same parameters and return values
but can be used with `await` in async code.

**Parameters:**

- **messages** (<code>list\[ChatMessage\] | str</code>) – A list of ChatMessage instances representing the input messages. If a string is provided, it is converted
  to a list containing a ChatMessage with user role.
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – A callback function that is called when a new token is received from the stream. Async callbacks are
  preferred; a sync callback is accepted but will run synchronously on the event loop and may block it.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments for text generation. These are merged per key with the
  `generation_kwargs` passed at initialization: keys provided here take precedence, keys set
  only at initialization are kept.
- **tools** (<code>ToolsType | None</code>) – A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
  If set, it will override the `tools` parameter provided during initialization.
- **tools_strict** (<code>bool | None</code>) – Whether to enable strict schema adherence for tool calls. If set to `True`, the model will follow exactly
  the schema provided in the `parameters` field of the tool definition, but this may increase latency.
  If set, it will override the `tools_strict` parameter set during component initialization.

**Returns:**

- <code>dict\[str, list\[ChatMessage\]\]</code> – A dictionary with the following key:
- `replies`: A list containing the generated responses as ChatMessage instances, including the
  model's reasoning when it returns any.

## haystack_integrations.components.rankers.otari.ranker

### OtariRanker

Ranks Documents by relevance to a query using an [Otari](https://github.com/mozilla-ai/otari) gateway.

It calls the `/rerank` endpoint of the gateway. By default, this component calls the otari.ai gateway. For an
account in otari.ai's EU region, whose API keys start with `otk_v1_eu_`, set `api_base_url` to
`"https://eu.api.otari.ai/api/v1"`. To call a gateway you run yourself, set `api_base_url`, for example to
`"http://localhost:8000/api/v1"`.

Name models with a selector the gateway accepts, such as `"cohere:rerank-v3.5"`.

### Usage example

```python
from haystack import Document
from haystack_integrations.components.rankers.otari import OtariRanker

# Reads the API key from the OTARI_API_KEY environment variable
ranker = OtariRanker()
docs = [
    Document(content="The capital of Brazil is Brasilia."),
    Document(content="The capital of France is Paris."),
]
result = ranker.run(query="What is the capital of France?", documents=docs)
print(result["documents"][0].content)
# >> The capital of France is Paris.
```

#### __init__

```python
__init__(
    *,
    model: str = "cohere:rerank-v3.5",
    api_key: Secret = Secret.from_env_var("OTARI_API_KEY"),
    api_base_url: str = "https://api.otari.ai/api/v1",
    top_k: int | None = None,
    score_threshold: float | None = None,
    meta_fields_to_embed: list[str] | None = None,
    meta_data_separator: str = "\n",
    max_tokens_per_doc: int | None = None,
    timeout: float = 30.0,
    max_retries: int = 3
) -> None
```

Creates an instance of OtariRanker.

**Parameters:**

- **model** (<code>str</code>) – The rerank model selector, for example `"cohere:rerank-v3.5"`.
  The gateway must have the provider configured.
- **api_key** (<code>Secret</code>) – The Otari API key: a key issued by your gateway, or an otari.ai API key.
  Defaults to the `OTARI_API_KEY` environment variable.
- **api_base_url** (<code>str</code>) – The API root of the Otari gateway, including the `/api/v1` path.
  Defaults to the otari.ai gateway. For an account in otari.ai's EU region, use
  `"https://eu.api.otari.ai/api/v1"`.
- **top_k** (<code>int | None</code>) – The maximum number of Documents to return. If `None`, all documents are returned.
- **score_threshold** (<code>float | None</code>) – If set, documents with a relevance score below this value are dropped.
  Applied after `top_k`, so the output may contain fewer than `top_k` documents.
- **meta_fields_to_embed** (<code>list\[str\] | None</code>) – List of meta fields that should be concatenated with the document
  content before reranking.
- **meta_data_separator** (<code>str</code>) – Separator used to concatenate the meta fields to the document content.
- **max_tokens_per_doc** (<code>int | None</code>) – The maximum number of tokens of each document that the model considers.
  Longer documents are truncated. If `None`, the provider's default applies.
- **timeout** (<code>float</code>) – The timeout for each request to the gateway, in seconds.
- **max_retries** (<code>int</code>) – The maximum number of retries after a failed request. Requests are retried on timeouts,
  connection errors, and the HTTP status codes 408, 418, 429, and 503.

**Raises:**

- <code>ValueError</code> – If `top_k` is not > 0.

#### run

```python
run(
    query: str,
    documents: list[Document],
    top_k: int | None = None,
    score_threshold: float | None = None,
) -> dict[str, list[Document] | dict[str, Any]]
```

Returns a list of Documents ranked by their relevance to the given query.

**Parameters:**

- **query** (<code>str</code>) – Query string.
- **documents** (<code>list\[Document\]</code>) – List of Documents to rank.
- **top_k** (<code>int | None</code>) – The maximum number of Documents to return. Overrides the value set at initialization.
- **score_threshold** (<code>float | None</code>) – Minimum relevance score required for a document to be returned. Overrides
  the value set at initialization.

**Returns:**

- <code>dict\[str, list\[Document\] | dict\[str, Any\]\]</code> – A dictionary with:
- `documents`: Documents sorted from most to least relevant.
- `meta`: Information about the model and usage.

**Raises:**

- <code>ValueError</code> – If `top_k` is not > 0.
- <code>RuntimeError</code> – If the gateway returns an error.
- <code>httpx.RequestError</code> – If the request still fails to reach the gateway after all retries.

#### run_async

```python
run_async(
    query: str,
    documents: list[Document],
    top_k: int | None = None,
    score_threshold: float | None = None,
) -> dict[str, list[Document] | dict[str, Any]]
```

Asynchronously returns a list of Documents ranked by their relevance to the given query.

**Parameters:**

- **query** (<code>str</code>) – Query string.
- **documents** (<code>list\[Document\]</code>) – List of Documents to rank.
- **top_k** (<code>int | None</code>) – The maximum number of Documents to return. Overrides the value set at initialization.
- **score_threshold** (<code>float | None</code>) – Minimum relevance score required for a document to be returned. Overrides
  the value set at initialization.

**Returns:**

- <code>dict\[str, list\[Document\] | dict\[str, Any\]\]</code> – A dictionary with:
- `documents`: Documents sorted from most to least relevant.
- `meta`: Information about the model and usage.

**Raises:**

- <code>ValueError</code> – If `top_k` is not > 0.
- <code>RuntimeError</code> – If the gateway returns an error.
- <code>httpx.RequestError</code> – If the request still fails to reach the gateway after all retries.
