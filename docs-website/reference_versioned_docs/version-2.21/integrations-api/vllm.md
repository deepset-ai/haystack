---
title: "vLLM"
id: integrations-vllm
description: "vLLM integration for Haystack"
slug: "/integrations-vllm"
---


## haystack_integrations.components.embedders.vllm.document_embedder

### VLLMDocumentEmbedder

A component for computing Document embeddings using models served with [vLLM](https://docs.vllm.ai/).

The embedding of each Document is stored in the `embedding` field of the Document.
It expects a vLLM server to be running and accessible at the `api_base_url` parameter and uses the
OpenAI-compatible Embeddings API exposed by vLLM.

### Starting the vLLM server

Before using this component, start a vLLM server with an embedding model:

```bash
vllm serve google/embeddinggemma-300m
```

For details on server options, see the [vLLM CLI docs](https://docs.vllm.ai/en/stable/cli/serve/).

### Usage example

```python
from haystack import Document
from haystack_integrations.components.embedders.vllm import VLLMDocumentEmbedder

doc = Document(content="I love pizza!")

document_embedder = VLLMDocumentEmbedder(model="google/embeddinggemma-300m")

result = document_embedder.run([doc])
print(result["documents"][0].embedding)
```

### Usage example with vLLM-specific parameters

Pass vLLM-specific parameters via the `extra_parameters` dictionary. They are forwarded as `extra_body`
to the OpenAI-compatible endpoint.

```python
document_embedder = VLLMDocumentEmbedder(
    model="google/embeddinggemma-300m",
    extra_parameters={"truncate_prompt_tokens": 256, "truncation_side": "right"},
)
```

#### __init__

```python
__init__(
    *,
    model: str,
    api_key: Secret | None = Secret.from_env_var("VLLM_API_KEY", strict=False),
    api_base_url: str = "http://localhost:8000/v1",
    prefix: str = "",
    suffix: str = "",
    dimensions: int | None = None,
    batch_size: int = 32,
    progress_bar: bool = True,
    meta_fields_to_embed: list[str] | None = None,
    embedding_separator: str = "\n",
    timeout: float | None = None,
    max_retries: int | None = None,
    http_client_kwargs: dict[str, Any] | None = None,
    raise_on_failure: bool = False,
    extra_parameters: dict[str, Any] | None = None
) -> None
```

Creates an instance of VLLMDocumentEmbedder.

**Parameters:**

- **model** (<code>str</code>) – The name of the model served by vLLM. Check
  [vLLM documentation](https://docs.vllm.ai/en/stable/models/pooling_models) for more information.
- **api_key** (<code>Secret | None</code>) – The vLLM API key. Defaults to the `VLLM_API_KEY` environment variable.
  Only required if the vLLM server was started with `--api-key`.
- **api_base_url** (<code>str</code>) – The base URL of the vLLM server.
- **prefix** (<code>str</code>) – A string to add at the beginning of each text.
- **suffix** (<code>str</code>) – A string to add at the end of each text.
- **dimensions** (<code>int | None</code>) – The number of dimensions of the resulting embedding. Only models trained with
  Matryoshka Representation Learning support this parameter. See
  [vLLM documentation](https://docs.vllm.ai/en/stable/models/pooling_models/embed/#matryoshka-embeddings)
  for more information.
- **batch_size** (<code>int</code>) – Number of documents to encode at once.
- **progress_bar** (<code>bool</code>) – Whether to show a progress bar.
- **meta_fields_to_embed** (<code>list\[str\] | None</code>) – List of meta fields to embed along with the document text.
- **embedding_separator** (<code>str</code>) – Separator used to concatenate the meta fields to the document text.
- **timeout** (<code>float | None</code>) – Timeout in seconds for vLLM client calls. If not set, the OpenAI client default applies.
- **max_retries** (<code>int | None</code>) – Maximum number of retries for failed requests. If not set, the OpenAI client
  default applies.
- **http_client_kwargs** (<code>dict\[str, Any\] | None</code>) – A dictionary of keyword arguments to configure a custom `httpx.Client` or
  `httpx.AsyncClient`. For more information, see the
  [HTTPX documentation](https://www.python-httpx.org/api/#client).
- **raise_on_failure** (<code>bool</code>) – Whether to raise an exception if the embedding request fails. If `False`,
  the component logs the error and continues processing the remaining documents.
- **extra_parameters** (<code>dict\[str, Any\] | None</code>) – Additional parameters forwarded as `extra_body` to the vLLM embeddings
  endpoint. Use this to pass parameters not part of the standard OpenAI Embeddings API, such as
  `truncate_prompt_tokens`, `truncation_side`, etc. See the
  [vLLM Embeddings API docs](https://docs.vllm.ai/en/stable/models/pooling_models/embed/#openai-compatible-embeddings-api).

#### warm_up

```python
warm_up() -> None
```

Create the OpenAI clients.

#### run

```python
run(documents: list[Document]) -> dict[str, list[Document] | dict[str, Any]]
```

Embed a list of Documents.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – Documents to embed.

**Returns:**

- <code>dict\[str, list\[Document\] | dict\[str, Any\]\]</code> – A dictionary with:
- `documents`: The input documents with their `embedding` field populated.
- `meta`: Information about the usage of the model.

#### run_async

```python
run_async(
    documents: list[Document],
) -> dict[str, list[Document] | dict[str, Any]]
```

Asynchronously embed a list of Documents.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – Documents to embed.

**Returns:**

- <code>dict\[str, list\[Document\] | dict\[str, Any\]\]</code> – A dictionary with:
- `documents`: The input documents with their `embedding` field populated.
- `meta`: Information about the usage of the model.

## haystack_integrations.components.embedders.vllm.text_embedder

### VLLMTextEmbedder

A component for embedding strings using models served with [vLLM](https://docs.vllm.ai/).

It expects a vLLM server to be running and accessible at the `api_base_url` parameter and uses the
OpenAI-compatible Embeddings API exposed by vLLM.

### Starting the vLLM server

Before using this component, start a vLLM server with an embedding model:

```bash
vllm serve google/embeddinggemma-300m
```

For details on server options, see the [vLLM CLI docs](https://docs.vllm.ai/en/stable/cli/serve/).

### Usage example

```python
from haystack_integrations.components.embedders.vllm import VLLMTextEmbedder

text_embedder = VLLMTextEmbedder(model="google/embeddinggemma-300m")
print(text_embedder.run("I love pizza!"))
```

### Usage example with vLLM-specific parameters

Pass vLLM-specific parameters via the `extra_parameters` dictionary. They are forwarded as `extra_body`
to the OpenAI-compatible endpoint.

```python
text_embedder = VLLMTextEmbedder(
    model="google/embeddinggemma-300m",
    extra_parameters={"truncate_prompt_tokens": 256, "truncation_side": "right"},
)
```

#### __init__

```python
__init__(
    *,
    model: str,
    api_key: Secret | None = Secret.from_env_var("VLLM_API_KEY", strict=False),
    api_base_url: str = "http://localhost:8000/v1",
    prefix: str = "",
    suffix: str = "",
    dimensions: int | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    http_client_kwargs: dict[str, Any] | None = None,
    extra_parameters: dict[str, Any] | None = None
) -> None
```

Creates an instance of VLLMTextEmbedder.

**Parameters:**

- **model** (<code>str</code>) – The name of the model served by vLLM (e.g., "intfloat/e5-mistral-7b-instruct").
- **api_key** (<code>Secret | None</code>) – The vLLM API key. Defaults to the `VLLM_API_KEY` environment variable.
  Only required if the vLLM server was started with `--api-key`.
- **api_base_url** (<code>str</code>) – The base URL of the vLLM server.
- **prefix** (<code>str</code>) – A string to add at the beginning of each text to embed.
- **suffix** (<code>str</code>) – A string to add at the end of each text to embed.
- **dimensions** (<code>int | None</code>) – The number of dimensions of the resulting embedding. Only models trained with
  Matryoshka Representation Learning support this parameter. See
  [vLLM documentation](https://docs.vllm.ai/en/stable/models/pooling_models/embed/#matryoshka-embeddings)
  for more information.
- **timeout** (<code>float | None</code>) – Timeout in seconds for vLLM client calls. If not set, the OpenAI client default applies.
- **max_retries** (<code>int | None</code>) – Maximum number of retries for failed requests. If not set, the OpenAI client
  default applies.
- **http_client_kwargs** (<code>dict\[str, Any\] | None</code>) – A dictionary of keyword arguments to configure a custom `httpx.Client` or
  `httpx.AsyncClient`. For more information, see the
  [HTTPX documentation](https://www.python-httpx.org/api/#client).
- **extra_parameters** (<code>dict\[str, Any\] | None</code>) – Additional parameters forwarded as `extra_body` to the vLLM embeddings
  endpoint. Use this to pass parameters not part of the standard OpenAI Embeddings API, such as
  `truncate_prompt_tokens`, `truncation_side`, `additional_data`, `use_activation`, etc. See the
  [vLLM Embeddings API docs](https://docs.vllm.ai/en/stable/models/pooling_models/embed/#openai-compatible-embeddings-api).

#### warm_up

```python
warm_up() -> None
```

Create the OpenAI clients.

#### run

```python
run(text: str) -> dict[str, list[float] | dict[str, Any]]
```

Embed a single string.

**Parameters:**

- **text** (<code>str</code>) – Text to embed.

**Returns:**

- <code>dict\[str, list\[float\] | dict\[str, Any\]\]</code> – A dictionary with:
- `embedding`: The embedding of the input text.
- `meta`: Information about the usage of the model.

#### run_async

```python
run_async(text: str) -> dict[str, list[float] | dict[str, Any]]
```

Asynchronously embed a single string.

**Parameters:**

- **text** (<code>str</code>) – Text to embed.

**Returns:**

- <code>dict\[str, list\[float\] | dict\[str, Any\]\]</code> – A dictionary with:
- `embedding`: The embedding of the input text.
- `meta`: Information about the usage of the model.

## haystack_integrations.components.generators.vllm.chat.chat_generator

### VLLMChatGenerator

A component for generating chat completions using models served with [vLLM](https://docs.vllm.ai/).

It expects a vLLM server to be running and accessible at the `api_base_url` parameter.

### Starting the vLLM server

Before using this component, start a vLLM server:

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507
```

For reasoning models, start the server with the appropriate reasoning parser:

```bash
vllm serve Qwen/Qwen3-0.6B --reasoning-parser qwen3
```

For tool calling, the server must be started with `--enable-auto-tool-choice` and `--tool-call-parser`:

```bash
vllm serve Qwen/Qwen3-0.6B --enable-auto-tool-choice --tool-call-parser hermes
```

The available tool call parsers depend on the model. See the
[vLLM tool calling docs](https://docs.vllm.ai/en/stable/features/tool_calling/) for the full list.

For details on server options, see the [vLLM CLI docs](https://docs.vllm.ai/en/stable/cli/serve/).

### Usage example

```python
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.vllm import VLLMChatGenerator

generator = VLLMChatGenerator(
    model="Qwen/Qwen3-0.6B",
    generation_kwargs={"max_tokens": 512, "temperature": 0.7},
)

messages = [ChatMessage.from_user("What's Natural Language Processing?")]
response = generator.run(messages=messages)
print(response["replies"][0].text)
```

### Usage example with vLLM-specific parameters

Pass the vLLM-specific parameters inside the `generation_kwargs`["extra_body"] dictionary.

```python
from haystack_integrations.components.generators.vllm import VLLMChatGenerator

generator = VLLMChatGenerator(
    model="Qwen/Qwen3-0.6B",
    generation_kwargs={
        "max_tokens": 512,
        "extra_body": {
            "top_k": 50,
            "min_tokens": 10,
            "repetition_penalty": 1.1,
        },
    },
)
```

### Usage example with tool calling

To use tool calling, start the vLLM server with `--enable-auto-tool-choice` and `--tool-call-parser`.

```python
from haystack.dataclasses import ChatMessage
from haystack.tools import tool
from haystack_integrations.components.generators.vllm import VLLMChatGenerator

@tool
def weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny"

generator = VLLMChatGenerator(model="Qwen/Qwen3-0.6B", tools=[weather])

messages = [ChatMessage.from_user("What is the weather in Paris?")]
response = generator.run(messages=messages)
print(response["replies"][0].tool_calls)
```

### Usage example with reasoning models

To use reasoning models, start the vLLM server with `--reasoning-parser`.

```python
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.vllm import VLLMChatGenerator

generator = VLLMChatGenerator(model="Qwen/Qwen3-0.6B")

messages = [ChatMessage.from_user("Solve step by step: what is 15 * 37?")]
response = generator.run(messages=messages)
reply = response["replies"][0]
if reply.reasoning:
    print("Reasoning:", reply.reasoning.reasoning_text)
print("Answer:", reply.text)
```

#### __init__

```python
__init__(
    *,
    model: str,
    api_key: Secret | None = Secret.from_env_var("VLLM_API_KEY", strict=False),
    streaming_callback: StreamingCallbackT | None = None,
    api_base_url: str = "http://localhost:8000/v1",
    generation_kwargs: dict[str, Any] | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    tools: ToolsType | None = None,
    http_client_kwargs: dict[str, Any] | None = None
) -> None
```

Creates an instance of VLLMChatGenerator.

**Parameters:**

- **model** (<code>str</code>) – The name of the model served by vLLM (e.g., "Qwen/Qwen3-0.6B").
- **api_key** (<code>Secret | None</code>) – The vLLM API key. Defaults to the `VLLM_API_KEY` environment variable.
  Only required if the vLLM server was started with `--api-key`.
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – A callback function that is called when a new token is received from the stream.
  The callback function accepts
  [StreamingChunk](https://docs.haystack.deepset.ai/docs/data-classes#streamingchunk)
  as an argument.
- **api_base_url** (<code>str</code>) – The base URL of the vLLM server.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional parameters for text generation. These parameters are sent directly to
  the vLLM OpenAI-compatible endpoint. See
  [vLLM documentation](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/)
  for more details.
  Some of the supported parameters:
- `max_tokens`: Maximum number of tokens to generate.
- `temperature`: Sampling temperature.
- `top_p`: Nucleus sampling parameter.
- `n`: Number of completions to generate for each prompt.
- `stop`: One or more sequences after which the model should stop generating tokens.
- `response_format`: A JSON schema or a Pydantic model that enforces the structure of the response.
- `extra_body`: A dictionary of vLLM-specific parameters not part of the standard OpenAI API
  (e.g., `top_k`, `min_tokens`, `repetition_penalty`).
- **timeout** (<code>float | None</code>) – Timeout for vLLM client calls. If not set, it defaults to the default set by the OpenAI client.
- **max_retries** (<code>int | None</code>) – Maximum number of retries to attempt for failed requests. If not set, it defaults to the default
  set by the OpenAI client.
- **tools** (<code>ToolsType | None</code>) – A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
  Each tool should have a unique name. Not all models support tools.
- **http_client_kwargs** (<code>dict\[str, Any\] | None</code>) – A dictionary of keyword arguments to configure a custom `httpx.Client` or `httpx.AsyncClient`.
  For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).

#### warm_up

```python
warm_up() -> None
```

Create the OpenAI clients and warm up tools.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize this component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – The serialized component as a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> VLLMChatGenerator
```

Deserialize this component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary representation of this component.

**Returns:**

- <code>VLLMChatGenerator</code> – The deserialized component instance.

#### run

```python
run(
    messages: list[ChatMessage],
    streaming_callback: StreamingCallbackT | None = None,
    generation_kwargs: dict[str, Any] | None = None,
    *,
    tools: ToolsType | None = None
) -> dict[str, list[ChatMessage]]
```

Run the VLLM chat generator on the given input data.

**Parameters:**

- **messages** (<code>list\[ChatMessage\]</code>) – A list of ChatMessage instances representing the input messages.
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – A callback function that is called when a new token is received from the stream.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments for text generation. These parameters will
  override the parameters passed during component initialization.
  For details on vLLM API parameters, see
  [vLLM documentation](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/).
- **tools** (<code>ToolsType | None</code>) – A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
  If set, it will override the `tools` parameter provided during initialization.

**Returns:**

- <code>dict\[str, list\[ChatMessage\]\]</code> – A dictionary with the following key:
- `replies`: A list containing the generated responses as ChatMessage instances.

#### run_async

```python
run_async(
    messages: list[ChatMessage],
    streaming_callback: StreamingCallbackT | None = None,
    generation_kwargs: dict[str, Any] | None = None,
    *,
    tools: ToolsType | None = None
) -> dict[str, list[ChatMessage]]
```

Run the VLLM chat generator on the given input data asynchronously.

**Parameters:**

- **messages** (<code>list\[ChatMessage\]</code>) – A list of ChatMessage instances representing the input messages.
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – A callback function that is called when a new token is received from the stream.
  Must be a coroutine.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments for text generation. These parameters will
  override the parameters passed during component initialization.
  For details on vLLM API parameters, see
  [vLLM documentation](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/).
- **tools** (<code>ToolsType | None</code>) – A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
  If set, it will override the `tools` parameter provided during initialization.

**Returns:**

- <code>dict\[str, list\[ChatMessage\]\]</code> – A dictionary with the following key:
- `replies`: A list containing the generated responses as ChatMessage instances.

## haystack_integrations.components.rankers.vllm.ranker

### VLLMRanker

Ranks Documents based on their similarity to a query using models served with [vLLM](https://docs.vllm.ai/).

It expects a vLLM server to be running and accessible at the `api_base_url` parameter and uses the
`/rerank` endpoint exposed by vLLM.

### Starting the vLLM server

Before using this component, start a vLLM server with a reranker model:

```bash
vllm serve BAAI/bge-reranker-base
```

For details on server options, see the [vLLM CLI docs](https://docs.vllm.ai/en/stable/cli/serve/).

### Usage example

```python
from haystack import Document
from haystack_integrations.components.rankers.vllm import VLLMRanker

ranker = VLLMRanker(model="BAAI/bge-reranker-base")
docs = [
    Document(content="The capital of Brazil is Brasilia."),
    Document(content="The capital of France is Paris."),
]
result = ranker.run(query="What is the capital of France?", documents=docs)
print(result["documents"][0].content)
```

### Usage example with vLLM-specific parameters

Pass vLLM-specific parameters via the `extra_parameters` dictionary. They are merged into the
request body sent to the `/rerank` endpoint.

```python
ranker = VLLMRanker(
    model="BAAI/bge-reranker-base",
    extra_parameters={"truncate_prompt_tokens": 256},
)
```

#### __init__

```python
__init__(
    *,
    model: str,
    api_key: Secret | None = Secret.from_env_var("VLLM_API_KEY", strict=False),
    api_base_url: str = "http://localhost:8000/v1",
    top_k: int | None = None,
    score_threshold: float | None = None,
    meta_fields_to_embed: list[str] | None = None,
    meta_data_separator: str = "\n",
    http_client_kwargs: dict[str, Any] | None = None,
    extra_parameters: dict[str, Any] | None = None
) -> None
```

Creates an instance of VLLMRanker.

**Parameters:**

- **model** (<code>str</code>) – The name of the reranker model served by vLLM. Check
  [vLLM documentation](https://docs.vllm.ai/en/stable/models/pooling_models/scoring/#supported-models) for
  information on supported models.
- **api_key** (<code>Secret | None</code>) – The vLLM API key. Defaults to the `VLLM_API_KEY` environment variable.
  Only required if the vLLM server was started with `--api-key`.
- **api_base_url** (<code>str</code>) – The base URL of the vLLM server.
- **top_k** (<code>int | None</code>) – The maximum number of Documents to return. If `None`, all documents are returned.
- **score_threshold** (<code>float | None</code>) – If set, documents with a relevance score below this value are dropped.
  Applied after `top_k`, so the output may contain fewer than `top_k` documents.
- **meta_fields_to_embed** (<code>list\[str\] | None</code>) – List of meta fields that should be concatenated with the document
  content before reranking.
- **meta_data_separator** (<code>str</code>) – Separator used to concatenate the meta fields to the document content.
- **http_client_kwargs** (<code>dict\[str, Any\] | None</code>) – A dictionary of keyword arguments to configure a custom `httpx.Client` or
  `httpx.AsyncClient`. For more information, see the
  [HTTPX documentation](https://www.python-httpx.org/api/#client).
- **extra_parameters** (<code>dict\[str, Any\] | None</code>) – Additional parameters merged into the request body sent to the vLLM
  `/rerank` endpoint. Use this to pass parameters not part of the standard rerank API, such as
  `truncate_prompt_tokens`. See the
  [vLLM docs](https://docs.vllm.ai/en/stable/models/pooling_models/scoring/#rerank-api) for more information.

**Raises:**

- <code>ValueError</code> – If `top_k` is not > 0.

#### warm_up

```python
warm_up() -> None
```

Create the httpx clients.

#### run

```python
run(
    query: str,
    documents: list[Document],
    top_k: int | None = None,
    score_threshold: float | None = None,
) -> dict[str, list[Document] | dict[str, Any]]
```

Returns a list of Documents ranked by their similarity to the given query.

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

#### run_async

```python
run_async(
    query: str,
    documents: list[Document],
    top_k: int | None = None,
    score_threshold: float | None = None,
) -> dict[str, list[Document] | dict[str, Any]]
```

Asynchronously returns a list of Documents ranked by their similarity to the given query.

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
