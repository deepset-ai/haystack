---
title: "Nvidia"
id: integrations-nvidia
description: "Nvidia integration for Haystack"
slug: "/integrations-nvidia"
---


## `haystack_integrations.components.embedders.nvidia.document_embedder`

### `NvidiaDocumentEmbedder`

A component for embedding documents using embedding models provided by
[NVIDIA NIMs](https://ai.nvidia.com).

Usage example:

```python
from haystack_integrations.components.embedders.nvidia import NvidiaDocumentEmbedder

doc = Document(content="I love pizza!")

text_embedder = NvidiaDocumentEmbedder(model="nvidia/nv-embedqa-e5-v5", api_url="https://integrate.api.nvidia.com/v1")
text_embedder.warm_up()

result = document_embedder.run([doc])
print(result["documents"][0].embedding)
```

#### `__init__`

```python
__init__(
    model: str | None = None,
    api_key: Secret | None = Secret.from_env_var("NVIDIA_API_KEY"),
    api_url: str = os.getenv("NVIDIA_API_URL", DEFAULT_API_URL),
    prefix: str = "",
    suffix: str = "",
    batch_size: int = 32,
    progress_bar: bool = True,
    meta_fields_to_embed: list[str] | None = None,
    embedding_separator: str = "\n",
    truncate: EmbeddingTruncateMode | str | None = None,
    timeout: float | None = None,
) -> None
```

Create a NvidiaTextEmbedder component.

**Parameters:**

- **model** (<code>str | None</code>) – Embedding model to use.
  If no specific model along with locally hosted API URL is provided,
  the system defaults to the available model found using /models API.
- **api_key** (<code>Secret | None</code>) – API key for the NVIDIA NIM.
- **api_url** (<code>str</code>) – Custom API URL for the NVIDIA NIM.
  Format for API URL is `http://host:port`
- **prefix** (<code>str</code>) – A string to add to the beginning of each text.
- **suffix** (<code>str</code>) – A string to add to the end of each text.
- **batch_size** (<code>int</code>) – Number of Documents to encode at once.
  Cannot be greater than 50.
- **progress_bar** (<code>bool</code>) – Whether to show a progress bar or not.
- **meta_fields_to_embed** (<code>list\[str\] | None</code>) – List of meta fields that should be embedded along with the Document text.
- **embedding_separator** (<code>str</code>) – Separator used to concatenate the meta fields to the Document text.
- **truncate** (<code>EmbeddingTruncateMode | str | None</code>) – Specifies how inputs longer than the maximum token length should be truncated.
  If None the behavior is model-dependent, see the official documentation for more information.
- **timeout** (<code>float | None</code>) – Timeout for request calls, if not set it is inferred from the `NVIDIA_TIMEOUT` environment variable
  or set to 60 by default.

#### `default_model`

```python
default_model() -> None
```

Set default model in local NIM mode.

#### `warm_up`

```python
warm_up() -> None
```

Initializes the component.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `available_models`

```python
available_models: list[Model]
```

Get a list of available models that work with NvidiaDocumentEmbedder.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> NvidiaDocumentEmbedder
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to deserialize from.

**Returns:**

- <code>NvidiaDocumentEmbedder</code> – The deserialized component.

#### `run`

```python
run(documents: list[Document]) -> dict[str, list[Document] | dict[str, Any]]
```

Embed a list of Documents.

The embedding of each Document is stored in the `embedding` field of the Document.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – A list of Documents to embed.

**Returns:**

- <code>dict\[str, list\[Document\] | dict\[str, Any\]\]</code> – A dictionary with the following keys and values:
- `documents` - List of processed Documents with embeddings.
- `meta` - Metadata on usage statistics, etc.

**Raises:**

- <code>TypeError</code> – If the input is not a list of Documents.

## `haystack_integrations.components.embedders.nvidia.text_embedder`

### `NvidiaTextEmbedder`

A component for embedding strings using embedding models provided by
[NVIDIA NIMs](https://ai.nvidia.com).

For models that differentiate between query and document inputs,
this component embeds the input string as a query.

Usage example:

```python
from haystack_integrations.components.embedders.nvidia import NvidiaTextEmbedder

text_to_embed = "I love pizza!"

text_embedder = NvidiaTextEmbedder(model="nvidia/nv-embedqa-e5-v5", api_url="https://integrate.api.nvidia.com/v1")
text_embedder.warm_up()

print(text_embedder.run(text_to_embed))
```

#### `__init__`

```python
__init__(
    model: str | None = None,
    api_key: Secret | None = Secret.from_env_var("NVIDIA_API_KEY"),
    api_url: str = os.getenv("NVIDIA_API_URL", DEFAULT_API_URL),
    prefix: str = "",
    suffix: str = "",
    truncate: EmbeddingTruncateMode | str | None = None,
    timeout: float | None = None,
)
```

Create a NvidiaTextEmbedder component.

**Parameters:**

- **model** (<code>str | None</code>) – Embedding model to use.
  If no specific model along with locally hosted API URL is provided,
  the system defaults to the available model found using /models API.
- **api_key** (<code>Secret | None</code>) – API key for the NVIDIA NIM.
- **api_url** (<code>str</code>) – Custom API URL for the NVIDIA NIM.
  Format for API URL is `http://host:port`
- **prefix** (<code>str</code>) – A string to add to the beginning of each text.
- **suffix** (<code>str</code>) – A string to add to the end of each text.
- **truncate** (<code>EmbeddingTruncateMode | str | None</code>) – Specifies how inputs longer that the maximum token length should be truncated.
  If None the behavior is model-dependent, see the official documentation for more information.
- **timeout** (<code>float | None</code>) – Timeout for request calls, if not set it is inferred from the `NVIDIA_TIMEOUT` environment variable
  or set to 60 by default.

#### `default_model`

```python
default_model()
```

Set default model in local NIM mode.

#### `warm_up`

```python
warm_up()
```

Initializes the component.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `available_models`

```python
available_models: list[Model]
```

Get a list of available models that work with NvidiaTextEmbedder.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> NvidiaTextEmbedder
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to deserialize from.

**Returns:**

- <code>NvidiaTextEmbedder</code> – The deserialized component.

#### `run`

```python
run(text: str) -> dict[str, list[float] | dict[str, Any]]
```

Embed a string.

**Parameters:**

- **text** (<code>str</code>) – The text to embed.

**Returns:**

- <code>dict\[str, list\[float\] | dict\[str, Any\]\]</code> – A dictionary with the following keys and values:
- `embedding` - Embedding of the text.
- `meta` - Metadata on usage statistics, etc.

**Raises:**

- <code>TypeError</code> – If the input is not a string.
- <code>ValueError</code> – If the input string is empty.

## `haystack_integrations.components.embedders.nvidia.truncate`

### `EmbeddingTruncateMode`

Bases: <code>Enum</code>

Specifies how inputs to the NVIDIA embedding components are truncated.
If START, the input will be truncated from the start.
If END, the input will be truncated from the end.
If NONE, an error will be returned (if the input is too long).

#### `from_str`

```python
from_str(string: str) -> EmbeddingTruncateMode
```

Create an truncate mode from a string.

**Parameters:**

- **string** (<code>str</code>) – String to convert.

**Returns:**

- <code>EmbeddingTruncateMode</code> – Truncate mode.

## `haystack_integrations.components.generators.nvidia.chat.chat_generator`

### `NvidiaChatGenerator`

Bases: <code>OpenAIChatGenerator</code>

Enables text generation using NVIDIA generative models.
For supported models, see [NVIDIA Docs](https://build.nvidia.com/models).

Users can pass any text generation parameters valid for the NVIDIA Chat Completion API
directly to this component via the `generation_kwargs` parameter in `__init__` or the `generation_kwargs`
parameter in `run` method.

This component uses the ChatMessage format for structuring both input and output,
ensuring coherent and contextually relevant responses in chat-based text generation scenarios.
Details on the ChatMessage format can be found in the
[Haystack docs](https://docs.haystack.deepset.ai/docs/data-classes#chatmessage)

For more details on the parameters supported by the NVIDIA API, refer to the
[NVIDIA Docs](https://build.nvidia.com/models).

Usage example:

```python
from haystack_integrations.components.generators.nvidia import NvidiaChatGenerator
from haystack.dataclasses import ChatMessage

messages = [ChatMessage.from_user("What's Natural Language Processing?")]

client = NvidiaChatGenerator()
response = client.run(messages)
print(response)
```

#### `__init__`

```python
__init__(
    *,
    api_key: Secret = Secret.from_env_var("NVIDIA_API_KEY"),
    model: str = "meta/llama-3.1-8b-instruct",
    streaming_callback: StreamingCallbackT | None = None,
    api_base_url: str | None = os.getenv("NVIDIA_API_URL", DEFAULT_API_URL),
    generation_kwargs: dict[str, Any] | None = None,
    tools: ToolsType | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    http_client_kwargs: dict[str, Any] | None = None
) -> None
```

Creates an instance of NvidiaChatGenerator.

**Parameters:**

- **api_key** (<code>Secret</code>) – The NVIDIA API key.
- **model** (<code>str</code>) – The name of the NVIDIA chat completion model to use.
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – A callback function that is called when a new token is received from the stream.
  The callback function accepts StreamingChunk as an argument.
- **api_base_url** (<code>str | None</code>) – The NVIDIA API Base url.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Other parameters to use for the model. These parameters are all sent directly to
  the NVIDIA API endpoint. See [NVIDIA API docs](https://docs.nvcf.nvidia.com/ai/generative-models/)
  for more details.
  Some of the supported parameters:
- `max_tokens`: The maximum number of tokens the output text can have.
- `temperature`: What sampling temperature to use. Higher values mean the model will take more risks.
  Try 0.9 for more creative applications and 0 (argmax sampling) for ones with a well-defined answer.
- `top_p`: An alternative to sampling with temperature, called nucleus sampling, where the model
  considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens
  comprising the top 10% probability mass are considered.
- `stream`: Whether to stream back partial progress. If set, tokens will be sent as data-only server-sent
  events as they become available, with the stream terminated by a data: [DONE] message.
- `response_format`: For NVIDIA NIM servers, this parameter has limited support.
  - The basic JSON mode with `{"type": "json_object"}` is supported by compatible models, to produce
    valid JSON output.
    To pass the JSON schema to the model, use the `guided_json` parameter in `extra_body`.
    For example:
  ```python
  generation_kwargs={
      "extra_body": {
          "nvext": {
              "guided_json": {
                  json_schema
          }
      }
  }
  ```
  For more details, see the [NVIDIA NIM documentation](https://docs.nvidia.com/nim/large-language-models/latest/structured-generation.html).
- **tools** (<code>ToolsType | None</code>) – A list of tools or a Toolset for which the model can prepare calls. This parameter can accept either a
  list of `Tool` objects or a `Toolset` instance.
- **timeout** (<code>float | None</code>) – The timeout for the NVIDIA API call.
- **max_retries** (<code>int | None</code>) – Maximum number of retries to contact NVIDIA after an internal error.
  If not set, it defaults to either the `NVIDIA_MAX_RETRIES` environment variable, or set to 5.
- **http_client_kwargs** (<code>dict\[str, Any\] | None</code>) – A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
  For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serialize this component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – The serialized component as a dictionary.

## `haystack_integrations.components.generators.nvidia.generator`

### `NvidiaGenerator`

Generates text using generative models hosted with
[NVIDIA NIM](https://ai.nvidia.com) on the [NVIDIA API Catalog](https://build.nvidia.com/explore/discover).

### Usage example

```python
from haystack_integrations.components.generators.nvidia import NvidiaGenerator

generator = NvidiaGenerator(
    model="meta/llama3-8b-instruct",
    model_arguments={
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": 1024,
    },
)
generator.warm_up()

result = generator.run(prompt="What is the answer?")
print(result["replies"])
print(result["meta"])
print(result["usage"])
```

You need an NVIDIA API key for this component to work.

#### `__init__`

```python
__init__(
    model: str | None = None,
    api_url: str = os.getenv("NVIDIA_API_URL", DEFAULT_API_URL),
    api_key: Secret | None = Secret.from_env_var("NVIDIA_API_KEY"),
    model_arguments: dict[str, Any] | None = None,
    timeout: float | None = None,
) -> None
```

Create a NvidiaGenerator component.

**Parameters:**

- **model** (<code>str | None</code>) – Name of the model to use for text generation.
  See the [NVIDIA NIMs](https://ai.nvidia.com)
  for more information on the supported models.
  `Note`: If no specific model along with locally hosted API URL is provided,
  the system defaults to the available model found using /models API.
  Check supported models at [NVIDIA NIM](https://ai.nvidia.com).
- **api_key** (<code>Secret | None</code>) – API key for the NVIDIA NIM. Set it as the `NVIDIA_API_KEY` environment
  variable or pass it here.
- **api_url** (<code>str</code>) – Custom API URL for the NVIDIA NIM.
- **model_arguments** (<code>dict\[str, Any\] | None</code>) – Additional arguments to pass to the model provider. These arguments are
  specific to a model.
  Search your model in the [NVIDIA NIM](https://ai.nvidia.com)
  to find the arguments it accepts.
- **timeout** (<code>float | None</code>) – Timeout for request calls, if not set it is inferred from the `NVIDIA_TIMEOUT` environment variable
  or set to 60 by default.

#### `default_model`

```python
default_model() -> None
```

Set default model in local NIM mode.

#### `warm_up`

```python
warm_up() -> None
```

Initializes the component.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `available_models`

```python
available_models: list[Model]
```

Get a list of available models that work with ChatNVIDIA.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> NvidiaGenerator
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>NvidiaGenerator</code> – Deserialized component.

#### `run`

```python
run(prompt: str) -> dict[str, list[str] | list[dict[str, Any]]]
```

Queries the model with the provided prompt.

**Parameters:**

- **prompt** (<code>str</code>) – Text to be sent to the generative model.

**Returns:**

- <code>dict\[str, list\[str\] | list\[dict\[str, Any\]\]\]</code> – A dictionary with the following keys:
- `replies` - Replies generated by the model.
- `meta` - Metadata for each reply.

## `haystack_integrations.components.rankers.nvidia.ranker`

### `NvidiaRanker`

A component for ranking documents using ranking models provided by
[NVIDIA NIMs](https://ai.nvidia.com).

Usage example:

```python
from haystack_integrations.components.rankers.nvidia import NvidiaRanker
from haystack import Document
from haystack.utils import Secret

ranker = NvidiaRanker(
    model="nvidia/nv-rerankqa-mistral-4b-v3",
    api_key=Secret.from_env_var("NVIDIA_API_KEY"),
)
ranker.warm_up()

query = "What is the capital of Germany?"
documents = [
    Document(content="Berlin is the capital of Germany."),
    Document(content="The capital of Germany is Berlin."),
    Document(content="Germany's capital is Berlin."),
]

result = ranker.run(query, documents, top_k=2)
print(result["documents"])
```

#### `__init__`

```python
__init__(
    model: str | None = None,
    truncate: RankerTruncateMode | str | None = None,
    api_url: str = os.getenv("NVIDIA_API_URL", DEFAULT_API_URL),
    api_key: Secret | None = Secret.from_env_var("NVIDIA_API_KEY"),
    top_k: int = 5,
    query_prefix: str = "",
    document_prefix: str = "",
    meta_fields_to_embed: list[str] | None = None,
    embedding_separator: str = "\n",
    timeout: float | None = None,
) -> None
```

Create a NvidiaRanker component.

**Parameters:**

- **model** (<code>str | None</code>) – Ranking model to use.
- **truncate** (<code>RankerTruncateMode | str | None</code>) – Truncation strategy to use. Can be "NONE", "END", or RankerTruncateMode. Defaults to NIM's default.
- **api_key** (<code>Secret | None</code>) – API key for the NVIDIA NIM.
- **api_url** (<code>str</code>) – Custom API URL for the NVIDIA NIM.
- **top_k** (<code>int</code>) – Number of documents to return.
- **query_prefix** (<code>str</code>) – A string to add at the beginning of the query text before ranking.
  Use it to prepend the text with an instruction, as required by reranking models like `bge`.
- **document_prefix** (<code>str</code>) – A string to add at the beginning of each document before ranking. You can use it to prepend the document
  with an instruction, as required by embedding models like `bge`.
- **meta_fields_to_embed** (<code>list\[str\] | None</code>) – List of metadata fields to embed with the document.
- **embedding_separator** (<code>str</code>) – Separator to concatenate metadata fields to the document.
- **timeout** (<code>float | None</code>) – Timeout for request calls, if not set it is inferred from the `NVIDIA_TIMEOUT` environment variable
  or set to 60 by default.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serialize the ranker to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary containing the ranker's attributes.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> NvidiaRanker
```

Deserialize the ranker from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – A dictionary containing the ranker's attributes.

**Returns:**

- <code>NvidiaRanker</code> – The deserialized ranker.

#### `warm_up`

```python
warm_up() -> None
```

Initialize the ranker.

**Raises:**

- <code>ValueError</code> – If the API key is required for hosted NVIDIA NIMs.

#### `run`

```python
run(
    query: str, documents: list[Document], top_k: int | None = None
) -> dict[str, list[Document]]
```

Rank a list of documents based on a given query.

**Parameters:**

- **query** (<code>str</code>) – The query to rank the documents against.
- **documents** (<code>list\[Document\]</code>) – The list of documents to rank.
- **top_k** (<code>int | None</code>) – The number of documents to return.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary containing the ranked documents.

**Raises:**

- <code>TypeError</code> – If the arguments are of the wrong type.

## `haystack_integrations.components.rankers.nvidia.truncate`

### `RankerTruncateMode`

Bases: <code>str</code>, <code>Enum</code>

Specifies how inputs to the NVIDIA ranker components are truncated.
If NONE, the input will not be truncated and an error returned instead.
If END, the input will be truncated from the end.

#### `from_str`

```python
from_str(string: str) -> RankerTruncateMode
```

Create an truncate mode from a string.

**Parameters:**

- **string** (<code>str</code>) – String to convert.

**Returns:**

- <code>RankerTruncateMode</code> – Truncate mode.
