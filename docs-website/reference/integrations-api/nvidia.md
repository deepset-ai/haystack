---
title: "Nvidia"
id: integrations-nvidia
description: "Nvidia integration for Haystack"
slug: "/integrations-nvidia"
---

<a id="haystack_integrations.components.embedders.nvidia.document_embedder"></a>

## Module haystack\_integrations.components.embedders.nvidia.document\_embedder

<a id="haystack_integrations.components.embedders.nvidia.document_embedder.NvidiaDocumentEmbedder"></a>

### NvidiaDocumentEmbedder

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

<a id="haystack_integrations.components.embedders.nvidia.document_embedder.NvidiaDocumentEmbedder.__init__"></a>

#### NvidiaDocumentEmbedder.\_\_init\_\_

```python
def __init__(model: str | None = None,
             api_key: Secret | None = Secret.from_env_var("NVIDIA_API_KEY"),
             api_url: str = os.getenv("NVIDIA_API_URL", DEFAULT_API_URL),
             prefix: str = "",
             suffix: str = "",
             batch_size: int = 32,
             progress_bar: bool = True,
             meta_fields_to_embed: list[str] | None = None,
             embedding_separator: str = "\n",
             truncate: EmbeddingTruncateMode | str | None = None,
             timeout: float | None = None) -> None
```

Create a NvidiaTextEmbedder component.

**Arguments**:

- `model`: Embedding model to use.
If no specific model along with locally hosted API URL is provided,
the system defaults to the available model found using /models API.
- `api_key`: API key for the NVIDIA NIM.
- `api_url`: Custom API URL for the NVIDIA NIM.
Format for API URL is `http://host:port`
- `prefix`: A string to add to the beginning of each text.
- `suffix`: A string to add to the end of each text.
- `batch_size`: Number of Documents to encode at once.
Cannot be greater than 50.
- `progress_bar`: Whether to show a progress bar or not.
- `meta_fields_to_embed`: List of meta fields that should be embedded along with the Document text.
- `embedding_separator`: Separator used to concatenate the meta fields to the Document text.
- `truncate`: Specifies how inputs longer than the maximum token length should be truncated.
If None the behavior is model-dependent, see the official documentation for more information.
- `timeout`: Timeout for request calls, if not set it is inferred from the `NVIDIA_TIMEOUT` environment variable
or set to 60 by default.

<a id="haystack_integrations.components.embedders.nvidia.document_embedder.NvidiaDocumentEmbedder.default_model"></a>

#### NvidiaDocumentEmbedder.default\_model

```python
def default_model() -> None
```

Set default model in local NIM mode.

<a id="haystack_integrations.components.embedders.nvidia.document_embedder.NvidiaDocumentEmbedder.warm_up"></a>

#### NvidiaDocumentEmbedder.warm\_up

```python
def warm_up() -> None
```

Initializes the component.

<a id="haystack_integrations.components.embedders.nvidia.document_embedder.NvidiaDocumentEmbedder.to_dict"></a>

#### NvidiaDocumentEmbedder.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.embedders.nvidia.document_embedder.NvidiaDocumentEmbedder.available_models"></a>

#### NvidiaDocumentEmbedder.available\_models

```python
@property
def available_models() -> list[Model]
```

Get a list of available models that work with NvidiaDocumentEmbedder.

<a id="haystack_integrations.components.embedders.nvidia.document_embedder.NvidiaDocumentEmbedder.from_dict"></a>

#### NvidiaDocumentEmbedder.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "NvidiaDocumentEmbedder"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: The dictionary to deserialize from.

**Returns**:

The deserialized component.

<a id="haystack_integrations.components.embedders.nvidia.document_embedder.NvidiaDocumentEmbedder.run"></a>

#### NvidiaDocumentEmbedder.run

```python
@component.output_types(documents=list[Document], meta=dict[str, Any])
def run(documents: list[Document]
        ) -> dict[str, list[Document] | dict[str, Any]]
```

Embed a list of Documents.

The embedding of each Document is stored in the `embedding` field of the Document.

**Arguments**:

- `documents`: A list of Documents to embed.

**Raises**:

- `TypeError`: If the input is not a list of Documents.

**Returns**:

A dictionary with the following keys and values:
- `documents` - List of processed Documents with embeddings.
- `meta` - Metadata on usage statistics, etc.

<a id="haystack_integrations.components.embedders.nvidia.text_embedder"></a>

## Module haystack\_integrations.components.embedders.nvidia.text\_embedder

<a id="haystack_integrations.components.embedders.nvidia.text_embedder.NvidiaTextEmbedder"></a>

### NvidiaTextEmbedder

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

<a id="haystack_integrations.components.embedders.nvidia.text_embedder.NvidiaTextEmbedder.__init__"></a>

#### NvidiaTextEmbedder.\_\_init\_\_

```python
def __init__(model: str | None = None,
             api_key: Secret | None = Secret.from_env_var("NVIDIA_API_KEY"),
             api_url: str = os.getenv("NVIDIA_API_URL", DEFAULT_API_URL),
             prefix: str = "",
             suffix: str = "",
             truncate: EmbeddingTruncateMode | str | None = None,
             timeout: float | None = None)
```

Create a NvidiaTextEmbedder component.

**Arguments**:

- `model`: Embedding model to use.
If no specific model along with locally hosted API URL is provided,
the system defaults to the available model found using /models API.
- `api_key`: API key for the NVIDIA NIM.
- `api_url`: Custom API URL for the NVIDIA NIM.
Format for API URL is `http://host:port`
- `prefix`: A string to add to the beginning of each text.
- `suffix`: A string to add to the end of each text.
- `truncate`: Specifies how inputs longer that the maximum token length should be truncated.
If None the behavior is model-dependent, see the official documentation for more information.
- `timeout`: Timeout for request calls, if not set it is inferred from the `NVIDIA_TIMEOUT` environment variable
or set to 60 by default.

<a id="haystack_integrations.components.embedders.nvidia.text_embedder.NvidiaTextEmbedder.default_model"></a>

#### NvidiaTextEmbedder.default\_model

```python
def default_model()
```

Set default model in local NIM mode.

<a id="haystack_integrations.components.embedders.nvidia.text_embedder.NvidiaTextEmbedder.warm_up"></a>

#### NvidiaTextEmbedder.warm\_up

```python
def warm_up()
```

Initializes the component.

<a id="haystack_integrations.components.embedders.nvidia.text_embedder.NvidiaTextEmbedder.to_dict"></a>

#### NvidiaTextEmbedder.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.embedders.nvidia.text_embedder.NvidiaTextEmbedder.available_models"></a>

#### NvidiaTextEmbedder.available\_models

```python
@property
def available_models() -> list[Model]
```

Get a list of available models that work with NvidiaTextEmbedder.

<a id="haystack_integrations.components.embedders.nvidia.text_embedder.NvidiaTextEmbedder.from_dict"></a>

#### NvidiaTextEmbedder.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "NvidiaTextEmbedder"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: The dictionary to deserialize from.

**Returns**:

The deserialized component.

<a id="haystack_integrations.components.embedders.nvidia.text_embedder.NvidiaTextEmbedder.run"></a>

#### NvidiaTextEmbedder.run

```python
@component.output_types(embedding=list[float], meta=dict[str, Any])
def run(text: str) -> dict[str, list[float] | dict[str, Any]]
```

Embed a string.

**Arguments**:

- `text`: The text to embed.

**Raises**:

- `TypeError`: If the input is not a string.
- `ValueError`: If the input string is empty.

**Returns**:

A dictionary with the following keys and values:
- `embedding` - Embedding of the text.
- `meta` - Metadata on usage statistics, etc.

<a id="haystack_integrations.components.embedders.nvidia.truncate"></a>

## Module haystack\_integrations.components.embedders.nvidia.truncate

<a id="haystack_integrations.components.embedders.nvidia.truncate.EmbeddingTruncateMode"></a>

### EmbeddingTruncateMode

Specifies how inputs to the NVIDIA embedding components are truncated.
If START, the input will be truncated from the start.
If END, the input will be truncated from the end.
If NONE, an error will be returned (if the input is too long).

<a id="haystack_integrations.components.embedders.nvidia.truncate.EmbeddingTruncateMode.from_str"></a>

#### EmbeddingTruncateMode.from\_str

```python
@classmethod
def from_str(cls, string: str) -> "EmbeddingTruncateMode"
```

Create an truncate mode from a string.

**Arguments**:

- `string`: String to convert.

**Returns**:

Truncate mode.

<a id="haystack_integrations.components.generators.nvidia.chat.chat_generator"></a>

## Module haystack\_integrations.components.generators.nvidia.chat.chat\_generator

<a id="haystack_integrations.components.generators.nvidia.chat.chat_generator.NvidiaChatGenerator"></a>

### NvidiaChatGenerator

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

<a id="haystack_integrations.components.generators.nvidia.chat.chat_generator.NvidiaChatGenerator.__init__"></a>

#### NvidiaChatGenerator.\_\_init\_\_

```python
def __init__(*,
             api_key: Secret = Secret.from_env_var("NVIDIA_API_KEY"),
             model: str = "meta/llama-3.1-8b-instruct",
             streaming_callback: StreamingCallbackT | None = None,
             api_base_url: str | None = os.getenv("NVIDIA_API_URL",
                                                  DEFAULT_API_URL),
             generation_kwargs: dict[str, Any] | None = None,
             tools: ToolsType | None = None,
             timeout: float | None = None,
             max_retries: int | None = None,
             http_client_kwargs: dict[str, Any] | None = None) -> None
```

Creates an instance of NvidiaChatGenerator.

**Arguments**:

- `api_key`: The NVIDIA API key.
- `model`: The name of the NVIDIA chat completion model to use.
- `streaming_callback`: A callback function that is called when a new token is received from the stream.
The callback function accepts StreamingChunk as an argument.
- `api_base_url`: The NVIDIA API Base url.
- `generation_kwargs`: Other parameters to use for the model. These parameters are all sent directly to
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
- `tools`: A list of tools or a Toolset for which the model can prepare calls. This parameter can accept either a
list of `Tool` objects or a `Toolset` instance.
- `timeout`: The timeout for the NVIDIA API call.
- `max_retries`: Maximum number of retries to contact NVIDIA after an internal error.
If not set, it defaults to either the `NVIDIA_MAX_RETRIES` environment variable, or set to 5.
- `http_client_kwargs`: A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/`client`).

<a id="haystack_integrations.components.generators.nvidia.chat.chat_generator.NvidiaChatGenerator.to_dict"></a>

#### NvidiaChatGenerator.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serialize this component to a dictionary.

**Returns**:

The serialized component as a dictionary.

<a id="haystack_integrations.components.generators.nvidia.generator"></a>

## Module haystack\_integrations.components.generators.nvidia.generator

<a id="haystack_integrations.components.generators.nvidia.generator.NvidiaGenerator"></a>

### NvidiaGenerator

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

<a id="haystack_integrations.components.generators.nvidia.generator.NvidiaGenerator.__init__"></a>

#### NvidiaGenerator.\_\_init\_\_

```python
def __init__(model: str | None = None,
             api_url: str = os.getenv("NVIDIA_API_URL", DEFAULT_API_URL),
             api_key: Secret | None = Secret.from_env_var("NVIDIA_API_KEY"),
             model_arguments: dict[str, Any] | None = None,
             timeout: float | None = None) -> None
```

Create a NvidiaGenerator component.

**Arguments**:

- `model`: Name of the model to use for text generation.
See the [NVIDIA NIMs](https://ai.nvidia.com)
for more information on the supported models.
`Note`: If no specific model along with locally hosted API URL is provided,
the system defaults to the available model found using /models API.
Check supported models at [NVIDIA NIM](https://ai.nvidia.com).
- `api_key`: API key for the NVIDIA NIM. Set it as the `NVIDIA_API_KEY` environment
variable or pass it here.
- `api_url`: Custom API URL for the NVIDIA NIM.
- `model_arguments`: Additional arguments to pass to the model provider. These arguments are
specific to a model.
Search your model in the [NVIDIA NIM](https://ai.nvidia.com)
to find the arguments it accepts.
- `timeout`: Timeout for request calls, if not set it is inferred from the `NVIDIA_TIMEOUT` environment variable
or set to 60 by default.

<a id="haystack_integrations.components.generators.nvidia.generator.NvidiaGenerator.default_model"></a>

#### NvidiaGenerator.default\_model

```python
def default_model() -> None
```

Set default model in local NIM mode.

<a id="haystack_integrations.components.generators.nvidia.generator.NvidiaGenerator.warm_up"></a>

#### NvidiaGenerator.warm\_up

```python
def warm_up() -> None
```

Initializes the component.

<a id="haystack_integrations.components.generators.nvidia.generator.NvidiaGenerator.to_dict"></a>

#### NvidiaGenerator.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.generators.nvidia.generator.NvidiaGenerator.available_models"></a>

#### NvidiaGenerator.available\_models

```python
@property
def available_models() -> list[Model]
```

Get a list of available models that work with ChatNVIDIA.

<a id="haystack_integrations.components.generators.nvidia.generator.NvidiaGenerator.from_dict"></a>

#### NvidiaGenerator.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "NvidiaGenerator"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.generators.nvidia.generator.NvidiaGenerator.run"></a>

#### NvidiaGenerator.run

```python
@component.output_types(replies=list[str], meta=list[dict[str, Any]])
def run(prompt: str) -> dict[str, list[str] | list[dict[str, Any]]]
```

Queries the model with the provided prompt.

**Arguments**:

- `prompt`: Text to be sent to the generative model.

**Returns**:

A dictionary with the following keys:
- `replies` - Replies generated by the model.
- `meta` - Metadata for each reply.

<a id="haystack_integrations.components.rankers.nvidia.ranker"></a>

## Module haystack\_integrations.components.rankers.nvidia.ranker

<a id="haystack_integrations.components.rankers.nvidia.ranker.NvidiaRanker"></a>

### NvidiaRanker

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

<a id="haystack_integrations.components.rankers.nvidia.ranker.NvidiaRanker.__init__"></a>

#### NvidiaRanker.\_\_init\_\_

```python
def __init__(model: str | None = None,
             truncate: RankerTruncateMode | str | None = None,
             api_url: str = os.getenv("NVIDIA_API_URL", DEFAULT_API_URL),
             api_key: Secret | None = Secret.from_env_var("NVIDIA_API_KEY"),
             top_k: int = 5,
             query_prefix: str = "",
             document_prefix: str = "",
             meta_fields_to_embed: list[str] | None = None,
             embedding_separator: str = "\n",
             timeout: float | None = None) -> None
```

Create a NvidiaRanker component.

**Arguments**:

- `model`: Ranking model to use.
- `truncate`: Truncation strategy to use. Can be "NONE", "END", or RankerTruncateMode. Defaults to NIM's default.
- `api_key`: API key for the NVIDIA NIM.
- `api_url`: Custom API URL for the NVIDIA NIM.
- `top_k`: Number of documents to return.
- `query_prefix`: A string to add at the beginning of the query text before ranking.
Use it to prepend the text with an instruction, as required by reranking models like `bge`.
- `document_prefix`: A string to add at the beginning of each document before ranking. You can use it to prepend the document
with an instruction, as required by embedding models like `bge`.
- `meta_fields_to_embed`: List of metadata fields to embed with the document.
- `embedding_separator`: Separator to concatenate metadata fields to the document.
- `timeout`: Timeout for request calls, if not set it is inferred from the `NVIDIA_TIMEOUT` environment variable
or set to 60 by default.

<a id="haystack_integrations.components.rankers.nvidia.ranker.NvidiaRanker.to_dict"></a>

#### NvidiaRanker.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serialize the ranker to a dictionary.

**Returns**:

A dictionary containing the ranker's attributes.

<a id="haystack_integrations.components.rankers.nvidia.ranker.NvidiaRanker.from_dict"></a>

#### NvidiaRanker.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "NvidiaRanker"
```

Deserialize the ranker from a dictionary.

**Arguments**:

- `data`: A dictionary containing the ranker's attributes.

**Returns**:

The deserialized ranker.

<a id="haystack_integrations.components.rankers.nvidia.ranker.NvidiaRanker.warm_up"></a>

#### NvidiaRanker.warm\_up

```python
def warm_up() -> None
```

Initialize the ranker.

**Raises**:

- `ValueError`: If the API key is required for hosted NVIDIA NIMs.

<a id="haystack_integrations.components.rankers.nvidia.ranker.NvidiaRanker.run"></a>

#### NvidiaRanker.run

```python
@component.output_types(documents=list[Document])
def run(query: str,
        documents: list[Document],
        top_k: int | None = None) -> dict[str, list[Document]]
```

Rank a list of documents based on a given query.

**Arguments**:

- `query`: The query to rank the documents against.
- `documents`: The list of documents to rank.
- `top_k`: The number of documents to return.

**Raises**:

- `TypeError`: If the arguments are of the wrong type.

**Returns**:

A dictionary containing the ranked documents.

<a id="haystack_integrations.components.rankers.nvidia.truncate"></a>

## Module haystack\_integrations.components.rankers.nvidia.truncate

<a id="haystack_integrations.components.rankers.nvidia.truncate.RankerTruncateMode"></a>

### RankerTruncateMode

Specifies how inputs to the NVIDIA ranker components are truncated.
If NONE, the input will not be truncated and an error returned instead.
If END, the input will be truncated from the end.

<a id="haystack_integrations.components.rankers.nvidia.truncate.RankerTruncateMode.from_str"></a>

#### RankerTruncateMode.from\_str

```python
@classmethod
def from_str(cls, string: str) -> "RankerTruncateMode"
```

Create an truncate mode from a string.

**Arguments**:

- `string`: String to convert.

**Returns**:

Truncate mode.

