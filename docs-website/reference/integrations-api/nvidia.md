---
title: "Nvidia"
id: integrations-nvidia
description: "Nvidia integration for Haystack"
slug: "/integrations-nvidia"
---

<a id="haystack_integrations.components.embedders.nvidia.document_embedder"></a>

# Module haystack\_integrations.components.embedders.nvidia.document\_embedder

<a id="haystack_integrations.components.embedders.nvidia.document_embedder.NvidiaDocumentEmbedder"></a>

## NvidiaDocumentEmbedder

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
def __init__(model: Optional[str] = None,
             api_key: Optional[Secret] = Secret.from_env_var("NVIDIA_API_KEY"),
             api_url: str = os.getenv("NVIDIA_API_URL", DEFAULT_API_URL),
             prefix: str = "",
             suffix: str = "",
             batch_size: int = 32,
             progress_bar: bool = True,
             meta_fields_to_embed: Optional[List[str]] = None,
             embedding_separator: str = "\n",
             truncate: Optional[Union[EmbeddingTruncateMode, str]] = None,
             timeout: Optional[float] = None)
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
def default_model()
```

Set default model in local NIM mode.

<a id="haystack_integrations.components.embedders.nvidia.document_embedder.NvidiaDocumentEmbedder.warm_up"></a>

#### NvidiaDocumentEmbedder.warm\_up

```python
def warm_up()
```

Initializes the component.

<a id="haystack_integrations.components.embedders.nvidia.document_embedder.NvidiaDocumentEmbedder.to_dict"></a>

#### NvidiaDocumentEmbedder.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.embedders.nvidia.document_embedder.NvidiaDocumentEmbedder.available_models"></a>

#### NvidiaDocumentEmbedder.available\_models

```python
@property
def available_models() -> List[Model]
```

Get a list of available models that work with NvidiaDocumentEmbedder.

<a id="haystack_integrations.components.embedders.nvidia.document_embedder.NvidiaDocumentEmbedder.from_dict"></a>

#### NvidiaDocumentEmbedder.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "NvidiaDocumentEmbedder"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: The dictionary to deserialize from.

**Returns**:

The deserialized component.

<a id="haystack_integrations.components.embedders.nvidia.document_embedder.NvidiaDocumentEmbedder.run"></a>

#### NvidiaDocumentEmbedder.run

```python
@component.output_types(documents=List[Document], meta=Dict[str, Any])
def run(
    documents: List[Document]
) -> Dict[str, Union[List[Document], Dict[str, Any]]]
```

Embed a list of Documents.

The embedding of each Document is stored in the `embedding` field of the Document.

**Arguments**:

- `documents`: A list of Documents to embed.

**Raises**:

- `RuntimeError`: If the component was not initialized.
- `TypeError`: If the input is not a string.

**Returns**:

A dictionary with the following keys and values:
- `documents` - List of processed Documents with embeddings.
- `meta` - Metadata on usage statistics, etc.

<a id="haystack_integrations.components.embedders.nvidia.text_embedder"></a>

# Module haystack\_integrations.components.embedders.nvidia.text\_embedder

<a id="haystack_integrations.components.embedders.nvidia.text_embedder.NvidiaTextEmbedder"></a>

## NvidiaTextEmbedder

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
def __init__(model: Optional[str] = None,
             api_key: Optional[Secret] = Secret.from_env_var("NVIDIA_API_KEY"),
             api_url: str = os.getenv("NVIDIA_API_URL", DEFAULT_API_URL),
             prefix: str = "",
             suffix: str = "",
             truncate: Optional[Union[EmbeddingTruncateMode, str]] = None,
             timeout: Optional[float] = None)
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
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.embedders.nvidia.text_embedder.NvidiaTextEmbedder.available_models"></a>

#### NvidiaTextEmbedder.available\_models

```python
@property
def available_models() -> List[Model]
```

Get a list of available models that work with NvidiaTextEmbedder.

<a id="haystack_integrations.components.embedders.nvidia.text_embedder.NvidiaTextEmbedder.from_dict"></a>

#### NvidiaTextEmbedder.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "NvidiaTextEmbedder"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: The dictionary to deserialize from.

**Returns**:

The deserialized component.

<a id="haystack_integrations.components.embedders.nvidia.text_embedder.NvidiaTextEmbedder.run"></a>

#### NvidiaTextEmbedder.run

```python
@component.output_types(embedding=List[float], meta=Dict[str, Any])
def run(text: str) -> Dict[str, Union[List[float], Dict[str, Any]]]
```

Embed a string.

**Arguments**:

- `text`: The text to embed.

**Raises**:

- `RuntimeError`: If the component was not initialized.
- `TypeError`: If the input is not a string.

**Returns**:

A dictionary with the following keys and values:
- `embedding` - Embedding of the text.
- `meta` - Metadata on usage statistics, etc.

<a id="haystack_integrations.components.embedders.nvidia.truncate"></a>

# Module haystack\_integrations.components.embedders.nvidia.truncate

<a id="haystack_integrations.components.embedders.nvidia.truncate.EmbeddingTruncateMode"></a>

## EmbeddingTruncateMode

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

<a id="haystack_integrations.components.generators.nvidia.generator"></a>

# Module haystack\_integrations.components.generators.nvidia.generator

<a id="haystack_integrations.components.generators.nvidia.generator.NvidiaGenerator"></a>

## NvidiaGenerator

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
def __init__(model: Optional[str] = None,
             api_url: str = os.getenv("NVIDIA_API_URL", DEFAULT_API_URL),
             api_key: Optional[Secret] = Secret.from_env_var("NVIDIA_API_KEY"),
             model_arguments: Optional[Dict[str, Any]] = None,
             timeout: Optional[float] = None)
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
def default_model()
```

Set default model in local NIM mode.

<a id="haystack_integrations.components.generators.nvidia.generator.NvidiaGenerator.warm_up"></a>

#### NvidiaGenerator.warm\_up

```python
def warm_up()
```

Initializes the component.

<a id="haystack_integrations.components.generators.nvidia.generator.NvidiaGenerator.to_dict"></a>

#### NvidiaGenerator.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.generators.nvidia.generator.NvidiaGenerator.available_models"></a>

#### NvidiaGenerator.available\_models

```python
@property
def available_models() -> List[Model]
```

Get a list of available models that work with ChatNVIDIA.

<a id="haystack_integrations.components.generators.nvidia.generator.NvidiaGenerator.from_dict"></a>

#### NvidiaGenerator.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "NvidiaGenerator"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.generators.nvidia.generator.NvidiaGenerator.run"></a>

#### NvidiaGenerator.run

```python
@component.output_types(replies=List[str], meta=List[Dict[str, Any]])
def run(prompt: str) -> Dict[str, Union[List[str], List[Dict[str, Any]]]]
```

Queries the model with the provided prompt.

**Arguments**:

- `prompt`: Text to be sent to the generative model.

**Returns**:

A dictionary with the following keys:
- `replies` - Replies generated by the model.
- `meta` - Metadata for each reply.

<a id="haystack_integrations.components.rankers.nvidia.ranker"></a>

# Module haystack\_integrations.components.rankers.nvidia.ranker

<a id="haystack_integrations.components.rankers.nvidia.ranker.NvidiaRanker"></a>

## NvidiaRanker

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
def __init__(model: Optional[str] = None,
             truncate: Optional[Union[RankerTruncateMode, str]] = None,
             api_url: str = os.getenv("NVIDIA_API_URL", DEFAULT_API_URL),
             api_key: Optional[Secret] = Secret.from_env_var("NVIDIA_API_KEY"),
             top_k: int = 5,
             query_prefix: str = "",
             document_prefix: str = "",
             meta_fields_to_embed: Optional[List[str]] = None,
             embedding_separator: str = "\n",
             timeout: Optional[float] = None)
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
def to_dict() -> Dict[str, Any]
```

Serialize the ranker to a dictionary.

**Returns**:

A dictionary containing the ranker's attributes.

<a id="haystack_integrations.components.rankers.nvidia.ranker.NvidiaRanker.from_dict"></a>

#### NvidiaRanker.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "NvidiaRanker"
```

Deserialize the ranker from a dictionary.

**Arguments**:

- `data`: A dictionary containing the ranker's attributes.

**Returns**:

The deserialized ranker.

<a id="haystack_integrations.components.rankers.nvidia.ranker.NvidiaRanker.warm_up"></a>

#### NvidiaRanker.warm\_up

```python
def warm_up()
```

Initialize the ranker.

**Raises**:

- `ValueError`: If the API key is required for hosted NVIDIA NIMs.

<a id="haystack_integrations.components.rankers.nvidia.ranker.NvidiaRanker.run"></a>

#### NvidiaRanker.run

```python
@component.output_types(documents=List[Document])
def run(query: str,
        documents: List[Document],
        top_k: Optional[int] = None) -> Dict[str, List[Document]]
```

Rank a list of documents based on a given query.

**Arguments**:

- `query`: The query to rank the documents against.
- `documents`: The list of documents to rank.
- `top_k`: The number of documents to return.

**Raises**:

- `RuntimeError`: If the ranker has not been loaded.
- `TypeError`: If the arguments are of the wrong type.

**Returns**:

A dictionary containing the ranked documents.

<a id="haystack_integrations.components.rankers.nvidia.truncate"></a>

# Module haystack\_integrations.components.rankers.nvidia.truncate

<a id="haystack_integrations.components.rankers.nvidia.truncate.RankerTruncateMode"></a>

## RankerTruncateMode

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
