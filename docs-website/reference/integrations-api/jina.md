---
title: "Jina"
id: integrations-jina
description: "Jina integration for Haystack"
slug: "/integrations-jina"
---


## `haystack_integrations.components.connectors.jina.reader`

### `JinaReaderConnector`

A component that interacts with Jina AI's reader service to process queries and return documents.

This component supports different modes of operation: `read`, `search`, and `ground`.

Usage example:

```python
from haystack_integrations.components.connectors.jina import JinaReaderConnector

reader = JinaReaderConnector(mode="read")
query = "https://example.com"
result = reader.run(query=query)
document = result["documents"][0]
print(document.content)

>>> "This domain is for use in illustrative examples..."
```

#### `__init__`

```python
__init__(
    mode: JinaReaderMode | str,
    api_key: Secret = Secret.from_env_var("JINA_API_KEY"),
    json_response: bool = True,
)
```

Initialize a JinaReader instance.

**Parameters:**

- **mode** (<code>JinaReaderMode | str</code>) – The operation mode for the reader (`read`, `search` or `ground`).
- `read`: process a URL and return the textual content of the page.
- `search`: search the web and return textual content of the most relevant pages.
- `ground`: call the grounding engine to perform fact checking.
  For more information on the modes, see the [Jina Reader documentation](https://jina.ai/reader/).
- **api_key** (<code>Secret</code>) – The Jina API key. It can be explicitly provided or automatically read from the
  environment variable JINA_API_KEY (recommended).
- **json_response** (<code>bool</code>) – Controls the response format from the Jina Reader API.
  If `True`, requests a JSON response, resulting in Documents with rich structured metadata.
  If `False`, requests a raw response, resulting in one Document with minimal metadata.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> JinaReaderConnector
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>JinaReaderConnector</code> – Deserialized component.

#### `run`

```python
run(
    query: str, headers: dict[str, str] | None = None
) -> dict[str, list[Document]]
```

Process the query/URL using the Jina AI reader service.

**Parameters:**

- **query** (<code>str</code>) – The query string or URL to process.
- **headers** (<code>dict\[str, str\] | None</code>) – Optional headers to include in the request for customization. Refer to the
  [Jina Reader documentation](https://jina.ai/reader/) for more information.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
  - `documents`: A list of `Document` objects.

## `haystack_integrations.components.embedders.jina.document_embedder`

### `JinaDocumentEmbedder`

A component for computing Document embeddings using Jina AI models.
The embedding of each Document is stored in the `embedding` field of the Document.

Usage example:

```python
from haystack import Document
from haystack_integrations.components.embedders.jina import JinaDocumentEmbedder

# Make sure that the environment variable JINA_API_KEY is set

document_embedder = JinaDocumentEmbedder(task="retrieval.query")

doc = Document(content="I love pizza!")

result = document_embedder.run([doc])
print(result['documents'][0].embedding)

# [0.017020374536514282, -0.023255806416273117, ...]
```

#### `__init__`

```python
__init__(
    api_key: Secret = Secret.from_env_var("JINA_API_KEY"),
    model: str = "jina-embeddings-v3",
    prefix: str = "",
    suffix: str = "",
    batch_size: int = 32,
    progress_bar: bool = True,
    meta_fields_to_embed: list[str] | None = None,
    embedding_separator: str = "\n",
    task: str | None = None,
    dimensions: int | None = None,
    late_chunking: bool | None = None,
)
```

Create a JinaDocumentEmbedder component.

**Parameters:**

- **api_key** (<code>Secret</code>) – The Jina API key.
- **model** (<code>str</code>) – The name of the Jina model to use.
  Check the list of available models on [Jina documentation](https://jina.ai/embeddings/).
- **prefix** (<code>str</code>) – A string to add to the beginning of each text.
- **suffix** (<code>str</code>) – A string to add to the end of each text.
- **batch_size** (<code>int</code>) – Number of Documents to encode at once.
- **progress_bar** (<code>bool</code>) – Whether to show a progress bar or not. Can be helpful to disable in production deployments
  to keep the logs clean.
- **meta_fields_to_embed** (<code>list\[str\] | None</code>) – List of meta fields that should be embedded along with the Document text.
- **embedding_separator** (<code>str</code>) – Separator used to concatenate the meta fields to the Document text.
- **task** (<code>str | None</code>) – The downstream task for which the embeddings will be used.
  The model will return the optimized embeddings for that task.
  Check the list of available tasks on [Jina documentation](https://jina.ai/embeddings/).
- **dimensions** (<code>int | None</code>) – Number of desired dimension.
  Smaller dimensions are easier to store and retrieve, with minimal performance impact thanks to MRL.
- **late_chunking** (<code>bool | None</code>) – A boolean to enable or disable late chunking.
  Apply the late chunking technique to leverage the model's long-context capabilities for
  generating contextual chunk embeddings.

The support of `task` and `late_chunking` parameters is only available for jina-embeddings-v3.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> JinaDocumentEmbedder
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>JinaDocumentEmbedder</code> – Deserialized component.

#### `run`

```python
run(documents: list[Document]) -> dict[str, Any]
```

Compute the embeddings for a list of Documents.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – A list of Documents to embed.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with following keys:
- `documents`: List of Documents, each with an `embedding` field containing the computed embedding.
- `meta`: A dictionary with metadata including the model name and usage statistics.

**Raises:**

- <code>TypeError</code> – If the input is not a list of Documents.

## `haystack_integrations.components.embedders.jina.document_image_embedder`

### `JinaDocumentImageEmbedder`

A component for computing Document embeddings based on images using Jina AI multimodal models.

The embedding of each Document is stored in the `embedding` field of the Document.

The JinaDocumentImageEmbedder supports models from the jina-clip series and jina-embeddings-v4
which can encode images into vector representations in the same embedding space as text.

Usage example:

```python
from haystack import Document
from haystack_integrations.components.embedders.jina import JinaDocumentImageEmbedder

# Make sure that the environment variable JINA_API_KEY is set

embedder = JinaDocumentImageEmbedder(model="jina-clip-v2")

documents = [
    Document(content="A photo of a cat", meta={"file_path": "cat.jpg"}),
    Document(content="A photo of a dog", meta={"file_path": "dog.jpg"}),
]

result = embedder.run(documents=documents)
documents_with_embeddings = result["documents"]
print(documents_with_embeddings[0].embedding)

# [0.017020374536514282, -0.023255806416273117, ...]
```

#### `__init__`

```python
__init__(
    *,
    api_key: Secret = Secret.from_env_var("JINA_API_KEY"),
    model: str = "jina-clip-v2",
    file_path_meta_field: str = "file_path",
    root_path: str | None = None,
    embedding_dimension: int | None = None,
    image_size: tuple[int, int] | None = None,
    batch_size: int = 5
)
```

Create a JinaDocumentImageEmbedder component.

**Parameters:**

- **api_key** (<code>Secret</code>) – The Jina API key. It can be explicitly provided or automatically read from the
  environment variable `JINA_API_KEY` (recommended).
- **model** (<code>str</code>) – The name of the Jina multimodal model to use.
  Supported models include:
- "jina-clip-v1"
- "jina-clip-v2" (default)
- "jina-embeddings-v4"
  Check the list of available models on [Jina documentation](https://jina.ai/embeddings/).
- **file_path_meta_field** (<code>str</code>) – The metadata field in the Document that contains the file path to the image or PDF.
- **root_path** (<code>str | None</code>) – The root directory path where document files are located. If provided, file paths in
  document metadata will be resolved relative to this path. If None, file paths are treated as absolute paths.
- **embedding_dimension** (<code>int | None</code>) – Number of desired dimensions for the embedding.
  Smaller dimensions are easier to store and retrieve, with minimal performance impact thanks to MRL.
  Only supported by jina-embeddings-v4.
- **image_size** (<code>tuple\[int, int\] | None</code>) – If provided, resizes the image to fit within the specified dimensions (width, height) while
  maintaining aspect ratio. This reduces file size, memory usage, and processing time.
- **batch_size** (<code>int</code>) – Number of images to send in each API request. Defaults to 5.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> JinaDocumentImageEmbedder
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>JinaDocumentImageEmbedder</code> – Deserialized component.

#### `run`

```python
run(documents: list[Document]) -> dict[str, list[Document]]
```

Embed a list of image documents.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – Documents to embed.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: Documents with embeddings.

## `haystack_integrations.components.embedders.jina.text_embedder`

### `JinaTextEmbedder`

A component for embedding strings using Jina AI models.

Usage example:

```python
from haystack_integrations.components.embedders.jina import JinaTextEmbedder

# Make sure that the environment variable JINA_API_KEY is set

text_embedder = JinaTextEmbedder(task="retrieval.query")

text_to_embed = "I love pizza!"

print(text_embedder.run(text_to_embed))

# {'embedding': [0.017020374536514282, -0.023255806416273117, ...],
# 'meta': {'model': 'jina-embeddings-v3',
#          'usage': {'prompt_tokens': 4, 'total_tokens': 4}}}
```

#### `__init__`

```python
__init__(
    api_key: Secret = Secret.from_env_var("JINA_API_KEY"),
    model: str = "jina-embeddings-v3",
    prefix: str = "",
    suffix: str = "",
    task: str | None = None,
    dimensions: int | None = None,
    late_chunking: bool | None = None,
)
```

Create a JinaTextEmbedder component.

**Parameters:**

- **api_key** (<code>Secret</code>) – The Jina API key. It can be explicitly provided or automatically read from the
  environment variable `JINA_API_KEY` (recommended).
- **model** (<code>str</code>) – The name of the Jina model to use.
  Check the list of available models on [Jina documentation](https://jina.ai/embeddings/).
- **prefix** (<code>str</code>) – A string to add to the beginning of each text.
- **suffix** (<code>str</code>) – A string to add to the end of each text.
- **task** (<code>str | None</code>) – The downstream task for which the embeddings will be used.
  The model will return the optimized embeddings for that task.
  Check the list of available tasks on [Jina documentation](https://jina.ai/embeddings/).
- **dimensions** (<code>int | None</code>) – Number of desired dimension.
  Smaller dimensions are easier to store and retrieve, with minimal performance impact thanks to MRL.
- **late_chunking** (<code>bool | None</code>) – A boolean to enable or disable late chunking.
  Apply the late chunking technique to leverage the model's long-context capabilities for
  generating contextual chunk embeddings.

The support of `task` and `late_chunking` parameters is only available for jina-embeddings-v3.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> JinaTextEmbedder
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>JinaTextEmbedder</code> – Deserialized component.

#### `run`

```python
run(text: str) -> dict[str, Any]
```

Embed a string.

**Parameters:**

- **text** (<code>str</code>) – The string to embed.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with following keys:
- `embedding`: The embedding of the input string.
- `meta`: A dictionary with metadata including the model name and usage statistics.

**Raises:**

- <code>TypeError</code> – If the input is not a string.

## `haystack_integrations.components.rankers.jina.ranker`

### `JinaRanker`

Ranks Documents based on their similarity to the query using Jina AI models.

Usage example:

```python
from haystack import Document
from haystack_integrations.components.rankers.jina import JinaRanker

ranker = JinaRanker()
docs = [Document(content="Paris"), Document(content="Berlin")]
query = "City in Germany"
result = ranker.run(query=query, documents=docs)
docs = result["documents"]
print(docs[0].content)
```

#### `__init__`

```python
__init__(
    model: str = "jina-reranker-v1-base-en",
    api_key: Secret = Secret.from_env_var("JINA_API_KEY"),
    top_k: int | None = None,
    score_threshold: float | None = None,
)
```

Creates an instance of JinaRanker.

**Parameters:**

- **api_key** (<code>Secret</code>) – The Jina API key. It can be explicitly provided or automatically read from the
  environment variable JINA_API_KEY (recommended).
- **model** (<code>str</code>) – The name of the Jina model to use. Check the list of available models on `https://jina.ai/reranker/`
- **top_k** (<code>int | None</code>) – The maximum number of Documents to return per query. If `None`, all documents are returned
- **score_threshold** (<code>float | None</code>) – If provided only returns documents with a score above this threshold.

**Raises:**

- <code>ValueError</code> – If `top_k` is not > 0.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> JinaRanker
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>JinaRanker</code> – Deserialized component.

#### `run`

```python
run(
    query: str,
    documents: list[Document],
    top_k: int | None = None,
    score_threshold: float | None = None,
)
```

Returns a list of Documents ranked by their similarity to the given query.

**Parameters:**

- **query** (<code>str</code>) – Query string.
- **documents** (<code>list\[Document\]</code>) – List of Documents.
- **top_k** (<code>int | None</code>) – The maximum number of Documents you want the Ranker to return.
- **score_threshold** (<code>float | None</code>) – If provided only returns documents with a score above this threshold.

**Returns:**

- – A dictionary with the following keys:
- `documents`: List of Documents most similar to the given query in descending order of similarity.

**Raises:**

- <code>ValueError</code> – If `top_k` is not > 0.
