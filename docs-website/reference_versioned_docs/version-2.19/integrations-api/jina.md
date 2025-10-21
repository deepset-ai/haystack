---
title: "Jina"
id: integrations-jina
description: "Jina integration for Haystack"
slug: "/integrations-jina"
---

<a id="haystack_integrations.components.embedders.jina.document_embedder"></a>

# Module haystack\_integrations.components.embedders.jina.document\_embedder

<a id="haystack_integrations.components.embedders.jina.document_embedder.JinaDocumentEmbedder"></a>

## JinaDocumentEmbedder

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

<a id="haystack_integrations.components.embedders.jina.document_embedder.JinaDocumentEmbedder.__init__"></a>

#### JinaDocumentEmbedder.\_\_init\_\_

```python
def __init__(api_key: Secret = Secret.from_env_var("JINA_API_KEY"),
             model: str = "jina-embeddings-v3",
             prefix: str = "",
             suffix: str = "",
             batch_size: int = 32,
             progress_bar: bool = True,
             meta_fields_to_embed: Optional[List[str]] = None,
             embedding_separator: str = "\n",
             task: Optional[str] = None,
             dimensions: Optional[int] = None,
             late_chunking: Optional[bool] = None)
```

Create a JinaDocumentEmbedder component.

**Arguments**:

- `api_key`: The Jina API key.
- `model`: The name of the Jina model to use.
Check the list of available models on [Jina documentation](https://jina.ai/embeddings/).
- `prefix`: A string to add to the beginning of each text.
- `suffix`: A string to add to the end of each text.
- `batch_size`: Number of Documents to encode at once.
- `progress_bar`: Whether to show a progress bar or not. Can be helpful to disable in production deployments
to keep the logs clean.
- `meta_fields_to_embed`: List of meta fields that should be embedded along with the Document text.
- `embedding_separator`: Separator used to concatenate the meta fields to the Document text.
- `task`: The downstream task for which the embeddings will be used.
The model will return the optimized embeddings for that task.
Check the list of available tasks on [Jina documentation](https://jina.ai/embeddings/).
- `dimensions`: Number of desired dimension.
Smaller dimensions are easier to store and retrieve, with minimal performance impact thanks to MRL.
- `late_chunking`: A boolean to enable or disable late chunking.
Apply the late chunking technique to leverage the model's long-context capabilities for
generating contextual chunk embeddings.

The support of `task` and `late_chunking` parameters is only available for jina-embeddings-v3.

<a id="haystack_integrations.components.embedders.jina.document_embedder.JinaDocumentEmbedder.to_dict"></a>

#### JinaDocumentEmbedder.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.embedders.jina.document_embedder.JinaDocumentEmbedder.from_dict"></a>

#### JinaDocumentEmbedder.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "JinaDocumentEmbedder"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.embedders.jina.document_embedder.JinaDocumentEmbedder.run"></a>

#### JinaDocumentEmbedder.run

```python
@component.output_types(documents=List[Document], meta=Dict[str, Any])
def run(documents: List[Document]) -> Dict[str, Any]
```

Compute the embeddings for a list of Documents.

**Arguments**:

- `documents`: A list of Documents to embed.

**Raises**:

- `TypeError`: If the input is not a list of Documents.

**Returns**:

A dictionary with following keys:
- `documents`: List of Documents, each with an `embedding` field containing the computed embedding.
- `meta`: A dictionary with metadata including the model name and usage statistics.

<a id="haystack_integrations.components.embedders.jina.document_image_embedder"></a>

# Module haystack\_integrations.components.embedders.jina.document\_image\_embedder

<a id="haystack_integrations.components.embedders.jina.document_image_embedder.JinaDocumentImageEmbedder"></a>

## JinaDocumentImageEmbedder

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

<a id="haystack_integrations.components.embedders.jina.document_image_embedder.JinaDocumentImageEmbedder.__init__"></a>

#### JinaDocumentImageEmbedder.\_\_init\_\_

```python
def __init__(*,
             api_key: Secret = Secret.from_env_var("JINA_API_KEY"),
             model: str = "jina-clip-v2",
             file_path_meta_field: str = "file_path",
             root_path: Optional[str] = None,
             embedding_dimension: Optional[int] = None,
             image_size: Optional[Tuple[int, int]] = None,
             batch_size: int = 5)
```

Create a JinaDocumentImageEmbedder component.

**Arguments**:

- `api_key`: The Jina API key. It can be explicitly provided or automatically read from the
environment variable `JINA_API_KEY` (recommended).
- `model`: The name of the Jina multimodal model to use.
Supported models include:
- "jina-clip-v1"
- "jina-clip-v2" (default)
- "jina-embeddings-v4"
Check the list of available models on [Jina documentation](https://jina.ai/embeddings/).
- `file_path_meta_field`: The metadata field in the Document that contains the file path to the image or PDF.
- `root_path`: The root directory path where document files are located. If provided, file paths in
document metadata will be resolved relative to this path. If None, file paths are treated as absolute paths.
- `embedding_dimension`: Number of desired dimensions for the embedding.
Smaller dimensions are easier to store and retrieve, with minimal performance impact thanks to MRL.
Only supported by jina-embeddings-v4.
- `image_size`: If provided, resizes the image to fit within the specified dimensions (width, height) while
maintaining aspect ratio. This reduces file size, memory usage, and processing time.
- `batch_size`: Number of images to send in each API request. Defaults to 5.

<a id="haystack_integrations.components.embedders.jina.document_image_embedder.JinaDocumentImageEmbedder.to_dict"></a>

#### JinaDocumentImageEmbedder.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.embedders.jina.document_image_embedder.JinaDocumentImageEmbedder.from_dict"></a>

#### JinaDocumentImageEmbedder.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "JinaDocumentImageEmbedder"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.embedders.jina.document_image_embedder.JinaDocumentImageEmbedder.run"></a>

#### JinaDocumentImageEmbedder.run

```python
@component.output_types(documents=List[Document])
def run(documents: List[Document]) -> Dict[str, List[Document]]
```

Embed a list of image documents.

**Arguments**:

- `documents`: Documents to embed.

**Returns**:

A dictionary with the following keys:
- `documents`: Documents with embeddings.

<a id="haystack_integrations.components.embedders.jina.text_embedder"></a>

# Module haystack\_integrations.components.embedders.jina.text\_embedder

<a id="haystack_integrations.components.embedders.jina.text_embedder.JinaTextEmbedder"></a>

## JinaTextEmbedder

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

<a id="haystack_integrations.components.embedders.jina.text_embedder.JinaTextEmbedder.__init__"></a>

#### JinaTextEmbedder.\_\_init\_\_

```python
def __init__(api_key: Secret = Secret.from_env_var("JINA_API_KEY"),
             model: str = "jina-embeddings-v3",
             prefix: str = "",
             suffix: str = "",
             task: Optional[str] = None,
             dimensions: Optional[int] = None,
             late_chunking: Optional[bool] = None)
```

Create a JinaTextEmbedder component.

**Arguments**:

- `api_key`: The Jina API key. It can be explicitly provided or automatically read from the
environment variable `JINA_API_KEY` (recommended).
- `model`: The name of the Jina model to use.
Check the list of available models on [Jina documentation](https://jina.ai/embeddings/).
- `prefix`: A string to add to the beginning of each text.
- `suffix`: A string to add to the end of each text.
- `task`: The downstream task for which the embeddings will be used.
The model will return the optimized embeddings for that task.
Check the list of available tasks on [Jina documentation](https://jina.ai/embeddings/).
- `dimensions`: Number of desired dimension.
Smaller dimensions are easier to store and retrieve, with minimal performance impact thanks to MRL.
- `late_chunking`: A boolean to enable or disable late chunking.
Apply the late chunking technique to leverage the model's long-context capabilities for
generating contextual chunk embeddings.

The support of `task` and `late_chunking` parameters is only available for jina-embeddings-v3.

<a id="haystack_integrations.components.embedders.jina.text_embedder.JinaTextEmbedder.to_dict"></a>

#### JinaTextEmbedder.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.embedders.jina.text_embedder.JinaTextEmbedder.from_dict"></a>

#### JinaTextEmbedder.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "JinaTextEmbedder"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.embedders.jina.text_embedder.JinaTextEmbedder.run"></a>

#### JinaTextEmbedder.run

```python
@component.output_types(embedding=List[float], meta=Dict[str, Any])
def run(text: str) -> Dict[str, Any]
```

Embed a string.

**Arguments**:

- `text`: The string to embed.

**Raises**:

- `TypeError`: If the input is not a string.

**Returns**:

A dictionary with following keys:
- `embedding`: The embedding of the input string.
- `meta`: A dictionary with metadata including the model name and usage statistics.

<a id="haystack_integrations.components.rankers.jina.ranker"></a>

# Module haystack\_integrations.components.rankers.jina.ranker

<a id="haystack_integrations.components.rankers.jina.ranker.JinaRanker"></a>

## JinaRanker

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

<a id="haystack_integrations.components.rankers.jina.ranker.JinaRanker.__init__"></a>

#### JinaRanker.\_\_init\_\_

```python
def __init__(model: str = "jina-reranker-v1-base-en",
             api_key: Secret = Secret.from_env_var("JINA_API_KEY"),
             top_k: Optional[int] = None,
             score_threshold: Optional[float] = None)
```

Creates an instance of JinaRanker.

**Arguments**:

- `api_key`: The Jina API key. It can be explicitly provided or automatically read from the
environment variable JINA_API_KEY (recommended).
- `model`: The name of the Jina model to use. Check the list of available models on `https://jina.ai/reranker/`
- `top_k`: The maximum number of Documents to return per query. If `None`, all documents are returned
- `score_threshold`: If provided only returns documents with a score above this threshold.

**Raises**:

- `ValueError`: If `top_k` is not > 0.

<a id="haystack_integrations.components.rankers.jina.ranker.JinaRanker.to_dict"></a>

#### JinaRanker.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.rankers.jina.ranker.JinaRanker.from_dict"></a>

#### JinaRanker.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "JinaRanker"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.rankers.jina.ranker.JinaRanker.run"></a>

#### JinaRanker.run

```python
@component.output_types(documents=List[Document])
def run(query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None)
```

Returns a list of Documents ranked by their similarity to the given query.

**Arguments**:

- `query`: Query string.
- `documents`: List of Documents.
- `top_k`: The maximum number of Documents you want the Ranker to return.
- `score_threshold`: If provided only returns documents with a score above this threshold.

**Raises**:

- `ValueError`: If `top_k` is not > 0.

**Returns**:

A dictionary with the following keys:
- `documents`: List of Documents most similar to the given query in descending order of similarity.

<a id="haystack_integrations.components.connectors.jina.reader"></a>

# Module haystack\_integrations.components.connectors.jina.reader

<a id="haystack_integrations.components.connectors.jina.reader.JinaReaderConnector"></a>

## JinaReaderConnector

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

<a id="haystack_integrations.components.connectors.jina.reader.JinaReaderConnector.__init__"></a>

#### JinaReaderConnector.\_\_init\_\_

```python
def __init__(mode: Union[JinaReaderMode, str],
             api_key: Secret = Secret.from_env_var("JINA_API_KEY"),
             json_response: bool = True)
```

Initialize a JinaReader instance.

**Arguments**:

- `mode`: The operation mode for the reader (`read`, `search` or `ground`).
- `read`: process a URL and return the textual content of the page.
- `search`: search the web and return textual content of the most relevant pages.
- `ground`: call the grounding engine to perform fact checking.
For more information on the modes, see the [Jina Reader documentation](https://jina.ai/reader/).
- `api_key`: The Jina API key. It can be explicitly provided or automatically read from the
environment variable JINA_API_KEY (recommended).
- `json_response`: Controls the response format from the Jina Reader API.
If `True`, requests a JSON response, resulting in Documents with rich structured metadata.
If `False`, requests a raw response, resulting in one Document with minimal metadata.

<a id="haystack_integrations.components.connectors.jina.reader.JinaReaderConnector.to_dict"></a>

#### JinaReaderConnector.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.connectors.jina.reader.JinaReaderConnector.from_dict"></a>

#### JinaReaderConnector.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "JinaReaderConnector"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.connectors.jina.reader.JinaReaderConnector.run"></a>

#### JinaReaderConnector.run

```python
@component.output_types(documents=List[Document])
def run(query: str,
        headers: Optional[Dict[str, str]] = None) -> Dict[str, List[Document]]
```

Process the query/URL using the Jina AI reader service.

**Arguments**:

- `query`: The query string or URL to process.
- `headers`: Optional headers to include in the request for customization. Refer to the
[Jina Reader documentation](https://jina.ai/reader/) for more information.

**Returns**:

A dictionary with the following keys:
- `documents`: A list of `Document` objects.
