---
title: "FastEmbed"
id: fastembed-embedders
description: "FastEmbed integration for Haystack"
slug: "/fastembed-embedders"
---


## `haystack_integrations.components.embedders.fastembed.fastembed_document_embedder`

### `FastembedDocumentEmbedder`

FastembedDocumentEmbedder computes Document embeddings using Fastembed embedding models.
The embedding of each Document is stored in the `embedding` field of the Document.

Usage example:

```python
# To use this component, install the "fastembed-haystack" package.
# pip install fastembed-haystack

from haystack_integrations.components.embedders.fastembed import FastembedDocumentEmbedder
from haystack.dataclasses import Document

doc_embedder = FastembedDocumentEmbedder(
    model="BAAI/bge-small-en-v1.5",
    batch_size=256,
)

# Text taken from PubMed QA Dataset (https://huggingface.co/datasets/pubmed_qa)
document_list = [
    Document(
        content=("Oxidative stress generated within inflammatory joints can produce autoimmune phenomena and joint "
                 "destruction. Radical species with oxidative activity, including reactive nitrogen species, "
                 "represent mediators of inflammation and cartilage damage."),
        meta={
            "pubid": "25,445,628",
            "long_answer": "yes",
        },
    ),
    Document(
        content=("Plasma levels of pancreatic polypeptide (PP) rise upon food intake. Although other pancreatic "
                 "islet hormones, such as insulin and glucagon, have been extensively investigated, PP secretion "
                 "and actions are still poorly understood."),
        meta={
            "pubid": "25,445,712",
            "long_answer": "yes",
        },
    ),
]

result = doc_embedder.run(document_list)
print(f"Document Text: {result['documents'][0].content}")
print(f"Document Embedding: {result['documents'][0].embedding}")
print(f"Embedding Dimension: {len(result['documents'][0].embedding)}")
```

#### `__init__`

```python
__init__(
    model: str = "BAAI/bge-small-en-v1.5",
    cache_dir: str | None = None,
    threads: int | None = None,
    prefix: str = "",
    suffix: str = "",
    batch_size: int = 256,
    progress_bar: bool = True,
    parallel: int | None = None,
    local_files_only: bool = False,
    meta_fields_to_embed: list[str] | None = None,
    embedding_separator: str = "\n",
) -> None
```

Create an FastembedDocumentEmbedder component.

**Parameters:**

- **model** (<code>str</code>) – Local path or name of the model in Hugging Face's model hub,
  such as `BAAI/bge-small-en-v1.5`.
- **cache_dir** (<code>str | None</code>) – The path to the cache directory.
  Can be set using the `FASTEMBED_CACHE_PATH` env variable.
  Defaults to `fastembed_cache` in the system's temp directory.
- **threads** (<code>int | None</code>) – The number of threads single onnxruntime session can use. Defaults to None.
- **prefix** (<code>str</code>) – A string to add to the beginning of each text.
- **suffix** (<code>str</code>) – A string to add to the end of each text.
- **batch_size** (<code>int</code>) – Number of strings to encode at once.
- **progress_bar** (<code>bool</code>) – If `True`, displays progress bar during embedding.
- **parallel** (<code>int | None</code>) – If > 1, data-parallel encoding will be used, recommended for offline encoding of large datasets.
  If 0, use all available cores.
  If None, don't use data-parallel processing, use default onnxruntime threading instead.
- **local_files_only** (<code>bool</code>) – If `True`, only use the model files in the `cache_dir`.
- **meta_fields_to_embed** (<code>list\[str\] | None</code>) – List of meta fields that should be embedded along with the Document content.
- **embedding_separator** (<code>str</code>) – Separator used to concatenate the meta fields to the Document content.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `warm_up`

```python
warm_up() -> None
```

Initializes the component.

#### `run`

```python
run(documents: list[Document]) -> dict[str, list[Document]]
```

Embeds a list of Documents.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – List of Documents to embed.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: List of Documents with each Document's `embedding` field set to the computed embeddings.

**Raises:**

- <code>TypeError</code> – If the input is not a list of Documents.

## `haystack_integrations.components.embedders.fastembed.fastembed_sparse_document_embedder`

### `FastembedSparseDocumentEmbedder`

FastembedSparseDocumentEmbedder computes Document embeddings using Fastembed sparse models.

Usage example:

```python
from haystack_integrations.components.embedders.fastembed import FastembedSparseDocumentEmbedder
from haystack.dataclasses import Document

sparse_doc_embedder = FastembedSparseDocumentEmbedder(
    model="prithivida/Splade_PP_en_v1",
    batch_size=32,
)

# Text taken from PubMed QA Dataset (https://huggingface.co/datasets/pubmed_qa)
document_list = [
    Document(
        content=("Oxidative stress generated within inflammatory joints can produce autoimmune phenomena and joint "
                 "destruction. Radical species with oxidative activity, including reactive nitrogen species, "
                 "represent mediators of inflammation and cartilage damage."),
        meta={
            "pubid": "25,445,628",
            "long_answer": "yes",
        },
    ),
    Document(
        content=("Plasma levels of pancreatic polypeptide (PP) rise upon food intake. Although other pancreatic "
                 "islet hormones, such as insulin and glucagon, have been extensively investigated, PP secretion "
                 "and actions are still poorly understood."),
        meta={
            "pubid": "25,445,712",
            "long_answer": "yes",
        },
    ),
]

result = sparse_doc_embedder.run(document_list)
print(f"Document Text: {result['documents'][0].content}")
print(f"Document Sparse Embedding: {result['documents'][0].sparse_embedding}")
print(f"Sparse Embedding Dimension: {len(result['documents'][0].sparse_embedding)}")
```

#### `__init__`

```python
__init__(
    model: str = "prithivida/Splade_PP_en_v1",
    cache_dir: str | None = None,
    threads: int | None = None,
    batch_size: int = 32,
    progress_bar: bool = True,
    parallel: int | None = None,
    local_files_only: bool = False,
    meta_fields_to_embed: list[str] | None = None,
    embedding_separator: str = "\n",
    model_kwargs: dict[str, Any] | None = None,
) -> None
```

Create an FastembedDocumentEmbedder component.

**Parameters:**

- **model** (<code>str</code>) – Local path or name of the model in Hugging Face's model hub,
  such as `prithivida/Splade_PP_en_v1`.
- **cache_dir** (<code>str | None</code>) – The path to the cache directory.
  Can be set using the `FASTEMBED_CACHE_PATH` env variable.
  Defaults to `fastembed_cache` in the system's temp directory.
- **threads** (<code>int | None</code>) – The number of threads single onnxruntime session can use.
- **batch_size** (<code>int</code>) – Number of strings to encode at once.
- **progress_bar** (<code>bool</code>) – If `True`, displays progress bar during embedding.
- **parallel** (<code>int | None</code>) – If > 1, data-parallel encoding will be used, recommended for offline encoding of large datasets.
  If 0, use all available cores.
  If None, don't use data-parallel processing, use default onnxruntime threading instead.
- **local_files_only** (<code>bool</code>) – If `True`, only use the model files in the `cache_dir`.
- **meta_fields_to_embed** (<code>list\[str\] | None</code>) – List of meta fields that should be embedded along with the Document content.
- **embedding_separator** (<code>str</code>) – Separator used to concatenate the meta fields to the Document content.
- **model_kwargs** (<code>dict\[str, Any\] | None</code>) – Dictionary containing model parameters such as `k`, `b`, `avg_len`, `language`.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `warm_up`

```python
warm_up() -> None
```

Initializes the component.

#### `run`

```python
run(documents: list[Document]) -> dict[str, list[Document]]
```

Embeds a list of Documents.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – List of Documents to embed.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: List of Documents with each Document's `sparse_embedding`
  field set to the computed embeddings.

**Raises:**

- <code>TypeError</code> – If the input is not a list of Documents.

## `haystack_integrations.components.embedders.fastembed.fastembed_sparse_text_embedder`

### `FastembedSparseTextEmbedder`

FastembedSparseTextEmbedder computes string embedding using fastembed sparse models.

Usage example:

```python
from haystack_integrations.components.embedders.fastembed import FastembedSparseTextEmbedder

text = ("It clearly says online this will work on a Mac OS system. "
        "The disk comes and it does not, only Windows. Do Not order this if you have a Mac!!")

sparse_text_embedder = FastembedSparseTextEmbedder(
    model="prithivida/Splade_PP_en_v1"
)

sparse_embedding = sparse_text_embedder.run(text)["sparse_embedding"]
```

#### `__init__`

```python
__init__(
    model: str = "prithivida/Splade_PP_en_v1",
    cache_dir: str | None = None,
    threads: int | None = None,
    progress_bar: bool = True,
    parallel: int | None = None,
    local_files_only: bool = False,
    model_kwargs: dict[str, Any] | None = None,
) -> None
```

Create a FastembedSparseTextEmbedder component.

**Parameters:**

- **model** (<code>str</code>) – Local path or name of the model in Fastembed's model hub, such as `prithivida/Splade_PP_en_v1`
- **cache_dir** (<code>str | None</code>) – The path to the cache directory.
  Can be set using the `FASTEMBED_CACHE_PATH` env variable.
  Defaults to `fastembed_cache` in the system's temp directory.
- **threads** (<code>int | None</code>) – The number of threads single onnxruntime session can use. Defaults to None.
- **progress_bar** (<code>bool</code>) – If `True`, displays progress bar during embedding.
- **parallel** (<code>int | None</code>) – If > 1, data-parallel encoding will be used, recommended for offline encoding of large datasets.
  If 0, use all available cores.
  If None, don't use data-parallel processing, use default onnxruntime threading instead.
- **local_files_only** (<code>bool</code>) – If `True`, only use the model files in the `cache_dir`.
- **model_kwargs** (<code>dict\[str, Any\] | None</code>) – Dictionary containing model parameters such as `k`, `b`, `avg_len`, `language`.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `warm_up`

```python
warm_up() -> None
```

Initializes the component.

#### `run`

```python
run(text: str) -> dict[str, SparseEmbedding]
```

Embeds text using the Fastembed model.

**Parameters:**

- **text** (<code>str</code>) – A string to embed.

**Returns:**

- <code>dict\[str, SparseEmbedding\]</code> – A dictionary with the following keys:
- `embedding`: A list of floats representing the embedding of the input text.

**Raises:**

- <code>TypeError</code> – If the input is not a string.

## `haystack_integrations.components.embedders.fastembed.fastembed_text_embedder`

### `FastembedTextEmbedder`

FastembedTextEmbedder computes string embedding using fastembed embedding models.

Usage example:

```python
from haystack_integrations.components.embedders.fastembed import FastembedTextEmbedder

text = ("It clearly says online this will work on a Mac OS system. "
        "The disk comes and it does not, only Windows. Do Not order this if you have a Mac!!")

text_embedder = FastembedTextEmbedder(
    model="BAAI/bge-small-en-v1.5"
)

embedding = text_embedder.run(text)["embedding"]
```

#### `__init__`

```python
__init__(
    model: str = "BAAI/bge-small-en-v1.5",
    cache_dir: str | None = None,
    threads: int | None = None,
    prefix: str = "",
    suffix: str = "",
    progress_bar: bool = True,
    parallel: int | None = None,
    local_files_only: bool = False,
) -> None
```

Create a FastembedTextEmbedder component.

**Parameters:**

- **model** (<code>str</code>) – Local path or name of the model in Fastembed's model hub, such as `BAAI/bge-small-en-v1.5`
- **cache_dir** (<code>str | None</code>) – The path to the cache directory.
  Can be set using the `FASTEMBED_CACHE_PATH` env variable.
  Defaults to `fastembed_cache` in the system's temp directory.
- **threads** (<code>int | None</code>) – The number of threads single onnxruntime session can use. Defaults to None.
- **prefix** (<code>str</code>) – A string to add to the beginning of each text.
- **suffix** (<code>str</code>) – A string to add to the end of each text.
- **progress_bar** (<code>bool</code>) – If `True`, displays progress bar during embedding.
- **parallel** (<code>int | None</code>) – If > 1, data-parallel encoding will be used, recommended for offline encoding of large datasets.
  If 0, use all available cores.
  If None, don't use data-parallel processing, use default onnxruntime threading instead.
- **local_files_only** (<code>bool</code>) – If `True`, only use the model files in the `cache_dir`.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `warm_up`

```python
warm_up() -> None
```

Initializes the component.

#### `run`

```python
run(text: str) -> dict[str, list[float]]
```

Embeds text using the Fastembed model.

**Parameters:**

- **text** (<code>str</code>) – A string to embed.

**Returns:**

- <code>dict\[str, list\[float\]\]</code> – A dictionary with the following keys:
- `embedding`: A list of floats representing the embedding of the input text.

**Raises:**

- <code>TypeError</code> – If the input is not a string.

## `haystack_integrations.components.rankers.fastembed.ranker`

### `FastembedRanker`

Ranks Documents based on their similarity to the query using
[Fastembed models](https://qdrant.github.io/fastembed/examples/Supported_Models/).

Documents are indexed from most to least semantically relevant to the query.

Usage example:

```python
from haystack import Document
from haystack_integrations.components.rankers.fastembed import FastembedRanker

ranker = FastembedRanker(model_name="Xenova/ms-marco-MiniLM-L-6-v2", top_k=2)

docs = [Document(content="Paris"), Document(content="Berlin")]
query = "What is the capital of germany?"
output = ranker.run(query=query, documents=docs)
print(output["documents"][0].content)

# Berlin
```

#### `__init__`

```python
__init__(
    model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2",
    top_k: int = 10,
    cache_dir: str | None = None,
    threads: int | None = None,
    batch_size: int = 64,
    parallel: int | None = None,
    local_files_only: bool = False,
    meta_fields_to_embed: list[str] | None = None,
    meta_data_separator: str = "\n",
)
```

Creates an instance of the 'FastembedRanker'.

**Parameters:**

- **model_name** (<code>str</code>) – Fastembed model name. Check the list of supported models in the [Fastembed documentation](https://qdrant.github.io/fastembed/examples/Supported_Models/).
- **top_k** (<code>int</code>) – The maximum number of documents to return.
- **cache_dir** (<code>str | None</code>) – The path to the cache directory.
  Can be set using the `FASTEMBED_CACHE_PATH` env variable.
  Defaults to `fastembed_cache` in the system's temp directory.
- **threads** (<code>int | None</code>) – The number of threads single onnxruntime session can use. Defaults to None.
- **batch_size** (<code>int</code>) – Number of strings to encode at once.
- **parallel** (<code>int | None</code>) – If > 1, data-parallel encoding will be used, recommended for offline encoding of large datasets.
  If 0, use all available cores.
  If None, don't use data-parallel processing, use default onnxruntime threading instead.
- **local_files_only** (<code>bool</code>) – If `True`, only use the model files in the `cache_dir`.
- **meta_fields_to_embed** (<code>list\[str\] | None</code>) – List of meta fields that should be concatenated
  with the document content for reranking.
- **meta_data_separator** (<code>str</code>) – Separator used to concatenate the meta fields
  to the Document content.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> FastembedRanker
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to deserialize from.

**Returns:**

- <code>FastembedRanker</code> – The deserialized component.

#### `warm_up`

```python
warm_up()
```

Initializes the component.

#### `run`

```python
run(
    query: str, documents: list[Document], top_k: int | None = None
) -> dict[str, list[Document]]
```

Returns a list of documents ranked by their similarity to the given query, using FastEmbed.

**Parameters:**

- **query** (<code>str</code>) – The input query to compare the documents to.
- **documents** (<code>list\[Document\]</code>) – A list of documents to be ranked.
- **top_k** (<code>int | None</code>) – The maximum number of documents to return.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: A list of documents closest to the query, sorted from most similar to least similar.

**Raises:**

- <code>ValueError</code> – If `top_k` is not > 0.
