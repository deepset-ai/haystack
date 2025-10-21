---
title: "FastEmbed"
id: fastembed-embedders
description: "FastEmbed integration for Haystack"
slug: "/fastembed-embedders"
---

<a id="haystack_integrations.components.embedders.fastembed.fastembed_document_embedder"></a>

# Module haystack\_integrations.components.embedders.fastembed.fastembed\_document\_embedder

<a id="haystack_integrations.components.embedders.fastembed.fastembed_document_embedder.FastembedDocumentEmbedder"></a>

## FastembedDocumentEmbedder

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

doc_embedder.warm_up()

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

<a id="haystack_integrations.components.embedders.fastembed.fastembed_document_embedder.FastembedDocumentEmbedder.__init__"></a>

#### FastembedDocumentEmbedder.\_\_init\_\_

```python
def __init__(model: str = "BAAI/bge-small-en-v1.5",
             cache_dir: Optional[str] = None,
             threads: Optional[int] = None,
             prefix: str = "",
             suffix: str = "",
             batch_size: int = 256,
             progress_bar: bool = True,
             parallel: Optional[int] = None,
             local_files_only: bool = False,
             meta_fields_to_embed: Optional[List[str]] = None,
             embedding_separator: str = "\n")
```

Create an FastembedDocumentEmbedder component.

**Arguments**:

- `model`: Local path or name of the model in Hugging Face's model hub,
such as `BAAI/bge-small-en-v1.5`.
- `cache_dir`: The path to the cache directory.
Can be set using the `FASTEMBED_CACHE_PATH` env variable.
Defaults to `fastembed_cache` in the system's temp directory.
- `threads`: The number of threads single onnxruntime session can use. Defaults to None.
- `prefix`: A string to add to the beginning of each text.
- `suffix`: A string to add to the end of each text.
- `batch_size`: Number of strings to encode at once.
- `progress_bar`: If `True`, displays progress bar during embedding.
- `parallel`: If > 1, data-parallel encoding will be used, recommended for offline encoding of large datasets.
If 0, use all available cores.
If None, don't use data-parallel processing, use default onnxruntime threading instead.
- `local_files_only`: If `True`, only use the model files in the `cache_dir`.
- `meta_fields_to_embed`: List of meta fields that should be embedded along with the Document content.
- `embedding_separator`: Separator used to concatenate the meta fields to the Document content.

<a id="haystack_integrations.components.embedders.fastembed.fastembed_document_embedder.FastembedDocumentEmbedder.to_dict"></a>

#### FastembedDocumentEmbedder.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.embedders.fastembed.fastembed_document_embedder.FastembedDocumentEmbedder.warm_up"></a>

#### FastembedDocumentEmbedder.warm\_up

```python
def warm_up()
```

Initializes the component.

<a id="haystack_integrations.components.embedders.fastembed.fastembed_document_embedder.FastembedDocumentEmbedder.run"></a>

#### FastembedDocumentEmbedder.run

```python
@component.output_types(documents=List[Document])
def run(documents: List[Document]) -> Dict[str, List[Document]]
```

Embeds a list of Documents.

**Arguments**:

- `documents`: List of Documents to embed.

**Returns**:

A dictionary with the following keys:
- `documents`: List of Documents with each Document's `embedding` field set to the computed embeddings.

<a id="haystack_integrations.components.embedders.fastembed.fastembed_text_embedder"></a>

# Module haystack\_integrations.components.embedders.fastembed.fastembed\_text\_embedder

<a id="haystack_integrations.components.embedders.fastembed.fastembed_text_embedder.FastembedTextEmbedder"></a>

## FastembedTextEmbedder

FastembedTextEmbedder computes string embedding using fastembed embedding models.

Usage example:
```python
from haystack_integrations.components.embedders.fastembed import FastembedTextEmbedder

text = ("It clearly says online this will work on a Mac OS system. "
        "The disk comes and it does not, only Windows. Do Not order this if you have a Mac!!")

text_embedder = FastembedTextEmbedder(
    model="BAAI/bge-small-en-v1.5"
)
text_embedder.warm_up()

embedding = text_embedder.run(text)["embedding"]
```

<a id="haystack_integrations.components.embedders.fastembed.fastembed_text_embedder.FastembedTextEmbedder.__init__"></a>

#### FastembedTextEmbedder.\_\_init\_\_

```python
def __init__(model: str = "BAAI/bge-small-en-v1.5",
             cache_dir: Optional[str] = None,
             threads: Optional[int] = None,
             prefix: str = "",
             suffix: str = "",
             progress_bar: bool = True,
             parallel: Optional[int] = None,
             local_files_only: bool = False)
```

Create a FastembedTextEmbedder component.

**Arguments**:

- `model`: Local path or name of the model in Fastembed's model hub, such as `BAAI/bge-small-en-v1.5`
- `cache_dir`: The path to the cache directory.
Can be set using the `FASTEMBED_CACHE_PATH` env variable.
Defaults to `fastembed_cache` in the system's temp directory.
- `threads`: The number of threads single onnxruntime session can use. Defaults to None.
- `prefix`: A string to add to the beginning of each text.
- `suffix`: A string to add to the end of each text.
- `progress_bar`: If `True`, displays progress bar during embedding.
- `parallel`: If > 1, data-parallel encoding will be used, recommended for offline encoding of large datasets.
If 0, use all available cores.
If None, don't use data-parallel processing, use default onnxruntime threading instead.
- `local_files_only`: If `True`, only use the model files in the `cache_dir`.

<a id="haystack_integrations.components.embedders.fastembed.fastembed_text_embedder.FastembedTextEmbedder.to_dict"></a>

#### FastembedTextEmbedder.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.embedders.fastembed.fastembed_text_embedder.FastembedTextEmbedder.warm_up"></a>

#### FastembedTextEmbedder.warm\_up

```python
def warm_up()
```

Initializes the component.

<a id="haystack_integrations.components.embedders.fastembed.fastembed_text_embedder.FastembedTextEmbedder.run"></a>

#### FastembedTextEmbedder.run

```python
@component.output_types(embedding=List[float])
def run(text: str) -> Dict[str, List[float]]
```

Embeds text using the Fastembed model.

**Arguments**:

- `text`: A string to embed.

**Raises**:

- `TypeError`: If the input is not a string.
- `RuntimeError`: If the embedding model has not been loaded.

**Returns**:

A dictionary with the following keys:
- `embedding`: A list of floats representing the embedding of the input text.

<a id="haystack_integrations.components.embedders.fastembed.fastembed_sparse_document_embedder"></a>

# Module haystack\_integrations.components.embedders.fastembed.fastembed\_sparse\_document\_embedder

<a id="haystack_integrations.components.embedders.fastembed.fastembed_sparse_document_embedder.FastembedSparseDocumentEmbedder"></a>

## FastembedSparseDocumentEmbedder

FastembedSparseDocumentEmbedder computes Document embeddings using Fastembed sparse models.

Usage example:
```python
from haystack_integrations.components.embedders.fastembed import FastembedSparseDocumentEmbedder
from haystack.dataclasses import Document

sparse_doc_embedder = FastembedSparseDocumentEmbedder(
    model="prithivida/Splade_PP_en_v1",
    batch_size=32,
)

sparse_doc_embedder.warm_up()

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

<a id="haystack_integrations.components.embedders.fastembed.fastembed_sparse_document_embedder.FastembedSparseDocumentEmbedder.__init__"></a>

#### FastembedSparseDocumentEmbedder.\_\_init\_\_

```python
def __init__(model: str = "prithivida/Splade_PP_en_v1",
             cache_dir: Optional[str] = None,
             threads: Optional[int] = None,
             batch_size: int = 32,
             progress_bar: bool = True,
             parallel: Optional[int] = None,
             local_files_only: bool = False,
             meta_fields_to_embed: Optional[List[str]] = None,
             embedding_separator: str = "\n",
             model_kwargs: Optional[Dict[str, Any]] = None)
```

Create an FastembedDocumentEmbedder component.

**Arguments**:

- `model`: Local path or name of the model in Hugging Face's model hub,
such as `prithivida/Splade_PP_en_v1`.
- `cache_dir`: The path to the cache directory.
Can be set using the `FASTEMBED_CACHE_PATH` env variable.
Defaults to `fastembed_cache` in the system's temp directory.
- `threads`: The number of threads single onnxruntime session can use.
- `batch_size`: Number of strings to encode at once.
- `progress_bar`: If `True`, displays progress bar during embedding.
- `parallel`: If > 1, data-parallel encoding will be used, recommended for offline encoding of large datasets.
If 0, use all available cores.
If None, don't use data-parallel processing, use default onnxruntime threading instead.
- `local_files_only`: If `True`, only use the model files in the `cache_dir`.
- `meta_fields_to_embed`: List of meta fields that should be embedded along with the Document content.
- `embedding_separator`: Separator used to concatenate the meta fields to the Document content.
- `model_kwargs`: Dictionary containing model parameters such as `k`, `b`, `avg_len`, `language`.

<a id="haystack_integrations.components.embedders.fastembed.fastembed_sparse_document_embedder.FastembedSparseDocumentEmbedder.to_dict"></a>

#### FastembedSparseDocumentEmbedder.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.embedders.fastembed.fastembed_sparse_document_embedder.FastembedSparseDocumentEmbedder.warm_up"></a>

#### FastembedSparseDocumentEmbedder.warm\_up

```python
def warm_up()
```

Initializes the component.

<a id="haystack_integrations.components.embedders.fastembed.fastembed_sparse_document_embedder.FastembedSparseDocumentEmbedder.run"></a>

#### FastembedSparseDocumentEmbedder.run

```python
@component.output_types(documents=List[Document])
def run(documents: List[Document]) -> Dict[str, List[Document]]
```

Embeds a list of Documents.

**Arguments**:

- `documents`: List of Documents to embed.

**Returns**:

A dictionary with the following keys:
- `documents`: List of Documents with each Document's `sparse_embedding`
field set to the computed embeddings.

<a id="haystack_integrations.components.embedders.fastembed.fastembed_sparse_text_embedder"></a>

# Module haystack\_integrations.components.embedders.fastembed.fastembed\_sparse\_text\_embedder

<a id="haystack_integrations.components.embedders.fastembed.fastembed_sparse_text_embedder.FastembedSparseTextEmbedder"></a>

## FastembedSparseTextEmbedder

FastembedSparseTextEmbedder computes string embedding using fastembed sparse models.

Usage example:
```python
from haystack_integrations.components.embedders.fastembed import FastembedSparseTextEmbedder

text = ("It clearly says online this will work on a Mac OS system. "
        "The disk comes and it does not, only Windows. Do Not order this if you have a Mac!!")

sparse_text_embedder = FastembedSparseTextEmbedder(
    model="prithivida/Splade_PP_en_v1"
)
sparse_text_embedder.warm_up()

sparse_embedding = sparse_text_embedder.run(text)["sparse_embedding"]
```

<a id="haystack_integrations.components.embedders.fastembed.fastembed_sparse_text_embedder.FastembedSparseTextEmbedder.__init__"></a>

#### FastembedSparseTextEmbedder.\_\_init\_\_

```python
def __init__(model: str = "prithivida/Splade_PP_en_v1",
             cache_dir: Optional[str] = None,
             threads: Optional[int] = None,
             progress_bar: bool = True,
             parallel: Optional[int] = None,
             local_files_only: bool = False,
             model_kwargs: Optional[Dict[str, Any]] = None)
```

Create a FastembedSparseTextEmbedder component.

**Arguments**:

- `model`: Local path or name of the model in Fastembed's model hub, such as `prithivida/Splade_PP_en_v1`
- `cache_dir`: The path to the cache directory.
Can be set using the `FASTEMBED_CACHE_PATH` env variable.
Defaults to `fastembed_cache` in the system's temp directory.
- `threads`: The number of threads single onnxruntime session can use. Defaults to None.
- `progress_bar`: If `True`, displays progress bar during embedding.
- `parallel`: If > 1, data-parallel encoding will be used, recommended for offline encoding of large datasets.
If 0, use all available cores.
If None, don't use data-parallel processing, use default onnxruntime threading instead.
- `local_files_only`: If `True`, only use the model files in the `cache_dir`.
- `model_kwargs`: Dictionary containing model parameters such as `k`, `b`, `avg_len`, `language`.

<a id="haystack_integrations.components.embedders.fastembed.fastembed_sparse_text_embedder.FastembedSparseTextEmbedder.to_dict"></a>

#### FastembedSparseTextEmbedder.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.embedders.fastembed.fastembed_sparse_text_embedder.FastembedSparseTextEmbedder.warm_up"></a>

#### FastembedSparseTextEmbedder.warm\_up

```python
def warm_up()
```

Initializes the component.

<a id="haystack_integrations.components.embedders.fastembed.fastembed_sparse_text_embedder.FastembedSparseTextEmbedder.run"></a>

#### FastembedSparseTextEmbedder.run

```python
@component.output_types(sparse_embedding=SparseEmbedding)
def run(text: str) -> Dict[str, SparseEmbedding]
```

Embeds text using the Fastembed model.

**Arguments**:

- `text`: A string to embed.

**Raises**:

- `TypeError`: If the input is not a string.
- `RuntimeError`: If the embedding model has not been loaded.

**Returns**:

A dictionary with the following keys:
- `embedding`: A list of floats representing the embedding of the input text.

<a id="haystack_integrations.components.rankers.fastembed.ranker"></a>

# Module haystack\_integrations.components.rankers.fastembed.ranker

<a id="haystack_integrations.components.rankers.fastembed.ranker.FastembedRanker"></a>

## FastembedRanker

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

<a id="haystack_integrations.components.rankers.fastembed.ranker.FastembedRanker.__init__"></a>

#### FastembedRanker.\_\_init\_\_

```python
def __init__(model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2",
             top_k: int = 10,
             cache_dir: Optional[str] = None,
             threads: Optional[int] = None,
             batch_size: int = 64,
             parallel: Optional[int] = None,
             local_files_only: bool = False,
             meta_fields_to_embed: Optional[List[str]] = None,
             meta_data_separator: str = "\n")
```

Creates an instance of the 'FastembedRanker'.

**Arguments**:

- `model_name`: Fastembed model name. Check the list of supported models in the [Fastembed documentation](https://qdrant.github.io/fastembed/examples/Supported_Models/).
- `top_k`: The maximum number of documents to return.
- `cache_dir`: The path to the cache directory.
Can be set using the `FASTEMBED_CACHE_PATH` env variable.
Defaults to `fastembed_cache` in the system's temp directory.
- `threads`: The number of threads single onnxruntime session can use. Defaults to None.
- `batch_size`: Number of strings to encode at once.
- `parallel`: If > 1, data-parallel encoding will be used, recommended for offline encoding of large datasets.
If 0, use all available cores.
If None, don't use data-parallel processing, use default onnxruntime threading instead.
- `local_files_only`: If `True`, only use the model files in the `cache_dir`.
- `meta_fields_to_embed`: List of meta fields that should be concatenated
with the document content for reranking.
- `meta_data_separator`: Separator used to concatenate the meta fields
to the Document content.

<a id="haystack_integrations.components.rankers.fastembed.ranker.FastembedRanker.to_dict"></a>

#### FastembedRanker.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.rankers.fastembed.ranker.FastembedRanker.from_dict"></a>

#### FastembedRanker.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "FastembedRanker"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: The dictionary to deserialize from.

**Returns**:

The deserialized component.

<a id="haystack_integrations.components.rankers.fastembed.ranker.FastembedRanker.warm_up"></a>

#### FastembedRanker.warm\_up

```python
def warm_up()
```

Initializes the component.

<a id="haystack_integrations.components.rankers.fastembed.ranker.FastembedRanker.run"></a>

#### FastembedRanker.run

```python
@component.output_types(documents=List[Document])
def run(query: str,
        documents: List[Document],
        top_k: Optional[int] = None) -> Dict[str, List[Document]]
```

Returns a list of documents ranked by their similarity to the given query, using FastEmbed.

**Arguments**:

- `query`: The input query to compare the documents to.
- `documents`: A list of documents to be ranked.
- `top_k`: The maximum number of documents to return.

**Raises**:

- `ValueError`: If `top_k` is not > 0.

**Returns**:

A dictionary with the following keys:
- `documents`: A list of documents closest to the query, sorted from most similar to least similar.
