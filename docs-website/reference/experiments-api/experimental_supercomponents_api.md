---
title: "SuperComponents"
id: experimental-supercomponents-api
description: "Pipelines wrapped as components."
slug: "/experimental-supercomponents-api"
---

<a id="haystack_experimental.super_components.indexers.sentence_transformers_document_indexer"></a>

## Module haystack\_experimental.super\_components.indexers.sentence\_transformers\_document\_indexer

<a id="haystack_experimental.super_components.indexers.sentence_transformers_document_indexer.SentenceTransformersDocumentIndexer"></a>

### SentenceTransformersDocumentIndexer

A document indexer that takes a list of documents, embeds them using SentenceTransformers, and stores them.

Usage:

```python
>>> from haystack import Document
>>> from haystack.document_stores.in_memory import InMemoryDocumentStore
>>> document_store = InMemoryDocumentStore()
>>> doc = Document(content="I love pizza!")
>>> indexer = SentenceTransformersDocumentIndexer(document_store=document_store)
>>> indexer.warm_up()
>>> result = indexer.run(documents=[doc])
>>> print(result)
{'documents_written': 1}
>>> document_store.count_documents()
1
```

<a id="haystack_experimental.super_components.indexers.sentence_transformers_document_indexer.SentenceTransformersDocumentIndexer.__init__"></a>

#### SentenceTransformersDocumentIndexer.\_\_init\_\_

```python
def __init__(
        document_store: DocumentStore,
        model: str = "sentence-transformers/all-mpnet-base-v2",
        device: Optional[ComponentDevice] = None,
        token: Optional[Secret] = Secret.from_env_var(
            ["HF_API_TOKEN", "HF_TOKEN"], strict=False),
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 32,
        progress_bar: bool = True,
        normalize_embeddings: bool = False,
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
        trust_remote_code: bool = False,
        truncate_dim: Optional[int] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        config_kwargs: Optional[Dict[str, Any]] = None,
        precision: Literal["float32", "int8", "uint8", "binary",
                           "ubinary"] = "float32",
        duplicate_policy: DuplicatePolicy = DuplicatePolicy.OVERWRITE) -> None
```

Initialize the SentenceTransformersDocumentIndexer component.

**Arguments**:

- `document_store`: The document store where the documents should be stored.
- `model`: The embedding model to use (local path or Hugging Face model ID).
- `device`: The device to use for loading the model.
- `token`: The API token to download private models from Hugging Face.
- `prefix`: String to add at the beginning of each document text.
- `suffix`: String to add at the end of each document text.
- `batch_size`: Number of documents to embed at once.
- `progress_bar`: If True, shows a progress bar when embedding documents.
- `normalize_embeddings`: If True, embeddings are L2 normalized.
- `meta_fields_to_embed`: List of metadata fields to embed along with the document text.
- `embedding_separator`: Separator used to concatenate metadata fields to document text.
- `trust_remote_code`: If True, allows custom models and scripts.
- `truncate_dim`: Dimension to truncate sentence embeddings to.
- `model_kwargs`: Additional keyword arguments for model initialization.
- `tokenizer_kwargs`: Additional keyword arguments for tokenizer initialization.
- `config_kwargs`: Additional keyword arguments for model configuration.
- `precision`: The precision to use for the embeddings.
- `duplicate_policy`: The duplicate policy to use when writing documents.

<a id="haystack_experimental.super_components.indexers.sentence_transformers_document_indexer.SentenceTransformersDocumentIndexer.to_dict"></a>

#### SentenceTransformersDocumentIndexer.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serialize this instance to a dictionary.

<a id="haystack_experimental.super_components.indexers.sentence_transformers_document_indexer.SentenceTransformersDocumentIndexer.from_dict"></a>

#### SentenceTransformersDocumentIndexer.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str,
                              Any]) -> "SentenceTransformersDocumentIndexer"
```

Load an instance of this component from a dictionary.

