---
title: "Solr"
id: integrations-solr
description: "Solr integration for Haystack"
slug: "/integrations-solr"
---


## haystack_integrations.components.retrievers.solr.bm25_retriever

### SolrBM25Retriever

Fetches documents from a `SolrDocumentStore` using Solr's BM25 similarity.

Usage example:

```python
from haystack_integrations.document_stores.solr import SolrDocumentStore
from haystack_integrations.components.retrievers.solr import SolrBM25Retriever

document_store = SolrDocumentStore(core="haystack")
retriever = SolrBM25Retriever(document_store=document_store)
result = retriever.run(query="Apache Solr")
```

#### __init__

```python
__init__(
    *,
    document_store: SolrDocumentStore,
    filters: dict[str, Any] | None = None,
    fuzziness: int = 0,
    top_k: int = 10,
    scale_score: bool = False,
    all_terms_must_match: bool = False,
    filter_policy: str | FilterPolicy = FilterPolicy.REPLACE,
    raise_on_failure: bool = True
) -> None
```

Create a `SolrBM25Retriever`.

**Parameters:**

- **document_store** (<code>SolrDocumentStore</code>) – the document store to search.
- **filters** (<code>dict\[str, Any\] | None</code>) – filters applied to the search. Combined with the filters passed to `run`
  according to `filter_policy`.
- **fuzziness** (<code>int</code>) – per-term edit distance. `0`, the default, disables fuzzy matching.
- **top_k** (<code>int</code>) – maximum number of documents to return.
- **scale_score** (<code>bool</code>) – whether to scale scores into the `(0, 1)` range.
- **all_terms_must_match** (<code>bool</code>) – whether every query term must match.
- **filter_policy** (<code>str | FilterPolicy</code>) – how runtime filters combine with the filters given here.
- **raise_on_failure** (<code>bool</code>) – whether a failing search raises, or logs and returns no documents.

**Raises:**

- <code>ValueError</code> – if `document_store` is not a `SolrDocumentStore`, or `top_k` is not positive.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> SolrBM25Retriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – dictionary to deserialize from.

**Returns:**

- <code>SolrBM25Retriever</code> – deserialized component.

#### run

```python
run(
    query: str,
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
    fuzziness: int | None = None,
    scale_score: bool | None = None,
    all_terms_must_match: bool | None = None,
) -> dict[str, list[Document]]
```

Retrieve documents matching `query`.

**Parameters:**

- **query** (<code>str</code>) – the query string.
- **filters** (<code>dict\[str, Any\] | None</code>) – filters applied to the search.
- **top_k** (<code>int | None</code>) – maximum number of documents to return.
- **fuzziness** (<code>int | None</code>) – per-term edit distance.
- **scale_score** (<code>bool | None</code>) – whether to scale scores into the `(0, 1)` range.
- **all_terms_must_match** (<code>bool | None</code>) – whether every query term must match.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – a dictionary with a `documents` key holding the retrieved documents.

**Raises:**

- <code>ValueError</code> – if `top_k` is not positive.

#### run_async

```python
run_async(
    query: str,
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
    fuzziness: int | None = None,
    scale_score: bool | None = None,
    all_terms_must_match: bool | None = None,
) -> dict[str, list[Document]]
```

Retrieve documents matching `query`, asynchronously.

**Parameters:**

- **query** (<code>str</code>) – the query string.
- **filters** (<code>dict\[str, Any\] | None</code>) – filters applied to the search.
- **top_k** (<code>int | None</code>) – maximum number of documents to return.
- **fuzziness** (<code>int | None</code>) – per-term edit distance.
- **scale_score** (<code>bool | None</code>) – whether to scale scores into the `(0, 1)` range.
- **all_terms_must_match** (<code>bool | None</code>) – whether every query term must match.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – a dictionary with a `documents` key holding the retrieved documents.

**Raises:**

- <code>ValueError</code> – if `top_k` is not positive.

#### close

```python
close() -> None
```

Close the underlying document store connection.

#### close_async

```python
close_async() -> None
```

Close the underlying document store async connection.

## haystack_integrations.components.retrievers.solr.embedding_retriever

### SolrEmbeddingRetriever

Fetches documents from a `SolrDocumentStore` using Solr's `{!knn}` dense vector search.

Usage example:

```python
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.document_stores.solr import SolrDocumentStore
from haystack_integrations.components.retrievers.solr import SolrEmbeddingRetriever

document_store = SolrDocumentStore(core="haystack", embedding_dim=384)
embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")

pipeline = Pipeline()
pipeline.add_component("embedder", embedder)
pipeline.add_component("retriever", SolrEmbeddingRetriever(document_store=document_store))
pipeline.connect("embedder.embedding", "retriever.query_embedding")

result = pipeline.run(data={"embedder": {"text": "Apache Solr"}})
```

#### __init__

```python
__init__(
    *,
    document_store: SolrDocumentStore,
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
    filter_policy: str | FilterPolicy = FilterPolicy.REPLACE,
    raise_on_failure: bool = True
) -> None
```

Create a `SolrEmbeddingRetriever`.

**Parameters:**

- **document_store** (<code>SolrDocumentStore</code>) – the document store to search.
- **filters** (<code>dict\[str, Any\] | None</code>) – filters applied to the search. Combined with the filters passed to `run`
  according to `filter_policy`. Filters act as a k-NN graph pre-filter, so the search still
  returns up to `top_k` documents.
- **top_k** (<code>int</code>) – maximum number of documents to return.
- **filter_policy** (<code>str | FilterPolicy</code>) – how runtime filters combine with the filters given here.
- **raise_on_failure** (<code>bool</code>) – whether a failing search raises, or logs and returns no documents.

**Raises:**

- <code>ValueError</code> – if `document_store` is not a `SolrDocumentStore`, or `top_k` is not positive.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> SolrEmbeddingRetriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – dictionary to deserialize from.

**Returns:**

- <code>SolrEmbeddingRetriever</code> – deserialized component.

#### run

```python
run(
    query_embedding: list[float],
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
) -> dict[str, list[Document]]
```

Retrieve documents similar to `query_embedding`.

**Parameters:**

- **query_embedding** (<code>list\[float\]</code>) – the query embedding.
- **filters** (<code>dict\[str, Any\] | None</code>) – filters applied to the search.
- **top_k** (<code>int | None</code>) – maximum number of documents to return.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – a dictionary with a `documents` key holding the retrieved documents.

**Raises:**

- <code>ValueError</code> – if `top_k` is not positive.

#### run_async

```python
run_async(
    query_embedding: list[float],
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
) -> dict[str, list[Document]]
```

Retrieve documents similar to `query_embedding`, asynchronously.

**Parameters:**

- **query_embedding** (<code>list\[float\]</code>) – the query embedding.
- **filters** (<code>dict\[str, Any\] | None</code>) – filters applied to the search.
- **top_k** (<code>int | None</code>) – maximum number of documents to return.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – a dictionary with a `documents` key holding the retrieved documents.

**Raises:**

- <code>ValueError</code> – if `top_k` is not positive.

#### close

```python
close() -> None
```

Close the underlying document store connection.

#### close_async

```python
close_async() -> None
```

Close the underlying document store async connection.

## haystack_integrations.components.retrievers.solr.solr_hybrid_retriever

### SolrHybridRetriever

Hybrid retrieval over a `SolrDocumentStore`, combining BM25 and dense vector search.

Wraps a pipeline that embeds the query, runs a BM25 and an embedding retriever over the same core,
and fuses the two result lists with a `DocumentJoiner`.

Usage example:

```python
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.document_stores.solr import SolrDocumentStore
from haystack_integrations.components.retrievers.solr import SolrHybridRetriever

document_store = SolrDocumentStore(core="haystack", embedding_dim=384)
retriever = SolrHybridRetriever(
    document_store=document_store,
    embedder=SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"),
)
retriever.warm_up()
result = retriever.run(query="Apache Solr")
```

#### __init__

```python
__init__(
    document_store: SolrDocumentStore,
    *,
    embedder: TextEmbedder,
    filters_bm25: dict[str, Any] | None = None,
    fuzziness: int = 0,
    top_k_bm25: int = 10,
    scale_score: bool = False,
    all_terms_must_match: bool = False,
    filter_policy_bm25: str | FilterPolicy = FilterPolicy.REPLACE,
    filters_embedding: dict[str, Any] | None = None,
    top_k_embedding: int = 10,
    filter_policy_embedding: str | FilterPolicy = FilterPolicy.REPLACE,
    join_mode: str | JoinMode = JoinMode.RECIPROCAL_RANK_FUSION,
    weights: list[float] | None = None,
    top_k: int | None = None,
    sort_by_score: bool = True,
    **kwargs: Any
) -> None
```

Create a `SolrHybridRetriever`.

**Parameters:**

- **document_store** (<code>SolrDocumentStore</code>) – the document store both retrievers search.
- **embedder** (<code>TextEmbedder</code>) – the text embedder turning the query into a vector.
- **filters_bm25** (<code>dict\[str, Any\] | None</code>) – filters for the BM25 branch.
- **fuzziness** (<code>int</code>) – per-term edit distance for the BM25 branch.
- **top_k_bm25** (<code>int</code>) – maximum number of documents from the BM25 branch.
- **scale_score** (<code>bool</code>) – whether to scale BM25 scores into the `(0, 1)` range.
- **all_terms_must_match** (<code>bool</code>) – whether every query term must match in the BM25 branch.
- **filter_policy_bm25** (<code>str | FilterPolicy</code>) – filter policy for the BM25 branch.
- **filters_embedding** (<code>dict\[str, Any\] | None</code>) – filters for the embedding branch.
- **top_k_embedding** (<code>int</code>) – maximum number of documents from the embedding branch.
- **filter_policy_embedding** (<code>str | FilterPolicy</code>) – filter policy for the embedding branch.
- **join_mode** (<code>str | JoinMode</code>) – how the two result lists are fused.
- **weights** (<code>list\[float\] | None</code>) – per-branch weights used by the joiner.
- **top_k** (<code>int | None</code>) – maximum number of documents returned after fusion.
- **sort_by_score** (<code>bool</code>) – whether the fused documents are sorted by score.
- **kwargs** (<code>Any</code>) – extra init arguments for the underlying retrievers, given as
  `bm25_retriever={...}` and/or `embedding_retriever={...}`.

**Raises:**

- <code>ValueError</code> – if `kwargs` contains a key other than those two.

#### warm_up

```python
warm_up() -> None
```

Warm up the underlying pipeline components.

#### run

```python
run(
    query: str,
    filters_bm25: dict[str, Any] | None = None,
    filters_embedding: dict[str, Any] | None = None,
    top_k_bm25: int | None = None,
    top_k_embedding: int | None = None,
) -> dict[str, list[Document]]
```

Run the hybrid retrieval pipeline and return the retrieved documents.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> SolrHybridRetriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – dictionary to deserialize from.

**Returns:**

- <code>SolrHybridRetriever</code> – deserialized component.

#### close

```python
close() -> None
```

Close the underlying document store connection.

#### close_async

```python
close_async() -> None
```

Close the underlying document store async connection.

## haystack_integrations.document_stores.solr.document_store

### SolrDocumentStore

A Document Store for [Apache Solr](https://solr.apache.org/).

Supports keyword search through Solr's BM25 similarity and dense vector search through
`DenseVectorField` and the `{!knn}` query parser. Requires **Solr 9.6 or newer**.

Usage example:

```python
from haystack import Document
from haystack_integrations.document_stores.solr import SolrDocumentStore

store = SolrDocumentStore(url="http://localhost:8983/solr", core="haystack", embedding_dim=768)
store.write_documents([Document(content="Apache Solr is a search platform.")])
```

Metadata is stored in Solr fields whose names encode the Python type of the value, so metadata
round-trips with its type intact. See the `schema` module for the details of that mapping. Metadata
keys become Solr field names and must therefore consist of letters, digits and underscores.

Two things Solr cannot do:

- `Document.sparse_embedding` is ignored, with a warning, because Solr has no sparse vector field.
- Comparing `content` with `==` is a phrase match against an analysed field rather than exact
  string equality. Filter on a metadata field when exact matching matters.

#### __init__

```python
__init__(
    *,
    url: str | None = None,
    core: str = "haystack",
    embedding_dim: int = 768,
    similarity_function: Literal[
        "cosine", "dot_product", "euclidean"
    ] = "cosine",
    return_embedding: bool = False,
    create_core: bool = False,
    manage_schema: bool = True,
    config_set: str = "_default",
    vector_field_type_params: dict[str, Any] | None = None,
    auth: tuple[Secret, Secret] | tuple[str, str] | None = (
        Secret.from_env_var("SOLR_USERNAME", strict=False),
        Secret.from_env_var("SOLR_PASSWORD", strict=False),
    ),
    verify_certs: bool = True,
    timeout: float = 30.0,
    batch_size: int = DEFAULT_BATCH_SIZE,
    commit: bool = True,
    commit_within_ms: int | None = None,
    query_page_size: int = DEFAULT_QUERY_PAGE_SIZE,
    **kwargs: Any
) -> None
```

Create a new `SolrDocumentStore`.

**Parameters:**

- **url** (<code>str | None</code>) – Solr base URL. Falls back to the `SOLR_URL` environment variable, then to
  `http://localhost:8983/solr`.
- **core** (<code>str</code>) – name of the Solr core (or SolrCloud collection) to read from and write to.
- **embedding_dim** (<code>int</code>) – dimension of the embeddings. Solr fixes a vector field's dimension when
  the field is created, so this cannot be changed for an existing core.
- **similarity_function** (<code>Literal['cosine', 'dot_product', 'euclidean']</code>) – vector similarity to use, one of `cosine`, `dot_product` or
  `euclidean`.
- **return_embedding** (<code>bool</code>) – whether `filter_documents` and the retrievers return embeddings.
  Leaving this `False` keeps large vectors off the wire.
- **create_core** (<code>bool</code>) – whether to create the core if it does not exist. Requires the `config_set`
  to be present in Solr's configset directory (`<solr_home>/configsets`), which is not the
  case for a stock installation, so this defaults to `False` and most deployments should
  create the core out of band.
- **manage_schema** (<code>bool</code>) – whether to create the fields the document store needs and disable Solr's
  schemaless field guessing. Set to `False` to manage the schema yourself, in which case
  `schema.schema_payload` is the definitive list of the fields and dynamic fields required.
- **config_set** (<code>str</code>) – configset used when `create_core` is enabled.
- **vector_field_type_params** (<code>dict\[str, Any\] | None</code>) – extra attributes for the vector field type, for example
  `{"hnswM": 32}` on Solr 10 or `{"hnswMaxConnections": 32}` on Solr 9. Left unset by default
  because Solr 10 renamed these attributes without a compatibility shim.
- **auth** (<code>tuple\[Secret, Secret\] | tuple\[str, str\] | None</code>) – username and password for basic authentication. Reads the `SOLR_USERNAME` and
  `SOLR_PASSWORD` environment variables by default. Pass `None` to disable authentication.
- **verify_certs** (<code>bool</code>) – whether to verify TLS certificates.
- **timeout** (<code>float</code>) – request timeout in seconds.
- **batch_size** (<code>int</code>) – number of documents sent per update request.
- **commit** (<code>bool</code>) – whether writes and deletes commit immediately, making them searchable at once.
- **commit_within_ms** (<code>int | None</code>) – ask Solr to commit within this many milliseconds instead of blocking.
- **query_page_size** (<code>int</code>) – number of documents fetched per page when paginating.
- **kwargs** (<code>Any</code>) – extra keyword arguments forwarded to the underlying `httpx` clients, for
  example `proxy` or `headers`.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> SolrDocumentStore
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – dictionary to deserialize from.

**Returns:**

- <code>SolrDocumentStore</code> – deserialized component.

#### close

```python
close() -> None
```

Close the underlying HTTP client. The store reconnects on the next call.

#### close_async

```python
close_async() -> None
```

Close the underlying async HTTP client. The store reconnects on the next call.

#### count_documents

```python
count_documents() -> int
```

Returns how many documents are present in the document store.

**Returns:**

- <code>int</code> – the number of documents.

#### count_documents_async

```python
count_documents_async() -> int
```

Returns how many documents are present in the document store.

**Returns:**

- <code>int</code> – the number of documents.

#### count_documents_by_filter

```python
count_documents_by_filter(filters: dict[str, Any]) -> int
```

Returns how many documents match the given filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – the filters to apply.

**Returns:**

- <code>int</code> – the number of matching documents.

#### count_documents_by_filter_async

```python
count_documents_by_filter_async(filters: dict[str, Any]) -> int
```

Returns how many documents match the given filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – the filters to apply.

**Returns:**

- <code>int</code> – the number of matching documents.

#### filter_documents

```python
filter_documents(filters: dict[str, Any] | None = None) -> list[Document]
```

Returns the documents that match the filters provided.

For a detailed specification of the filters, refer to the
[documentation](https://docs.haystack.deepset.ai/docs/metadata-filtering).

All Haystack operators are supported: `==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `not in`, and the
`AND`, `OR` and `NOT` logical operators. Three behaviours are worth knowing:

- `>`, `>=`, `<` and `<=` accept numbers and ISO-8601 date strings. Any other string raises a
  `FilterError`, because Solr would compare it lexicographically and quietly give an answer
  nobody meant.
- Because the value's Python type selects the Solr field, `{"field": "meta.page", "value": 100}`
  and `{"field": "meta.page", "value": "100"}` match different documents.
- `==` on `content` is a phrase match against an analysed field, not exact equality.

**Parameters:**

- **filters** (<code>dict\[str, Any\] | None</code>) – the filters to apply to the document list.

**Returns:**

- <code>list\[Document\]</code> – a list of Documents that match the given filters.

**Raises:**

- <code>FilterError</code> – if the filters are malformed, or compare a value Solr cannot order.

#### filter_documents_async

```python
filter_documents_async(filters: dict[str, Any] | None = None) -> list[Document]
```

Returns the documents that match the filters provided.

See `filter_documents` for the supported operators and their caveats.

**Parameters:**

- **filters** (<code>dict\[str, Any\] | None</code>) – the filters to apply to the document list.

**Returns:**

- <code>list\[Document\]</code> – a list of Documents that match the given filters.

**Raises:**

- <code>FilterError</code> – if the filters are malformed, or compare a value Solr cannot order.

#### write_documents

```python
write_documents(
    documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
) -> int
```

Writes Documents to Solr.

Metadata keys must consist of letters, digits and underscores only, because each key becomes a
Solr field name. Sparse embeddings are dropped, as Solr has no sparse vector field.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – a list of Documents to write.
- **policy** (<code>DuplicatePolicy</code>) – the policy to apply when a Document with the same id already exists.
  The default `DuplicatePolicy.NONE` resolves to `DuplicatePolicy.FAIL`.

**Returns:**

- <code>int</code> – the number of Documents written.

**Raises:**

- <code>ValueError</code> – if `documents` is not a list of Documents, or a metadata key cannot be
  expressed as a Solr field name.
- <code>DuplicateDocumentError</code> – if `policy` is `FAIL` (or the default `NONE`) and a Document
  already exists.

#### write_documents_async

```python
write_documents_async(
    documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
) -> int
```

Writes Documents to Solr.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – a list of Documents to write.
- **policy** (<code>DuplicatePolicy</code>) – the policy to apply when a Document with the same id already exists.
  The default `DuplicatePolicy.NONE` resolves to `DuplicatePolicy.FAIL`.

**Returns:**

- <code>int</code> – the number of Documents written.

**Raises:**

- <code>ValueError</code> – if `documents` is not a list of Documents, or a metadata key cannot be
  expressed as a Solr field name.
- <code>DuplicateDocumentError</code> – if `policy` is `FAIL` (or the default `NONE`) and a Document
  already exists.

#### delete_documents

```python
delete_documents(document_ids: list[str]) -> None
```

Deletes all documents with the given ids.

**Parameters:**

- **document_ids** (<code>list\[str\]</code>) – the ids of the documents to delete.

#### delete_documents_async

```python
delete_documents_async(document_ids: list[str]) -> None
```

Deletes all documents with the given ids.

**Parameters:**

- **document_ids** (<code>list\[str\]</code>) – the ids of the documents to delete.

#### delete_all_documents

```python
delete_all_documents() -> None
```

Deletes all documents in the core, leaving the schema in place.

#### delete_all_documents_async

```python
delete_all_documents_async() -> None
```

Deletes all documents in the core, leaving the schema in place.

#### delete_by_filter

```python
delete_by_filter(filters: dict[str, Any]) -> int
```

Deletes all documents matching the given filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – the filters selecting the documents to delete.

**Returns:**

- <code>int</code> – the number of documents deleted. The count is taken with a separate query before
  the delete is issued, so a concurrent write landing in between can make it differ from
  the number of documents the delete actually removes.

#### delete_by_filter_async

```python
delete_by_filter_async(filters: dict[str, Any]) -> int
```

Deletes all documents matching the given filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – the filters selecting the documents to delete.

**Returns:**

- <code>int</code> – the number of documents deleted. The count is taken with a separate query before
  the delete is issued, so a concurrent write landing in between can make it differ from
  the number of documents the delete actually removes.

#### update_by_filter

```python
update_by_filter(filters: dict[str, Any], meta: dict[str, Any]) -> int
```

Merges `meta` into the metadata of every document matching `filters`.

Matching documents are read, merged and rewritten in full rather than updated in place. A Solr
atomic update sets one field at a time, which would leave the previous value behind in another
field whenever a metadata value changes Python type, since the type is part of the field name.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – the filters selecting the documents to update.
- **meta** (<code>dict\[str, Any\]</code>) – the metadata to merge into each matching document.

**Returns:**

- <code>int</code> – the number of documents updated.

#### update_by_filter_async

```python
update_by_filter_async(filters: dict[str, Any], meta: dict[str, Any]) -> int
```

Merges `meta` into the metadata of every document matching `filters`.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – the filters selecting the documents to update.
- **meta** (<code>dict\[str, Any\]</code>) – the metadata to merge into each matching document.

**Returns:**

- <code>int</code> – the number of documents updated.

#### get_metadata_fields_info

```python
get_metadata_fields_info() -> dict[str, dict[str, str]]
```

Returns the metadata fields present in the core and their types.

**Returns:**

- <code>dict\[str, dict\[str, str\]\]</code> – a mapping of metadata field name to a dict with a `type` key.

#### get_metadata_fields_info_async

```python
get_metadata_fields_info_async() -> dict[str, dict[str, str]]
```

Returns the metadata fields present in the core and their types.

**Returns:**

- <code>dict\[str, dict\[str, str\]\]</code> – a mapping of metadata field name to a dict with a `type` key.

#### count_unique_metadata_by_filter

```python
count_unique_metadata_by_filter(
    filters: dict[str, Any], metadata_fields: list[str]
) -> dict[str, int]
```

Counts the distinct values of each given metadata field among documents matching `filters`.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – the filters restricting which documents are considered.
- **metadata_fields** (<code>list\[str\]</code>) – the metadata fields to count distinct values for.

**Returns:**

- <code>dict\[str, int\]</code> – a mapping of metadata field name to its number of distinct values.

#### count_unique_metadata_by_filter_async

```python
count_unique_metadata_by_filter_async(
    filters: dict[str, Any], metadata_fields: list[str]
) -> dict[str, int]
```

Counts the distinct values of each given metadata field among documents matching `filters`.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – the filters restricting which documents are considered.
- **metadata_fields** (<code>list\[str\]</code>) – the metadata fields to count distinct values for.

**Returns:**

- <code>dict\[str, int\]</code> – a mapping of metadata field name to its number of distinct values.

#### get_metadata_field_min_max

```python
get_metadata_field_min_max(
    metadata_field: str,
) -> dict[str, float | int | None]
```

Returns the minimum and maximum value of a numeric metadata field.

**Parameters:**

- **metadata_field** (<code>str</code>) – the metadata field, with or without a `meta.` prefix.

**Returns:**

- <code>dict\[str, float | int | None\]</code> – a dict with `min` and `max` keys, both `None` when the field has no numeric values.

#### get_metadata_field_min_max_async

```python
get_metadata_field_min_max_async(
    metadata_field: str,
) -> dict[str, float | int | None]
```

Returns the minimum and maximum value of a numeric metadata field.

**Parameters:**

- **metadata_field** (<code>str</code>) – the metadata field, with or without a `meta.` prefix.

**Returns:**

- <code>dict\[str, float | int | None\]</code> – a dict with `min` and `max` keys, both `None` when the field has no numeric values.

#### get_metadata_field_unique_values

```python
get_metadata_field_unique_values(
    metadata_field: str,
    search_term: str | None = None,
    from_: int = 0,
    size: int = 10,
    filters: dict[str, Any] | None = None,
) -> tuple[list[Any], int]
```

Returns the distinct values of a metadata field, paginated.

**Parameters:**

- **metadata_field** (<code>str</code>) – the metadata field, with or without a `meta.` prefix.
- **search_term** (<code>str | None</code>) – when given, only values containing it (case-insensitively) are returned.
- **from\_** (<code>int</code>) – index of the first value to return.
- **size** (<code>int</code>) – how many values to return.
- **filters** (<code>dict\[str, Any\] | None</code>) – filters restricting which documents are considered.

**Returns:**

- <code>tuple\[list\[Any\], int\]</code> – a `(values, total_count)` pair, where `total_count` counts all matching values.

#### get_metadata_field_unique_values_async

```python
get_metadata_field_unique_values_async(
    metadata_field: str,
    search_term: str | None = None,
    from_: int = 0,
    size: int = 10,
    filters: dict[str, Any] | None = None,
) -> tuple[list[Any], int]
```

Returns the distinct values of a metadata field, paginated.

**Parameters:**

- **metadata_field** (<code>str</code>) – the metadata field, with or without a `meta.` prefix.
- **search_term** (<code>str | None</code>) – when given, only values containing it (case-insensitively) are returned.
- **from\_** (<code>int</code>) – index of the first value to return.
- **size** (<code>int</code>) – how many values to return.
- **filters** (<code>dict\[str, Any\] | None</code>) – filters restricting which documents are considered.

**Returns:**

- <code>tuple\[list\[Any\], int\]</code> – a `(values, total_count)` pair, where `total_count` counts all matching values.

## haystack_integrations.document_stores.solr.filters

Translation of Haystack filters into Solr filter query (`fq`) clauses.

### escape_query_chars

```python
escape_query_chars(value: str) -> str
```

Escape the Lucene syntax characters in `value`.

**Parameters:**

- **value** (<code>str</code>) – the raw string.

**Returns:**

- <code>str</code> – the string with every syntax character and every whitespace run backslash-escaped.

### normalize_filters

```python
normalize_filters(filters: dict[str, Any]) -> str
```

Convert Haystack filters into a single Solr filter query clause.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – the filters to convert, in Haystack's comparison/logic dictionary form.

**Returns:**

- <code>str</code> – a clause suitable for Solr's `fq` parameter or for a delete-by-query.

**Raises:**

- <code>FilterError</code> – if `filters` is malformed or uses an unsupported operator or value type.

## haystack_integrations.document_stores.solr.schema

Mapping between Haystack `Document`s and Solr documents.

Solr is strongly typed: a field's type is fixed the first time the field is created and a value of the
wrong type is rejected. Haystack metadata, on the other hand, is an arbitrary `dict[str, Any]` whose
value types are only known at write time. To reconcile the two, every metadata entry is stored in a
Solr field whose name encodes the Python type of the value:

```
meta.page  = "100"  ->  meta_s_page = "100"     (string)
meta.page  = 100    ->  meta_l_page = 100       (plong)
```

The type code lives in the *prefix* rather than the suffix because Solr dynamic field patterns accept
only a leading or a trailing wildcard - `meta_*_s` is not a legal pattern, while `meta_s_*` is.

Encoding the type in the field name buys two properties that a single JSON blob or Solr's schemaless
type inference cannot provide:

- metadata round-trips with its Python type intact, so `{"page": "100"}` never comes back as
  `{"page": 100}`;
- values that merely share a string form stay distinct, so the int `1`, the str `"1"`, the float `1.0`
  and the bool `True` occupy four different fields and are reported as four distinct values.

### type_code_for_value

```python
type_code_for_value(value: Any) -> str
```

Return the type code under which `value` is stored.

Homogeneous lists of scalars use the multi-valued code for their element type. Everything else -
dicts, mixed lists, nested structures - falls back to a JSON-encoded string.

**Parameters:**

- **value** (<code>Any</code>) – the metadata value to classify.

**Returns:**

- <code>str</code> – one of the codes in `ALL_TYPE_CODES`.

### meta_field_name

```python
meta_field_name(key: str, type_code: str) -> str
```

Build the Solr field name holding metadata `key` at `type_code`.

**Parameters:**

- **key** (<code>str</code>) – the Haystack metadata key.
- **type_code** (<code>str</code>) – one of the codes in `ALL_TYPE_CODES`.

**Returns:**

- <code>str</code> – the Solr field name, e.g. `meta_s_page`.

### parse_meta_field_name

```python
parse_meta_field_name(field: str) -> tuple[str, str] | None
```

Invert `meta_field_name`.

**Parameters:**

- **field** (<code>str</code>) – a Solr field name.

**Returns:**

- <code>tuple\[str, str\] | None</code> – a `(type_code, key)` pair, or `None` if `field` is not a metadata field. Type codes
  contain no underscore, so a single split is unambiguous even when the key does.

### validate_meta_keys

```python
validate_meta_keys(meta: dict[str, Any]) -> None
```

Reject metadata keys that cannot be expressed as a Solr field name.

**Parameters:**

- **meta** (<code>dict\[str, Any\]</code>) – the metadata of a single document.

**Raises:**

- <code>ValueError</code> – if any key contains a character outside `[A-Za-z0-9_]`. Silently rewriting
  such keys would let two distinct keys collide, so the write is refused instead.

### document_to_solr

```python
document_to_solr(document: Document) -> dict[str, Any]
```

Convert a Haystack `Document` into a Solr document.

**Parameters:**

- **document** (<code>Document</code>) – the document to convert.

**Returns:**

- <code>dict\[str, Any\]</code> – a JSON-serializable dict ready to be posted to Solr's update handler.

**Raises:**

- <code>ValueError</code> – if a metadata key cannot be expressed as a Solr field name.

### solr_to_document

```python
solr_to_document(
    solr_document: dict[str, Any], *, score: float | None = None
) -> Document
```

Convert a Solr document back into a Haystack `Document`.

**Parameters:**

- **solr_document** (<code>dict\[str, Any\]</code>) – a single entry from a Solr query response.
- **score** (<code>float | None</code>) – the relevance score to attach, when the document came from a retrieval query.

**Returns:**

- <code>Document</code> – the reconstructed document.

### vector_field_type_name

```python
vector_field_type_name(embedding_dim: int) -> str
```

Return the name of the `DenseVectorField` type backing embeddings of `embedding_dim` dimensions.

**Parameters:**

- **embedding_dim** (<code>int</code>) – the embedding dimension.

**Returns:**

- <code>str</code> – the Solr field type name.

### schema_payload

```python
schema_payload(
    *,
    embedding_dim: int,
    similarity_function: str,
    existing_field_types: set[str],
    existing_fields: set[str],
    existing_dynamic_fields: set[str],
    vector_field_type_params: dict[str, Any] | None = None
) -> dict[str, Any]
```

Build an idempotent Schema API payload creating only what the core is missing.

**Parameters:**

- **embedding_dim** (<code>int</code>) – dimension of the `DenseVectorField` backing embeddings.
- **similarity_function** (<code>str</code>) – `cosine`, `dot_product` or `euclidean`.
- **existing_field_types** (<code>set\[str\]</code>) – field type names already defined in the core.
- **existing_fields** (<code>set\[str\]</code>) – field names already defined in the core.
- **existing_dynamic_fields** (<code>set\[str\]</code>) – dynamic field patterns already defined in the core.
- **vector_field_type_params** (<code>dict\[str, Any\] | None</code>) – extra attributes for the vector field type, for example
  `{"hnswM": 32}` on Solr 10 or `{"hnswMaxConnections": 32}` on Solr 9. Left unset by default so
  that one payload is valid on both major versions, which renamed these attributes.

**Returns:**

- <code>dict\[str, Any\]</code> – the Schema API payload. Empty when the core already has everything.
