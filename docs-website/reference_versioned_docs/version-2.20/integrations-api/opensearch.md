---
title: "OpenSearch"
id: integrations-opensearch
description: "OpenSearch integration for Haystack"
slug: "/integrations-opensearch"
---

<a id="haystack_integrations.components.retrievers.opensearch.bm25_retriever"></a>

## Module haystack\_integrations.components.retrievers.opensearch.bm25\_retriever

<a id="haystack_integrations.components.retrievers.opensearch.bm25_retriever.OpenSearchBM25Retriever"></a>

### OpenSearchBM25Retriever

Fetches documents from OpenSearchDocumentStore using the keyword-based BM25 algorithm.

BM25 computes a weighted word overlap between the query string and a document to determine its similarity.

<a id="haystack_integrations.components.retrievers.opensearch.bm25_retriever.OpenSearchBM25Retriever.__init__"></a>

#### OpenSearchBM25Retriever.\_\_init\_\_

```python
def __init__(*,
             document_store: OpenSearchDocumentStore,
             filters: dict[str, Any] | None = None,
             fuzziness: int | str = "AUTO",
             top_k: int = 10,
             scale_score: bool = False,
             all_terms_must_match: bool = False,
             filter_policy: str | FilterPolicy = FilterPolicy.REPLACE,
             custom_query: dict[str, Any] | None = None,
             raise_on_failure: bool = True)
```

Creates the OpenSearchBM25Retriever component.

**Arguments**:

- `document_store`: An instance of OpenSearchDocumentStore to use with the Retriever.
- `filters`: Filters to narrow down the search for documents in the Document Store.
- `fuzziness`: Determines how approximate string matching is applied in full-text queries.
This parameter sets the number of character edits (insertions, deletions, or substitutions)
required to transform one word into another. For example, the "fuzziness" between the words
"wined" and "wind" is 1 because only one edit is needed to match them.

Use "AUTO" (the default) for automatic adjustment based on term length, which is optimal for
most scenarios. For detailed guidance, refer to the
[OpenSearch fuzzy query documentation](https://opensearch.org/docs/latest/query-dsl/term/fuzzy/).
- `top_k`: Maximum number of documents to return.
- `scale_score`: If `True`, scales the score of retrieved documents to a range between 0 and 1.
This is useful when comparing documents across different indexes.
- `all_terms_must_match`: If `True`, all terms in the query string must be present in the
retrieved documents. This is useful when searching for short text where even one term
can make a difference.
- `filter_policy`: Policy to determine how filters are applied. Possible options:
- `replace`: Runtime filters replace initialization filters. Use this policy to change the filtering scope
for specific queries.
- `merge`: Runtime filters are merged with initialization filters.
- `custom_query`: The query containing a mandatory `$query` and an optional `$filters` placeholder.
    **An example custom_query:**

    ```python
    {
        "query": {
            "bool": {
                "should": [{"multi_match": {
                    "query": "$query",                 // mandatory query placeholder
                    "type": "most_fields",
                    "fields": ["content", "title"]}}],
                "filter": "$filters"                  // optional filter placeholder
            }
        }
    }
    ```

An example `run()` method for this `custom_query`:

```python
retriever.run(
    query="Why did the revenue increase?",
    filters={
        "operator": "AND",
        "conditions": [
            {"field": "meta.years", "operator": "==", "value": "2019"},
            {"field": "meta.quarters", "operator": "in", "value": ["Q1", "Q2"]},
        ],
    },
)
```
- `raise_on_failure`: Whether to raise an exception if the API call fails. Otherwise log a warning and return an empty list.

**Raises**:

- `ValueError`: If `document_store` is not an instance of OpenSearchDocumentStore.

<a id="haystack_integrations.components.retrievers.opensearch.bm25_retriever.OpenSearchBM25Retriever.to_dict"></a>

#### OpenSearchBM25Retriever.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.retrievers.opensearch.bm25_retriever.OpenSearchBM25Retriever.from_dict"></a>

#### OpenSearchBM25Retriever.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "OpenSearchBM25Retriever"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.retrievers.opensearch.bm25_retriever.OpenSearchBM25Retriever.run"></a>

#### OpenSearchBM25Retriever.run

```python
@component.output_types(documents=list[Document])
def run(
    query: str,
    filters: dict[str, Any] | None = None,
    all_terms_must_match: bool | None = None,
    top_k: int | None = None,
    fuzziness: int | str | None = None,
    scale_score: bool | None = None,
    custom_query: dict[str, Any] | None = None,
    document_store: OpenSearchDocumentStore | None = None
) -> dict[str, list[Document]]
```

Retrieve documents using BM25 retrieval.

**Arguments**:

- `query`: The query string.
- `filters`: Filters applied to the retrieved documents. The way runtime filters are applied depends on
the `filter_policy` specified at Retriever's initialization.
- `all_terms_must_match`: If `True`, all terms in the query string must be present in the
retrieved documents.
- `top_k`: Maximum number of documents to return.
- `fuzziness`: Fuzziness parameter for full-text queries to apply approximate string matching.
For more information, see [OpenSearch fuzzy query](https://opensearch.org/docs/latest/query-dsl/term/fuzzy/).
- `scale_score`: If `True`, scales the score of retrieved documents to a range between 0 and 1.
This is useful when comparing documents across different indexes.
- `custom_query`: A custom OpenSearch query. It must include a `$query` and may optionally
include a `$filters` placeholder.

    **An example custom_query:**

    ```python
    {
        "query": {
            "bool": {
                "should": [{"multi_match": {
                    "query": "$query",                 // mandatory query placeholder
                    "type": "most_fields",
                    "fields": ["content", "title"]}}],
                "filter": "$filters"                  // optional filter placeholder
            }
        }
    }
    ```

**For this custom_query, a sample `run()` could be:**

```python
retriever.run(
    query="Why did the revenue increase?",
    filters={
        "operator": "AND",
        "conditions": [
            {"field": "meta.years", "operator": "==", "value": "2019"},
            {"field": "meta.quarters", "operator": "in", "value": ["Q1", "Q2"]},
        ],
    },
)
```
- `document_store`: Optionally, an instance of OpenSearchDocumentStore to use with the Retriever

**Returns**:

A dictionary containing the retrieved documents with the following structure:
- documents: List of retrieved Documents.

<a id="haystack_integrations.components.retrievers.opensearch.bm25_retriever.OpenSearchBM25Retriever.run_async"></a>

#### OpenSearchBM25Retriever.run\_async

```python
@component.output_types(documents=list[Document])
async def run_async(
    query: str,
    filters: dict[str, Any] | None = None,
    all_terms_must_match: bool | None = None,
    top_k: int | None = None,
    fuzziness: int | str | None = None,
    scale_score: bool | None = None,
    custom_query: dict[str, Any] | None = None,
    document_store: OpenSearchDocumentStore | None = None
) -> dict[str, list[Document]]
```

Asynchronously retrieve documents using BM25 retrieval.

**Arguments**:

- `query`: The query string.
- `filters`: Filters applied to the retrieved documents. The way runtime filters are applied depends on
the `filter_policy` specified at Retriever's initialization.
- `all_terms_must_match`: If `True`, all terms in the query string must be present in the
retrieved documents.
- `top_k`: Maximum number of documents to return.
- `fuzziness`: Fuzziness parameter for full-text queries to apply approximate string matching.
For more information, see [OpenSearch fuzzy query](https://opensearch.org/docs/latest/query-dsl/term/fuzzy/).
- `scale_score`: If `True`, scales the score of retrieved documents to a range between 0 and 1.
This is useful when comparing documents across different indexes.
- `custom_query`: A custom OpenSearch query. It must include a `$query` and may optionally
include a `$filters` placeholder.
- `document_store`: Optionally, an instance of OpenSearchDocumentStore to use with the Retriever

**Returns**:

A dictionary containing the retrieved documents with the following structure:
- documents: List of retrieved Documents.

<a id="haystack_integrations.components.retrievers.opensearch.embedding_retriever"></a>

## Module haystack\_integrations.components.retrievers.opensearch.embedding\_retriever

<a id="haystack_integrations.components.retrievers.opensearch.embedding_retriever.OpenSearchEmbeddingRetriever"></a>

### OpenSearchEmbeddingRetriever

Retrieves documents from the OpenSearchDocumentStore using a vector similarity metric.

 Must be connected to the OpenSearchDocumentStore to run.

<a id="haystack_integrations.components.retrievers.opensearch.embedding_retriever.OpenSearchEmbeddingRetriever.__init__"></a>

#### OpenSearchEmbeddingRetriever.\_\_init\_\_

```python
def __init__(*,
             document_store: OpenSearchDocumentStore,
             filters: dict[str, Any] | None = None,
             top_k: int = 10,
             filter_policy: str | FilterPolicy = FilterPolicy.REPLACE,
             custom_query: dict[str, Any] | None = None,
             raise_on_failure: bool = True,
             efficient_filtering: bool = False)
```

Create the OpenSearchEmbeddingRetriever component.

**Arguments**:

- `document_store`: An instance of OpenSearchDocumentStore to use with the Retriever.
- `filters`: Filters applied when fetching documents from the Document Store.
Filters are applied during the approximate kNN search to ensure the Retriever returns
`top_k` matching documents.
- `top_k`: Maximum number of documents to return.
- `filter_policy`: Policy to determine how filters are applied. Possible options:
- `merge`: Runtime filters are merged with initialization filters.
- `replace`: Runtime filters replace initialization filters. Use this policy to change the filtering scope.
- `custom_query`: The custom OpenSearch query containing a mandatory `$query_embedding` and
an optional `$filters` placeholder.

    **An example custom_query:**

    ```python
    {
        "query": {
            "bool": {
                "must": [
                    {
                        "knn": {
                            "embedding": {
                                "vector": "$query_embedding",   // mandatory query placeholder
                                "k": 10000,
                            }
                        }
                    }
                ],
                "filter": "$filters"                            // optional filter placeholder
            }
        }
    }
    ```

For this `custom_query`, an example `run()` could be:

```python
retriever.run(
    query_embedding=embedding,
    filters={
        "operator": "AND",
        "conditions": [
            {"field": "meta.years", "operator": "==", "value": "2019"},
            {"field": "meta.quarters", "operator": "in", "value": ["Q1", "Q2"]},
        ],
    },
)
```
- `raise_on_failure`: If `True`, raises an exception if the API call fails.
If `False`, logs a warning and returns an empty list.
- `efficient_filtering`: If `True`, the filter will be applied during the approximate kNN search.
This is only supported for knn engines "faiss" and "lucene" and does not work with the default "nmslib".

**Raises**:

- `ValueError`: If `document_store` is not an instance of OpenSearchDocumentStore.

<a id="haystack_integrations.components.retrievers.opensearch.embedding_retriever.OpenSearchEmbeddingRetriever.to_dict"></a>

#### OpenSearchEmbeddingRetriever.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.retrievers.opensearch.embedding_retriever.OpenSearchEmbeddingRetriever.from_dict"></a>

#### OpenSearchEmbeddingRetriever.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "OpenSearchEmbeddingRetriever"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.retrievers.opensearch.embedding_retriever.OpenSearchEmbeddingRetriever.run"></a>

#### OpenSearchEmbeddingRetriever.run

```python
@component.output_types(documents=list[Document])
def run(
    query_embedding: list[float],
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
    custom_query: dict[str, Any] | None = None,
    efficient_filtering: bool | None = None,
    document_store: OpenSearchDocumentStore | None = None
) -> dict[str, list[Document]]
```

Retrieve documents using a vector similarity metric.

**Arguments**:

- `query_embedding`: Embedding of the query.
- `filters`: Filters applied when fetching documents from the Document Store.
Filters are applied during the approximate kNN search to ensure the Retriever returns `top_k` matching
documents.
The way runtime filters are applied depends on the `filter_policy` selected when initializing the Retriever.
- `top_k`: Maximum number of documents to return.
- `custom_query`: A custom OpenSearch query containing a mandatory `$query_embedding` and an
optional `$filters` placeholder.

    **An example custom_query:**

    ```python
    {
        "query": {
            "bool": {
                "must": [
                    {
                        "knn": {
                            "embedding": {
                                "vector": "$query_embedding",   // mandatory query placeholder
                                "k": 10000,
                            }
                        }
                    }
                ],
                "filter": "$filters"                            // optional filter placeholder
            }
        }
    }
    ```

For this `custom_query`, an example `run()` could be:

```python
retriever.run(
    query_embedding=embedding,
    filters={
        "operator": "AND",
        "conditions": [
            {"field": "meta.years", "operator": "==", "value": "2019"},
            {"field": "meta.quarters", "operator": "in", "value": ["Q1", "Q2"]},
        ],
    },
)
```
- `efficient_filtering`: If `True`, the filter will be applied during the approximate kNN search.
This is only supported for knn engines "faiss" and "lucene" and does not work with the default "nmslib".
- `document_store`: Optional instance of OpenSearchDocumentStore to use with the Retriever.

**Returns**:

Dictionary with key "documents" containing the retrieved Documents.
- documents: List of Document similar to `query_embedding`.

<a id="haystack_integrations.components.retrievers.opensearch.embedding_retriever.OpenSearchEmbeddingRetriever.run_async"></a>

#### OpenSearchEmbeddingRetriever.run\_async

```python
@component.output_types(documents=list[Document])
async def run_async(
    query_embedding: list[float],
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
    custom_query: dict[str, Any] | None = None,
    efficient_filtering: bool | None = None,
    document_store: OpenSearchDocumentStore | None = None
) -> dict[str, list[Document]]
```

Asynchronously retrieve documents using a vector similarity metric.

**Arguments**:

- `query_embedding`: Embedding of the query.
- `filters`: Filters applied when fetching documents from the Document Store.
Filters are applied during the approximate kNN search to ensure the Retriever
  returns `top_k` matching documents.
The way runtime filters are applied depends on the `filter_policy` selected when initializing the Retriever.
- `top_k`: Maximum number of documents to return.
- `custom_query`: A custom OpenSearch query containing a mandatory `$query_embedding` and an
optional `$filters` placeholder.

    **An example custom_query:**

    ```python
    {
        "query": {
            "bool": {
                "must": [
                    {
                        "knn": {
                            "embedding": {
                                "vector": "$query_embedding",   // mandatory query placeholder
                                "k": 10000,
                            }
                        }
                    }
                ],
                "filter": "$filters"                            // optional filter placeholder
            }
        }
    }
    ```

For this `custom_query`, an example `run()` could be:

```python
retriever.run(
    query_embedding=embedding,
    filters={
        "operator": "AND",
        "conditions": [
            {"field": "meta.years", "operator": "==", "value": "2019"},
            {"field": "meta.quarters", "operator": "in", "value": ["Q1", "Q2"]},
        ],
    },
)
```
- `efficient_filtering`: If `True`, the filter will be applied during the approximate kNN search.
This is only supported for knn engines "faiss" and "lucene" and does not work with the default "nmslib".
- `document_store`: Optional instance of OpenSearchDocumentStore to use with the Retriever.

**Returns**:

Dictionary with key "documents" containing the retrieved Documents.
- documents: List of Document similar to `query_embedding`.

<a id="haystack_integrations.components.retrievers.opensearch.metadata_retriever"></a>

## Module haystack\_integrations.components.retrievers.opensearch.metadata\_retriever

<a id="haystack_integrations.components.retrievers.opensearch.metadata_retriever.OpenSearchMetadataRetriever"></a>

### OpenSearchMetadataRetriever

Retrieves and ranks metadata from documents stored in an OpenSearchDocumentStore.

It searches specified metadata fields for matches to a given query, ranks the results based on relevance using
Jaccard similarity, and returns the top-k results containing only the specified metadata fields. Additionally, it
adds a boost to the score of exact matches.

The search is designed for metadata fields whose values are **text** (strings). It uses prefix, wildcard and fuzzy
matching to find candidate documents; these query types operate only on text/keyword fields in OpenSearch.

Metadata fields with **non-string types** (integers, floats, booleans, lists of non-strings) are indexed by
OpenSearch as numeric, boolean, or array types. Those field types do not support prefix, wildcard, or full-text
match queries, so documents are typically not found when you search only by such fields.

**Mixed types** in the same metadata field (e.g. a list containing both strings and numbers) are not supported.

Must be connected to the OpenSearchDocumentStore to run.

**Example**:

    ```python
    from haystack import Document
    from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore
    from haystack_integrations.components.retrievers.opensearch import OpenSearchMetadataRetriever

    # Create documents with metadata
    docs = [
        Document(
            content="Python programming guide",
            meta={"category": "Python", "status": "active", "priority": 1, "author": "John Doe"}
        ),
        Document(
            content="Java tutorial",
            meta={"category": "Java", "status": "active", "priority": 2, "author": "Jane Smith"}
        ),
        Document(
            content="Python advanced topics",
            meta={"category": "Python", "status": "inactive", "priority": 3, "author": "John Doe"}
        ),
    ]
    document_store.write_documents(docs, refresh=True)

    # Create retriever specifying which metadata fields to search and return
    retriever = OpenSearchMetadataRetriever(
        document_store=document_store,
        metadata_fields=["category", "status", "priority"],
        top_k=10,
    )

    # Search for metadata
    result = retriever.run(query="Python")

    # Result structure:
    # {
    #     "metadata": [
    #         {"category": "Python", "status": "active", "priority": 1},
    #         {"category": "Python", "status": "inactive", "priority": 3},
    #     ]
    # }
    #
    # Note: Only the specified metadata_fields are returned in the results.
    # Other metadata fields (like "author") and document content are excluded.
    ```

<a id="haystack_integrations.components.retrievers.opensearch.metadata_retriever.OpenSearchMetadataRetriever.__init__"></a>

#### OpenSearchMetadataRetriever.\_\_init\_\_

```python
def __init__(*,
             document_store: OpenSearchDocumentStore,
             metadata_fields: list[str],
             top_k: int = 20,
             exact_match_weight: float = 0.6,
             mode: Literal["strict", "fuzzy"] = "fuzzy",
             fuzziness: int | Literal["AUTO"] = 2,
             prefix_length: int = 0,
             max_expansions: int = 200,
             tie_breaker: float = 0.7,
             jaccard_n: int = 3,
             raise_on_failure: bool = True)
```

Create the OpenSearchMetadataRetriever component.

**Arguments**:

- `document_store`: An instance of OpenSearchDocumentStore to use with the Retriever.
- `metadata_fields`: List of metadata field names to search within each document's metadata.
- `top_k`: Maximum number of top results to return based on relevance. Default is 20.
- `exact_match_weight`: Weight to boost the score of exact matches in metadata fields.
Default is 0.6. It's used on both "strict" and "fuzzy" modes and applied after the search executes.
- `mode`: Search mode. "strict" uses prefix and wildcard matching,
"fuzzy" uses fuzzy matching with dis_max queries. Default is "fuzzy".
In both modes, results are scored using Jaccard similarity (n-gram based)
computed server-side via a Painless script; n is controlled by jaccard_n.
- `fuzziness`: Maximum allowed Damerau-Levenshtein distance (edit distance) for fuzzy matching.
Accepts an integer (e.g., 0, 1, 2) or "AUTO" which chooses based on term length.
Default is 2. Only applies when mode is "fuzzy".
- `prefix_length`: Number of leading characters that must match exactly before fuzzy matching applies.
Default is 0 (no prefix requirement). Only applies when mode is "fuzzy".
- `max_expansions`: Maximum number of term variations the fuzzy query can generate.
Default is 200. Only applies when mode is "fuzzy".
- `tie_breaker`: Weight (0..1) for other matching clauses in the dis_max query.
Boosts documents that match multiple clauses. Default is 0.7. Only applies when mode is "fuzzy".
- `jaccard_n`: N-gram size for Jaccard similarity scoring. Default 3; larger n favors longer token matches.
- `raise_on_failure`: If `True`, raises an exception if the API call fails.
If `False`, logs a warning and returns an empty list.

**Raises**:

- `ValueError`: If `document_store` is not an instance of OpenSearchDocumentStore.

<a id="haystack_integrations.components.retrievers.opensearch.metadata_retriever.OpenSearchMetadataRetriever.to_dict"></a>

#### OpenSearchMetadataRetriever.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.retrievers.opensearch.metadata_retriever.OpenSearchMetadataRetriever.from_dict"></a>

#### OpenSearchMetadataRetriever.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "OpenSearchMetadataRetriever"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.retrievers.opensearch.metadata_retriever.OpenSearchMetadataRetriever.run"></a>

#### OpenSearchMetadataRetriever.run

```python
@component.output_types(metadata=list[dict[str, Any]])
def run(query: str,
        *,
        document_store: OpenSearchDocumentStore | None = None,
        metadata_fields: list[str] | None = None,
        top_k: int | None = None,
        exact_match_weight: float | None = None,
        mode: Literal["strict", "fuzzy"] | None = None,
        fuzziness: int | Literal["AUTO"] | None = None,
        prefix_length: int | None = None,
        max_expansions: int | None = None,
        tie_breaker: float | None = None,
        jaccard_n: int | None = None,
        filters: dict[str, Any] | None = None
        ) -> dict[str, list[dict[str, Any]]]
```

Execute a search query against the metadata fields of documents stored in the Document Store.

**Arguments**:

- `query`: The search query string, which can contain multiple comma-separated parts.
Each part will be searched across all specified fields.
- `document_store`: The Document Store to run the query against.
If not provided, the one provided in `__init__` is used.
- `metadata_fields`: List of metadata field names to search within.
If not provided, the fields provided in `__init__` are used.
- `top_k`: Maximum number of top results to return based on relevance.
The search retrieves up to 1000 hits from OpenSearch, then applies boosting and filters
the results to the top_k most relevant matches.
If not provided, the top_k provided in `__init__` is used.
- `exact_match_weight`: Weight to boost the score of exact matches in metadata fields.
If not provided, the exact_match_weight provided in `__init__` is used.
- `mode`: Search mode. "strict" uses prefix and wildcard matching,
"fuzzy" uses fuzzy matching with dis_max queries.
In both modes, results are scored using Jaccard similarity (n-gram based) via a Painless script.
If not provided, the mode provided in `__init__` is used.
- `fuzziness`: Maximum allowed Damerau-Levenshtein distance (edit distance) for fuzzy matching.
Accepts an integer (e.g., 0, 1, 2) or "AUTO" which chooses based on term length.
Only applies when mode is "fuzzy". If not provided, the fuzziness provided in `__init__` is used.
- `prefix_length`: Number of leading characters that must match exactly before fuzzy matching applies.
Only applies when mode is "fuzzy". If not provided, the prefix_length provided in `__init__` is used.
- `max_expansions`: Maximum number of term variations the fuzzy query can generate.
Only applies when mode is "fuzzy". If not provided, the max_expansions provided in `__init__` is used.
- `tie_breaker`: Weight (0..1) for other matching clauses; boosts docs matching multiple
clauses. Only applies when mode is "fuzzy". If not provided, the tie_breaker provided in `__init__` is used.
- `jaccard_n`: N-gram size for Jaccard similarity scoring. If not provided, the jaccard_n from `__init__`
is used.
- `filters`: Additional filters to apply to the search query.

**Returns**:

A dictionary containing the top-k retrieved metadata results.
Example:
    ```python
    from haystack import Document

    # First, add a document with matching metadata to the store
    store.write_documents([
        Document(
            content="Python programming guide",
            meta={"category": "Python", "status": "active", "priority": 1}
        )
    ])

    retriever = OpenSearchMetadataRetriever(
        document_store=store,
        metadata_fields=["category", "status", "priority"]
    )
    result = retriever.run(query="Python, active")
    # Returns: {"metadata": [{"category": "Python", "status": "active", "priority": 1}]}
    ```

<a id="haystack_integrations.components.retrievers.opensearch.metadata_retriever.OpenSearchMetadataRetriever.run_async"></a>

#### OpenSearchMetadataRetriever.run\_async

```python
@component.output_types(metadata=list[dict[str, Any]])
async def run_async(
        query: str,
        *,
        document_store: OpenSearchDocumentStore | None = None,
        metadata_fields: list[str] | None = None,
        top_k: int | None = None,
        exact_match_weight: float | None = None,
        mode: Literal["strict", "fuzzy"] | None = None,
        fuzziness: int | Literal["AUTO"] | None = None,
        prefix_length: int | None = None,
        max_expansions: int | None = None,
        tie_breaker: float | None = None,
        jaccard_n: int | None = None,
        filters: dict[str, Any] | None = None
) -> dict[str, list[dict[str, Any]]]
```

Asynchronously execute a search query against the metadata fields of documents stored in the Document Store.

**Arguments**:

- `query`: The search query string, which can contain multiple comma-separated parts.
Each part will be searched across all specified fields.
- `document_store`: The Document Store to run the query against.
If not provided, the one provided in `__init__` is used.
- `metadata_fields`: List of metadata field names to search within.
If not provided, the fields provided in `__init__` are used.
- `top_k`: Maximum number of top results to return based on relevance.
The search retrieves up to 1000 hits from OpenSearch, then applies boosting and filters
the results to the top_k most relevant matches.
If not provided, the top_k provided in `__init__` is used.
- `exact_match_weight`: Weight to boost the score of exact matches in metadata fields.
If not provided, the exact_match_weight provided in `__init__` is used.
- `mode`: Search mode. "strict" uses prefix and wildcard matching,
"fuzzy" uses fuzzy matching with dis_max queries.
In both modes, results are scored using Jaccard similarity (n-gram based) via a Painless script.
If not provided, the mode provided in `__init__` is used.
- `fuzziness`: Maximum allowed Damerau-Levenshtein distance (edit distance) for fuzzy matching.
Accepts an integer (e.g., 0, 1, 2) or "AUTO" which chooses based on term length.
Only applies when mode is "fuzzy". If not provided, the fuzziness provided in `__init__` is used.
- `prefix_length`: Number of leading characters that must match exactly before fuzzy matching applies.
Only applies when mode is "fuzzy". If not provided, the prefix_length provided in `__init__` is used.
- `max_expansions`: Maximum number of term variations the fuzzy query can generate.
Only applies when mode is "fuzzy". If not provided, the max_expansions provided in `__init__` is used.
- `tie_breaker`: Weight (0..1) for other matching clauses; boosts docs matching multiple clauses.
Only applies when mode is "fuzzy". If not provided, the tie_breaker provided in `__init__` is used.
- `jaccard_n`: N-gram size for Jaccard similarity scoring. If not provided, the jaccard_n from `__init__`
is used.
- `filters`: Additional filters to apply to the search query.

**Returns**:

A dictionary containing the top-k retrieved metadata results.
Example:
    ```python
    from haystack import Document

    # First, add a document with matching metadata to the store
    await store.write_documents_async([
        Document(
            content="Python programming guide",
            meta={"category": "Python", "status": "active", "priority": 1}
        )
    ])

    retriever = OpenSearchMetadataRetriever(
        document_store=store,
        metadata_fields=["category", "status", "priority"]
    )
    result = await retriever.run_async(query="Python, active")
    # Returns: {"metadata": [{"category": "Python", "status": "active", "priority": 1}]}
    ```

<a id="haystack_integrations.components.retrievers.opensearch.open_search_hybrid_retriever"></a>

## Module haystack\_integrations.components.retrievers.opensearch.open\_search\_hybrid\_retriever

<a id="haystack_integrations.components.retrievers.opensearch.open_search_hybrid_retriever.OpenSearchHybridRetriever"></a>

### OpenSearchHybridRetriever

A hybrid retriever that combines embedding-based and keyword-based retrieval from OpenSearch.

Example usage:

Make sure you have "sentence-transformers>=3.0.0":

    pip install haystack-ai datasets "sentence-transformers>=3.0.0"


And OpenSearch running. You can run OpenSearch with Docker:

    docker run -d --name opensearch-nosec -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node"
    -e "DISABLE_SECURITY_PLUGIN=true" opensearchproject/opensearch:2.12.0

```python
from haystack import Document
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack_integrations.components.retrievers.opensearch import OpenSearchHybridRetriever
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore

# Initialize the document store
doc_store = OpenSearchDocumentStore(
    hosts=["<http://localhost:9200>"],
    index="document_store",
    embedding_dim=384,
)

# Create some sample documents
docs = [
    Document(content="Machine learning is a subset of artificial intelligence."),
    Document(content="Deep learning is a subset of machine learning."),
    Document(content="Natural language processing is a field of AI."),
    Document(content="Reinforcement learning is a type of machine learning."),
    Document(content="Supervised learning is a type of machine learning."),
]

# Embed the documents and add them to the document store
doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
doc_embedder.warm_up()
docs = doc_embedder.run(docs)
doc_store.write_documents(docs['documents'])

# Initialize some haystack text embedder, in this case the SentenceTransformersTextEmbedder
embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")

# Initialize the hybrid retriever
retriever = OpenSearchHybridRetriever(
    document_store=doc_store,
    embedder=embedder,
    top_k_bm25=3,
    top_k_embedding=3,
    join_mode="reciprocal_rank_fusion"
)

# Run the retriever
results = retriever.run(query="What is reinforcement learning?", filters_bm25=None, filters_embedding=None)

>> results['documents']
{'documents': [Document(id=..., content: 'Reinforcement learning is a type of machine learning.', score: 1.0),
  Document(id=..., content: 'Supervised learning is a type of machine learning.', score: 0.9760624679979518),
  Document(id=..., content: 'Deep learning is a subset of machine learning.', score: 0.4919354838709677),
  Document(id=..., content: 'Machine learning is a subset of artificial intelligence.', score: 0.4841269841269841)]}
  ```

<a id="haystack_integrations.components.retrievers.opensearch.open_search_hybrid_retriever.OpenSearchHybridRetriever.__init__"></a>

#### OpenSearchHybridRetriever.\_\_init\_\_

```python
def __init__(document_store: OpenSearchDocumentStore,
             *,
             embedder: TextEmbedder,
             filters_bm25: dict[str, Any] | None = None,
             fuzziness: int | str = "AUTO",
             top_k_bm25: int = 10,
             scale_score: bool = False,
             all_terms_must_match: bool = False,
             filter_policy_bm25: str | FilterPolicy = FilterPolicy.REPLACE,
             custom_query_bm25: dict[str, Any] | None = None,
             filters_embedding: dict[str, Any] | None = None,
             top_k_embedding: int = 10,
             filter_policy_embedding: str
             | FilterPolicy = FilterPolicy.REPLACE,
             custom_query_embedding: dict[str, Any] | None = None,
             join_mode: str | JoinMode = JoinMode.RECIPROCAL_RANK_FUSION,
             weights: list[float] | None = None,
             top_k: int | None = None,
             sort_by_score: bool = True,
             **kwargs: Any) -> None
```

Initialize the OpenSearchHybridRetriever, a super component to retrieve documents from OpenSearch using

both embedding-based and keyword-based retrieval methods.

We don't explicitly define all the init parameters of the components in the constructor, for each
of the components, since that would be around 20+ parameters. Instead, we define the most important ones
and pass the rest as kwargs. This is to keep the constructor clean and easy to read.

If you need to pass extra parameters to the components, you can do so by passing them as kwargs. It expects
a dictionary with the component name as the key and the parameters as the value. The component name should be:

    - "bm25_retriever" -> OpenSearchBM25Retriever
    - "embedding_retriever" -> OpenSearchEmbeddingRetriever

**Arguments**:

- `document_store`: The OpenSearchDocumentStore to use for retrieval.
- `embedder`: A TextEmbedder to use for embedding the query.
See `haystack.components.embedders.types.protocol.TextEmbedder` for more information.
- `filters_bm25`: Filters for the BM25 retriever.
- `fuzziness`: The fuzziness for the BM25 retriever.
- `top_k_bm25`: The number of results to return from the BM25 retriever.
- `scale_score`: Whether to scale the score for the BM25 retriever.
- `all_terms_must_match`: Whether all terms must match for the BM25 retriever.
- `filter_policy_bm25`: The filter policy for the BM25 retriever.
- `custom_query_bm25`: A custom query for the BM25 retriever.
- `filters_embedding`: Filters for the embedding retriever.
- `top_k_embedding`: The number of results to return from the embedding retriever.
- `filter_policy_embedding`: The filter policy for the embedding retriever.
- `custom_query_embedding`: A custom query for the embedding retriever.
- `join_mode`: The mode to use for joining the results from the BM25 and embedding retrievers.
- `weights`: The weights for the joiner.
- `top_k`: The number of results to return from the joiner.
- `sort_by_score`: Whether to sort the results by score.
- `**kwargs`: Additional keyword arguments. Use the following keys to pass extra parameters to the retrievers:
- "bm25_retriever" -> OpenSearchBM25Retriever
- "embedding_retriever" -> OpenSearchEmbeddingRetriever

<a id="haystack_integrations.components.retrievers.opensearch.open_search_hybrid_retriever.OpenSearchHybridRetriever.to_dict"></a>

#### OpenSearchHybridRetriever.to\_dict

```python
def to_dict()
```

Serialize OpenSearchHybridRetriever to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.retrievers.opensearch.sql_retriever"></a>

## Module haystack\_integrations.components.retrievers.opensearch.sql\_retriever

<a id="haystack_integrations.components.retrievers.opensearch.sql_retriever.OpenSearchSQLRetriever"></a>

### OpenSearchSQLRetriever

Executes raw OpenSearch SQL queries against an OpenSearchDocumentStore.

This component allows you to execute SQL queries directly against the OpenSearch index,
which is useful for fetching metadata, aggregations, and other structured data at runtime.

Returns the raw JSON response from the OpenSearch SQL API.

<a id="haystack_integrations.components.retrievers.opensearch.sql_retriever.OpenSearchSQLRetriever.__init__"></a>

#### OpenSearchSQLRetriever.\_\_init\_\_

```python
def __init__(*,
             document_store: OpenSearchDocumentStore,
             raise_on_failure: bool = True,
             fetch_size: int | None = None)
```

Creates the OpenSearchSQLRetriever component.

**Arguments**:

- `document_store`: An instance of OpenSearchDocumentStore to use with the Retriever.
- `raise_on_failure`: Whether to raise an exception if the API call fails. Otherwise, log a warning and return None.
- `fetch_size`: Optional number of results to fetch per page. If not provided, the default
fetch size set in OpenSearch is used.

**Raises**:

- `ValueError`: If `document_store` is not an instance of OpenSearchDocumentStore.

<a id="haystack_integrations.components.retrievers.opensearch.sql_retriever.OpenSearchSQLRetriever.to_dict"></a>

#### OpenSearchSQLRetriever.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.retrievers.opensearch.sql_retriever.OpenSearchSQLRetriever.from_dict"></a>

#### OpenSearchSQLRetriever.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "OpenSearchSQLRetriever"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.retrievers.opensearch.sql_retriever.OpenSearchSQLRetriever.run"></a>

#### OpenSearchSQLRetriever.run

```python
@component.output_types(result=dict[str, Any])
def run(query: str,
        document_store: OpenSearchDocumentStore | None = None,
        fetch_size: int | None = None) -> dict[str, dict[str, Any]]
```

Execute a raw OpenSearch SQL query against the index.

**Arguments**:

- `query`: The OpenSearch SQL query to execute.
- `document_store`: Optionally, an instance of OpenSearchDocumentStore to use with the Retriever.
- `fetch_size`: Optional number of results to fetch per page. If not provided, uses the value
specified during initialization, or the default fetch size set in OpenSearch.

**Returns**:

A dictionary containing the raw JSON response from OpenSearch SQL API:
- result: The raw JSON response from OpenSearch (dict) or None on error.

Example:
    ```python
    retriever = OpenSearchSQLRetriever(document_store=document_store)
    result = retriever.run(
        query="SELECT content, category FROM my_index WHERE category = 'A'"
    )
    # result["result"] contains the raw OpenSearch JSON response
    # For regular queries: result["result"]["hits"]["hits"] contains documents
    # For aggregate queries: result["result"]["aggregations"] contains aggregations
    ```

<a id="haystack_integrations.components.retrievers.opensearch.sql_retriever.OpenSearchSQLRetriever.run_async"></a>

#### OpenSearchSQLRetriever.run\_async

```python
@component.output_types(result=dict[str, Any])
async def run_async(
        query: str,
        document_store: OpenSearchDocumentStore | None = None,
        fetch_size: int | None = None) -> dict[str, dict[str, Any]]
```

Asynchronously execute a raw OpenSearch SQL query against the index.

**Arguments**:

- `query`: The OpenSearch SQL query to execute.
- `document_store`: Optionally, an instance of OpenSearchDocumentStore to use with the Retriever.
- `fetch_size`: Optional number of results to fetch per page. If not provided, uses the value
specified during initialization, or the default fetch size set in OpenSearch.

**Returns**:

A dictionary containing the raw JSON response from OpenSearch SQL API:
- result: The raw JSON response from OpenSearch (dict) or None on error.

Example:
    ```python
    retriever = OpenSearchSQLRetriever(document_store=document_store)
    result = await retriever.run_async(
        query="SELECT content, category FROM my_index WHERE category = 'A'"
    )
    # result["result"] contains the raw OpenSearch JSON response
    # For regular queries: result["result"]["hits"]["hits"] contains documents
    # For aggregate queries: result["result"]["aggregations"] contains aggregations
    ```

<a id="haystack_integrations.document_stores.opensearch.document_store"></a>

## Module haystack\_integrations.document\_stores.opensearch.document\_store

<a id="haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore"></a>

### OpenSearchDocumentStore

An instance of an OpenSearch database you can use to store all types of data.

This document store is a thin wrapper around the OpenSearch client.
It allows you to store and retrieve documents from an OpenSearch index.

Usage example:
```python
from haystack_integrations.document_stores.opensearch import (
    OpenSearchDocumentStore,
)
from haystack import Document

document_store = OpenSearchDocumentStore(hosts="localhost:9200")

document_store.write_documents(
    [
        Document(content="My first document", id="1"),
        Document(content="My second document", id="2"),
    ]
)

print(document_store.count_documents())
# 2

print(document_store.filter_documents())
# [Document(id='1', content='My first document', ...), Document(id='2', content='My second document', ...)]
```

<a id="haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore.__init__"></a>

#### OpenSearchDocumentStore.\_\_init\_\_

```python
def __init__(
        *,
        hosts: Hosts | None = None,
        index: str = "default",
        max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES,
        embedding_dim: int = 768,
        return_embedding: bool = False,
        method: dict[str, Any] | None = None,
        mappings: dict[str, Any] | None = None,
        settings: dict[str, Any] | None = DEFAULT_SETTINGS,
        create_index: bool = True,
        http_auth: Any = (
            Secret.from_env_var("OPENSEARCH_USERNAME",
                                strict=False),  # noqa: B008
            Secret.from_env_var("OPENSEARCH_PASSWORD",
                                strict=False),  # noqa: B008
        ),
        use_ssl: bool | None = None,
        verify_certs: bool | None = None,
        timeout: int | None = None,
        **kwargs: Any) -> None
```

Creates a new OpenSearchDocumentStore instance.

The ``embeddings_dim``, ``method``, ``mappings``, and ``settings`` arguments are only used if the index does not
exist and needs to be created. If the index already exists, its current configurations will be used.

For more information on connection parameters, see the [official OpenSearch documentation](https://opensearch.org/docs/latest/clients/python-low-level/`connecting`-to-opensearch)

**Arguments**:

- `hosts`: List of hosts running the OpenSearch client. Defaults to None
- `index`: Name of index in OpenSearch, if it doesn't exist it will be created. Defaults to "default"
- `max_chunk_bytes`: Maximum size of the requests in bytes. Defaults to 100MB
- `embedding_dim`: Dimension of the embeddings. Defaults to 768
- `return_embedding`: Whether to return the embedding of the retrieved Documents. This parameter also applies to the
`filter_documents` and `filter_documents_async` methods.
- `method`: The method definition of the underlying configuration of the approximate k-NN algorithm. Please
see the [official OpenSearch docs](https://opensearch.org/docs/latest/search-plugins/knn/knn-index/`method`-definitions)
for more information. Defaults to None
- `mappings`: The mapping of how the documents are stored and indexed. Please see the [official OpenSearch docs](https://opensearch.org/docs/latest/field-types/)
for more information. If None, it uses the embedding_dim and method arguments to create default mappings.
Defaults to None
- `settings`: The settings of the index to be created. Please see the [official OpenSearch docs](https://opensearch.org/docs/latest/search-plugins/knn/knn-index/`index`-settings)
for more information. Defaults to `{"index.knn": True}`.
- `create_index`: Whether to create the index if it doesn't exist. Defaults to True
- `http_auth`: http_auth param passed to the underlying connection class.
For basic authentication with default connection class `Urllib3HttpConnection` this can be
- a tuple of (username, password)
- a list of [username, password]
- a string of "username:password"
If not provided, will read values from OPENSEARCH_USERNAME and OPENSEARCH_PASSWORD environment variables.
For AWS authentication with `Urllib3HttpConnection` pass an instance of `AWSAuth`.
Defaults to None
- `use_ssl`: Whether to use SSL. Defaults to None
- `verify_certs`: Whether to verify certificates. Defaults to None
- `timeout`: Timeout in seconds. Defaults to None
- `**kwargs`: Optional arguments that ``OpenSearch`` takes. For the full list of supported kwargs,
see the [official OpenSearch reference](https://opensearch-project.github.io/opensearch-py/api-ref/clients/opensearch_client.html)

<a id="haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore.create_index"></a>

#### OpenSearchDocumentStore.create\_index

```python
def create_index(index: str | None = None,
                 mappings: dict[str, Any] | None = None,
                 settings: dict[str, Any] | None = None) -> None
```

Creates an index in OpenSearch.

Note that this method ignores the `create_index` argument from the constructor.

**Arguments**:

- `index`: Name of the index to create. If None, the index name from the constructor is used.
- `mappings`: The mapping of how the documents are stored and indexed. Please see the [official OpenSearch docs](https://opensearch.org/docs/latest/field-types/)
for more information. If None, the mappings from the constructor are used.
- `settings`: The settings of the index to be created. Please see the [official OpenSearch docs](https://opensearch.org/docs/latest/search-plugins/knn/knn-index/`index`-settings)
for more information. If None, the settings from the constructor are used.

<a id="haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore.to_dict"></a>

#### OpenSearchDocumentStore.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore.from_dict"></a>

#### OpenSearchDocumentStore.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "OpenSearchDocumentStore"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore.count_documents"></a>

#### OpenSearchDocumentStore.count\_documents

```python
def count_documents() -> int
```

Returns how many documents are present in the document store.

<a id="haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore.count_documents_async"></a>

#### OpenSearchDocumentStore.count\_documents\_async

```python
async def count_documents_async() -> int
```

Asynchronously returns the total number of documents in the document store.

<a id="haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore.filter_documents"></a>

#### OpenSearchDocumentStore.filter\_documents

```python
def filter_documents(filters: dict[str, Any] | None = None) -> list[Document]
```

Returns the documents that match the filters provided.

For a detailed specification of the filters,
refer to the [documentation](https://docs.haystack.deepset.ai/docs/metadata-filtering)

**Arguments**:

- `filters`: The filters to apply to the document list.

**Returns**:

A list of Documents that match the given filters.

<a id="haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore.filter_documents_async"></a>

#### OpenSearchDocumentStore.filter\_documents\_async

```python
async def filter_documents_async(
        filters: dict[str, Any] | None = None) -> list[Document]
```

Asynchronously returns the documents that match the filters provided.

For a detailed specification of the filters,
refer to the [documentation](https://docs.haystack.deepset.ai/docs/metadata-filtering)

**Arguments**:

- `filters`: The filters to apply to the document list.

**Returns**:

A list of Documents that match the given filters.

<a id="haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore.write_documents"></a>

#### OpenSearchDocumentStore.write\_documents

```python
def write_documents(
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE,
        refresh: Literal["wait_for", True, False] = "wait_for") -> int
```

Writes documents to the document store.

**Arguments**:

- `documents`: A list of Documents to write to the document store.
- `policy`: The duplicate policy to use when writing documents.
- `refresh`: Controls when changes are made visible to search operations.
- `True`: Force refresh immediately after the operation.
- `False`: Do not refresh (better performance for bulk operations).
- `"wait_for"`: Wait for the next refresh cycle (default, ensures read-your-writes consistency).
For more details, see the [OpenSearch refresh documentation](https://opensearch.org/docs/latest/api-reference/document-apis/index-document/).

**Raises**:

- `DuplicateDocumentError`: If a document with the same id already exists in the document store
and the policy is set to `DuplicatePolicy.FAIL` (or not specified).

**Returns**:

The number of documents written to the document store.

<a id="haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore.write_documents_async"></a>

#### OpenSearchDocumentStore.write\_documents\_async

```python
async def write_documents_async(
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE,
        refresh: Literal["wait_for", True, False] = "wait_for") -> int
```

Asynchronously writes documents to the document store.

**Arguments**:

- `documents`: A list of Documents to write to the document store.
- `policy`: The duplicate policy to use when writing documents.
- `refresh`: Controls when changes are made visible to search operations.
- `True`: Force refresh immediately after the operation.
- `False`: Do not refresh (better performance for bulk operations).
- `"wait_for"`: Wait for the next refresh cycle (default, ensures read-your-writes consistency).
For more details, see the [OpenSearch refresh documentation](https://opensearch.org/docs/latest/api-reference/document-apis/index-document/).

**Returns**:

The number of documents written to the document store.

<a id="haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore.delete_documents"></a>

#### OpenSearchDocumentStore.delete\_documents

```python
def delete_documents(document_ids: list[str],
                     refresh: Literal["wait_for", True, False] = "wait_for",
                     routing: dict[str, str] | None = None) -> None
```

Deletes documents that match the provided `document_ids` from the document store.

**Arguments**:

- `document_ids`: the document ids to delete
- `refresh`: Controls when changes are made visible to search operations.
- `True`: Force refresh immediately after the operation.
- `False`: Do not refresh (better performance for bulk operations).
- `"wait_for"`: Wait for the next refresh cycle (default, ensures read-your-writes consistency).
For more details, see the [OpenSearch refresh documentation](https://opensearch.org/docs/latest/api-reference/document-apis/index-document/).
- `routing`: A dictionary mapping document IDs to their routing values.
Routing values are used to determine the shard where documents are stored.
If provided, the routing value for each document will be used during deletion.

<a id="haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore.delete_documents_async"></a>

#### OpenSearchDocumentStore.delete\_documents\_async

```python
async def delete_documents_async(
        document_ids: list[str],
        refresh: Literal["wait_for", True, False] = "wait_for",
        routing: dict[str, str] | None = None) -> None
```

Asynchronously deletes documents that match the provided `document_ids` from the document store.

**Arguments**:

- `document_ids`: the document ids to delete
- `refresh`: Controls when changes are made visible to search operations.
- `True`: Force refresh immediately after the operation.
- `False`: Do not refresh (better performance for bulk operations).
- `"wait_for"`: Wait for the next refresh cycle (default, ensures read-your-writes consistency).
For more details, see the [OpenSearch refresh documentation](https://opensearch.org/docs/latest/api-reference/document-apis/index-document/).
- `routing`: A dictionary mapping document IDs to their routing values.
Routing values are used to determine the shard where documents are stored.
If provided, the routing value for each document will be used during deletion.

<a id="haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore.delete_all_documents"></a>

#### OpenSearchDocumentStore.delete\_all\_documents

```python
def delete_all_documents(recreate_index: bool = False,
                         refresh: bool = True) -> None
```

Deletes all documents in the document store.

**Arguments**:

- `recreate_index`: If True, the index will be deleted and recreated with the original mappings and
settings. If False, all documents will be deleted using the `delete_by_query` API.
- `refresh`: If True, OpenSearch refreshes all shards involved in the delete by query after the request
completes. If False, no refresh is performed. For more details, see the
[OpenSearch delete_by_query refresh documentation](https://opensearch.org/docs/latest/api-reference/document-apis/delete-by-query/).

<a id="haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore.delete_all_documents_async"></a>

#### OpenSearchDocumentStore.delete\_all\_documents\_async

```python
async def delete_all_documents_async(recreate_index: bool = False,
                                     refresh: bool = True) -> None
```

Asynchronously deletes all documents in the document store.

**Arguments**:

- `recreate_index`: If True, the index will be deleted and recreated with the original mappings and
settings. If False, all documents will be deleted using the `delete_by_query` API.
- `refresh`: If True, OpenSearch refreshes all shards involved in the delete by query after the request
completes. If False, no refresh is performed. For more details, see the
[OpenSearch delete_by_query refresh documentation](https://opensearch.org/docs/latest/api-reference/document-apis/delete-by-query/).

<a id="haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore.delete_by_filter"></a>

#### OpenSearchDocumentStore.delete\_by\_filter

```python
def delete_by_filter(filters: dict[str, Any], refresh: bool = False) -> int
```

Deletes all documents that match the provided filters.

**Arguments**:

- `filters`: The filters to apply to select documents for deletion.
For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- `refresh`: If True, OpenSearch refreshes all shards involved in the delete by query after the request
completes. If False, no refresh is performed. For more details, see the
[OpenSearch delete_by_query refresh documentation](https://opensearch.org/docs/latest/api-reference/document-apis/delete-by-query/).

**Returns**:

The number of documents deleted.

<a id="haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore.delete_by_filter_async"></a>

#### OpenSearchDocumentStore.delete\_by\_filter\_async

```python
async def delete_by_filter_async(filters: dict[str, Any],
                                 refresh: bool = False) -> int
```

Asynchronously deletes all documents that match the provided filters.

**Arguments**:

- `filters`: The filters to apply to select documents for deletion.
For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- `refresh`: If True, OpenSearch refreshes all shards involved in the delete by query after the request
completes. If False, no refresh is performed. For more details, see the
[OpenSearch delete_by_query refresh documentation](https://opensearch.org/docs/latest/api-reference/document-apis/delete-by-query/).

**Returns**:

The number of documents deleted.

<a id="haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore.update_by_filter"></a>

#### OpenSearchDocumentStore.update\_by\_filter

```python
def update_by_filter(filters: dict[str, Any],
                     meta: dict[str, Any],
                     refresh: bool = False) -> int
```

Updates the metadata of all documents that match the provided filters.

**Arguments**:

- `filters`: The filters to apply to select documents for updating.
For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- `meta`: The metadata fields to update.
- `refresh`: If True, OpenSearch refreshes all shards involved in the update by query after the request
completes. If False, no refresh is performed. For more details, see the
[OpenSearch update_by_query refresh documentation](https://opensearch.org/docs/latest/api-reference/document-apis/update-by-query/).

**Returns**:

The number of documents updated.

<a id="haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore.update_by_filter_async"></a>

#### OpenSearchDocumentStore.update\_by\_filter\_async

```python
async def update_by_filter_async(filters: dict[str, Any],
                                 meta: dict[str, Any],
                                 refresh: bool = False) -> int
```

Asynchronously updates the metadata of all documents that match the provided filters.

**Arguments**:

- `filters`: The filters to apply to select documents for updating.
For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- `meta`: The metadata fields to update.
- `refresh`: If True, OpenSearch refreshes all shards involved in the update by query after the request
completes. If False, no refresh is performed. For more details, see the
[OpenSearch update_by_query refresh documentation](https://opensearch.org/docs/latest/api-reference/document-apis/update-by-query/).

**Returns**:

The number of documents updated.

<a id="haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore.count_documents_by_filter"></a>

#### OpenSearchDocumentStore.count\_documents\_by\_filter

```python
def count_documents_by_filter(filters: dict[str, Any]) -> int
```

Returns the number of documents that match the provided filters.

**Arguments**:

- `filters`: The filters to apply to count documents.
For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)

**Returns**:

The number of documents that match the filters.

<a id="haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore.count_documents_by_filter_async"></a>

#### OpenSearchDocumentStore.count\_documents\_by\_filter\_async

```python
async def count_documents_by_filter_async(filters: dict[str, Any]) -> int
```

Asynchronously returns the number of documents that match the provided filters.

**Arguments**:

- `filters`: The filters to apply to count documents.
For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)

**Returns**:

The number of documents that match the filters.

<a id="haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore.count_unique_metadata_by_filter"></a>

#### OpenSearchDocumentStore.count\_unique\_metadata\_by\_filter

```python
def count_unique_metadata_by_filter(
        filters: dict[str, Any], metadata_fields: list[str]) -> dict[str, int]
```

Returns the number of unique values for each specified metadata field of the documents

that match the provided filters.

**Arguments**:

- `filters`: The filters to apply to count documents.
For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- `metadata_fields`: List of field names to calculate unique values for.
Field names can include or omit the "meta." prefix.

**Raises**:

- `ValueError`: If any of the requested fields don't exist in the index mapping.

**Returns**:

A dictionary mapping each metadata field name to the count of its unique values among the filtered
documents.

<a id="haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore.count_unique_metadata_by_filter_async"></a>

#### OpenSearchDocumentStore.count\_unique\_metadata\_by\_filter\_async

```python
async def count_unique_metadata_by_filter_async(
        filters: dict[str, Any], metadata_fields: list[str]) -> dict[str, int]
```

Asynchronously returns the number of unique values for each specified metadata field of the documents

that match the provided filters.

**Arguments**:

- `filters`: The filters to apply to count documents.
For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- `metadata_fields`: List of field names to calculate unique values for.
Field names can include or omit the "meta." prefix.

**Raises**:

- `ValueError`: If any of the requested fields don't exist in the index mapping.

**Returns**:

A dictionary mapping each metadata field name to the count of its unique values among the filtered
documents.

<a id="haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore.get_metadata_fields_info"></a>

#### OpenSearchDocumentStore.get\_metadata\_fields\_info

```python
def get_metadata_fields_info() -> dict[str, dict[str, str]]
```

Returns the information about the fields in the index.

If we populated the index with documents like:

```python
    Document(content="Doc 1", meta={"category": "A", "status": "active", "priority": 1})
    Document(content="Doc 2", meta={"category": "B", "status": "inactive"})
```

This method would return:

```python
    {
        'content': {'type': 'text'},
        'category': {'type': 'keyword'},
        'status': {'type': 'keyword'},
        'priority': {'type': 'long'},
    }
```

**Returns**:

The information about the fields in the index.

<a id="haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore.get_metadata_fields_info_async"></a>

#### OpenSearchDocumentStore.get\_metadata\_fields\_info\_async

```python
async def get_metadata_fields_info_async() -> dict[str, dict[str, str]]
```

Asynchronously returns the information about the fields in the index.

If we populated the index with documents like:

```python
    Document(content="Doc 1", meta={"category": "A", "status": "active", "priority": 1})
    Document(content="Doc 2", meta={"category": "B", "status": "inactive"})
```

This method would return:

```python
    {
        'content': {'type': 'text'},
        'category': {'type': 'keyword'},
        'status': {'type': 'keyword'},
        'priority': {'type': 'long'},
    }
```

**Returns**:

The information about the fields in the index.

<a id="haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore.get_metadata_field_min_max"></a>

#### OpenSearchDocumentStore.get\_metadata\_field\_min\_max

```python
def get_metadata_field_min_max(metadata_field: str) -> dict[str, int | None]
```

Returns the minimum and maximum values for the given metadata field.

**Arguments**:

- `metadata_field`: The metadata field to get the minimum and maximum values for.

**Returns**:

A dictionary with the keys "min" and "max", where each value is the minimum or maximum value of the
metadata field across all documents.

<a id="haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore.get_metadata_field_min_max_async"></a>

#### OpenSearchDocumentStore.get\_metadata\_field\_min\_max\_async

```python
async def get_metadata_field_min_max_async(
        metadata_field: str) -> dict[str, int | None]
```

Asynchronously returns the minimum and maximum values for the given metadata field.

**Arguments**:

- `metadata_field`: The metadata field to get the minimum and maximum values for.

**Returns**:

A dictionary with the keys "min" and "max", where each value is the minimum or maximum value of the
metadata field across all documents.

<a id="haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore.get_metadata_field_unique_values"></a>

#### OpenSearchDocumentStore.get\_metadata\_field\_unique\_values

```python
def get_metadata_field_unique_values(
    metadata_field: str,
    search_term: str | None = None,
    size: int | None = 10000,
    after: dict[str, Any] | None = None
) -> tuple[list[str], dict[str, Any] | None]
```

Returns unique values for a metadata field, optionally filtered by a search term in the content.

Uses composite aggregations for proper pagination beyond 10k results.

**Arguments**:

- `metadata_field`: The metadata field to get unique values for.
- `search_term`: Optional search term to filter documents by matching in the content field.
- `size`: The number of unique values to return per page. Defaults to 10000.
- `after`: Optional pagination key from the previous response. Use None for the first page.
For subsequent pages, pass the `after_key` from the previous response.

**Returns**:

A tuple containing (list of unique values, after_key for pagination).
The after_key is None when there are no more results. Use it in the `after` parameter
for the next page.

<a id="haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore.get_metadata_field_unique_values_async"></a>

#### OpenSearchDocumentStore.get\_metadata\_field\_unique\_values\_async

```python
async def get_metadata_field_unique_values_async(
    metadata_field: str,
    search_term: str | None = None,
    size: int | None = 10000,
    after: dict[str, Any] | None = None
) -> tuple[list[str], dict[str, Any] | None]
```

Asynchronously returns unique values for a metadata field, optionally filtered by a search term in the content.

Uses composite aggregations for proper pagination beyond 10k results.

**Arguments**:

- `metadata_field`: The metadata field to get unique values for.
- `search_term`: Optional search term to filter documents by matching in the content field.
- `size`: The number of unique values to return per page. Defaults to 10000.
- `after`: Optional pagination key from the previous response. Use None for the first page.
For subsequent pages, pass the `after_key` from the previous response.

**Returns**:

A tuple containing (list of unique values, after_key for pagination).
The after_key is None when there are no more results. Use it in the `after` parameter
for the next page.

<a id="haystack_integrations.document_stores.opensearch.filters"></a>

## Module haystack\_integrations.document\_stores.opensearch.filters

<a id="haystack_integrations.document_stores.opensearch.filters.normalize_filters"></a>

#### normalize\_filters

```python
def normalize_filters(filters: dict[str, Any]) -> dict[str, Any]
```

Converts Haystack filters in OpenSearch compatible filters.

