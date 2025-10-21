---
title: "OpenSearch"
id: integrations-opensearch
description: "OpenSearch integration for Haystack"
slug: "/integrations-opensearch"
---

<a id="haystack_integrations.components.retrievers.opensearch.bm25_retriever"></a>

# Module haystack\_integrations.components.retrievers.opensearch.bm25\_retriever

<a id="haystack_integrations.components.retrievers.opensearch.bm25_retriever.OpenSearchBM25Retriever"></a>

## OpenSearchBM25Retriever

Fetches documents from OpenSearchDocumentStore using the keyword-based BM25 algorithm.

BM25 computes a weighted word overlap between the query string and a document to determine its similarity.

<a id="haystack_integrations.components.retrievers.opensearch.bm25_retriever.OpenSearchBM25Retriever.__init__"></a>

#### OpenSearchBM25Retriever.\_\_init\_\_

```python
def __init__(*,
             document_store: OpenSearchDocumentStore,
             filters: Optional[Dict[str, Any]] = None,
             fuzziness: Union[int, str] = "AUTO",
             top_k: int = 10,
             scale_score: bool = False,
             all_terms_must_match: bool = False,
             filter_policy: Union[str, FilterPolicy] = FilterPolicy.REPLACE,
             custom_query: Optional[Dict[str, Any]] = None,
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
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.retrievers.opensearch.bm25_retriever.OpenSearchBM25Retriever.from_dict"></a>

#### OpenSearchBM25Retriever.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "OpenSearchBM25Retriever"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.retrievers.opensearch.bm25_retriever.OpenSearchBM25Retriever.run"></a>

#### OpenSearchBM25Retriever.run

```python
@component.output_types(documents=List[Document])
def run(
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    all_terms_must_match: Optional[bool] = None,
    top_k: Optional[int] = None,
    fuzziness: Optional[Union[int, str]] = None,
    scale_score: Optional[bool] = None,
    custom_query: Optional[Dict[str, Any]] = None,
    document_store: Optional[OpenSearchDocumentStore] = None
) -> Dict[str, List[Document]]
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
@component.output_types(documents=List[Document])
async def run_async(
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    all_terms_must_match: Optional[bool] = None,
    top_k: Optional[int] = None,
    fuzziness: Optional[Union[int, str]] = None,
    scale_score: Optional[bool] = None,
    custom_query: Optional[Dict[str, Any]] = None,
    document_store: Optional[OpenSearchDocumentStore] = None
) -> Dict[str, List[Document]]
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

# Module haystack\_integrations.components.retrievers.opensearch.embedding\_retriever

<a id="haystack_integrations.components.retrievers.opensearch.embedding_retriever.OpenSearchEmbeddingRetriever"></a>

## OpenSearchEmbeddingRetriever

Retrieves documents from the OpenSearchDocumentStore using a vector similarity metric.

 Must be connected to the OpenSearchDocumentStore to run.

<a id="haystack_integrations.components.retrievers.opensearch.embedding_retriever.OpenSearchEmbeddingRetriever.__init__"></a>

#### OpenSearchEmbeddingRetriever.\_\_init\_\_

```python
def __init__(*,
             document_store: OpenSearchDocumentStore,
             filters: Optional[Dict[str, Any]] = None,
             top_k: int = 10,
             filter_policy: Union[str, FilterPolicy] = FilterPolicy.REPLACE,
             custom_query: Optional[Dict[str, Any]] = None,
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
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.retrievers.opensearch.embedding_retriever.OpenSearchEmbeddingRetriever.from_dict"></a>

#### OpenSearchEmbeddingRetriever.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "OpenSearchEmbeddingRetriever"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.retrievers.opensearch.embedding_retriever.OpenSearchEmbeddingRetriever.run"></a>

#### OpenSearchEmbeddingRetriever.run

```python
@component.output_types(documents=List[Document])
def run(
    query_embedding: List[float],
    filters: Optional[Dict[str, Any]] = None,
    top_k: Optional[int] = None,
    custom_query: Optional[Dict[str, Any]] = None,
    efficient_filtering: Optional[bool] = None,
    document_store: Optional[OpenSearchDocumentStore] = None
) -> Dict[str, List[Document]]
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
@component.output_types(documents=List[Document])
async def run_async(
    query_embedding: List[float],
    filters: Optional[Dict[str, Any]] = None,
    top_k: Optional[int] = None,
    custom_query: Optional[Dict[str, Any]] = None,
    efficient_filtering: Optional[bool] = None,
    document_store: Optional[OpenSearchDocumentStore] = None
) -> Dict[str, List[Document]]
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

<a id="haystack_integrations.components.retrievers.opensearch.open_search_hybrid_retriever"></a>

# Module haystack\_integrations.components.retrievers.opensearch.open\_search\_hybrid\_retriever

<a id="haystack_integrations.components.retrievers.opensearch.open_search_hybrid_retriever.OpenSearchHybridRetriever"></a>

## OpenSearchHybridRetriever

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
             filters_bm25: Optional[Dict[str, Any]] = None,
             fuzziness: Union[int, str] = "AUTO",
             top_k_bm25: int = 10,
             scale_score: bool = False,
             all_terms_must_match: bool = False,
             filter_policy_bm25: Union[str,
                                       FilterPolicy] = FilterPolicy.REPLACE,
             custom_query_bm25: Optional[Dict[str, Any]] = None,
             filters_embedding: Optional[Dict[str, Any]] = None,
             top_k_embedding: int = 10,
             filter_policy_embedding: Union[
                 str, FilterPolicy] = FilterPolicy.REPLACE,
             custom_query_embedding: Optional[Dict[str, Any]] = None,
             join_mode: Union[str, JoinMode] = JoinMode.RECIPROCAL_RANK_FUSION,
             weights: Optional[List[float]] = None,
             top_k: Optional[int] = None,
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

<a id="haystack_integrations.document_stores.opensearch.document_store"></a>

# Module haystack\_integrations.document\_stores.opensearch.document\_store

<a id="haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore"></a>

## OpenSearchDocumentStore

An instance of an OpenSearch database you can use to store all types of data.

This document store is a thin wrapper around the OpenSearch client.
It allows you to store and retrieve documents from an OpenSearch index.

Usage example:
```python
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore
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
        hosts: Optional[Hosts] = None,
        index: str = "default",
        max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES,
        embedding_dim: int = 768,
        return_embedding: bool = False,
        method: Optional[Dict[str, Any]] = None,
        mappings: Optional[Dict[str, Any]] = None,
        settings: Optional[Dict[str, Any]] = DEFAULT_SETTINGS,
        create_index: bool = True,
        http_auth: Any = (
            Secret.from_env_var("OPENSEARCH_USERNAME",
                                strict=False),  # noqa: B008
            Secret.from_env_var("OPENSEARCH_PASSWORD",
                                strict=False),  # noqa: B008
        ),
        use_ssl: Optional[bool] = None,
        verify_certs: Optional[bool] = None,
        timeout: Optional[int] = None,
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
def create_index(index: Optional[str] = None,
                 mappings: Optional[Dict[str, Any]] = None,
                 settings: Optional[Dict[str, Any]] = None) -> None
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
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore.from_dict"></a>

#### OpenSearchDocumentStore.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "OpenSearchDocumentStore"
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
def filter_documents(
        filters: Optional[Dict[str, Any]] = None) -> List[Document]
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
        filters: Optional[Dict[str, Any]] = None) -> List[Document]
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
def write_documents(documents: List[Document],
                    policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int
```

Writes documents to the document store.

**Arguments**:

- `documents`: A list of Documents to write to the document store.
- `policy`: The duplicate policy to use when writing documents.

**Raises**:

- `DuplicateDocumentError`: If a document with the same id already exists in the document store
and the policy is set to `DuplicatePolicy.FAIL` (or not specified).

**Returns**:

The number of documents written to the document store.

<a id="haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore.write_documents_async"></a>

#### OpenSearchDocumentStore.write\_documents\_async

```python
async def write_documents_async(
        documents: List[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int
```

Asynchronously writes documents to the document store.

**Arguments**:

- `documents`: A list of Documents to write to the document store.
- `policy`: The duplicate policy to use when writing documents.

**Returns**:

The number of documents written to the document store.

<a id="haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore.delete_documents"></a>

#### OpenSearchDocumentStore.delete\_documents

```python
def delete_documents(document_ids: List[str]) -> None
```

Deletes documents that match the provided `document_ids` from the document store.

**Arguments**:

- `document_ids`: the document ids to delete

<a id="haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore.delete_documents_async"></a>

#### OpenSearchDocumentStore.delete\_documents\_async

```python
async def delete_documents_async(document_ids: List[str]) -> None
```

Asynchronously deletes documents that match the provided `document_ids` from the document store.

**Arguments**:

- `document_ids`: the document ids to delete

<a id="haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore.delete_all_documents"></a>

#### OpenSearchDocumentStore.delete\_all\_documents

```python
def delete_all_documents(recreate_index: bool = False) -> None
```

Deletes all documents in the document store.

**Arguments**:

- `recreate_index`: If True, the index will be deleted and recreated with the original mappings and
settings. If False, all documents will be deleted using the `delete_by_query` API.

<a id="haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore.delete_all_documents_async"></a>

#### OpenSearchDocumentStore.delete\_all\_documents\_async

```python
async def delete_all_documents_async(recreate_index: bool = False) -> None
```

Asynchronously deletes all documents in the document store.

**Arguments**:

- `recreate_index`: If True, the index will be deleted and recreated with the original mappings and
settings. If False, all documents will be deleted using the `delete_by_query` API.

<a id="haystack_integrations.document_stores.opensearch.filters"></a>

# Module haystack\_integrations.document\_stores.opensearch.filters

<a id="haystack_integrations.document_stores.opensearch.filters.normalize_filters"></a>

#### normalize\_filters

```python
def normalize_filters(filters: Dict[str, Any]) -> Dict[str, Any]
```

Converts Haystack filters in OpenSearch compatible filters.
