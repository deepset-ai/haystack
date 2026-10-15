---
title: "Vespa"
id: integrations-vespa
description: "Vespa integration for Haystack"
slug: "/integrations-vespa"
---


## haystack_integrations.components.retrievers.vespa.embedding_retriever

### VespaEmbeddingRetriever

Retrieve documents from Vespa using dense vector similarity.

#### __init__

```python
__init__(
    *,
    document_store: VespaDocumentStore,
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
    ranking: str | None = DEFAULT_SEMANTIC_RANKING,
    query_tensor_name: str = "query_embedding",
    target_hits: int | None = None
) -> None
```

Create a Vespa embedding retriever.

**Parameters:**

- **document_store** (<code>VespaDocumentStore</code>) – Configured `VespaDocumentStore` for your application, for example
  `VespaDocumentStore(url="http://localhost", schema="doc", namespace="doc")` aligned with your
  Vespa schema. See https://docs.vespa.ai/en/basics/documents.html and the integration package README.
- **filters** (<code>dict\[str, Any\] | None</code>) – Optional static Haystack metadata filters unless overridden in :meth:`run`, for example
  `{"field": "meta.category", "operator": "==", "value": "news"}`. See
  https://docs.haystack.deepset.ai/docs/metadata-filtering and https://docs.vespa.ai/en/query-language.html.
- **top_k** (<code>int</code>) – Default maximum number of documents to return per query (for example `10`).
- **ranking** (<code>str | None</code>) – Vespa rank profile used after nearest-neighbor retrieval, for example `semantic` for a
  profile that scores with `closeness(field, embedding)`. Defaults to `semantic`. Pass `None` to use the
  schema default profile. See https://docs.vespa.ai/en/basics/ranking.html.
- **query_tensor_name** (<code>str</code>) – Name of the query tensor in YQL and in `input.query(...)` in your rank profile.
  For example `query_embedding` matches the default `semantic` profile. See
  https://docs.vespa.ai/en/nearest-neighbor-search.html.
- **target_hits** (<code>int | None</code>) – Optional nearest-neighbor `targetHits` value, for example `10` or `100`: how many
  neighbors are considered per content node before first-phase ranking. See
  https://docs.vespa.ai/en/nearest-neighbor-search.html.

**Raises:**

- <code>ValueError</code> – If `document_store` is not an instance of VespaDocumentStore.

#### run

```python
run(
    query_embedding: list[float],
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
) -> dict[str, list[Document]]
```

Retrieve documents from Vespa.

**Parameters:**

- **query_embedding** (<code>list\[float\]</code>) – Dense query embedding.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied when fetching documents from the Document Store.
- **top_k** (<code>int | None</code>) – Maximum number of documents to return.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – Retrieved documents.

## haystack_integrations.components.retrievers.vespa.keyword_retriever

### VespaKeywordRetriever

Retrieve documents from Vespa using lexical search.

#### __init__

```python
__init__(
    *,
    document_store: VespaDocumentStore,
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
    ranking: str | None = DEFAULT_BM25_RANKING
) -> None
```

Create a Vespa keyword retriever.

**Parameters:**

- **document_store** (<code>VespaDocumentStore</code>) – Configured `VespaDocumentStore` for your application, for example
  `VespaDocumentStore(url="http://localhost", schema="doc", namespace="doc")` so it matches the deployed
  schema and endpoint. See https://docs.vespa.ai/en/basics/documents.html and the integration package README.
- **filters** (<code>dict\[str, Any\] | None</code>) – Optional static Haystack metadata filters applied on each retrieval unless overridden in
  :meth:`run`, for example `{"field": "meta.category", "operator": "==", "value": "news"}`. See
  https://docs.haystack.deepset.ai/docs/metadata-filtering and https://docs.vespa.ai/en/query-language.html.
- **top_k** (<code>int</code>) – Default maximum number of documents to return per query (for example `10`).
- **ranking** (<code>str | None</code>) – Vespa rank profile for lexical matches, for example `bm25` for a profile that uses
  `bm25(content)`. Defaults to `bm25`. Pass `None` to use the schema default. See
  https://docs.vespa.ai/en/basics/ranking.html.

**Raises:**

- <code>ValueError</code> – If `document_store` is not an instance of VespaDocumentStore.

#### run

```python
run(
    query: str, filters: dict[str, Any] | None = None, top_k: int | None = None
) -> dict[str, list[Document]]
```

Retrieve documents from Vespa.

**Parameters:**

- **query** (<code>str</code>) – Query text.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied when fetching documents from the Document Store.
- **top_k** (<code>int | None</code>) – Maximum number of documents to return.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – Retrieved documents.

## haystack_integrations.document_stores.vespa.document_store

### VespaDocumentStore

Document store backed by an existing [Vespa](https://vespa.ai/) application.

#### __init__

```python
__init__(
    *,
    url: str | None = None,
    port: int = 8080,
    cert: Secret | None = None,
    key: Secret | None = None,
    vespa_cloud_secret_token: Secret | None = None,
    additional_headers: dict[str, str] | None = None,
    content_cluster_name: str = "content",
    schema: str = "doc",
    namespace: str | None = None,
    groupname: str | None = None,
    content_field: str = "content",
    embedding_field: str = "embedding",
    id_field: str = "id",
    metadata_fields: list[str] | None = None,
    query_limit: int = DEFAULT_QUERY_LIMIT
) -> None
```

Create a new Vespa document store.

**Parameters:**

- **url** (<code>str | None</code>) – Vespa endpoint base URL. If omitted, the `VESPA_URL` environment variable is used.
- **port** (<code>int</code>) – Vespa HTTP port.
- **cert** (<code>Secret | None</code>) – Secret resolving to the data plane certificate file path for mTLS authentication.
- **key** (<code>Secret | None</code>) – Secret resolving to the data plane key file path for mTLS authentication.
- **vespa_cloud_secret_token** (<code>Secret | None</code>) – Vespa Cloud data plane secret token for token authentication.
  If omitted, the `VESPA_CLOUD_SECRET_TOKEN` environment variable is used when set, matching pyvespa.
- **additional_headers** (<code>dict\[str, str\] | None</code>) – Additional headers to send to the Vespa application.
- **content_cluster_name** (<code>str</code>) – Vespa content cluster name.
- **schema** (<code>str</code>) – Vespa schema name to read from and write to.
- **namespace** (<code>str | None</code>) – Vespa namespace. Defaults to the schema name when omitted.
- **groupname** (<code>str | None</code>) – Optional Vespa group name.
- **content_field** (<code>str</code>) – Vespa field containing the document text.
- **embedding_field** (<code>str</code>) – Vespa field containing the dense embedding.
- **id_field** (<code>str</code>) – Optional Vespa field containing the document id in query responses.
  Vespa document IDs are always written via `data_id`. If this field is missing in the
  schema or summaries, the integration falls back to parsing the Vespa document path.
- **metadata_fields** (<code>list\[str\] | None</code>) – Optional allowlist of metadata fields to feed and return.
- **query_limit** (<code>int</code>) – Maximum number of documents returned by bulk queries. Defaults to 400 to
  stay within Vespa's common query hit limit unless explicitly overridden.

#### app

```python
app: Any
```

Return the underlying `pyvespa` `Vespa` HTTP client.

It is built from this store's `url`, `port`, and authentication settings
(`cert`, `key`, `vespa_cloud_secret_token`, `additional_headers`) so mTLS, bearer token,
and custom headers from the constructor (or environment) are applied.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the document store to a dictionary.

Uses the same init-parameter names as :meth:`__init__` and `default_to_dict` so nested serialization stays
aligned with Haystack's default component serialization.

**Returns:**

- <code>dict\[str, Any\]</code> – Serialized document store data.

#### count_documents

```python
count_documents() -> int
```

Return the total number of documents in Vespa.

**Returns:**

- <code>int</code> – Document count.

#### count_documents_by_filter

```python
count_documents_by_filter(filters: dict[str, Any]) -> int
```

Return the number of documents matching the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – Haystack metadata filters.

**Returns:**

- <code>int</code> – Count of matching documents.

#### write_documents

```python
write_documents(
    documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
) -> int
```

Write documents to Vespa.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – Documents to store.
- **policy** (<code>DuplicatePolicy</code>) – Duplicate handling policy.

**Returns:**

- <code>int</code> – Number of documents written.

#### delete_documents

```python
delete_documents(document_ids: list[str]) -> None
```

Delete documents by id.

**Parameters:**

- **document_ids** (<code>list\[str\]</code>) – Document ids to delete.

#### delete_all_documents

```python
delete_all_documents() -> None
```

Delete all documents for this store's schema, namespace, and content cluster.

Implemented with pyvespa `Vespa.delete_all_docs` (Document V1 bulk delete).

#### delete_by_filter

```python
delete_by_filter(filters: dict[str, Any]) -> int
```

Delete all documents matching the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – Haystack metadata filters.

**Returns:**

- <code>int</code> – Number of deleted documents.

#### update_by_filter

```python
update_by_filter(filters: dict[str, Any], meta: dict[str, Any]) -> int
```

Update metadata fields for documents matching the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – Haystack metadata filters.
- **meta** (<code>dict\[str, Any\]</code>) – Metadata values to merge into the matched documents.

**Returns:**

- <code>int</code> – Number of updated documents.

#### get_documents_by_id

```python
get_documents_by_id(document_ids: list[str]) -> list[Document]
```

Retrieve documents by their ids.

**Parameters:**

- **document_ids** (<code>list\[str\]</code>) – Document ids to fetch.

**Returns:**

- <code>list\[Document\]</code> – Matching documents.

#### filter_documents

```python
filter_documents(filters: dict[str, Any] | None = None) -> list[Document]
```

Retrieve documents matching the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\] | None</code>) – Haystack metadata filters.

**Returns:**

- <code>list\[Document\]</code> – Matching documents.

#### get_metadata_fields_info

```python
get_metadata_fields_info() -> dict[str, dict[str, str]]
```

Return best-effort metadata field information based on configured fields.

**Returns:**

- <code>dict\[str, dict\[str, str\]\]</code> – Field metadata information.

## haystack_integrations.document_stores.vespa.filters
