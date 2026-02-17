---
title: "Weaviate"
id: integrations-weaviate
description: "Weaviate integration for Haystack"
slug: "/integrations-weaviate"
---


## `haystack_integrations.components.retrievers.weaviate.bm25_retriever`

### `WeaviateBM25Retriever`

A component for retrieving documents from Weaviate using the BM25 algorithm.

Example usage:

```python
from haystack_integrations.document_stores.weaviate.document_store import (
    WeaviateDocumentStore,
)
from haystack_integrations.components.retrievers.weaviate.bm25_retriever import (
    WeaviateBM25Retriever,
)

document_store = WeaviateDocumentStore(url="http://localhost:8080")
retriever = WeaviateBM25Retriever(document_store=document_store)
retriever.run(query="How to make a pizza", top_k=3)
```

#### `__init__`

```python
__init__(
    *,
    document_store: WeaviateDocumentStore,
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
    filter_policy: str | FilterPolicy = FilterPolicy.REPLACE
)
```

Create a new instance of WeaviateBM25Retriever.

**Parameters:**

- **document_store** (<code>WeaviateDocumentStore</code>) – Instance of WeaviateDocumentStore that will be used from this retriever.
- **filters** (<code>dict\[str, Any\] | None</code>) – Custom filters applied when running the retriever
- **top_k** (<code>int</code>) – Maximum number of documents to return
- **filter_policy** (<code>str | FilterPolicy</code>) – Policy to determine how filters are applied.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> WeaviateBM25Retriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>WeaviateBM25Retriever</code> – Deserialized component.

#### `run`

```python
run(
    query: str, filters: dict[str, Any] | None = None, top_k: int | None = None
) -> dict[str, list[Document]]
```

Retrieves documents from Weaviate using the BM25 algorithm.

**Parameters:**

- **query** (<code>str</code>) – The query text.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved Documents. The way runtime filters are applied depends on
  the `filter_policy` chosen at retriever initialization. See init method docstring for more
  details.
- **top_k** (<code>int | None</code>) – The maximum number of documents to return.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: List of documents returned by the search engine.

#### `run_async`

```python
run_async(
    query: str, filters: dict[str, Any] | None = None, top_k: int | None = None
) -> dict[str, list[Document]]
```

Asynchronously retrieves documents from Weaviate using the BM25 algorithm.

**Parameters:**

- **query** (<code>str</code>) – The query text.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved Documents. The way runtime filters are applied depends on
  the `filter_policy` chosen at retriever initialization. See init method docstring for more
  details.
- **top_k** (<code>int | None</code>) – The maximum number of documents to return.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: List of documents returned by the search engine.

## `haystack_integrations.components.retrievers.weaviate.embedding_retriever`

### `WeaviateEmbeddingRetriever`

A retriever that uses Weaviate's vector search to find similar documents based on the embeddings of the query.

#### `__init__`

```python
__init__(
    *,
    document_store: WeaviateDocumentStore,
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
    distance: float | None = None,
    certainty: float | None = None,
    filter_policy: str | FilterPolicy = FilterPolicy.REPLACE
)
```

Creates a new instance of WeaviateEmbeddingRetriever.

**Parameters:**

- **document_store** (<code>WeaviateDocumentStore</code>) – Instance of WeaviateDocumentStore that will be used from this retriever.
- **filters** (<code>dict\[str, Any\] | None</code>) – Custom filters applied when running the retriever.
- **top_k** (<code>int</code>) – Maximum number of documents to return.
- **distance** (<code>float | None</code>) – The maximum allowed distance between Documents' embeddings.
- **certainty** (<code>float | None</code>) – Normalized distance between the result item and the search vector.
- **filter_policy** (<code>str | FilterPolicy</code>) – Policy to determine how filters are applied.

**Raises:**

- <code>ValueError</code> – If both `distance` and `certainty` are provided.
  See https://weaviate.io/developers/weaviate/api/graphql/search-operators#variables to learn more about
  `distance` and `certainty` parameters.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> WeaviateEmbeddingRetriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>WeaviateEmbeddingRetriever</code> – Deserialized component.

#### `run`

```python
run(
    query_embedding: list[float],
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
    distance: float | None = None,
    certainty: float | None = None,
) -> dict[str, list[Document]]
```

Retrieves documents from Weaviate using the vector search.

**Parameters:**

- **query_embedding** (<code>list\[float\]</code>) – Embedding of the query.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved Documents. The way runtime filters are applied depends on
  the `filter_policy` chosen at retriever initialization. See init method docstring for more
  details.
- **top_k** (<code>int | None</code>) – The maximum number of documents to return.
- **distance** (<code>float | None</code>) – The maximum allowed distance between Documents' embeddings.
- **certainty** (<code>float | None</code>) – Normalized distance between the result item and the search vector.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: List of documents returned by the search engine.

**Raises:**

- <code>ValueError</code> – If both `distance` and `certainty` are provided.
  See https://weaviate.io/developers/weaviate/api/graphql/search-operators#variables to learn more about
  `distance` and `certainty` parameters.

#### `run_async`

```python
run_async(
    query_embedding: list[float],
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
    distance: float | None = None,
    certainty: float | None = None,
) -> dict[str, list[Document]]
```

Asynchronously retrieves documents from Weaviate using the vector search.

**Parameters:**

- **query_embedding** (<code>list\[float\]</code>) – Embedding of the query.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved Documents. The way runtime filters are applied depends on
  the `filter_policy` chosen at retriever initialization. See init method docstring for more
  details.
- **top_k** (<code>int | None</code>) – The maximum number of documents to return.
- **distance** (<code>float | None</code>) – The maximum allowed distance between Documents' embeddings.
- **certainty** (<code>float | None</code>) – Normalized distance between the result item and the search vector.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: List of documents returned by the search engine.

**Raises:**

- <code>ValueError</code> – If both `distance` and `certainty` are provided.
  See https://weaviate.io/developers/weaviate/api/graphql/search-operators#variables to learn more about
  `distance` and `certainty` parameters.

## `haystack_integrations.components.retrievers.weaviate.hybrid_retriever`

### `WeaviateHybridRetriever`

A retriever that uses Weaviate's hybrid search to find similar documents based on the embeddings of the query.

#### `__init__`

```python
__init__(
    *,
    document_store: WeaviateDocumentStore,
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
    alpha: float | None = None,
    max_vector_distance: float | None = None,
    filter_policy: str | FilterPolicy = FilterPolicy.REPLACE
)
```

Creates a new instance of WeaviateHybridRetriever.

**Parameters:**

- **document_store** (<code>WeaviateDocumentStore</code>) – Instance of WeaviateDocumentStore that will be used from this retriever.
- **filters** (<code>dict\[str, Any\] | None</code>) – Custom filters applied when running the retriever.
- **top_k** (<code>int</code>) – Maximum number of documents to return.
- **alpha** (<code>float | None</code>) – Blending factor for hybrid retrieval in Weaviate. Must be in the range `[0.0, 1.0]`.

Weaviate hybrid search combines keyword (BM25) and vector scores into a single ranking. `alpha` controls
how much each part contributes to the final score:

- `alpha = 0.0`: only keyword (BM25) scoring is used.
- `alpha = 1.0`: only vector similarity scoring is used.
- Values in between blend the two; higher values favor the vector score, lower values favor BM25.

If `None`, the Weaviate server default is used.

See the official Weaviate docs on Hybrid Search parameters for more details:

- [Hybrid search parameters](https://weaviate.io/developers/weaviate/search/hybrid#parameters)
- [Hybrid Search](https://docs.weaviate.io/weaviate/concepts/search/hybrid-search)
- **max_vector_distance** (<code>float | None</code>) – Optional threshold that restricts the vector part of the hybrid search to candidates within a maximum
  vector distance. Candidates with a distance larger than this threshold are excluded from the vector portion
  before blending.

Use this to prune low-quality vector matches while still benefitting from keyword recall. Leave `None` to
use Weaviate's default behavior without an explicit cutoff.

See the official Weaviate docs on Hybrid Search parameters for more details:

- [Hybrid search parameters](https://weaviate.io/developers/weaviate/search/hybrid#parameters)
- [Hybrid Search](https://docs.weaviate.io/weaviate/concepts/search/hybrid-search)
- **filter_policy** (<code>str | FilterPolicy</code>) – Policy to determine how filters are applied.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> WeaviateHybridRetriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>WeaviateHybridRetriever</code> – Deserialized component.

#### `run`

```python
run(
    query: str,
    query_embedding: list[float],
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
    alpha: float | None = None,
    max_vector_distance: float | None = None,
) -> dict[str, list[Document]]
```

Retrieves documents from Weaviate using hybrid search.

**Parameters:**

- **query** (<code>str</code>) – The query text.
- **query_embedding** (<code>list\[float\]</code>) – Embedding of the query.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved Documents. The way runtime filters are applied depends on
  the `filter_policy` chosen at retriever initialization. See init method docstring for more
  details.
- **top_k** (<code>int | None</code>) – The maximum number of documents to return.
- **alpha** (<code>float | None</code>) – Blending factor for hybrid retrieval in Weaviate. Must be in the range `[0.0, 1.0]`.

Weaviate hybrid search combines keyword (BM25) and vector scores into a single ranking. `alpha` controls
how much each part contributes to the final score:

- `alpha = 0.0`: only keyword (BM25) scoring is used.
- `alpha = 1.0`: only vector similarity scoring is used.
- Values in between blend the two; higher values favor the vector score, lower values favor BM25.

If `None`, the Weaviate server default is used.

See the official Weaviate docs on Hybrid Search parameters for more details:

- [Hybrid search parameters](https://weaviate.io/developers/weaviate/search/hybrid#parameters)
- [Hybrid Search](https://docs.weaviate.io/weaviate/concepts/search/hybrid-search)
- **max_vector_distance** (<code>float | None</code>) – Optional threshold that restricts the vector part of the hybrid search to candidates within a maximum
  vector distance. Candidates with a distance larger than this threshold are excluded from the vector portion
  before blending.

Use this to prune low-quality vector matches while still benefitting from keyword recall. Leave `None` to
use Weaviate's default behavior without an explicit cutoff.

See the official Weaviate docs on Hybrid Search parameters for more details:

- [Hybrid search parameters](https://weaviate.io/developers/weaviate/search/hybrid#parameters)
- [Hybrid Search](https://docs.weaviate.io/weaviate/concepts/search/hybrid-search)

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: List of documents returned by the search engine.

#### `run_async`

```python
run_async(
    query: str,
    query_embedding: list[float],
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
    alpha: float | None = None,
    max_vector_distance: float | None = None,
) -> dict[str, list[Document]]
```

Asynchronously retrieves documents from Weaviate using hybrid search.

**Parameters:**

- **query** (<code>str</code>) – The query text.
- **query_embedding** (<code>list\[float\]</code>) – Embedding of the query.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved Documents. The way runtime filters are applied depends on
  the `filter_policy` chosen at retriever initialization. See init method docstring for more
  details.
- **top_k** (<code>int | None</code>) – The maximum number of documents to return.
- **alpha** (<code>float | None</code>) – Blending factor for hybrid retrieval in Weaviate. Must be in the range `[0.0, 1.0]`.

Weaviate hybrid search combines keyword (BM25) and vector scores into a single ranking. `alpha` controls
how much each part contributes to the final score:

- `alpha = 0.0`: only keyword (BM25) scoring is used.
- `alpha = 1.0`: only vector similarity scoring is used.
- Values in between blend the two; higher values favor the vector score, lower values favor BM25.

If `None`, the Weaviate server default is used.

See the official Weaviate docs on Hybrid Search parameters for more details:

- [Hybrid search parameters](https://weaviate.io/developers/weaviate/search/hybrid#parameters)
- [Hybrid Search](https://docs.weaviate.io/weaviate/concepts/search/hybrid-search)
- **max_vector_distance** (<code>float | None</code>) – Optional threshold that restricts the vector part of the hybrid search to candidates within a maximum
  vector distance. Candidates with a distance larger than this threshold are excluded from the vector portion
  before blending.

Use this to prune low-quality vector matches while still benefitting from keyword recall. Leave `None` to
use Weaviate's default behavior without an explicit cutoff.

See the official Weaviate docs on Hybrid Search parameters for more details:

- [Hybrid search parameters](https://weaviate.io/developers/weaviate/search/hybrid#parameters)
- [Hybrid Search](https://docs.weaviate.io/weaviate/concepts/search/hybrid-search)

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: List of documents returned by the search engine.

## `haystack_integrations.document_stores.weaviate.auth`

### `SupportedAuthTypes`

Bases: <code>Enum</code>

Supported auth credentials for WeaviateDocumentStore.

### `AuthCredentials`

Bases: <code>ABC</code>

Base class for all auth credentials supported by WeaviateDocumentStore.
Can be used to deserialize from dict any of the supported auth credentials.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Converts the object to a dictionary representation for serialization.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> AuthCredentials
```

Converts a dictionary representation to an auth credentials object.

#### `resolve_value`

```python
resolve_value()
```

Resolves all the secrets in the auth credentials object and returns the corresponding Weaviate object.
All subclasses must implement this method.

### `AuthApiKey`

Bases: <code>AuthCredentials</code>

AuthCredentials for API key authentication.
By default it will load `api_key` from the environment variable `WEAVIATE_API_KEY`.

### `AuthBearerToken`

Bases: <code>AuthCredentials</code>

AuthCredentials for Bearer token authentication.
By default it will load `access_token` from the environment variable `WEAVIATE_ACCESS_TOKEN`,
and `refresh_token` from the environment variable
`WEAVIATE_REFRESH_TOKEN`.
`WEAVIATE_REFRESH_TOKEN` environment variable is optional.

### `AuthClientCredentials`

Bases: <code>AuthCredentials</code>

AuthCredentials for client credentials authentication.
By default it will load `client_secret` from the environment variable `WEAVIATE_CLIENT_SECRET`, and
`scope` from the environment variable `WEAVIATE_SCOPE`.
`WEAVIATE_SCOPE` environment variable is optional, if set it can either be a string or a list of space
separated strings. e.g "scope1" or "scope1 scope2".

### `AuthClientPassword`

Bases: <code>AuthCredentials</code>

AuthCredentials for username and password authentication.
By default it will load `username` from the environment variable `WEAVIATE_USERNAME`,
`password` from the environment variable `WEAVIATE_PASSWORD`, and
`scope` from the environment variable `WEAVIATE_SCOPE`.
`WEAVIATE_SCOPE` environment variable is optional, if set it can either be a string or a list of space
separated strings. e.g "scope1" or "scope1 scope2".

## `haystack_integrations.document_stores.weaviate.document_store`

### `WeaviateDocumentStore`

A WeaviateDocumentStore instance you
can use with Weaviate Cloud Services or self-hosted instances.

Usage example with Weaviate Cloud Services:

```python
import os
from haystack_integrations.document_stores.weaviate.auth import AuthApiKey
from haystack_integrations.document_stores.weaviate.document_store import (
    WeaviateDocumentStore,
)

os.environ["WEAVIATE_API_KEY"] = "MY_API_KEY"

document_store = WeaviateDocumentStore(
    url="rAnD0mD1g1t5.something.weaviate.cloud",
    auth_client_secret=AuthApiKey(),
)
```

Usage example with self-hosted Weaviate:

```python
from haystack_integrations.document_stores.weaviate.document_store import (
    WeaviateDocumentStore,
)

document_store = WeaviateDocumentStore(url="http://localhost:8080")
```

#### `__init__`

```python
__init__(
    *,
    url: str | None = None,
    collection_settings: dict[str, Any] | None = None,
    auth_client_secret: AuthCredentials | None = None,
    additional_headers: dict | None = None,
    embedded_options: EmbeddedOptions | None = None,
    additional_config: AdditionalConfig | None = None,
    grpc_port: int = 50051,
    grpc_secure: bool = False
)
```

Create a new instance of WeaviateDocumentStore and connects to the Weaviate instance.

**Parameters:**

- **url** (<code>str | None</code>) – The URL to the weaviate instance.
- **collection_settings** (<code>dict\[str, Any\] | None</code>) – The collection settings to use. If `None`, it will use a collection named `default` with the following
  properties:
- \_original_id: text
- content: text
- blob_data: blob
- blob_mime_type: text
- score: number
  The Document `meta` fields are omitted in the default collection settings as we can't make assumptions
  on the structure of the meta field.
  We heavily recommend to create a custom collection with the correct meta properties
  for your use case.
  Another option is relying on the automatic schema generation, but that's not recommended for
  production use.
  See the official [Weaviate documentation](https://weaviate.io/developers/weaviate/manage-data/collections)
  for more information on collections and their properties.
- **auth_client_secret** (<code>AuthCredentials | None</code>) – Authentication credentials. Can be one of the following types depending on the authentication mode:
- `AuthBearerToken` to use existing access and (optionally, but recommended) refresh tokens
- `AuthClientPassword` to use username and password for oidc Resource Owner Password flow
- `AuthClientCredentials` to use a client secret for oidc client credential flow
- `AuthApiKey` to use an API key
- **additional_headers** (<code>dict | None</code>) – Additional headers to include in the requests. Can be used to set OpenAI/HuggingFace keys.
  OpenAI/HuggingFace key looks like this:

```
{"X-OpenAI-Api-Key": "<THE-KEY>"}, {"X-HuggingFace-Api-Key": "<THE-KEY>"}
```

- **embedded_options** (<code>EmbeddedOptions | None</code>) – If set, create an embedded Weaviate cluster inside the client. For a full list of options see
  `weaviate.embedded.EmbeddedOptions`.
- **additional_config** (<code>AdditionalConfig | None</code>) – Additional and advanced configuration options for weaviate.
- **grpc_port** (<code>int</code>) – The port to use for the gRPC connection.
- **grpc_secure** (<code>bool</code>) – Whether to use a secure channel for the underlying gRPC API.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> WeaviateDocumentStore
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to deserialize from.

**Returns:**

- <code>WeaviateDocumentStore</code> – The deserialized component.

#### `count_documents`

```python
count_documents() -> int
```

Returns the number of documents present in the DocumentStore.

#### `count_documents_by_filter`

```python
count_documents_by_filter(filters: dict[str, Any]) -> int
```

Returns the number of documents that match the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to count documents.
  For filter syntax, see
  [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering).

**Returns:**

- <code>int</code> – The number of documents that match the filters.

#### `count_documents_by_filter_async`

```python
count_documents_by_filter_async(filters: dict[str, Any]) -> int
```

Asynchronously returns the number of documents that match the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to count documents.
  For filter syntax, see
  [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering).

**Returns:**

- <code>int</code> – The number of documents that match the filters.

#### `get_metadata_fields_info`

```python
get_metadata_fields_info() -> dict[str, dict[str, str]]
```

Returns metadata field names and their types, excluding special fields.

Special fields (content, blob_data, blob_mime_type, \_original_id, score) are excluded
as they are not user metadata fields.

**Returns:**

- <code>dict\[str, dict\[str, str\]\]</code> – A dictionary where keys are field names and values are dictionaries
  containing type information, e.g.:

```python
{
    'number': {'type': 'int'},
    'date': {'type': 'date'},
    'category': {'type': 'text'},
    'status': {'type': 'text'}
}
```

#### `get_metadata_fields_info_async`

```python
get_metadata_fields_info_async() -> dict[str, dict[str, str]]
```

Asynchronously returns metadata field names and their types, excluding special fields.

Special fields (content, blob_data, blob_mime_type, \_original_id, score) are excluded
as they are not user metadata fields.

**Returns:**

- <code>dict\[str, dict\[str, str\]\]</code> – A dictionary where keys are field names and values are dictionaries
  containing type information, e.g.:

```python
{
    'number': {'type': 'int'},
    'date': {'type': 'date'},
    'category': {'type': 'text'},
    'status': {'type': 'text'}
}
```

#### `get_metadata_field_min_max`

```python
get_metadata_field_min_max(metadata_field: str) -> dict[str, Any]
```

Returns the minimum and maximum values for a numeric or date metadata field.

**Parameters:**

- **metadata_field** (<code>str</code>) – The metadata field name to get min/max for.
  Can be prefixed with 'meta.' (e.g., 'meta.year' or 'year').

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with 'min' and 'max' keys containing the respective values.

**Raises:**

- <code>ValueError</code> – If the field is not found or doesn't support min/max operations.

#### `get_metadata_field_min_max_async`

```python
get_metadata_field_min_max_async(metadata_field: str) -> dict[str, Any]
```

Asynchronously returns the minimum and maximum values for a numeric or date metadata field.

**Parameters:**

- **metadata_field** (<code>str</code>) – The metadata field name to get min/max for.
  Can be prefixed with 'meta.' (e.g., 'meta.year' or 'year').

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with 'min' and 'max' keys containing the respective values.

**Raises:**

- <code>ValueError</code> – If the field is not found or doesn't support min/max operations.

#### `count_unique_metadata_by_filter`

```python
count_unique_metadata_by_filter(
    filters: dict[str, Any], metadata_fields: list[str]
) -> dict[str, int]
```

Returns the count of unique values for each specified metadata field.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply when counting unique values.
  For filter syntax, see
  [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering).
- **metadata_fields** (<code>list\[str\]</code>) – List of metadata field names to count unique values for.
  Field names can be prefixed with 'meta.' (e.g., 'meta.category' or 'category').

**Returns:**

- <code>dict\[str, int\]</code> – A dictionary mapping field names to counts of unique values.

**Raises:**

- <code>ValueError</code> – If any of the requested fields don't exist in the collection schema.

#### `count_unique_metadata_by_filter_async`

```python
count_unique_metadata_by_filter_async(
    filters: dict[str, Any], metadata_fields: list[str]
) -> dict[str, int]
```

Asynchronously returns the count of unique values for each specified metadata field.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply when counting unique values.
  For filter syntax, see
  [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering).
- **metadata_fields** (<code>list\[str\]</code>) – List of metadata field names to count unique values for.
  Field names can be prefixed with 'meta.' (e.g., 'meta.category' or 'category').

**Returns:**

- <code>dict\[str, int\]</code> – A dictionary mapping field names to counts of unique values.

**Raises:**

- <code>ValueError</code> – If any of the requested fields don't exist in the collection schema.

#### `get_metadata_field_unique_values`

```python
get_metadata_field_unique_values(
    metadata_field: str,
    search_term: str | None = None,
    from_: int = 0,
    size: int = 10000,
) -> tuple[list[str], int]
```

Returns unique values for a metadata field with pagination support.

**Parameters:**

- **metadata_field** (<code>str</code>) – The metadata field name to get unique values for.
  Can be prefixed with 'meta.' (e.g., 'meta.category' or 'category').
- **search_term** (<code>str | None</code>) – Optional term to filter documents by content before
  extracting unique values. If provided, only documents whose content
  contains this term will be considered.
  Note: Uses substring matching (case-sensitive, no stemming).
- **from\_** (<code>int</code>) – The starting offset for pagination (0-indexed). Defaults to 0.
- **size** (<code>int</code>) – The maximum number of unique values to return. Defaults to 10000.

**Returns:**

- <code>tuple\[list\[str\], int\]</code> – A tuple of (list of unique values, total count of unique values).

**Raises:**

- <code>ValueError</code> – If the field is not found in the collection schema.

#### `get_metadata_field_unique_values_async`

```python
get_metadata_field_unique_values_async(
    metadata_field: str,
    search_term: str | None = None,
    from_: int = 0,
    size: int = 10000,
) -> tuple[list[str], int]
```

Asynchronously returns unique values for a metadata field with pagination support.

**Parameters:**

- **metadata_field** (<code>str</code>) – The metadata field name to get unique values for.
  Can be prefixed with 'meta.' (e.g., 'meta.category' or 'category').
- **search_term** (<code>str | None</code>) – Optional term to filter documents by content before
  extracting unique values. If provided, only documents whose content
  contains this term will be considered.
  Note: Uses substring matching (case-sensitive, no stemming).
- **from\_** (<code>int</code>) – The starting offset for pagination (0-indexed). Defaults to 0.
- **size** (<code>int</code>) – The maximum number of unique values to return. Defaults to 10000.

**Returns:**

- <code>tuple\[list\[str\], int\]</code> – A tuple of (list of unique values, total count of unique values).

**Raises:**

- <code>ValueError</code> – If the field is not found in the collection schema.

#### `filter_documents`

```python
filter_documents(filters: dict[str, Any] | None = None) -> list[Document]
```

Returns the documents that match the filters provided.

For a detailed specification of the filters, refer to the
DocumentStore.filter_documents() protocol documentation.

Note: The `contains` filter operator is case-sensitive (substring
matching). For case-insensitive matching, normalize the value before
building the filter.

**Parameters:**

- **filters** (<code>dict\[str, Any\] | None</code>) – The filters to apply to the document list.

**Returns:**

- <code>list\[Document\]</code> – A list of Documents that match the given filters.

#### `write_documents`

```python
write_documents(
    documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
) -> int
```

Writes documents to Weaviate using the specified policy.
We recommend using a OVERWRITE policy as it's faster than other policies for Weaviate since it uses
the batch API.
We can't use the batch API for other policies as it doesn't return any information whether the document
already exists or not. That prevents us from returning errors when using the FAIL policy or skipping a
Document when using the SKIP policy.

#### `delete_documents`

```python
delete_documents(document_ids: list[str]) -> None
```

Deletes all documents with matching document_ids from the DocumentStore.

**Parameters:**

- **document_ids** (<code>list\[str\]</code>) – The object_ids to delete.

#### `delete_all_documents`

```python
delete_all_documents(
    *, recreate_index: bool = False, batch_size: int = 1000
) -> None
```

Deletes all documents in a collection.

If recreate_index is False, it keeps the collection but deletes documents iteratively.
If recreate_index is True, the collection is dropped and faithfully recreated.
This is recommended for performance reasons.

**Parameters:**

- **recreate_index** (<code>bool</code>) – Use drop and recreate strategy. (recommended for performance)
- **batch_size** (<code>int</code>) – Only relevant if recreate_index is false. Defines the deletion batch size.
  Note that this parameter needs to be less or equal to the set `QUERY_MAXIMUM_RESULTS` variable
  set for the weaviate deployment (default is 10000).
  Reference: https://docs.weaviate.io/weaviate/manage-objects/delete#delete-all-objects

#### `delete_by_filter`

```python
delete_by_filter(filters: dict[str, Any]) -> int
```

Deletes all documents that match the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to select documents for deletion.
  For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)

**Returns:**

- <code>int</code> – The number of documents deleted.

#### `delete_by_filter_async`

```python
delete_by_filter_async(filters: dict[str, Any]) -> int
```

Asynchronously deletes all documents that match the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to select documents for deletion.
  For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)

**Returns:**

- <code>int</code> – The number of documents deleted.

#### `update_by_filter`

```python
update_by_filter(filters: dict[str, Any], meta: dict[str, Any]) -> int
```

Updates the metadata of all documents that match the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to select documents for updating.
  For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- **meta** (<code>dict\[str, Any\]</code>) – The metadata fields to update. These will be merged with existing metadata.

**Returns:**

- <code>int</code> – The number of documents updated.

#### `update_by_filter_async`

```python
update_by_filter_async(filters: dict[str, Any], meta: dict[str, Any]) -> int
```

Asynchronously updates the metadata of all documents that match the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to select documents for updating.
  For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- **meta** (<code>dict\[str, Any\]</code>) – The metadata fields to update. These will be merged with existing metadata.

**Returns:**

- <code>int</code> – The number of documents updated.
