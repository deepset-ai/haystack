---
title: "Weaviate"
id: integrations-weaviate
description: "Weaviate integration for Haystack"
slug: "/integrations-weaviate"
---

<a id="haystack_integrations.components.retrievers.weaviate.bm25_retriever"></a>

## Module haystack\_integrations.components.retrievers.weaviate.bm25\_retriever

<a id="haystack_integrations.components.retrievers.weaviate.bm25_retriever.WeaviateBM25Retriever"></a>

### WeaviateBM25Retriever

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

<a id="haystack_integrations.components.retrievers.weaviate.bm25_retriever.WeaviateBM25Retriever.__init__"></a>

#### WeaviateBM25Retriever.\_\_init\_\_

```python
def __init__(*,
             document_store: WeaviateDocumentStore,
             filters: dict[str, Any] | None = None,
             top_k: int = 10,
             filter_policy: str | FilterPolicy = FilterPolicy.REPLACE)
```

Create a new instance of WeaviateBM25Retriever.

**Arguments**:

- `document_store`: Instance of WeaviateDocumentStore that will be used from this retriever.
- `filters`: Custom filters applied when running the retriever
- `top_k`: Maximum number of documents to return
- `filter_policy`: Policy to determine how filters are applied.

<a id="haystack_integrations.components.retrievers.weaviate.bm25_retriever.WeaviateBM25Retriever.to_dict"></a>

#### WeaviateBM25Retriever.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.retrievers.weaviate.bm25_retriever.WeaviateBM25Retriever.from_dict"></a>

#### WeaviateBM25Retriever.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "WeaviateBM25Retriever"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.retrievers.weaviate.bm25_retriever.WeaviateBM25Retriever.run"></a>

#### WeaviateBM25Retriever.run

```python
@component.output_types(documents=list[Document])
def run(query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None) -> dict[str, list[Document]]
```

Retrieves documents from Weaviate using the BM25 algorithm.

**Arguments**:

- `query`: The query text.
- `filters`: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
the `filter_policy` chosen at retriever initialization. See init method docstring for more
details.
- `top_k`: The maximum number of documents to return.

**Returns**:

A dictionary with the following keys:
- `documents`: List of documents returned by the search engine.

<a id="haystack_integrations.components.retrievers.weaviate.bm25_retriever.WeaviateBM25Retriever.run_async"></a>

#### WeaviateBM25Retriever.run\_async

```python
@component.output_types(documents=list[Document])
async def run_async(query: str,
                    filters: dict[str, Any] | None = None,
                    top_k: int | None = None) -> dict[str, list[Document]]
```

Asynchronously retrieves documents from Weaviate using the BM25 algorithm.

**Arguments**:

- `query`: The query text.
- `filters`: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
the `filter_policy` chosen at retriever initialization. See init method docstring for more
details.
- `top_k`: The maximum number of documents to return.

**Returns**:

A dictionary with the following keys:
- `documents`: List of documents returned by the search engine.

<a id="haystack_integrations.components.retrievers.weaviate.embedding_retriever"></a>

## Module haystack\_integrations.components.retrievers.weaviate.embedding\_retriever

<a id="haystack_integrations.components.retrievers.weaviate.embedding_retriever.WeaviateEmbeddingRetriever"></a>

### WeaviateEmbeddingRetriever

A retriever that uses Weaviate's vector search to find similar documents based on the embeddings of the query.

<a id="haystack_integrations.components.retrievers.weaviate.embedding_retriever.WeaviateEmbeddingRetriever.__init__"></a>

#### WeaviateEmbeddingRetriever.\_\_init\_\_

```python
def __init__(*,
             document_store: WeaviateDocumentStore,
             filters: dict[str, Any] | None = None,
             top_k: int = 10,
             distance: float | None = None,
             certainty: float | None = None,
             filter_policy: str | FilterPolicy = FilterPolicy.REPLACE)
```

Creates a new instance of WeaviateEmbeddingRetriever.

**Arguments**:

- `document_store`: Instance of WeaviateDocumentStore that will be used from this retriever.
- `filters`: Custom filters applied when running the retriever.
- `top_k`: Maximum number of documents to return.
- `distance`: The maximum allowed distance between Documents' embeddings.
- `certainty`: Normalized distance between the result item and the search vector.
- `filter_policy`: Policy to determine how filters are applied.

**Raises**:

- `ValueError`: If both `distance` and `certainty` are provided.
See https://weaviate.io/developers/weaviate/api/graphql/search-operators#variables to learn more about
`distance` and `certainty` parameters.

<a id="haystack_integrations.components.retrievers.weaviate.embedding_retriever.WeaviateEmbeddingRetriever.to_dict"></a>

#### WeaviateEmbeddingRetriever.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.retrievers.weaviate.embedding_retriever.WeaviateEmbeddingRetriever.from_dict"></a>

#### WeaviateEmbeddingRetriever.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "WeaviateEmbeddingRetriever"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.retrievers.weaviate.embedding_retriever.WeaviateEmbeddingRetriever.run"></a>

#### WeaviateEmbeddingRetriever.run

```python
@component.output_types(documents=list[Document])
def run(query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        distance: float | None = None,
        certainty: float | None = None) -> dict[str, list[Document]]
```

Retrieves documents from Weaviate using the vector search.

**Arguments**:

- `query_embedding`: Embedding of the query.
- `filters`: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
the `filter_policy` chosen at retriever initialization. See init method docstring for more
details.
- `top_k`: The maximum number of documents to return.
- `distance`: The maximum allowed distance between Documents' embeddings.
- `certainty`: Normalized distance between the result item and the search vector.

**Raises**:

- `ValueError`: If both `distance` and `certainty` are provided.
See https://weaviate.io/developers/weaviate/api/graphql/search-operators#variables to learn more about
`distance` and `certainty` parameters.

**Returns**:

A dictionary with the following keys:
- `documents`: List of documents returned by the search engine.

<a id="haystack_integrations.components.retrievers.weaviate.embedding_retriever.WeaviateEmbeddingRetriever.run_async"></a>

#### WeaviateEmbeddingRetriever.run\_async

```python
@component.output_types(documents=list[Document])
async def run_async(
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        distance: float | None = None,
        certainty: float | None = None) -> dict[str, list[Document]]
```

Asynchronously retrieves documents from Weaviate using the vector search.

**Arguments**:

- `query_embedding`: Embedding of the query.
- `filters`: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
the `filter_policy` chosen at retriever initialization. See init method docstring for more
details.
- `top_k`: The maximum number of documents to return.
- `distance`: The maximum allowed distance between Documents' embeddings.
- `certainty`: Normalized distance between the result item and the search vector.

**Raises**:

- `ValueError`: If both `distance` and `certainty` are provided.
See https://weaviate.io/developers/weaviate/api/graphql/search-operators#variables to learn more about
`distance` and `certainty` parameters.

**Returns**:

A dictionary with the following keys:
- `documents`: List of documents returned by the search engine.

<a id="haystack_integrations.components.retrievers.weaviate.hybrid_retriever"></a>

## Module haystack\_integrations.components.retrievers.weaviate.hybrid\_retriever

<a id="haystack_integrations.components.retrievers.weaviate.hybrid_retriever.WeaviateHybridRetriever"></a>

### WeaviateHybridRetriever

A retriever that uses Weaviate's hybrid search to find similar documents based on the embeddings of the query.

<a id="haystack_integrations.components.retrievers.weaviate.hybrid_retriever.WeaviateHybridRetriever.__init__"></a>

#### WeaviateHybridRetriever.\_\_init\_\_

```python
def __init__(*,
             document_store: WeaviateDocumentStore,
             filters: dict[str, Any] | None = None,
             top_k: int = 10,
             alpha: float | None = None,
             max_vector_distance: float | None = None,
             filter_policy: str | FilterPolicy = FilterPolicy.REPLACE)
```

Creates a new instance of WeaviateHybridRetriever.

**Arguments**:

- `document_store`: Instance of WeaviateDocumentStore that will be used from this retriever.
- `filters`: Custom filters applied when running the retriever.
- `top_k`: Maximum number of documents to return.
- `alpha`: Blending factor for hybrid retrieval in Weaviate. Must be in the range `[0.0, 1.0]`.
Weaviate hybrid search combines keyword (BM25) and vector scores into a single ranking. `alpha` controls
how much each part contributes to the final score:

- `alpha = 0.0`: only keyword (BM25) scoring is used.
- `alpha = 1.0`: only vector similarity scoring is used.
- Values in between blend the two; higher values favor the vector score, lower values favor BM25.

If `None`, the Weaviate server default is used.

See the official Weaviate docs on Hybrid Search parameters for more details:
- [Hybrid search parameters](https://weaviate.io/developers/weaviate/search/hybrid#parameters)
- [Hybrid Search](https://docs.weaviate.io/weaviate/concepts/search/hybrid-search)
- `max_vector_distance`: Optional threshold that restricts the vector part of the hybrid search to candidates within a maximum
vector distance. Candidates with a distance larger than this threshold are excluded from the vector portion
before blending.

Use this to prune low-quality vector matches while still benefitting from keyword recall. Leave `None` to
use Weaviate's default behavior without an explicit cutoff.

See the official Weaviate docs on Hybrid Search parameters for more details:
- [Hybrid search parameters](https://weaviate.io/developers/weaviate/search/hybrid#parameters)
- [Hybrid Search](https://docs.weaviate.io/weaviate/concepts/search/hybrid-search)
- `filter_policy`: Policy to determine how filters are applied.

<a id="haystack_integrations.components.retrievers.weaviate.hybrid_retriever.WeaviateHybridRetriever.to_dict"></a>

#### WeaviateHybridRetriever.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.retrievers.weaviate.hybrid_retriever.WeaviateHybridRetriever.from_dict"></a>

#### WeaviateHybridRetriever.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "WeaviateHybridRetriever"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.retrievers.weaviate.hybrid_retriever.WeaviateHybridRetriever.run"></a>

#### WeaviateHybridRetriever.run

```python
@component.output_types(documents=list[Document])
def run(query: str,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        alpha: float | None = None,
        max_vector_distance: float | None = None) -> dict[str, list[Document]]
```

Retrieves documents from Weaviate using hybrid search.

**Arguments**:

- `query`: The query text.
- `query_embedding`: Embedding of the query.
- `filters`: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
the `filter_policy` chosen at retriever initialization. See init method docstring for more
details.
- `top_k`: The maximum number of documents to return.
- `alpha`: Blending factor for hybrid retrieval in Weaviate. Must be in the range `[0.0, 1.0]`.
Weaviate hybrid search combines keyword (BM25) and vector scores into a single ranking. `alpha` controls
how much each part contributes to the final score:

- `alpha = 0.0`: only keyword (BM25) scoring is used.
- `alpha = 1.0`: only vector similarity scoring is used.
- Values in between blend the two; higher values favor the vector score, lower values favor BM25.

If `None`, the Weaviate server default is used.

See the official Weaviate docs on Hybrid Search parameters for more details:
- [Hybrid search parameters](https://weaviate.io/developers/weaviate/search/hybrid#parameters)
- [Hybrid Search](https://docs.weaviate.io/weaviate/concepts/search/hybrid-search)
- `max_vector_distance`: Optional threshold that restricts the vector part of the hybrid search to candidates within a maximum
vector distance. Candidates with a distance larger than this threshold are excluded from the vector portion
before blending.

Use this to prune low-quality vector matches while still benefitting from keyword recall. Leave `None` to
use Weaviate's default behavior without an explicit cutoff.

See the official Weaviate docs on Hybrid Search parameters for more details:
- [Hybrid search parameters](https://weaviate.io/developers/weaviate/search/hybrid#parameters)
- [Hybrid Search](https://docs.weaviate.io/weaviate/concepts/search/hybrid-search)

**Returns**:

A dictionary with the following keys:
- `documents`: List of documents returned by the search engine.

<a id="haystack_integrations.components.retrievers.weaviate.hybrid_retriever.WeaviateHybridRetriever.run_async"></a>

#### WeaviateHybridRetriever.run\_async

```python
@component.output_types(documents=list[Document])
async def run_async(
        query: str,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        alpha: float | None = None,
        max_vector_distance: float | None = None) -> dict[str, list[Document]]
```

Asynchronously retrieves documents from Weaviate using hybrid search.

**Arguments**:

- `query`: The query text.
- `query_embedding`: Embedding of the query.
- `filters`: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
the `filter_policy` chosen at retriever initialization. See init method docstring for more
details.
- `top_k`: The maximum number of documents to return.
- `alpha`: Blending factor for hybrid retrieval in Weaviate. Must be in the range `[0.0, 1.0]`.
Weaviate hybrid search combines keyword (BM25) and vector scores into a single ranking. `alpha` controls
how much each part contributes to the final score:

- `alpha = 0.0`: only keyword (BM25) scoring is used.
- `alpha = 1.0`: only vector similarity scoring is used.
- Values in between blend the two; higher values favor the vector score, lower values favor BM25.

If `None`, the Weaviate server default is used.

See the official Weaviate docs on Hybrid Search parameters for more details:
- [Hybrid search parameters](https://weaviate.io/developers/weaviate/search/hybrid#parameters)
- [Hybrid Search](https://docs.weaviate.io/weaviate/concepts/search/hybrid-search)
- `max_vector_distance`: Optional threshold that restricts the vector part of the hybrid search to candidates within a maximum
vector distance. Candidates with a distance larger than this threshold are excluded from the vector portion
before blending.

Use this to prune low-quality vector matches while still benefitting from keyword recall. Leave `None` to
use Weaviate's default behavior without an explicit cutoff.

See the official Weaviate docs on Hybrid Search parameters for more details:
- [Hybrid search parameters](https://weaviate.io/developers/weaviate/search/hybrid#parameters)
- [Hybrid Search](https://docs.weaviate.io/weaviate/concepts/search/hybrid-search)

**Returns**:

A dictionary with the following keys:
- `documents`: List of documents returned by the search engine.

<a id="haystack_integrations.document_stores.weaviate.auth"></a>

## Module haystack\_integrations.document\_stores.weaviate.auth

<a id="haystack_integrations.document_stores.weaviate.auth.SupportedAuthTypes"></a>

### SupportedAuthTypes

Supported auth credentials for WeaviateDocumentStore.

<a id="haystack_integrations.document_stores.weaviate.auth.AuthCredentials"></a>

### AuthCredentials

Base class for all auth credentials supported by WeaviateDocumentStore.
Can be used to deserialize from dict any of the supported auth credentials.

<a id="haystack_integrations.document_stores.weaviate.auth.AuthCredentials.to_dict"></a>

#### AuthCredentials.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Converts the object to a dictionary representation for serialization.

<a id="haystack_integrations.document_stores.weaviate.auth.AuthCredentials.from_dict"></a>

#### AuthCredentials.from\_dict

```python
@staticmethod
def from_dict(data: dict[str, Any]) -> "AuthCredentials"
```

Converts a dictionary representation to an auth credentials object.

<a id="haystack_integrations.document_stores.weaviate.auth.AuthCredentials.resolve_value"></a>

#### AuthCredentials.resolve\_value

```python
@abstractmethod
def resolve_value()
```

Resolves all the secrets in the auth credentials object and returns the corresponding Weaviate object.
All subclasses must implement this method.

<a id="haystack_integrations.document_stores.weaviate.auth.AuthApiKey"></a>

### AuthApiKey

AuthCredentials for API key authentication.
By default it will load `api_key` from the environment variable `WEAVIATE_API_KEY`.

<a id="haystack_integrations.document_stores.weaviate.auth.AuthBearerToken"></a>

### AuthBearerToken

AuthCredentials for Bearer token authentication.
By default it will load `access_token` from the environment variable `WEAVIATE_ACCESS_TOKEN`,
and `refresh_token` from the environment variable
`WEAVIATE_REFRESH_TOKEN`.
`WEAVIATE_REFRESH_TOKEN` environment variable is optional.

<a id="haystack_integrations.document_stores.weaviate.auth.AuthClientCredentials"></a>

### AuthClientCredentials

AuthCredentials for client credentials authentication.
By default it will load `client_secret` from the environment variable `WEAVIATE_CLIENT_SECRET`, and
`scope` from the environment variable `WEAVIATE_SCOPE`.
`WEAVIATE_SCOPE` environment variable is optional, if set it can either be a string or a list of space
separated strings. e.g "scope1" or "scope1 scope2".

<a id="haystack_integrations.document_stores.weaviate.auth.AuthClientPassword"></a>

### AuthClientPassword

AuthCredentials for username and password authentication.
By default it will load `username` from the environment variable `WEAVIATE_USERNAME`,
`password` from the environment variable `WEAVIATE_PASSWORD`, and
`scope` from the environment variable `WEAVIATE_SCOPE`.
`WEAVIATE_SCOPE` environment variable is optional, if set it can either be a string or a list of space
separated strings. e.g "scope1" or "scope1 scope2".

<a id="haystack_integrations.document_stores.weaviate.document_store"></a>

## Module haystack\_integrations.document\_stores.weaviate.document\_store

<a id="haystack_integrations.document_stores.weaviate.document_store.WeaviateDocumentStore"></a>

### WeaviateDocumentStore

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

<a id="haystack_integrations.document_stores.weaviate.document_store.WeaviateDocumentStore.__init__"></a>

#### WeaviateDocumentStore.\_\_init\_\_

```python
def __init__(*,
             url: str | None = None,
             collection_settings: dict[str, Any] | None = None,
             auth_client_secret: AuthCredentials | None = None,
             additional_headers: dict | None = None,
             embedded_options: EmbeddedOptions | None = None,
             additional_config: AdditionalConfig | None = None,
             grpc_port: int = 50051,
             grpc_secure: bool = False)
```

Create a new instance of WeaviateDocumentStore and connects to the Weaviate instance.

**Arguments**:

- `url`: The URL to the weaviate instance.
- `collection_settings`: The collection settings to use. If `None`, it will use a collection named `default` with the following
properties:
- _original_id: text
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
- `auth_client_secret`: Authentication credentials. Can be one of the following types depending on the authentication mode:
- `AuthBearerToken` to use existing access and (optionally, but recommended) refresh tokens
- `AuthClientPassword` to use username and password for oidc Resource Owner Password flow
- `AuthClientCredentials` to use a client secret for oidc client credential flow
- `AuthApiKey` to use an API key
- `additional_headers`: Additional headers to include in the requests. Can be used to set OpenAI/HuggingFace keys.
OpenAI/HuggingFace key looks like this:
```
{"X-OpenAI-Api-Key": "<THE-KEY>"}, {"X-HuggingFace-Api-Key": "<THE-KEY>"}
```
- `embedded_options`: If set, create an embedded Weaviate cluster inside the client. For a full list of options see
`weaviate.embedded.EmbeddedOptions`.
- `additional_config`: Additional and advanced configuration options for weaviate.
- `grpc_port`: The port to use for the gRPC connection.
- `grpc_secure`: Whether to use a secure channel for the underlying gRPC API.

<a id="haystack_integrations.document_stores.weaviate.document_store.WeaviateDocumentStore.to_dict"></a>

#### WeaviateDocumentStore.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.document_stores.weaviate.document_store.WeaviateDocumentStore.from_dict"></a>

#### WeaviateDocumentStore.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "WeaviateDocumentStore"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: The dictionary to deserialize from.

**Returns**:

The deserialized component.

<a id="haystack_integrations.document_stores.weaviate.document_store.WeaviateDocumentStore.count_documents"></a>

#### WeaviateDocumentStore.count\_documents

```python
def count_documents() -> int
```

Returns the number of documents present in the DocumentStore.

<a id="haystack_integrations.document_stores.weaviate.document_store.WeaviateDocumentStore.count_documents_by_filter"></a>

#### WeaviateDocumentStore.count\_documents\_by\_filter

```python
def count_documents_by_filter(filters: dict[str, Any]) -> int
```

Returns the number of documents that match the provided filters.

**Arguments**:

- `filters`: The filters to apply to count documents.
For filter syntax, see
[Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering).

**Returns**:

The number of documents that match the filters.

<a id="haystack_integrations.document_stores.weaviate.document_store.WeaviateDocumentStore.count_documents_by_filter_async"></a>

#### WeaviateDocumentStore.count\_documents\_by\_filter\_async

```python
async def count_documents_by_filter_async(filters: dict[str, Any]) -> int
```

Asynchronously returns the number of documents that match the provided filters.

**Arguments**:

- `filters`: The filters to apply to count documents.
For filter syntax, see
[Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering).

**Returns**:

The number of documents that match the filters.

<a id="haystack_integrations.document_stores.weaviate.document_store.WeaviateDocumentStore.get_metadata_fields_info"></a>

#### WeaviateDocumentStore.get\_metadata\_fields\_info

```python
def get_metadata_fields_info() -> dict[str, dict[str, str]]
```

Returns metadata field names and their types, excluding special fields.

Special fields (content, blob_data, blob_mime_type, _original_id, score) are excluded
as they are not user metadata fields.

**Returns**:

A dictionary where keys are field names and values are dictionaries
containing type information, e.g.:
```python
{"number": {"type": "int"}, "date": {"type": "date"}, "category": {"type": "text"}, "status": {"type": "text"}}
```

<a id="haystack_integrations.document_stores.weaviate.document_store.WeaviateDocumentStore.get_metadata_fields_info_async"></a>

#### WeaviateDocumentStore.get\_metadata\_fields\_info\_async

```python
async def get_metadata_fields_info_async() -> dict[str, dict[str, str]]
```

Asynchronously returns metadata field names and their types, excluding special fields.

Special fields (content, blob_data, blob_mime_type, _original_id, score) are excluded
as they are not user metadata fields.

**Returns**:

A dictionary where keys are field names and values are dictionaries
containing type information, e.g.:
```python
{"number": {"type": "int"}, "date": {"type": "date"}, "category": {"type": "text"}, "status": {"type": "text"}}
```

<a id="haystack_integrations.document_stores.weaviate.document_store.WeaviateDocumentStore.get_metadata_field_min_max"></a>

#### WeaviateDocumentStore.get\_metadata\_field\_min\_max

```python
def get_metadata_field_min_max(metadata_field: str) -> dict[str, Any]
```

Returns the minimum and maximum values for a numeric or date metadata field.

**Arguments**:

- `metadata_field`: The metadata field name to get min/max for.
Can be prefixed with 'meta.' (e.g., 'meta.year' or 'year').

**Raises**:

- `ValueError`: If the field is not found or doesn't support min/max operations.

**Returns**:

A dictionary with 'min' and 'max' keys containing the respective values.

<a id="haystack_integrations.document_stores.weaviate.document_store.WeaviateDocumentStore.get_metadata_field_min_max_async"></a>

#### WeaviateDocumentStore.get\_metadata\_field\_min\_max\_async

```python
async def get_metadata_field_min_max_async(
        metadata_field: str) -> dict[str, Any]
```

Asynchronously returns the minimum and maximum values for a numeric or date metadata field.

**Arguments**:

- `metadata_field`: The metadata field name to get min/max for.
Can be prefixed with 'meta.' (e.g., 'meta.year' or 'year').

**Raises**:

- `ValueError`: If the field is not found or doesn't support min/max operations.

**Returns**:

A dictionary with 'min' and 'max' keys containing the respective values.

<a id="haystack_integrations.document_stores.weaviate.document_store.WeaviateDocumentStore.count_unique_metadata_by_filter"></a>

#### WeaviateDocumentStore.count\_unique\_metadata\_by\_filter

```python
def count_unique_metadata_by_filter(
        filters: dict[str, Any], metadata_fields: list[str]) -> dict[str, int]
```

Returns the count of unique values for each specified metadata field.

**Arguments**:

- `filters`: The filters to apply when counting unique values.
For filter syntax, see
[Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering).
- `metadata_fields`: List of metadata field names to count unique values for.
Field names can be prefixed with 'meta.' (e.g., 'meta.category' or 'category').

**Raises**:

- `ValueError`: If any of the requested fields don't exist in the collection schema.

**Returns**:

A dictionary mapping field names to counts of unique values.

<a id="haystack_integrations.document_stores.weaviate.document_store.WeaviateDocumentStore.count_unique_metadata_by_filter_async"></a>

#### WeaviateDocumentStore.count\_unique\_metadata\_by\_filter\_async

```python
async def count_unique_metadata_by_filter_async(
        filters: dict[str, Any], metadata_fields: list[str]) -> dict[str, int]
```

Asynchronously returns the count of unique values for each specified metadata field.

**Arguments**:

- `filters`: The filters to apply when counting unique values.
For filter syntax, see
[Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering).
- `metadata_fields`: List of metadata field names to count unique values for.
Field names can be prefixed with 'meta.' (e.g., 'meta.category' or 'category').

**Raises**:

- `ValueError`: If any of the requested fields don't exist in the collection schema.

**Returns**:

A dictionary mapping field names to counts of unique values.

<a id="haystack_integrations.document_stores.weaviate.document_store.WeaviateDocumentStore.get_metadata_field_unique_values"></a>

#### WeaviateDocumentStore.get\_metadata\_field\_unique\_values

```python
def get_metadata_field_unique_values(
        metadata_field: str,
        search_term: str | None = None,
        from_: int = 0,
        size: int = 10000) -> tuple[list[str], int]
```

Returns unique values for a metadata field with pagination support.

**Arguments**:

- `metadata_field`: The metadata field name to get unique values for.
Can be prefixed with 'meta.' (e.g., 'meta.category' or 'category').
- `search_term`: Optional term to filter documents by content before
extracting unique values. If provided, only documents whose content
contains this term will be considered.
Note: Uses substring matching (case-sensitive, no stemming).
- `from_`: The starting offset for pagination (0-indexed). Defaults to 0.
- `size`: The maximum number of unique values to return. Defaults to 10000.

**Raises**:

- `ValueError`: If the field is not found in the collection schema.

**Returns**:

A tuple of (list of unique values, total count of unique values).

<a id="haystack_integrations.document_stores.weaviate.document_store.WeaviateDocumentStore.get_metadata_field_unique_values_async"></a>

#### WeaviateDocumentStore.get\_metadata\_field\_unique\_values\_async

```python
async def get_metadata_field_unique_values_async(
        metadata_field: str,
        search_term: str | None = None,
        from_: int = 0,
        size: int = 10000) -> tuple[list[str], int]
```

Asynchronously returns unique values for a metadata field with pagination support.

**Arguments**:

- `metadata_field`: The metadata field name to get unique values for.
Can be prefixed with 'meta.' (e.g., 'meta.category' or 'category').
- `search_term`: Optional term to filter documents by content before
extracting unique values. If provided, only documents whose content
contains this term will be considered.
Note: Uses substring matching (case-sensitive, no stemming).
- `from_`: The starting offset for pagination (0-indexed). Defaults to 0.
- `size`: The maximum number of unique values to return. Defaults to 10000.

**Raises**:

- `ValueError`: If the field is not found in the collection schema.

**Returns**:

A tuple of (list of unique values, total count of unique values).

<a id="haystack_integrations.document_stores.weaviate.document_store.WeaviateDocumentStore.filter_documents"></a>

#### WeaviateDocumentStore.filter\_documents

```python
def filter_documents(filters: dict[str, Any] | None = None) -> list[Document]
```

Returns the documents that match the filters provided.

For a detailed specification of the filters, refer to the
DocumentStore.filter_documents() protocol documentation.

Note: The ``contains`` filter operator is case-sensitive (substring
matching). For case-insensitive matching, normalize the value before
building the filter.

**Arguments**:

- `filters`: The filters to apply to the document list.

**Returns**:

A list of Documents that match the given filters.

<a id="haystack_integrations.document_stores.weaviate.document_store.WeaviateDocumentStore.write_documents"></a>

#### WeaviateDocumentStore.write\_documents

```python
def write_documents(documents: list[Document],
                    policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int
```

Writes documents to Weaviate using the specified policy.
We recommend using a OVERWRITE policy as it's faster than other policies for Weaviate since it uses
the batch API.
We can't use the batch API for other policies as it doesn't return any information whether the document
already exists or not. That prevents us from returning errors when using the FAIL policy or skipping a
Document when using the SKIP policy.

<a id="haystack_integrations.document_stores.weaviate.document_store.WeaviateDocumentStore.delete_documents"></a>

#### WeaviateDocumentStore.delete\_documents

```python
def delete_documents(document_ids: list[str]) -> None
```

Deletes all documents with matching document_ids from the DocumentStore.

**Arguments**:

- `document_ids`: The object_ids to delete.

<a id="haystack_integrations.document_stores.weaviate.document_store.WeaviateDocumentStore.delete_all_documents"></a>

#### WeaviateDocumentStore.delete\_all\_documents

```python
def delete_all_documents(*,
                         recreate_index: bool = False,
                         batch_size: int = 1000) -> None
```

Deletes all documents in a collection.

If recreate_index is False, it keeps the collection but deletes documents iteratively.
If recreate_index is True, the collection is dropped and faithfully recreated.
This is recommended for performance reasons.

**Arguments**:

- `recreate_index`: Use drop and recreate strategy. (recommended for performance)
- `batch_size`: Only relevant if recreate_index is false. Defines the deletion batch size.
Note that this parameter needs to be less or equal to the set `QUERY_MAXIMUM_RESULTS` variable
set for the weaviate deployment (default is 10000).
Reference: https://docs.weaviate.io/weaviate/manage-objects/delete#delete-all-objects

<a id="haystack_integrations.document_stores.weaviate.document_store.WeaviateDocumentStore.delete_by_filter"></a>

#### WeaviateDocumentStore.delete\_by\_filter

```python
def delete_by_filter(filters: dict[str, Any]) -> int
```

Deletes all documents that match the provided filters.

**Arguments**:

- `filters`: The filters to apply to select documents for deletion.
For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)

**Returns**:

The number of documents deleted.

<a id="haystack_integrations.document_stores.weaviate.document_store.WeaviateDocumentStore.delete_by_filter_async"></a>

#### WeaviateDocumentStore.delete\_by\_filter\_async

```python
async def delete_by_filter_async(filters: dict[str, Any]) -> int
```

Asynchronously deletes all documents that match the provided filters.

**Arguments**:

- `filters`: The filters to apply to select documents for deletion.
For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)

**Returns**:

The number of documents deleted.

<a id="haystack_integrations.document_stores.weaviate.document_store.WeaviateDocumentStore.update_by_filter"></a>

#### WeaviateDocumentStore.update\_by\_filter

```python
def update_by_filter(filters: dict[str, Any], meta: dict[str, Any]) -> int
```

Updates the metadata of all documents that match the provided filters.

**Arguments**:

- `filters`: The filters to apply to select documents for updating.
For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- `meta`: The metadata fields to update. These will be merged with existing metadata.

**Returns**:

The number of documents updated.

<a id="haystack_integrations.document_stores.weaviate.document_store.WeaviateDocumentStore.update_by_filter_async"></a>

#### WeaviateDocumentStore.update\_by\_filter\_async

```python
async def update_by_filter_async(filters: dict[str, Any],
                                 meta: dict[str, Any]) -> int
```

Asynchronously updates the metadata of all documents that match the provided filters.

**Arguments**:

- `filters`: The filters to apply to select documents for updating.
For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- `meta`: The metadata fields to update. These will be merged with existing metadata.

**Returns**:

The number of documents updated.
