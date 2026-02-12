---
title: "MongoDB Atlas"
id: integrations-mongodb-atlas
description: "MongoDB Atlas integration for Haystack"
slug: "/integrations-mongodb-atlas"
---

<a id="haystack_integrations.components.retrievers.mongodb_atlas.embedding_retriever"></a>

## Module haystack\_integrations.components.retrievers.mongodb\_atlas.embedding\_retriever

<a id="haystack_integrations.components.retrievers.mongodb_atlas.embedding_retriever.MongoDBAtlasEmbeddingRetriever"></a>

### MongoDBAtlasEmbeddingRetriever

Retrieves documents from the MongoDBAtlasDocumentStore by embedding similarity.

The similarity is dependent on the vector_search_index used in the MongoDBAtlasDocumentStore and the chosen metric
during the creation of the index (i.e. cosine, dot product, or euclidean). See MongoDBAtlasDocumentStore for more
information.

Usage example:
```python
import numpy as np
from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore
from haystack_integrations.components.retrievers.mongodb_atlas import MongoDBAtlasEmbeddingRetriever

store = MongoDBAtlasDocumentStore(database_name="haystack_integration_test",
                                  collection_name="test_embeddings_collection",
                                  vector_search_index="cosine_index",
                                  full_text_search_index="full_text_index")
retriever = MongoDBAtlasEmbeddingRetriever(document_store=store)

results = retriever.run(query_embedding=np.random.random(768).tolist())
print(results["documents"])
```

The example above retrieves the 10 most similar documents to a random query embedding from the
MongoDBAtlasDocumentStore. Note that dimensions of the query_embedding must match the dimensions of the embeddings
stored in the MongoDBAtlasDocumentStore.

<a id="haystack_integrations.components.retrievers.mongodb_atlas.embedding_retriever.MongoDBAtlasEmbeddingRetriever.__init__"></a>

#### MongoDBAtlasEmbeddingRetriever.\_\_init\_\_

```python
def __init__(*,
             document_store: MongoDBAtlasDocumentStore,
             filters: dict[str, Any] | None = None,
             top_k: int = 10,
             filter_policy: str | FilterPolicy = FilterPolicy.REPLACE)
```

Create the MongoDBAtlasDocumentStore component.

**Arguments**:

- `document_store`: An instance of MongoDBAtlasDocumentStore.
- `filters`: Filters applied to the retrieved Documents. Make sure that the fields used in the filters are
included in the configuration of the `vector_search_index`. The configuration must be done manually
in the Web UI of MongoDB Atlas.
- `top_k`: Maximum number of Documents to return.
- `filter_policy`: Policy to determine how filters are applied.

**Raises**:

- `ValueError`: If `document_store` is not an instance of `MongoDBAtlasDocumentStore`.

<a id="haystack_integrations.components.retrievers.mongodb_atlas.embedding_retriever.MongoDBAtlasEmbeddingRetriever.to_dict"></a>

#### MongoDBAtlasEmbeddingRetriever.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.retrievers.mongodb_atlas.embedding_retriever.MongoDBAtlasEmbeddingRetriever.from_dict"></a>

#### MongoDBAtlasEmbeddingRetriever.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "MongoDBAtlasEmbeddingRetriever"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.retrievers.mongodb_atlas.embedding_retriever.MongoDBAtlasEmbeddingRetriever.run"></a>

#### MongoDBAtlasEmbeddingRetriever.run

```python
@component.output_types(documents=list[Document])
def run(query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None) -> dict[str, list[Document]]
```

Retrieve documents from the MongoDBAtlasDocumentStore, based on the provided embedding similarity.

**Arguments**:

- `query_embedding`: Embedding of the query.
- `filters`: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
the `filter_policy` chosen at retriever initialization. See init method docstring for more
details.
- `top_k`: Maximum number of Documents to return. Overrides the value specified at initialization.

**Returns**:

A dictionary with the following keys:
- `documents`: List of Documents most similar to the given `query_embedding`

<a id="haystack_integrations.components.retrievers.mongodb_atlas.embedding_retriever.MongoDBAtlasEmbeddingRetriever.run_async"></a>

#### MongoDBAtlasEmbeddingRetriever.run\_async

```python
@component.output_types(documents=list[Document])
async def run_async(query_embedding: list[float],
                    filters: dict[str, Any] | None = None,
                    top_k: int | None = None) -> dict[str, list[Document]]
```

Asynchronously retrieve documents from the MongoDBAtlasDocumentStore, based on the provided embedding

similarity.

**Arguments**:

- `query_embedding`: Embedding of the query.
- `filters`: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
the `filter_policy` chosen at retriever initialization. See init method docstring for more
details.
- `top_k`: Maximum number of Documents to return. Overrides the value specified at initialization.

**Returns**:

A dictionary with the following keys:
- `documents`: List of Documents most similar to the given `query_embedding`

<a id="haystack_integrations.components.retrievers.mongodb_atlas.full_text_retriever"></a>

## Module haystack\_integrations.components.retrievers.mongodb\_atlas.full\_text\_retriever

<a id="haystack_integrations.components.retrievers.mongodb_atlas.full_text_retriever.MongoDBAtlasFullTextRetriever"></a>

### MongoDBAtlasFullTextRetriever

Retrieves documents from the MongoDBAtlasDocumentStore by full-text search.

The full-text search is dependent on the full_text_search_index used in the MongoDBAtlasDocumentStore.
See MongoDBAtlasDocumentStore for more information.

Usage example:
```python
from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore
from haystack_integrations.components.retrievers.mongodb_atlas import MongoDBAtlasFullTextRetriever

store = MongoDBAtlasDocumentStore(database_name="your_existing_db",
                                  collection_name="your_existing_collection",
                                  vector_search_index="your_existing_index",
                                  full_text_search_index="your_existing_index")
retriever = MongoDBAtlasFullTextRetriever(document_store=store)

results = retriever.run(query="Lorem ipsum")
print(results["documents"])
```

The example above retrieves the 10 most similar documents to the query "Lorem ipsum" from the
MongoDBAtlasDocumentStore.

<a id="haystack_integrations.components.retrievers.mongodb_atlas.full_text_retriever.MongoDBAtlasFullTextRetriever.__init__"></a>

#### MongoDBAtlasFullTextRetriever.\_\_init\_\_

```python
def __init__(*,
             document_store: MongoDBAtlasDocumentStore,
             filters: dict[str, Any] | None = None,
             top_k: int = 10,
             filter_policy: str | FilterPolicy = FilterPolicy.REPLACE)
```

**Arguments**:

- `document_store`: An instance of MongoDBAtlasDocumentStore.
- `filters`: Filters applied to the retrieved Documents. Make sure that the fields used in the filters are
included in the configuration of the `full_text_search_index`. The configuration must be done manually
in the Web UI of MongoDB Atlas.
- `top_k`: Maximum number of Documents to return.
- `filter_policy`: Policy to determine how filters are applied.

**Raises**:

- `ValueError`: If `document_store` is not an instance of MongoDBAtlasDocumentStore.

<a id="haystack_integrations.components.retrievers.mongodb_atlas.full_text_retriever.MongoDBAtlasFullTextRetriever.to_dict"></a>

#### MongoDBAtlasFullTextRetriever.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.retrievers.mongodb_atlas.full_text_retriever.MongoDBAtlasFullTextRetriever.from_dict"></a>

#### MongoDBAtlasFullTextRetriever.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "MongoDBAtlasFullTextRetriever"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.retrievers.mongodb_atlas.full_text_retriever.MongoDBAtlasFullTextRetriever.run"></a>

#### MongoDBAtlasFullTextRetriever.run

```python
@component.output_types(documents=list[Document])
def run(query: str | list[str],
        fuzzy: dict[str, int] | None = None,
        match_criteria: Literal["any", "all"] | None = None,
        score: dict[str, dict] | None = None,
        synonyms: str | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int = 10) -> dict[str, list[Document]]
```

Retrieve documents from the MongoDBAtlasDocumentStore by full-text search.

**Arguments**:

- `query`: The query string or a list of query strings to search for.
If the query contains multiple terms, Atlas Search evaluates each term separately for matches.
- `fuzzy`: Enables finding strings similar to the search term(s).
Note, `fuzzy` cannot be used with `synonyms`. Configurable options include `maxEdits`, `prefixLength`,
and `maxExpansions`. For more details refer to MongoDB Atlas
[documentation](https://www.mongodb.com/docs/atlas/atlas-search/text/`fields`).
- `match_criteria`: Defines how terms in the query are matched. Supported options are `"any"` and `"all"`.
For more details refer to MongoDB Atlas
[documentation](https://www.mongodb.com/docs/atlas/atlas-search/text/`fields`).
- `score`: Specifies the scoring method for matching results. Supported options include `boost`, `constant`,
and `function`. For more details refer to MongoDB Atlas
[documentation](https://www.mongodb.com/docs/atlas/atlas-search/text/`fields`).
- `synonyms`: The name of the synonym mapping definition in the index. This value cannot be an empty string.
Note, `synonyms` can not be used with `fuzzy`.
- `filters`: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
the `filter_policy` chosen at retriever initialization. See init method docstring for more
details.
- `top_k`: Maximum number of Documents to return. Overrides the value specified at initialization.

**Returns**:

A dictionary with the following keys:
- `documents`: List of Documents most similar to the given `query`

<a id="haystack_integrations.components.retrievers.mongodb_atlas.full_text_retriever.MongoDBAtlasFullTextRetriever.run_async"></a>

#### MongoDBAtlasFullTextRetriever.run\_async

```python
@component.output_types(documents=list[Document])
async def run_async(query: str | list[str],
                    fuzzy: dict[str, int] | None = None,
                    match_criteria: Literal["any", "all"] | None = None,
                    score: dict[str, dict] | None = None,
                    synonyms: str | None = None,
                    filters: dict[str, Any] | None = None,
                    top_k: int = 10) -> dict[str, list[Document]]
```

Asynchronously retrieve documents from the MongoDBAtlasDocumentStore by full-text search.

**Arguments**:

- `query`: The query string or a list of query strings to search for.
If the query contains multiple terms, Atlas Search evaluates each term separately for matches.
- `fuzzy`: Enables finding strings similar to the search term(s).
Note, `fuzzy` cannot be used with `synonyms`. Configurable options include `maxEdits`, `prefixLength`,
and `maxExpansions`. For more details refer to MongoDB Atlas
[documentation](https://www.mongodb.com/docs/atlas/atlas-search/text/`fields`).
- `match_criteria`: Defines how terms in the query are matched. Supported options are `"any"` and `"all"`.
For more details refer to MongoDB Atlas
[documentation](https://www.mongodb.com/docs/atlas/atlas-search/text/`fields`).
- `score`: Specifies the scoring method for matching results. Supported options include `boost`, `constant`,
and `function`. For more details refer to MongoDB Atlas
[documentation](https://www.mongodb.com/docs/atlas/atlas-search/text/`fields`).
- `synonyms`: The name of the synonym mapping definition in the index. This value cannot be an empty string.
Note, `synonyms` can not be used with `fuzzy`.
- `filters`: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
the `filter_policy` chosen at retriever initialization. See init method docstring for more
details.
- `top_k`: Maximum number of Documents to return. Overrides the value specified at initialization.

**Returns**:

A dictionary with the following keys:
- `documents`: List of Documents most similar to the given `query`

<a id="haystack_integrations.document_stores.mongodb_atlas.document_store"></a>

## Module haystack\_integrations.document\_stores.mongodb\_atlas.document\_store

<a id="haystack_integrations.document_stores.mongodb_atlas.document_store.MongoDBAtlasDocumentStore"></a>

### MongoDBAtlasDocumentStore

A MongoDBAtlasDocumentStore implementation that uses the
[MongoDB Atlas](https://www.mongodb.com/atlas/database) service that is easy to deploy, operate, and scale.

To connect to MongoDB Atlas, you need to provide a connection string in the format:
`"mongodb+srv://{mongo_atlas_username}:{mongo_atlas_password}@{mongo_atlas_host}/?{mongo_atlas_params_string}"`.

This connection string can be obtained on the MongoDB Atlas Dashboard by clicking on the `CONNECT` button, selecting
Python as the driver, and copying the connection string. The connection string can be provided as an environment
variable `MONGO_CONNECTION_STRING` or directly as a parameter to the `MongoDBAtlasDocumentStore` constructor.

After providing the connection string, you'll need to specify the `database_name` and `collection_name` to use.
Most likely that you'll create these via the MongoDB Atlas web UI but one can also create them via the MongoDB
Python driver. Creating databases and collections is beyond the scope of MongoDBAtlasDocumentStore. The primary
purpose of this document store is to read and write documents to an existing collection.

Users must provide both a `vector_search_index` for vector search operations and a `full_text_search_index`
for full-text search operations. The `vector_search_index` supports a chosen metric
(e.g., cosine, dot product, or Euclidean), while the `full_text_search_index` enables efficient text-based searches.
Both indexes can be created through the Atlas web UI.

For more details on MongoDB Atlas, see the official
MongoDB Atlas [documentation](https://www.mongodb.com/docs/atlas/getting-started/).

Usage example:
```python
from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore

store = MongoDBAtlasDocumentStore(database_name="your_existing_db",
                                  collection_name="your_existing_collection",
                                  vector_search_index="your_existing_index",
                                  full_text_search_index="your_existing_index")
print(store.count_documents())
```

<a id="haystack_integrations.document_stores.mongodb_atlas.document_store.MongoDBAtlasDocumentStore.__init__"></a>

#### MongoDBAtlasDocumentStore.\_\_init\_\_

```python
def __init__(*,
             mongo_connection_string: Secret = Secret.from_env_var(
                 "MONGO_CONNECTION_STRING"),
             database_name: str,
             collection_name: str,
             vector_search_index: str,
             full_text_search_index: str,
             embedding_field: str = "embedding",
             content_field: str = "content")
```

Creates a new MongoDBAtlasDocumentStore instance.

**Arguments**:

- `mongo_connection_string`: MongoDB Atlas connection string in the format:
`"mongodb+srv://{mongo_atlas_username}:{mongo_atlas_password}@{mongo_atlas_host}/?{mongo_atlas_params_string}"`.
This can be obtained on the MongoDB Atlas Dashboard by clicking on the `CONNECT` button.
This value will be read automatically from the env var "MONGO_CONNECTION_STRING".
- `database_name`: Name of the database to use.
- `collection_name`: Name of the collection to use. To use this document store for embedding retrieval,
this collection needs to have a vector search index set up on the `embedding` field.
- `vector_search_index`: The name of the vector search index to use for vector search operations.
Create a vector_search_index in the Atlas web UI and specify the init params of MongoDBAtlasDocumentStore.             For more details refer to MongoDB
Atlas [documentation](https://www.mongodb.com/docs/atlas/atlas-vector-search/create-index/`std`-label-avs-create-index).
- `full_text_search_index`: The name of the search index to use for full-text search operations.
Create a full_text_search_index in the Atlas web UI and specify the init params of
MongoDBAtlasDocumentStore. For more details refer to MongoDB Atlas
[documentation](https://www.mongodb.com/docs/atlas/atlas-search/create-index/).
- `embedding_field`: The name of the field containing document embeddings. Default is "embedding".
- `content_field`: The name of the field containing the document content. Default is "content".
This field allows defining which field to load into the Haystack Document object as content.
It can be particularly useful when integrating with an existing collection for retrieval. We discourage
using this parameter when working with collections created by Haystack.

**Raises**:

- `ValueError`: If the collection name contains invalid characters.

<a id="haystack_integrations.document_stores.mongodb_atlas.document_store.MongoDBAtlasDocumentStore.__del__"></a>

#### MongoDBAtlasDocumentStore.\_\_del\_\_

```python
def __del__() -> None
```

Destructor method to close MongoDB connections when the instance is destroyed.

<a id="haystack_integrations.document_stores.mongodb_atlas.document_store.MongoDBAtlasDocumentStore.to_dict"></a>

#### MongoDBAtlasDocumentStore.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.document_stores.mongodb_atlas.document_store.MongoDBAtlasDocumentStore.from_dict"></a>

#### MongoDBAtlasDocumentStore.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "MongoDBAtlasDocumentStore"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.document_stores.mongodb_atlas.document_store.MongoDBAtlasDocumentStore.count_documents"></a>

#### MongoDBAtlasDocumentStore.count\_documents

```python
def count_documents() -> int
```

Returns how many documents are present in the document store.

**Returns**:

The number of documents in the document store.

<a id="haystack_integrations.document_stores.mongodb_atlas.document_store.MongoDBAtlasDocumentStore.count_documents_async"></a>

#### MongoDBAtlasDocumentStore.count\_documents\_async

```python
async def count_documents_async() -> int
```

Asynchronously returns how many documents are present in the document store.

**Returns**:

The number of documents in the document store.

<a id="haystack_integrations.document_stores.mongodb_atlas.document_store.MongoDBAtlasDocumentStore.count_documents_by_filter"></a>

#### MongoDBAtlasDocumentStore.count\_documents\_by\_filter

```python
def count_documents_by_filter(filters: dict[str, Any]) -> int
```

Applies a filter and counts the documents that matched it.

**Arguments**:

- `filters`: The filters to apply to the document list.

**Returns**:

The number of documents that match the filter.

<a id="haystack_integrations.document_stores.mongodb_atlas.document_store.MongoDBAtlasDocumentStore.count_documents_by_filter_async"></a>

#### MongoDBAtlasDocumentStore.count\_documents\_by\_filter\_async

```python
async def count_documents_by_filter_async(filters: dict[str, Any]) -> int
```

Asynchronously applies a filter and counts the documents that matched it.

**Arguments**:

- `filters`: The filters to apply to the document list.

**Returns**:

The number of documents that match the filter.

<a id="haystack_integrations.document_stores.mongodb_atlas.document_store.MongoDBAtlasDocumentStore.count_unique_metadata_by_filter"></a>

#### MongoDBAtlasDocumentStore.count\_unique\_metadata\_by\_filter

```python
def count_unique_metadata_by_filter(
        filters: dict[str, Any], metadata_fields: list[str]) -> dict[str, int]
```

Applies a filter selecting documents and counts the unique values for each meta field of the matched documents.

**Arguments**:

- `filters`: The filters to apply to the document list.
- `metadata_fields`: The metadata fields to count unique values for.

**Returns**:

A dictionary where the keys are the metadata field names and the values are the count of unique
values.

<a id="haystack_integrations.document_stores.mongodb_atlas.document_store.MongoDBAtlasDocumentStore.count_unique_metadata_by_filter_async"></a>

#### MongoDBAtlasDocumentStore.count\_unique\_metadata\_by\_filter\_async

```python
async def count_unique_metadata_by_filter_async(
        filters: dict[str, Any], metadata_fields: list[str]) -> dict[str, int]
```

Asynchronously applies a filter selecting documents and counts the unique values for each meta field of the

matched documents.

**Arguments**:

- `filters`: The filters to apply to the document list.
- `metadata_fields`: The metadata fields to count unique values for.

**Returns**:

A dictionary where the keys are the metadata field names and the values are the count of unique
values.

<a id="haystack_integrations.document_stores.mongodb_atlas.document_store.MongoDBAtlasDocumentStore.get_metadata_fields_info"></a>

#### MongoDBAtlasDocumentStore.get\_metadata\_fields\_info

```python
def get_metadata_fields_info() -> dict[str, dict]
```

Returns the metadata fields and their corresponding types.

Since MongoDB is schemaless, this method samples the latest 50 documents to infer the fields and their types.

**Returns**:

A dictionary where the keys are the metadata field names and the values are dictionary with 'type'.

<a id="haystack_integrations.document_stores.mongodb_atlas.document_store.MongoDBAtlasDocumentStore.get_metadata_fields_info_async"></a>

#### MongoDBAtlasDocumentStore.get\_metadata\_fields\_info\_async

```python
async def get_metadata_fields_info_async() -> dict[str, dict]
```

Asynchronously returns the metadata fields and their corresponding types.

Since MongoDB is schemaless, this method samples the latest 50 documents to infer the fields and their types.

**Returns**:

A dictionary where the keys are the metadata field names and the values are dictionary with 'type'.

<a id="haystack_integrations.document_stores.mongodb_atlas.document_store.MongoDBAtlasDocumentStore.get_metadata_field_min_max"></a>

#### MongoDBAtlasDocumentStore.get\_metadata\_field\_min\_max

```python
def get_metadata_field_min_max(metadata_field: str) -> dict[str, Any]
```

For a given metadata field, find its max and min value.

**Arguments**:

- `metadata_field`: The metadata field to get the min and max values for.

**Returns**:

A dictionary with 'min' and 'max' keys.

<a id="haystack_integrations.document_stores.mongodb_atlas.document_store.MongoDBAtlasDocumentStore.get_metadata_field_min_max_async"></a>

#### MongoDBAtlasDocumentStore.get\_metadata\_field\_min\_max\_async

```python
async def get_metadata_field_min_max_async(
        metadata_field: str) -> dict[str, Any]
```

Asynchronously for a given metadata field, find its max and min value.

**Arguments**:

- `metadata_field`: The metadata field to get the min and max values for.

**Returns**:

A dictionary with 'min' and 'max' keys.

<a id="haystack_integrations.document_stores.mongodb_atlas.document_store.MongoDBAtlasDocumentStore.get_metadata_field_unique_values"></a>

#### MongoDBAtlasDocumentStore.get\_metadata\_field\_unique\_values

```python
def get_metadata_field_unique_values(metadata_field: str,
                                     search_term: str | None = None,
                                     from_: int = 0,
                                     size: int = 10) -> tuple[list[str], int]
```

Retrieves unique values for a field matching a search_term or all possible values if no search term is given.

**Arguments**:

- `metadata_field`: The metadata field to retrieve unique values for.
- `search_term`: The search term to filter values. Matches as a case-insensitive substring.
- `from_`: The starting index for pagination.
- `size`: The number of values to return.

**Returns**:

A tuple containing a list of unique values and the total count of unique values matching the
search term.

<a id="haystack_integrations.document_stores.mongodb_atlas.document_store.MongoDBAtlasDocumentStore.get_metadata_field_unique_values_async"></a>

#### MongoDBAtlasDocumentStore.get\_metadata\_field\_unique\_values\_async

```python
async def get_metadata_field_unique_values_async(
        metadata_field: str,
        search_term: str | None = None,
        from_: int = 0,
        size: int = 10) -> tuple[list[str], int]
```

Asynchronously retrieves unique values for a field matching a search_term or all possible values if no search

term is given.

**Arguments**:

- `metadata_field`: The metadata field to retrieve unique values for.
- `search_term`: The search term to filter values. Matches as a case-insensitive substring.
- `from_`: The starting index for pagination.
- `size`: The number of values to return.

**Returns**:

A tuple containing a list of unique values and the total count of unique values matching the
search term.

<a id="haystack_integrations.document_stores.mongodb_atlas.document_store.MongoDBAtlasDocumentStore.filter_documents"></a>

#### MongoDBAtlasDocumentStore.filter\_documents

```python
def filter_documents(filters: dict[str, Any] | None = None) -> list[Document]
```

Returns the documents that match the filters provided.

For a detailed specification of the filters,
refer to the Haystack [documentation](https://docs.haystack.deepset.ai/docs/metadata-filtering).

**Arguments**:

- `filters`: The filters to apply. It returns only the documents that match the filters.

**Returns**:

A list of Documents that match the given filters.

<a id="haystack_integrations.document_stores.mongodb_atlas.document_store.MongoDBAtlasDocumentStore.filter_documents_async"></a>

#### MongoDBAtlasDocumentStore.filter\_documents\_async

```python
async def filter_documents_async(
        filters: dict[str, Any] | None = None) -> list[Document]
```

Asynchronously returns the documents that match the filters provided.

For a detailed specification of the filters,
refer to the Haystack [documentation](https://docs.haystack.deepset.ai/docs/metadata-filtering).

**Arguments**:

- `filters`: The filters to apply. It returns only the documents that match the filters.

**Returns**:

A list of Documents that match the given filters.

<a id="haystack_integrations.document_stores.mongodb_atlas.document_store.MongoDBAtlasDocumentStore.write_documents"></a>

#### MongoDBAtlasDocumentStore.write\_documents

```python
def write_documents(documents: list[Document],
                    policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int
```

Writes documents into the MongoDB Atlas collection.

**Arguments**:

- `documents`: A list of Documents to write to the document store.
- `policy`: The duplicate policy to use when writing documents.

**Raises**:

- `DuplicateDocumentError`: If a document with the same ID already exists in the document store
and the policy is set to DuplicatePolicy.FAIL (or not specified).
- `ValueError`: If the documents are not of type Document.

**Returns**:

The number of documents written to the document store.

<a id="haystack_integrations.document_stores.mongodb_atlas.document_store.MongoDBAtlasDocumentStore.write_documents_async"></a>

#### MongoDBAtlasDocumentStore.write\_documents\_async

```python
async def write_documents_async(
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int
```

Writes documents into the MongoDB Atlas collection.

**Arguments**:

- `documents`: A list of Documents to write to the document store.
- `policy`: The duplicate policy to use when writing documents.

**Raises**:

- `DuplicateDocumentError`: If a document with the same ID already exists in the document store
and the policy is set to DuplicatePolicy.FAIL (or not specified).
- `ValueError`: If the documents are not of type Document.

**Returns**:

The number of documents written to the document store.

<a id="haystack_integrations.document_stores.mongodb_atlas.document_store.MongoDBAtlasDocumentStore.delete_documents"></a>

#### MongoDBAtlasDocumentStore.delete\_documents

```python
def delete_documents(document_ids: list[str]) -> None
```

Deletes all documents with a matching document_ids from the document store.

**Arguments**:

- `document_ids`: the document ids to delete

<a id="haystack_integrations.document_stores.mongodb_atlas.document_store.MongoDBAtlasDocumentStore.delete_documents_async"></a>

#### MongoDBAtlasDocumentStore.delete\_documents\_async

```python
async def delete_documents_async(document_ids: list[str]) -> None
```

Asynchronously deletes all documents with a matching document_ids from the document store.

**Arguments**:

- `document_ids`: the document ids to delete

<a id="haystack_integrations.document_stores.mongodb_atlas.document_store.MongoDBAtlasDocumentStore.delete_by_filter"></a>

#### MongoDBAtlasDocumentStore.delete\_by\_filter

```python
def delete_by_filter(filters: dict[str, Any]) -> int
```

Deletes all documents that match the provided filters.

**Arguments**:

- `filters`: The filters to apply to select documents for deletion.
For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)

**Returns**:

The number of documents deleted.

<a id="haystack_integrations.document_stores.mongodb_atlas.document_store.MongoDBAtlasDocumentStore.delete_by_filter_async"></a>

#### MongoDBAtlasDocumentStore.delete\_by\_filter\_async

```python
async def delete_by_filter_async(filters: dict[str, Any]) -> int
```

Asynchronously deletes all documents that match the provided filters.

**Arguments**:

- `filters`: The filters to apply to select documents for deletion.
For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)

**Returns**:

The number of documents deleted.

<a id="haystack_integrations.document_stores.mongodb_atlas.document_store.MongoDBAtlasDocumentStore.update_by_filter"></a>

#### MongoDBAtlasDocumentStore.update\_by\_filter

```python
def update_by_filter(filters: dict[str, Any], meta: dict[str, Any]) -> int
```

Updates the metadata of all documents that match the provided filters.

**Arguments**:

- `filters`: The filters to apply to select documents for updating.
For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- `meta`: The metadata fields to update.

**Returns**:

The number of documents updated.

<a id="haystack_integrations.document_stores.mongodb_atlas.document_store.MongoDBAtlasDocumentStore.update_by_filter_async"></a>

#### MongoDBAtlasDocumentStore.update\_by\_filter\_async

```python
async def update_by_filter_async(filters: dict[str, Any],
                                 meta: dict[str, Any]) -> int
```

Asynchronously updates the metadata of all documents that match the provided filters.

**Arguments**:

- `filters`: The filters to apply to select documents for updating.
For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- `meta`: The metadata fields to update.

**Returns**:

The number of documents updated.

<a id="haystack_integrations.document_stores.mongodb_atlas.document_store.MongoDBAtlasDocumentStore.delete_all_documents"></a>

#### MongoDBAtlasDocumentStore.delete\_all\_documents

```python
def delete_all_documents(*, recreate_collection: bool = False) -> None
```

Deletes all documents in the document store.

**Arguments**:

- `recreate_collection`: If True, the collection will be dropped and recreated with the original
configuration and indexes. If False, all documents will be deleted while preserving the collection.
Recreating the collection is faster for very large collections.

<a id="haystack_integrations.document_stores.mongodb_atlas.document_store.MongoDBAtlasDocumentStore.delete_all_documents_async"></a>

#### MongoDBAtlasDocumentStore.delete\_all\_documents\_async

```python
async def delete_all_documents_async(*,
                                     recreate_collection: bool = False
                                     ) -> None
```

Asynchronously deletes all documents in the document store.

**Arguments**:

- `recreate_collection`: If True, the collection will be dropped and recreated with the original
configuration and indexes. If False, all documents will be deleted while preserving the collection.
Recreating the collection is faster for very large collections.

