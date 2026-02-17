---
title: "MongoDB Atlas"
id: integrations-mongodb-atlas
description: "MongoDB Atlas integration for Haystack"
slug: "/integrations-mongodb-atlas"
---


## `haystack_integrations.components.retrievers.mongodb_atlas.embedding_retriever`

### `MongoDBAtlasEmbeddingRetriever`

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

#### `__init__`

```python
__init__(
    *,
    document_store: MongoDBAtlasDocumentStore,
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
    filter_policy: str | FilterPolicy = FilterPolicy.REPLACE
)
```

Create the MongoDBAtlasDocumentStore component.

**Parameters:**

- **document_store** (<code>MongoDBAtlasDocumentStore</code>) – An instance of MongoDBAtlasDocumentStore.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved Documents. Make sure that the fields used in the filters are
  included in the configuration of the `vector_search_index`. The configuration must be done manually
  in the Web UI of MongoDB Atlas.
- **top_k** (<code>int</code>) – Maximum number of Documents to return.
- **filter_policy** (<code>str | FilterPolicy</code>) – Policy to determine how filters are applied.

**Raises:**

- <code>ValueError</code> – If `document_store` is not an instance of `MongoDBAtlasDocumentStore`.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> MongoDBAtlasEmbeddingRetriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>MongoDBAtlasEmbeddingRetriever</code> – Deserialized component.

#### `run`

```python
run(
    query_embedding: list[float],
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
) -> dict[str, list[Document]]
```

Retrieve documents from the MongoDBAtlasDocumentStore, based on the provided embedding similarity.

**Parameters:**

- **query_embedding** (<code>list\[float\]</code>) – Embedding of the query.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved Documents. The way runtime filters are applied depends on
  the `filter_policy` chosen at retriever initialization. See init method docstring for more
  details.
- **top_k** (<code>int | None</code>) – Maximum number of Documents to return. Overrides the value specified at initialization.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: List of Documents most similar to the given `query_embedding`

#### `run_async`

```python
run_async(
    query_embedding: list[float],
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
) -> dict[str, list[Document]]
```

Asynchronously retrieve documents from the MongoDBAtlasDocumentStore, based on the provided embedding
similarity.

**Parameters:**

- **query_embedding** (<code>list\[float\]</code>) – Embedding of the query.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved Documents. The way runtime filters are applied depends on
  the `filter_policy` chosen at retriever initialization. See init method docstring for more
  details.
- **top_k** (<code>int | None</code>) – Maximum number of Documents to return. Overrides the value specified at initialization.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: List of Documents most similar to the given `query_embedding`

## `haystack_integrations.components.retrievers.mongodb_atlas.full_text_retriever`

### `MongoDBAtlasFullTextRetriever`

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

#### `__init__`

```python
__init__(
    *,
    document_store: MongoDBAtlasDocumentStore,
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
    filter_policy: str | FilterPolicy = FilterPolicy.REPLACE
)
```

**Parameters:**

- **document_store** (<code>MongoDBAtlasDocumentStore</code>) – An instance of MongoDBAtlasDocumentStore.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved Documents. Make sure that the fields used in the filters are
  included in the configuration of the `full_text_search_index`. The configuration must be done manually
  in the Web UI of MongoDB Atlas.
- **top_k** (<code>int</code>) – Maximum number of Documents to return.
- **filter_policy** (<code>str | FilterPolicy</code>) – Policy to determine how filters are applied.

**Raises:**

- <code>ValueError</code> – If `document_store` is not an instance of MongoDBAtlasDocumentStore.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> MongoDBAtlasFullTextRetriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>MongoDBAtlasFullTextRetriever</code> – Deserialized component.

#### `run`

```python
run(
    query: str | list[str],
    fuzzy: dict[str, int] | None = None,
    match_criteria: Literal["any", "all"] | None = None,
    score: dict[str, dict] | None = None,
    synonyms: str | None = None,
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
) -> dict[str, list[Document]]
```

Retrieve documents from the MongoDBAtlasDocumentStore by full-text search.

**Parameters:**

- **query** (<code>str | list\[str\]</code>) – The query string or a list of query strings to search for.
  If the query contains multiple terms, Atlas Search evaluates each term separately for matches.
- **fuzzy** (<code>dict\[str, int\] | None</code>) – Enables finding strings similar to the search term(s).
  Note, `fuzzy` cannot be used with `synonyms`. Configurable options include `maxEdits`, `prefixLength`,
  and `maxExpansions`. For more details refer to MongoDB Atlas
  [documentation](https://www.mongodb.com/docs/atlas/atlas-search/text/#fields).
- **match_criteria** (<code>Literal['any', 'all'] | None</code>) – Defines how terms in the query are matched. Supported options are `"any"` and `"all"`.
  For more details refer to MongoDB Atlas
  [documentation](https://www.mongodb.com/docs/atlas/atlas-search/text/#fields).
- **score** (<code>dict\[str, dict\] | None</code>) – Specifies the scoring method for matching results. Supported options include `boost`, `constant`,
  and `function`. For more details refer to MongoDB Atlas
  [documentation](https://www.mongodb.com/docs/atlas/atlas-search/text/#fields).
- **synonyms** (<code>str | None</code>) – The name of the synonym mapping definition in the index. This value cannot be an empty string.
  Note, `synonyms` can not be used with `fuzzy`.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved Documents. The way runtime filters are applied depends on
  the `filter_policy` chosen at retriever initialization. See init method docstring for more
  details.
- **top_k** (<code>int</code>) – Maximum number of Documents to return. Overrides the value specified at initialization.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: List of Documents most similar to the given `query`

#### `run_async`

```python
run_async(
    query: str | list[str],
    fuzzy: dict[str, int] | None = None,
    match_criteria: Literal["any", "all"] | None = None,
    score: dict[str, dict] | None = None,
    synonyms: str | None = None,
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
) -> dict[str, list[Document]]
```

Asynchronously retrieve documents from the MongoDBAtlasDocumentStore by full-text search.

**Parameters:**

- **query** (<code>str | list\[str\]</code>) – The query string or a list of query strings to search for.
  If the query contains multiple terms, Atlas Search evaluates each term separately for matches.
- **fuzzy** (<code>dict\[str, int\] | None</code>) – Enables finding strings similar to the search term(s).
  Note, `fuzzy` cannot be used with `synonyms`. Configurable options include `maxEdits`, `prefixLength`,
  and `maxExpansions`. For more details refer to MongoDB Atlas
  [documentation](https://www.mongodb.com/docs/atlas/atlas-search/text/#fields).
- **match_criteria** (<code>Literal['any', 'all'] | None</code>) – Defines how terms in the query are matched. Supported options are `"any"` and `"all"`.
  For more details refer to MongoDB Atlas
  [documentation](https://www.mongodb.com/docs/atlas/atlas-search/text/#fields).
- **score** (<code>dict\[str, dict\] | None</code>) – Specifies the scoring method for matching results. Supported options include `boost`, `constant`,
  and `function`. For more details refer to MongoDB Atlas
  [documentation](https://www.mongodb.com/docs/atlas/atlas-search/text/#fields).
- **synonyms** (<code>str | None</code>) – The name of the synonym mapping definition in the index. This value cannot be an empty string.
  Note, `synonyms` can not be used with `fuzzy`.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved Documents. The way runtime filters are applied depends on
  the `filter_policy` chosen at retriever initialization. See init method docstring for more
  details.
- **top_k** (<code>int</code>) – Maximum number of Documents to return. Overrides the value specified at initialization.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: List of Documents most similar to the given `query`

## `haystack_integrations.document_stores.mongodb_atlas.document_store`

### `MongoDBAtlasDocumentStore`

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

#### `__init__`

```python
__init__(
    *,
    mongo_connection_string: Secret = Secret.from_env_var(
        "MONGO_CONNECTION_STRING"
    ),
    database_name: str,
    collection_name: str,
    vector_search_index: str,
    full_text_search_index: str,
    embedding_field: str = "embedding",
    content_field: str = "content"
)
```

Creates a new MongoDBAtlasDocumentStore instance.

**Parameters:**

- **mongo_connection_string** (<code>Secret</code>) – MongoDB Atlas connection string in the format:
  `"mongodb+srv://{mongo_atlas_username}:{mongo_atlas_password}@{mongo_atlas_host}/?{mongo_atlas_params_string}"`.
  This can be obtained on the MongoDB Atlas Dashboard by clicking on the `CONNECT` button.
  This value will be read automatically from the env var "MONGO_CONNECTION_STRING".
- **database_name** (<code>str</code>) – Name of the database to use.
- **collection_name** (<code>str</code>) – Name of the collection to use. To use this document store for embedding retrieval,
  this collection needs to have a vector search index set up on the `embedding` field.
- **vector_search_index** (<code>str</code>) – The name of the vector search index to use for vector search operations.
  Create a vector_search_index in the Atlas web UI and specify the init params of MongoDBAtlasDocumentStore. For more details refer to MongoDB
  Atlas [documentation](https://www.mongodb.com/docs/atlas/atlas-vector-search/create-index/#std-label-avs-create-index).
- **full_text_search_index** (<code>str</code>) – The name of the search index to use for full-text search operations.
  Create a full_text_search_index in the Atlas web UI and specify the init params of
  MongoDBAtlasDocumentStore. For more details refer to MongoDB Atlas
  [documentation](https://www.mongodb.com/docs/atlas/atlas-search/create-index/).
- **embedding_field** (<code>str</code>) – The name of the field containing document embeddings. Default is "embedding".
- **content_field** (<code>str</code>) – The name of the field containing the document content. Default is "content".
  This field allows defining which field to load into the Haystack Document object as content.
  It can be particularly useful when integrating with an existing collection for retrieval. We discourage
  using this parameter when working with collections created by Haystack.

**Raises:**

- <code>ValueError</code> – If the collection name contains invalid characters.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> MongoDBAtlasDocumentStore
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>MongoDBAtlasDocumentStore</code> – Deserialized component.

#### `count_documents`

```python
count_documents() -> int
```

Returns how many documents are present in the document store.

**Returns:**

- <code>int</code> – The number of documents in the document store.

#### `count_documents_async`

```python
count_documents_async() -> int
```

Asynchronously returns how many documents are present in the document store.

**Returns:**

- <code>int</code> – The number of documents in the document store.

#### `count_documents_by_filter`

```python
count_documents_by_filter(filters: dict[str, Any]) -> int
```

Applies a filter and counts the documents that matched it.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to the document list.

**Returns:**

- <code>int</code> – The number of documents that match the filter.

#### `count_documents_by_filter_async`

```python
count_documents_by_filter_async(filters: dict[str, Any]) -> int
```

Asynchronously applies a filter and counts the documents that matched it.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to the document list.

**Returns:**

- <code>int</code> – The number of documents that match the filter.

#### `count_unique_metadata_by_filter`

```python
count_unique_metadata_by_filter(
    filters: dict[str, Any], metadata_fields: list[str]
) -> dict[str, int]
```

Applies a filter selecting documents and counts the unique values for each meta field of the matched documents.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to the document list.
- **metadata_fields** (<code>list\[str\]</code>) – The metadata fields to count unique values for.

**Returns:**

- <code>dict\[str, int\]</code> – A dictionary where the keys are the metadata field names and the values are the count of unique
  values.

#### `count_unique_metadata_by_filter_async`

```python
count_unique_metadata_by_filter_async(
    filters: dict[str, Any], metadata_fields: list[str]
) -> dict[str, int]
```

Asynchronously applies a filter selecting documents and counts the unique values for each meta field of the
matched documents.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to the document list.
- **metadata_fields** (<code>list\[str\]</code>) – The metadata fields to count unique values for.

**Returns:**

- <code>dict\[str, int\]</code> – A dictionary where the keys are the metadata field names and the values are the count of unique
  values.

#### `get_metadata_fields_info`

```python
get_metadata_fields_info() -> dict[str, dict]
```

Returns the metadata fields and their corresponding types.

Since MongoDB is schemaless, this method samples the latest 50 documents to infer the fields and their types.

**Returns:**

- <code>dict\[str, dict\]</code> – A dictionary where the keys are the metadata field names and the values are dictionary with 'type'.

#### `get_metadata_fields_info_async`

```python
get_metadata_fields_info_async() -> dict[str, dict]
```

Asynchronously returns the metadata fields and their corresponding types.

Since MongoDB is schemaless, this method samples the latest 50 documents to infer the fields and their types.

**Returns:**

- <code>dict\[str, dict\]</code> – A dictionary where the keys are the metadata field names and the values are dictionary with 'type'.

#### `get_metadata_field_min_max`

```python
get_metadata_field_min_max(metadata_field: str) -> dict[str, Any]
```

For a given metadata field, find its max and min value.

**Parameters:**

- **metadata_field** (<code>str</code>) – The metadata field to get the min and max values for.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with 'min' and 'max' keys.

#### `get_metadata_field_min_max_async`

```python
get_metadata_field_min_max_async(metadata_field: str) -> dict[str, Any]
```

Asynchronously for a given metadata field, find its max and min value.

**Parameters:**

- **metadata_field** (<code>str</code>) – The metadata field to get the min and max values for.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with 'min' and 'max' keys.

#### `get_metadata_field_unique_values`

```python
get_metadata_field_unique_values(
    metadata_field: str,
    search_term: str | None = None,
    from_: int = 0,
    size: int = 10,
) -> tuple[list[str], int]
```

Retrieves unique values for a field matching a search_term or all possible values if no search term is given.

**Parameters:**

- **metadata_field** (<code>str</code>) – The metadata field to retrieve unique values for.
- **search_term** (<code>str | None</code>) – The search term to filter values. Matches as a case-insensitive substring.
- **from\_** (<code>int</code>) – The starting index for pagination.
- **size** (<code>int</code>) – The number of values to return.

**Returns:**

- <code>tuple\[list\[str\], int\]</code> – A tuple containing a list of unique values and the total count of unique values matching the
  search term.

#### `get_metadata_field_unique_values_async`

```python
get_metadata_field_unique_values_async(
    metadata_field: str,
    search_term: str | None = None,
    from_: int = 0,
    size: int = 10,
) -> tuple[list[str], int]
```

Asynchronously retrieves unique values for a field matching a search_term or all possible values if no search
term is given.

**Parameters:**

- **metadata_field** (<code>str</code>) – The metadata field to retrieve unique values for.
- **search_term** (<code>str | None</code>) – The search term to filter values. Matches as a case-insensitive substring.
- **from\_** (<code>int</code>) – The starting index for pagination.
- **size** (<code>int</code>) – The number of values to return.

**Returns:**

- <code>tuple\[list\[str\], int\]</code> – A tuple containing a list of unique values and the total count of unique values matching the
  search term.

#### `filter_documents`

```python
filter_documents(filters: dict[str, Any] | None = None) -> list[Document]
```

Returns the documents that match the filters provided.

For a detailed specification of the filters,
refer to the Haystack [documentation](https://docs.haystack.deepset.ai/docs/metadata-filtering).

**Parameters:**

- **filters** (<code>dict\[str, Any\] | None</code>) – The filters to apply. It returns only the documents that match the filters.

**Returns:**

- <code>list\[Document\]</code> – A list of Documents that match the given filters.

#### `filter_documents_async`

```python
filter_documents_async(filters: dict[str, Any] | None = None) -> list[Document]
```

Asynchronously returns the documents that match the filters provided.

For a detailed specification of the filters,
refer to the Haystack [documentation](https://docs.haystack.deepset.ai/docs/metadata-filtering).

**Parameters:**

- **filters** (<code>dict\[str, Any\] | None</code>) – The filters to apply. It returns only the documents that match the filters.

**Returns:**

- <code>list\[Document\]</code> – A list of Documents that match the given filters.

#### `write_documents`

```python
write_documents(
    documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
) -> int
```

Writes documents into the MongoDB Atlas collection.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – A list of Documents to write to the document store.
- **policy** (<code>DuplicatePolicy</code>) – The duplicate policy to use when writing documents.

**Returns:**

- <code>int</code> – The number of documents written to the document store.

**Raises:**

- <code>DuplicateDocumentError</code> – If a document with the same ID already exists in the document store
  and the policy is set to DuplicatePolicy.FAIL (or not specified).
- <code>ValueError</code> – If the documents are not of type Document.

#### `write_documents_async`

```python
write_documents_async(
    documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
) -> int
```

Writes documents into the MongoDB Atlas collection.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – A list of Documents to write to the document store.
- **policy** (<code>DuplicatePolicy</code>) – The duplicate policy to use when writing documents.

**Returns:**

- <code>int</code> – The number of documents written to the document store.

**Raises:**

- <code>DuplicateDocumentError</code> – If a document with the same ID already exists in the document store
  and the policy is set to DuplicatePolicy.FAIL (or not specified).
- <code>ValueError</code> – If the documents are not of type Document.

#### `delete_documents`

```python
delete_documents(document_ids: list[str]) -> None
```

Deletes all documents with a matching document_ids from the document store.

**Parameters:**

- **document_ids** (<code>list\[str\]</code>) – the document ids to delete

#### `delete_documents_async`

```python
delete_documents_async(document_ids: list[str]) -> None
```

Asynchronously deletes all documents with a matching document_ids from the document store.

**Parameters:**

- **document_ids** (<code>list\[str\]</code>) – the document ids to delete

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
- **meta** (<code>dict\[str, Any\]</code>) – The metadata fields to update.

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
- **meta** (<code>dict\[str, Any\]</code>) – The metadata fields to update.

**Returns:**

- <code>int</code> – The number of documents updated.

#### `delete_all_documents`

```python
delete_all_documents(*, recreate_collection: bool = False) -> None
```

Deletes all documents in the document store.

**Parameters:**

- **recreate_collection** (<code>bool</code>) – If True, the collection will be dropped and recreated with the original
  configuration and indexes. If False, all documents will be deleted while preserving the collection.
  Recreating the collection is faster for very large collections.

#### `delete_all_documents_async`

```python
delete_all_documents_async(*, recreate_collection: bool = False) -> None
```

Asynchronously deletes all documents in the document store.

**Parameters:**

- **recreate_collection** (<code>bool</code>) – If True, the collection will be dropped and recreated with the original
  configuration and indexes. If False, all documents will be deleted while preserving the collection.
  Recreating the collection is faster for very large collections.

## `haystack_integrations.document_stores.mongodb_atlas.filters`
