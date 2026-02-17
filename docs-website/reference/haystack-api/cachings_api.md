---
title: "Caching"
id: caching-api
description: "Checks if any document coming from the given URL is already present in the store."
slug: "/caching-api"
---


## `cache_checker`

### `CacheChecker`

Checks for the presence of documents in a Document Store based on a specified field in each document's metadata.

If matching documents are found, they are returned as "hits". If not found in the cache, the items
are returned as "misses".

### Usage example

```python
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.caching.cache_checker import CacheChecker

docstore = InMemoryDocumentStore()
documents = [
    Document(content="doc1", meta={"url": "https://example.com/1"}),
    Document(content="doc2", meta={"url": "https://example.com/2"}),
    Document(content="doc3", meta={"url": "https://example.com/1"}),
    Document(content="doc4", meta={"url": "https://example.com/2"}),
]
docstore.write_documents(documents)
checker = CacheChecker(docstore, cache_field="url")
results = checker.run(items=["https://example.com/1", "https://example.com/5"])
assert results == {"hits": [documents[0], documents[2]], "misses": ["https://example.com/5"]}
```

#### `__init__`

```python
__init__(document_store: DocumentStore, cache_field: str)
```

Creates a CacheChecker component.

**Parameters:**

- **document_store** (<code>DocumentStore</code>) – Document Store to check for the presence of specific documents.
- **cache_field** (<code>str</code>) – Name of the document's metadata field
  to check for cache hits.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> CacheChecker
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>CacheChecker</code> – Deserialized component.

#### `run`

```python
run(items: list[Any])
```

Checks if any document associated with the specified cache field is already present in the store.

**Parameters:**

- **items** (<code>list\[Any\]</code>) – Values to be checked against the cache field.

**Returns:**

- – A dictionary with two keys:
- `hits` - Documents that matched with at least one of the items.
- `misses` - Items that were not present in any documents.
