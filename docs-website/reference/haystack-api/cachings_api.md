---
title: "Caching"
id: caching-api
description: "Checks if any document coming from the given URL is already present in the store."
slug: "/caching-api"
---


## cache_checker

### CacheChecker

Checks for the presence of documents in a Document Store based on a specified field in each document's metadata.

If matching fresh documents are found, they are returned as "hits".
If no matching documents are found, or all matching documents have expired,
the items are returned as "misses".

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

#### __init__

```python
__init__(
    document_store: DocumentStore,
    cache_field: str,
    *,
    ttl: float | timedelta | None = None,
    time_field: str = "cached_at"
) -> None
```

Creates a CacheChecker component.

**Parameters:**

- **document_store** (<code>DocumentStore</code>) – Document Store to check for the presence of specific documents.
- **cache_field** (<code>str</code>) – Name of the document's metadata field
  to check for cache hits.
- **ttl** (<code>float | timedelta | None</code>) – Time-to-live for cache entries, as a `timedelta` or a number of seconds. Must be positive,
  otherwise a `ValueError` is raised.
  If set, a matching document only counts as a hit if the timestamp in `time_field` is not older than `ttl`.
  Documents with a missing, expired or future timestamp are returned as misses.
  If `None`, cache entries never expire.
- **time_field** (<code>str</code>) – Name of the document's metadata field holding the cache timestamp (a `meta.` prefix is ignored).
  Must contain a `datetime` or an ISO 8601 string when `ttl` is set.
  Timezone-aware values are recommended. Malformed values are not handled by the component;
  the Document Store decides (for example, `InMemoryDocumentStore` raises a `FilterError`).

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> CacheChecker
```

Deserialize the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary containing the component type and initialization parameters.

**Returns:**

- <code>CacheChecker</code> – A CacheChecker instance reconstructed from the dictionary.

#### run

```python
run(items: list[Any]) -> dict[str, Any]
```

Checks if any document associated with the specified cache field is already present in the store.

**Parameters:**

- **items** (<code>list\[Any\]</code>) – Values to be checked against the cache field.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with two keys:
- `hits` - Documents matching the requested items and satisfying the TTL filter, if configured.
- `misses` - Items with no matching documents or whose matching documents are all expired.

#### run_async

```python
run_async(items: list[Any]) -> dict[str, Any]
```

Asynchronously checks if any document associated with the specified cache field is already present in the store.

**Parameters:**

- **items** (<code>list\[Any\]</code>) – Values to be checked against the cache field.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with two keys:
- `hits` - Documents matching the requested items and satisfying the TTL filter, if configured.
- `misses` - Items with no matching documents or whose matching documents are all expired.

#### close

```python
close() -> None
```

Release the synchronous resources of the underlying Document Store.

#### close_async

```python
close_async() -> None
```

Release the asynchronous resources of the underlying Document Store.
