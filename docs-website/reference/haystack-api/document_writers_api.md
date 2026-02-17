---
title: "Document Writers"
id: document-writers-api
description: "Writes Documents to a DocumentStore."
slug: "/document-writers-api"
---


## `haystack.components.writers.document_writer`

### `haystack.components.writers.document_writer.DocumentWriter`

Writes documents to a DocumentStore.

### Usage example

```python
from haystack import Document
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
docs = [
    Document(content="Python is a popular programming language"),
]
doc_store = InMemoryDocumentStore()
writer = DocumentWriter(document_store=doc_store)
writer.run(docs)
```

#### `__init__`

```python
__init__(
    document_store: DocumentStore,
    policy: DuplicatePolicy = DuplicatePolicy.NONE,
)
```

Create a DocumentWriter component.

**Parameters:**

- **document_store** (<code>DocumentStore</code>) – The instance of the document store where you want to store your documents.
- **policy** (<code>DuplicatePolicy</code>) – The policy to apply when a Document with the same ID already exists in the DocumentStore.
- `DuplicatePolicy.NONE`: Default policy, relies on the DocumentStore settings.
- `DuplicatePolicy.SKIP`: Skips documents with the same ID and doesn't write them to the DocumentStore.
- `DuplicatePolicy.OVERWRITE`: Overwrites documents with the same ID.
- `DuplicatePolicy.FAIL`: Raises an error if a Document with the same ID is already in the DocumentStore.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> DocumentWriter
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to deserialize from.

**Returns:**

- <code>DocumentWriter</code> – The deserialized component.

**Raises:**

- <code>DeserializationError</code> – If the document store is not properly specified in the serialization data or its type cannot be imported.

#### `run`

```python
run(
    documents: list[Document], policy: DuplicatePolicy | None = None
) -> dict[str, int]
```

Run the DocumentWriter on the given input data.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – A list of documents to write to the document store.
- **policy** (<code>DuplicatePolicy | None</code>) – The policy to use when encountering duplicate documents.

**Returns:**

- <code>dict\[str, int\]</code> – Number of documents written to the document store.

**Raises:**

- <code>ValueError</code> – If the specified document store is not found.

#### `run_async`

```python
run_async(
    documents: list[Document], policy: DuplicatePolicy | None = None
) -> dict[str, int]
```

Asynchronously run the DocumentWriter on the given input data.

This is the asynchronous version of the `run` method. It has the same parameters and return values
but can be used with `await` in async code.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – A list of documents to write to the document store.
- **policy** (<code>DuplicatePolicy | None</code>) – The policy to use when encountering duplicate documents.

**Returns:**

- <code>dict\[str, int\]</code> – Number of documents written to the document store.

**Raises:**

- <code>ValueError</code> – If the specified document store is not found.
- <code>TypeError</code> – If the specified document store does not implement `write_documents_async`.
