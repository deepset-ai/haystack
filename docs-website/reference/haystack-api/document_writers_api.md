---
title: "Document Writers"
id: document-writers-api
description: "Writes Documents to a DocumentStore."
slug: "/document-writers-api"
---

<a id="document_writer"></a>

# Module document\_writer

<a id="document_writer.DocumentWriter"></a>

## DocumentWriter

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

<a id="document_writer.DocumentWriter.__init__"></a>

#### DocumentWriter.\_\_init\_\_

```python
def __init__(document_store: DocumentStore,
             policy: DuplicatePolicy = DuplicatePolicy.NONE)
```

Create a DocumentWriter component.

**Arguments**:

- `document_store`: The instance of the document store where you want to store your documents.
- `policy`: The policy to apply when a Document with the same ID already exists in the DocumentStore.
- `DuplicatePolicy.NONE`: Default policy, relies on the DocumentStore settings.
- `DuplicatePolicy.SKIP`: Skips documents with the same ID and doesn't write them to the DocumentStore.
- `DuplicatePolicy.OVERWRITE`: Overwrites documents with the same ID.
- `DuplicatePolicy.FAIL`: Raises an error if a Document with the same ID is already in the DocumentStore.

<a id="document_writer.DocumentWriter.to_dict"></a>

#### DocumentWriter.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="document_writer.DocumentWriter.from_dict"></a>

#### DocumentWriter.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "DocumentWriter"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: The dictionary to deserialize from.

**Raises**:

- `DeserializationError`: If the document store is not properly specified in the serialization data or its type cannot be imported.

**Returns**:

The deserialized component.

<a id="document_writer.DocumentWriter.run"></a>

#### DocumentWriter.run

```python
@component.output_types(documents_written=int)
def run(documents: list[Document], policy: Optional[DuplicatePolicy] = None)
```

Run the DocumentWriter on the given input data.

**Arguments**:

- `documents`: A list of documents to write to the document store.
- `policy`: The policy to use when encountering duplicate documents.

**Raises**:

- `ValueError`: If the specified document store is not found.

**Returns**:

Number of documents written to the document store.

<a id="document_writer.DocumentWriter.run_async"></a>

#### DocumentWriter.run\_async

```python
@component.output_types(documents_written=int)
async def run_async(documents: list[Document],
                    policy: Optional[DuplicatePolicy] = None)
```

Asynchronously run the DocumentWriter on the given input data.

This is the asynchronous version of the `run` method. It has the same parameters and return values
but can be used with `await` in async code.

**Arguments**:

- `documents`: A list of documents to write to the document store.
- `policy`: The policy to use when encountering duplicate documents.

**Raises**:

- `ValueError`: If the specified document store is not found.
- `TypeError`: If the specified document store does not implement `write_documents_async`.

**Returns**:

Number of documents written to the document store.

