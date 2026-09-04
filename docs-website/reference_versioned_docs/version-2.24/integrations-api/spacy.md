---
title: "Spacy"
id: integrations-spacy
description: "Spacy integration for Haystack"
slug: "/integrations-spacy"
---


## haystack_integrations.components.extractors.spacy.named_entity_extractor

### NamedEntityAnnotation

Describes a single NER annotation.

**Parameters:**

- **entity** (<code>str</code>) – Entity label.
- **start** (<code>int</code>) – Start index of the entity in the document.
- **end** (<code>int</code>) – End index of the entity in the document.
- **score** (<code>float | None</code>) – Score calculated by the model.

### SpacyNamedEntityExtractor

Annotates named entities in a collection of documents.

The component can be used with any [spaCy model](https://spacy.io/models) that contains
an NER component. Annotations are stored as metadata in the documents.

Usage example:

```python
from haystack import Document

from haystack_integrations.components.extractors.spacy import SpacyNamedEntityExtractor

documents = [
    Document(content="I'm Merlin, the happy pig!"),
    Document(content="My name is Clara and I live in Berkeley, California."),
]
extractor = SpacyNamedEntityExtractor(model="en_core_web_sm")
results = extractor.run(documents=documents)["documents"]
annotations = [SpacyNamedEntityExtractor.get_stored_annotations(doc) for doc in results]
print(annotations)
```

#### __init__

```python
__init__(
    *,
    model: str,
    pipeline_kwargs: dict[str, Any] | None = None,
    device: ComponentDevice | None = None
) -> None
```

Create a Named Entity extractor component.

**Parameters:**

- **model** (<code>str</code>) – Name of the spaCy model or a path to the model on
  the local disk.
- **pipeline_kwargs** (<code>dict\[str, Any\] | None</code>) – Keyword arguments passed to the pipeline. The
  pipeline can override these arguments.
- **device** (<code>ComponentDevice | None</code>) – The device on which the model is loaded. If `None`,
  the default device is automatically selected.

**Raises:**

- <code>ValueError</code> – If the device represents multiple devices, which the
  spaCy backend does not support.

#### warm_up

```python
warm_up() -> None
```

Initialize the component.

**Raises:**

- <code>ComponentError</code> – If the component fails to initialize successfully.

#### run

```python
run(documents: list[Document], batch_size: int = 1) -> dict[str, Any]
```

Annotate named entities in each document and store the annotations in the document's metadata.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – Documents to process.
- **batch_size** (<code>int</code>) – Batch size used for processing the documents.

**Returns:**

- <code>dict\[str, Any\]</code> – Processed documents.

**Raises:**

- <code>ComponentError</code> – If the model fails to process a document.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> SpacyNamedEntityExtractor
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>SpacyNamedEntityExtractor</code> – Deserialized component.

#### initialized

```python
initialized: bool
```

Returns if the extractor is ready to annotate text.

#### get_stored_annotations

```python
get_stored_annotations(
    document: Document,
) -> list[NamedEntityAnnotation] | None
```

Returns the document's named entity annotations stored in its metadata, if any.

**Parameters:**

- **document** (<code>Document</code>) – Document whose annotations are to be fetched.

**Returns:**

- <code>list\[NamedEntityAnnotation\] | None</code> – The stored annotations.
