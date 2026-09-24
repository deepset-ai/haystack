---
title: "Langdetect"
id: integrations-langdetect
description: "Langdetect integration for Haystack"
slug: "/integrations-langdetect"
---


## haystack_integrations.components.classifiers.langdetect.document_language_classifier

### DocumentLanguageClassifier

Classifies the language of each document and adds it to its metadata.

Provide a list of languages during initialization. If the document's text doesn't match any of the
specified languages, the metadata value is set to "unmatched".
To route documents based on their language, use the MetadataRouter component after DocumentLanguageClassifier.
For routing plain text, use the TextLanguageRouter component instead.

### Usage example

```python
from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.classifiers.langdetect import DocumentLanguageClassifier
from haystack.components.routers import MetadataRouter
from haystack.components.writers import DocumentWriter

docs = [Document(id="1", content="This is an English document"),
        Document(id="2", content="Este es un documento en español")]

document_store = InMemoryDocumentStore()

p = Pipeline()
p.add_component(instance=DocumentLanguageClassifier(languages=["en"]), name="language_classifier")
p.add_component(
instance=MetadataRouter(rules={
    "en": {
        "field": "meta.language",
        "operator": "==",
        "value": "en"
    }
}),
name="router")
p.add_component(instance=DocumentWriter(document_store=document_store), name="writer")
p.connect("language_classifier.documents", "router.documents")
p.connect("router.en", "writer.documents")

p.run({"language_classifier": {"documents": docs}})

written_docs = document_store.filter_documents()
assert len(written_docs) == 1
assert written_docs[0] == Document(id="1", content="This is an English document", meta={"language": "en"})
```

#### __init__

```python
__init__(languages: list[str] | None = None) -> None
```

Initializes the DocumentLanguageClassifier component.

**Parameters:**

- **languages** (<code>list\[str\] | None</code>) – A list of ISO language codes.
  See the supported languages in [`langdetect` documentation](https://github.com/Mimino666/langdetect#languages).
  If not specified, defaults to ["en"].

#### run

```python
run(documents: list[Document]) -> dict[str, list[Document]]
```

Classifies the language of each document and adds it to its metadata.

If the document's text doesn't match any of the languages specified at initialization,
sets the metadata value to "unmatched".

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – A list of documents for language classification.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following key:
- `documents`: A list of documents with an added `language` metadata field.

**Raises:**

- <code>TypeError</code> – if the input is not a list of Documents.

## haystack_integrations.components.routers.langdetect.text_language_router

### TextLanguageRouter

Routes text strings to different output connections based on their language.

Provide a list of languages during initialization. If the document's text doesn't match any of the
specified languages, the metadata value is set to "unmatched".
For routing documents based on their language, use the DocumentLanguageClassifier component,
followed by the MetaDataRouter.

### Usage example

```python
from haystack import Pipeline, Document
from haystack_integrations.components.routers.langdetect import TextLanguageRouter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever

document_store = InMemoryDocumentStore()
document_store.write_documents([Document(content="Elvis Presley was an American singer and actor.")])

p = Pipeline()
p.add_component(instance=TextLanguageRouter(languages=["en"]), name="text_language_router")
p.add_component(instance=InMemoryBM25Retriever(document_store=document_store), name="retriever")
p.connect("text_language_router.en", "retriever.query")

result = p.run({"text_language_router": {"text": "Who was Elvis Presley?"}})
assert result["retriever"]["documents"][0].content == "Elvis Presley was an American singer and actor."

result = p.run({"text_language_router": {"text": "ένα ελληνικό κείμενο"}})
assert result["text_language_router"]["unmatched"] == "ένα ελληνικό κείμενο"
```

#### __init__

```python
__init__(languages: list[str] | None = None) -> None
```

Initialize the TextLanguageRouter component.

**Parameters:**

- **languages** (<code>list\[str\] | None</code>) – A list of ISO language codes.
  See the supported languages in [`langdetect` documentation](https://github.com/Mimino666/langdetect#languages).
  If not specified, defaults to ["en"].

#### run

```python
run(text: str) -> dict[str, str]
```

Routes the text strings to different output connections based on their language.

If the document's text doesn't match any of the specified languages, the metadata value is set to "unmatched".

**Parameters:**

- **text** (<code>str</code>) – A text string to route.

**Returns:**

- <code>dict\[str, str\]</code> – A dictionary in which the key is the language (or `"unmatched"`),
  and the value is the text.

**Raises:**

- <code>TypeError</code> – If the input is not a string.
