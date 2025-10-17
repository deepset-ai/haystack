---
title: "Classifiers"
id: classifiers-api
description: "Classify documents based on the provided labels."
slug: "/classifiers-api"
---

<a id="document_language_classifier"></a>

# Module document\_language\_classifier

<a id="document_language_classifier.DocumentLanguageClassifier"></a>

## DocumentLanguageClassifier

Classifies the language of each document and adds it to its metadata.

Provide a list of languages during initialization. If the document's text doesn't match any of the
specified languages, the metadata value is set to "unmatched".
To route documents based on their language, use the MetadataRouter component after DocumentLanguageClassifier.
For routing plain text, use the TextLanguageRouter component instead.

### Usage example

```python
from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.classifiers import DocumentLanguageClassifier
from haystack.components.routers import MetadataRouter
from haystack.components.writers import DocumentWriter

docs = [Document(id="1", content="This is an English document"),
        Document(id="2", content="Este es un documento en español")]

document_store = InMemoryDocumentStore()

p = Pipeline()
p.add_component(instance=DocumentLanguageClassifier(languages=["en"]), name="language_classifier")
p.add_component(instance=MetadataRouter(rules={"en": {"language": {"$eq": "en"}}}), name="router")
p.add_component(instance=DocumentWriter(document_store=document_store), name="writer")
p.connect("language_classifier.documents", "router.documents")
p.connect("router.en", "writer.documents")

p.run({"language_classifier": {"documents": docs}})

written_docs = document_store.filter_documents()
assert len(written_docs) == 1
assert written_docs[0] == Document(id="1", content="This is an English document", meta={"language": "en"})
```

<a id="document_language_classifier.DocumentLanguageClassifier.__init__"></a>

#### DocumentLanguageClassifier.\_\_init\_\_

```python
def __init__(languages: Optional[list[str]] = None)
```

Initializes the DocumentLanguageClassifier component.

**Arguments**:

- `languages`: A list of ISO language codes.
See the supported languages in [`langdetect` documentation](https://github.com/Mimino666/langdetect#languages).
If not specified, defaults to ["en"].

<a id="document_language_classifier.DocumentLanguageClassifier.run"></a>

#### DocumentLanguageClassifier.run

```python
@component.output_types(documents=list[Document])
def run(documents: list[Document])
```

Classifies the language of each document and adds it to its metadata.

If the document's text doesn't match any of the languages specified at initialization,
sets the metadata value to "unmatched".

**Arguments**:

- `documents`: A list of documents for language classification.

**Raises**:

- `TypeError`: if the input is not a list of Documents.

**Returns**:

A dictionary with the following key:
- `documents`: A list of documents with an added `language` metadata field.

<a id="zero_shot_document_classifier"></a>

# Module zero\_shot\_document\_classifier

<a id="zero_shot_document_classifier.TransformersZeroShotDocumentClassifier"></a>

## TransformersZeroShotDocumentClassifier

Performs zero-shot classification of documents based on given labels and adds the predicted label to their metadata.

The component uses a Hugging Face pipeline for zero-shot classification.
Provide the model and the set of labels to be used for categorization during initialization.
Additionally, you can configure the component to allow multiple labels to be true.

Classification is run on the document's content field by default. If you want it to run on another field, set the
`classification_field` to one of the document's metadata fields.

Available models for the task of zero-shot-classification include:
    - `valhalla/distilbart-mnli-12-3`
    - `cross-encoder/nli-distilroberta-base`
    - `cross-encoder/nli-deberta-v3-xsmall`

### Usage example

The following is a pipeline that classifies documents based on predefined classification labels
retrieved from a search pipeline:

```python
from haystack import Document
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.core.pipeline import Pipeline
from haystack.components.classifiers import TransformersZeroShotDocumentClassifier

documents = [Document(id="0", content="Today was a nice day!"),
             Document(id="1", content="Yesterday was a bad day!")]

document_store = InMemoryDocumentStore()
retriever = InMemoryBM25Retriever(document_store=document_store)
document_classifier = TransformersZeroShotDocumentClassifier(
    model="cross-encoder/nli-deberta-v3-xsmall",
    labels=["positive", "negative"],
)

document_store.write_documents(documents)

pipeline = Pipeline()
pipeline.add_component(instance=retriever, name="retriever")
pipeline.add_component(instance=document_classifier, name="document_classifier")
pipeline.connect("retriever", "document_classifier")

queries = ["How was your day today?", "How was your day yesterday?"]
expected_predictions = ["positive", "negative"]

for idx, query in enumerate(queries):
    result = pipeline.run({"retriever": {"query": query, "top_k": 1}})
    assert result["document_classifier"]["documents"][0].to_dict()["id"] == str(idx)
    assert (result["document_classifier"]["documents"][0].to_dict()["classification"]["label"]
            == expected_predictions[idx])
```

<a id="zero_shot_document_classifier.TransformersZeroShotDocumentClassifier.__init__"></a>

#### TransformersZeroShotDocumentClassifier.\_\_init\_\_

```python
def __init__(model: str,
             labels: list[str],
             multi_label: bool = False,
             classification_field: Optional[str] = None,
             device: Optional[ComponentDevice] = None,
             token: Optional[Secret] = Secret.from_env_var(
                 ["HF_API_TOKEN", "HF_TOKEN"], strict=False),
             huggingface_pipeline_kwargs: Optional[dict[str, Any]] = None)
```

Initializes the TransformersZeroShotDocumentClassifier.

See the Hugging Face [website](https://huggingface.co/models?pipeline_tag=zero-shot-classification&sort=downloads&search=nli)
for the full list of zero-shot classification models (NLI) models.

**Arguments**:

- `model`: The name or path of a Hugging Face model for zero shot document classification.
- `labels`: The set of possible class labels to classify each document into, for example,
["positive", "negative"]. The labels depend on the selected model.
- `multi_label`: Whether or not multiple candidate labels can be true.
If `False`, the scores are normalized such that
the sum of the label likelihoods for each sequence is 1. If `True`, the labels are considered
independent and probabilities are normalized for each candidate by doing a softmax of the entailment
score vs. the contradiction score.
- `classification_field`: Name of document's meta field to be used for classification.
If not set, `Document.content` is used by default.
- `device`: The device on which the model is loaded. If `None`, the default device is automatically
selected. If a device/device map is specified in `huggingface_pipeline_kwargs`, it overrides this parameter.
- `token`: The Hugging Face token to use as HTTP bearer authorization.
Check your HF token in your [account settings](https://huggingface.co/settings/tokens).
- `huggingface_pipeline_kwargs`: Dictionary containing keyword arguments used to initialize the
Hugging Face pipeline for text classification.

<a id="zero_shot_document_classifier.TransformersZeroShotDocumentClassifier.warm_up"></a>

#### TransformersZeroShotDocumentClassifier.warm\_up

```python
def warm_up()
```

Initializes the component.

<a id="zero_shot_document_classifier.TransformersZeroShotDocumentClassifier.to_dict"></a>

#### TransformersZeroShotDocumentClassifier.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="zero_shot_document_classifier.TransformersZeroShotDocumentClassifier.from_dict"></a>

#### TransformersZeroShotDocumentClassifier.from\_dict

```python
@classmethod
def from_dict(
        cls, data: dict[str, Any]) -> "TransformersZeroShotDocumentClassifier"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="zero_shot_document_classifier.TransformersZeroShotDocumentClassifier.run"></a>

#### TransformersZeroShotDocumentClassifier.run

```python
@component.output_types(documents=list[Document])
def run(documents: list[Document], batch_size: int = 1)
```

Classifies the documents based on the provided labels and adds them to their metadata.

The classification results are stored in the `classification` dict within
each document's metadata. If `multi_label` is set to `True`, the scores for each label are available under
the `details` key within the `classification` dictionary.

**Arguments**:

- `documents`: Documents to process.
- `batch_size`: Batch size used for processing the content in each document.

**Returns**:

A dictionary with the following key:
- `documents`: A list of documents with an added metadata field called `classification`.

