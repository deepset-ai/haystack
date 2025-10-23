---
title: "Preprocessors"
id: experimental-preprocessors-api
description: "Pipelines wrapped as components."
slug: "/experimental-preprocessors-api"
---

<a id="haystack_experimental.components.preprocessors.embedding_based_document_splitter"></a>

## Module haystack\_experimental.components.preprocessors.embedding\_based\_document\_splitter

<a id="haystack_experimental.components.preprocessors.embedding_based_document_splitter.EmbeddingBasedDocumentSplitter"></a>

### EmbeddingBasedDocumentSplitter

Splits documents based on embedding similarity using cosine distances between sequential sentence groups.

This component first splits text into sentences, optionally groups them, calculates embeddings for each group,
and then uses cosine distance between sequential embeddings to determine split points. Any distance above
the specified percentile is treated as a break point. The component also tracks page numbers based on form feed
characters (``) in the original document.

This component is inspired by [5 Levels of Text Splitting](
    https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb
) by Greg Kamradt.

### Usage example

```python
from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack_experimental.components.preprocessors import EmbeddingBasedDocumentSplitter

# Create a document with content that has a clear topic shift
doc = Document(
    content="This is a first sentence. This is a second sentence. This is a third sentence. "
    "Completely different topic. The same completely different topic."
)

# Initialize the embedder to calculate semantic similarities
embedder = SentenceTransformersDocumentEmbedder()

# Configure the splitter with parameters that control splitting behavior
splitter = EmbeddingBasedDocumentSplitter(
    document_embedder=embedder,
    sentences_per_group=2,      # Group 2 sentences before calculating embeddings
    percentile=0.95,            # Split when cosine distance exceeds 95th percentile
    min_length=50,              # Merge splits shorter than 50 characters
    max_length=1000             # Further split chunks longer than 1000 characters
)
splitter.warm_up()
result = splitter.run(documents=[doc])

# The result contains a list of Document objects, each representing a semantic chunk
# Each split document includes metadata: source_id, split_id, and page_number
print(f"Original document split into {len(result['documents'])} chunks")
for i, split_doc in enumerate(result['documents']):
    print(f"Chunk {i}: {split_doc.content[:50]}...")
```

<a id="haystack_experimental.components.preprocessors.embedding_based_document_splitter.EmbeddingBasedDocumentSplitter.__init__"></a>

#### EmbeddingBasedDocumentSplitter.\_\_init\_\_

```python
def __init__(*,
             document_embedder: DocumentEmbedder,
             sentences_per_group: int = 3,
             percentile: float = 0.95,
             min_length: int = 50,
             max_length: int = 1000,
             language: Language = "en",
             use_split_rules: bool = True,
             extend_abbreviations: bool = True)
```

Initialize EmbeddingBasedDocumentSplitter.

**Arguments**:

- `document_embedder`: The DocumentEmbedder to use for calculating embeddings.
- `sentences_per_group`: Number of sentences to group together before embedding.
- `percentile`: Percentile threshold for cosine distance. Distances above this percentile
are treated as break points.
- `min_length`: Minimum length of splits in characters. Splits below this length will be merged.
- `max_length`: Maximum length of splits in characters. Splits above this length will be recursively split.
- `language`: Language for sentence tokenization.
- `use_split_rules`: Whether to use additional split rules for sentence tokenization. Applies additional
split rules from SentenceSplitter to the sentence spans.
- `extend_abbreviations`: If True, the abbreviations used by NLTK's PunktTokenizer are extended by a list
of curated abbreviations. Currently supported languages are: en, de.
If False, the default abbreviations are used.

<a id="haystack_experimental.components.preprocessors.embedding_based_document_splitter.EmbeddingBasedDocumentSplitter.warm_up"></a>

#### EmbeddingBasedDocumentSplitter.warm\_up

```python
def warm_up() -> None
```

Warm up the component by initializing the sentence splitter.

<a id="haystack_experimental.components.preprocessors.embedding_based_document_splitter.EmbeddingBasedDocumentSplitter.run"></a>

#### EmbeddingBasedDocumentSplitter.run

```python
@component.output_types(documents=List[Document])
def run(documents: List[Document]) -> Dict[str, List[Document]]
```

Split documents based on embedding similarity.

**Arguments**:

- `documents`: The documents to split.

**Raises**:

- `None`: - `RuntimeError`: If the component wasn't warmed up.
- `TypeError`: If the input is not a list of Documents.
- `ValueError`: If the document content is None or empty.

**Returns**:

A dictionary with the following key:
- `documents`: List of documents with the split texts. Each document includes:
- A metadata field `source_id` to track the original document.
- A metadata field `split_id` to track the split number.
- A metadata field `page_number` to track the original page number.
- All other metadata copied from the original document.

<a id="haystack_experimental.components.preprocessors.embedding_based_document_splitter.EmbeddingBasedDocumentSplitter.to_dict"></a>

#### EmbeddingBasedDocumentSplitter.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

<a id="haystack_experimental.components.preprocessors.embedding_based_document_splitter.EmbeddingBasedDocumentSplitter.from_dict"></a>

#### EmbeddingBasedDocumentSplitter.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingBasedDocumentSplitter"
```

Deserializes the component from a dictionary.

