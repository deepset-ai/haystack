---
title: "Chonkie"
id: integrations-chonkie
description: "Chonkie integration for Haystack"
slug: "/integrations-chonkie"
---


## haystack_integrations.components.preprocessors.chonkie.recursive_splitter

### ChonkieRecursiveDocumentSplitter

A Document Splitter that uses Chonkie's RecursiveChunker to split documents.

### Usage example

```python
from haystack import Document
from haystack_integrations.components.preprocessors.chonkie import ChonkieRecursiveDocumentSplitter

chunker = ChonkieRecursiveDocumentSplitter(chunk_size=512)
documents = [Document(content="Hello world. This is a test.")]
result = chunker.run(documents=documents)
print(result["documents"])
```

#### __init__

```python
__init__(
    *,
    tokenizer: str = "character",
    chunk_size: int = 2048,
    min_characters_per_chunk: int = 24,
    rules: RecursiveRules | dict[str, Any] | None = None,
    skip_empty_documents: bool = True,
    page_break_character: str = "\x0c"
) -> None
```

Initializes the ChonkieRecursiveDocumentSplitter.

**Parameters:**

- **tokenizer** (<code>str</code>) – The tokenizer to use for chunking. Defaults to "character".
  Common options include "character", "gpt2", and "cl100k_base".
  See the [Chonkie documentation](https://docs.chonkie.ai/) for more information on available tokenizers.
- **chunk_size** (<code>int</code>) – The maximum number of tokens per chunk. The actual length depends on the chosen tokenizer.
- **min_characters_per_chunk** (<code>int</code>) – The minimum number of characters per chunk.
- **rules** (<code>RecursiveRules | dict\[str, Any\] | None</code>) – Custom rules for recursive chunking. If None, default rules are used.
  See the [Chonkie documentation](https://docs.chonkie.ai/) for more information.
- **skip_empty_documents** (<code>bool</code>) – Whether to skip empty documents.
- **page_break_character** (<code>str</code>) – The character to use for page breaks.

#### run

```python
run(documents: list[Document]) -> dict[str, list[Document]]
```

Splits a list of documents into smaller chunks.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – The list of documents to split.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the "documents" key containing the list of chunks.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> ChonkieRecursiveDocumentSplitter
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>ChonkieRecursiveDocumentSplitter</code> – Deserialized component.

## haystack_integrations.components.preprocessors.chonkie.semantic_splitter

### ChonkieSemanticDocumentSplitter

A Document Splitter that uses Chonkie's SemanticChunker to split documents.

### Usage example

```python
from haystack import Document
from haystack_integrations.components.preprocessors.chonkie import ChonkieSemanticDocumentSplitter

chunker = ChonkieSemanticDocumentSplitter(chunk_size=512)
documents = [Document(content="Hello world. This is a test.")]
result = chunker.run(documents=documents)
print(result["documents"])
```

#### __init__

```python
__init__(
    *,
    embedding_model: Any = "minishlab/potion-base-32M",
    threshold: float = 0.8,
    chunk_size: int = 2048,
    similarity_window: int = 3,
    min_sentences_per_chunk: int = 1,
    min_characters_per_sentence: int = 24,
    delim: Any = None,
    include_delim: str = "prev",
    skip_window: int = 0,
    filter_window: int = 5,
    filter_polyorder: int = 3,
    filter_tolerance: float = 0.2,
    skip_empty_documents: bool = True,
    page_break_character: str = "\x0c"
) -> None
```

Initializes the ChonkieSemanticDocumentSplitter.

**Parameters:**

- **embedding_model** (<code>Any</code>) – The embedding model to use for semantic similarity.
  See the [Chonkie documentation](https://docs.chonkie.ai/) for more information on supported models.
- **threshold** (<code>float</code>) – The semantic similarity threshold.
- **chunk_size** (<code>int</code>) – The maximum number of tokens per chunk. The actual length depends on the
  embedding model's tokenizer.
- **similarity_window** (<code>int</code>) – The window size for similarity calculations.
- **min_sentences_per_chunk** (<code>int</code>) – The minimum number of sentences per chunk.
- **min_characters_per_sentence** (<code>int</code>) – The minimum number of characters per sentence.
- **delim** (<code>Any</code>) – Delimiters to use for splitting. If None, default delimiters are used.
- **include_delim** (<code>str</code>) – Whether to include the delimiter in the chunks.
- **skip_window** (<code>int</code>) – The skip window for similarity calculations.
- **filter_window** (<code>int</code>) – The filter window for similarity calculations.
- **filter_polyorder** (<code>int</code>) – The polynomial order for similarity filtering.
- **filter_tolerance** (<code>float</code>) – The tolerance for similarity filtering.
- **skip_empty_documents** (<code>bool</code>) – Whether to skip empty documents.
- **page_break_character** (<code>str</code>) – The character to use for page breaks.

#### warm_up

```python
warm_up() -> None
```

Initializes the component by loading the embedding model.

#### run

```python
run(documents: list[Document]) -> dict[str, list[Document]]
```

Splits a list of documents into smaller semantic chunks.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – The list of documents to split.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the "documents" key containing the list of chunks.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> ChonkieSemanticDocumentSplitter
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>ChonkieSemanticDocumentSplitter</code> – Deserialized component.

## haystack_integrations.components.preprocessors.chonkie.sentence_splitter

### ChonkieSentenceDocumentSplitter

A Document Splitter that uses Chonkie's SentenceChunker to split documents.

### Usage example

```python
from haystack import Document
from haystack_integrations.components.preprocessors.chonkie import ChonkieSentenceDocumentSplitter

chunker = ChonkieSentenceDocumentSplitter(chunk_size=512)
documents = [Document(content="Hello world. This is a test.")]
result = chunker.run(documents=documents)
print(result["documents"])
```

#### __init__

```python
__init__(
    *,
    tokenizer: str = "character",
    chunk_size: int = 2048,
    chunk_overlap: int = 0,
    min_sentences_per_chunk: int = 1,
    min_characters_per_sentence: int = 12,
    approximate: bool = False,
    delim: Any = None,
    include_delim: str = "prev",
    skip_empty_documents: bool = True,
    page_break_character: str = "\x0c"
) -> None
```

Initializes the ChonkieSentenceDocumentSplitter.

**Parameters:**

- **tokenizer** (<code>str</code>) – The tokenizer to use for chunking. Defaults to "character".
  Common options include "character", "gpt2", and "cl100k_base".
  See the [Chonkie documentation](https://docs.chonkie.ai/) for more information on available tokenizers.
- **chunk_size** (<code>int</code>) – The maximum number of tokens per chunk. The actual length depends on the chosen tokenizer.
- **chunk_overlap** (<code>int</code>) – The overlap between consecutive chunks.
- **min_sentences_per_chunk** (<code>int</code>) – The minimum number of sentences per chunk.
- **min_characters_per_sentence** (<code>int</code>) – The minimum number of characters per sentence.
- **approximate** (<code>bool</code>) – Whether to use approximate chunking.
- **delim** (<code>Any</code>) – Delimiters to use for splitting. If None, default delimiters are used.
- **include_delim** (<code>str</code>) – Whether to include the delimiter in the chunks ("prev" or "next").
- **skip_empty_documents** (<code>bool</code>) – Whether to skip empty documents.
- **page_break_character** (<code>str</code>) – The character to use for page breaks.

#### run

```python
run(documents: list[Document]) -> dict[str, list[Document]]
```

Splits a list of documents into smaller sentence-based chunks.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – The list of documents to split.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the "documents" key containing the list of chunks.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> ChonkieSentenceDocumentSplitter
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>ChonkieSentenceDocumentSplitter</code> – Deserialized component.

## haystack_integrations.components.preprocessors.chonkie.token_splitter

### ChonkieTokenDocumentSplitter

A Document Splitter that uses Chonkie's TokenChunker to split documents.

### Usage example

```python
from haystack import Document
from haystack_integrations.components.preprocessors.chonkie import ChonkieTokenDocumentSplitter

chunker = ChonkieTokenDocumentSplitter(chunk_size=512, chunk_overlap=50)
documents = [Document(content="Hello world. This is a test.")]
result = chunker.run(documents=documents)
print(result["documents"])
```

#### __init__

```python
__init__(
    *,
    tokenizer: str = "character",
    chunk_size: int = 2048,
    chunk_overlap: int = 0,
    skip_empty_documents: bool = True,
    page_break_character: str = "\x0c"
) -> None
```

Initializes the ChonkieTokenDocumentSplitter.

**Parameters:**

- **tokenizer** (<code>str</code>) – The tokenizer to use for chunking. Defaults to "character".
  Common options include "character", "gpt2", and "cl100k_base".
  See the [Chonkie documentation](https://docs.chonkie.ai/) for more information on available tokenizers.
- **chunk_size** (<code>int</code>) – The maximum number of tokens per chunk. The actual length depends on the chosen tokenizer.
- **chunk_overlap** (<code>int</code>) – The overlap between consecutive chunks.
- **skip_empty_documents** (<code>bool</code>) – Whether to skip empty documents.
- **page_break_character** (<code>str</code>) – The character to use for page breaks.

#### run

```python
run(documents: list[Document]) -> dict[str, list[Document]]
```

Splits a list of documents into smaller token-based chunks.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – The list of documents to split.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the "documents" key containing the list of chunks.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> ChonkieTokenDocumentSplitter
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>ChonkieTokenDocumentSplitter</code> – Deserialized component.
