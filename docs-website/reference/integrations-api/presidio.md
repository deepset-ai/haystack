---
title: "Presidio"
id: integrations-presidio
description: "Presidio integration for Haystack"
slug: "/integrations-presidio"
---


## haystack_integrations.components.extractors.presidio.presidio_entity_extractor

### PresidioEntityExtractor

Detects PII entities in Haystack Documents using Microsoft Presidio Analyzer.

See [Presidio Analyzer](https://microsoft.github.io/presidio/) for details.

Accepts a list of Documents and returns new Documents with detected PII entities stored
in each Document's metadata under the key `"entities"`. Each entry in the list contains
the entity type, start/end character offsets, and the confidence score.

Original Documents are not mutated. Documents without text content are passed through unchanged.

The analyzer engine is loaded on the first call to `run()`,
or by calling `warm_up()` explicitly beforehand.

### Usage example

```python
from haystack import Document
from haystack_integrations.components.extractors.presidio import PresidioEntityExtractor

extractor = PresidioEntityExtractor()
result = extractor.run(documents=[Document(content="Contact Alice at alice@example.com")])
print(result["documents"][0].meta["entities"])
# [{"entity_type": "PERSON", "start": 8, "end": 13, "score": 0.85},
#  {"entity_type": "EMAIL_ADDRESS", "start": 17, "end": 34, "score": 1.0}]
```

#### __init__

```python
__init__(
    *,
    language: str = "en",
    entities: list[str] | None = None,
    score_threshold: float = 0.35
) -> None
```

Initializes the PresidioEntityExtractor.

**Parameters:**

- **language** (<code>str</code>) – Language code for PII detection. Defaults to `"en"`.
  See [Presidio supported languages](https://microsoft.github.io/presidio/analyzer/languages/).
- **entities** (<code>list\[str\] | None</code>) – List of PII entity types to detect (e.g. `["PERSON", "EMAIL_ADDRESS"]`).
  If `None`, all supported entity types are detected.
  See [Presidio supported entities](https://microsoft.github.io/presidio/supported_entities/).
- **score_threshold** (<code>float</code>) – Minimum confidence score (0-1) for a detected entity to be included. Defaults to `0.35`.
  See [Presidio analyzer documentation](https://microsoft.github.io/presidio/analyzer/).

#### warm_up

```python
warm_up() -> None
```

Initializes the Presidio analyzer engine.

This method loads the underlying NLP models. In a Haystack Pipeline,
this is called automatically before the first run.

#### run

```python
run(documents: list[Document]) -> dict[str, list[Document]]
```

Detects PII entities in the provided Documents.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – List of Documents to analyze for PII entities.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with key `documents` containing Documents with detected entities
  stored in metadata under the key `"entities"`.

## haystack_integrations.components.preprocessors.presidio.presidio_document_cleaner

### PresidioDocumentCleaner

Anonymizes PII in Haystack Documents using [Microsoft Presidio](https://microsoft.github.io/presidio/).

Accepts a list of Documents, detects personally identifiable information (PII) in their
text content, and returns new Documents with PII replaced by entity type placeholders
(e.g. `<PERSON>`, `<EMAIL_ADDRESS>`). Original Documents are not mutated.

Documents without text content are passed through unchanged.

The analyzer and anonymizer engines are loaded on the first call to `run()`,
or by calling `warm_up()` explicitly beforehand.

### Usage example

```python
from haystack import Document
from haystack_integrations.components.preprocessors.presidio import PresidioDocumentCleaner

cleaner = PresidioDocumentCleaner()
result = cleaner.run(documents=[Document(content="My name is John and my email is john@example.com")])
print(result["documents"][0].content)
# My name is <PERSON> and my email is <EMAIL_ADDRESS>
```

#### __init__

```python
__init__(
    *,
    language: str = "en",
    entities: list[str] | None = None,
    score_threshold: float = 0.35
) -> None
```

Initializes the PresidioDocumentCleaner.

**Parameters:**

- **language** (<code>str</code>) – Language code for PII detection. Defaults to `"en"`.
  See [Presidio supported languages](https://microsoft.github.io/presidio/analyzer/languages/).
- **entities** (<code>list\[str\] | None</code>) – List of PII entity types to detect and anonymize (e.g. `["PERSON", "EMAIL_ADDRESS"]`).
  If `None`, all supported entity types are used.
  See [Presidio supported entities](https://microsoft.github.io/presidio/supported_entities/).
- **score_threshold** (<code>float</code>) – Minimum confidence score (0-1) for a detected entity to be anonymized. Defaults to `0.35`.
  See [Presidio analyzer documentation](https://microsoft.github.io/presidio/analyzer/).

#### warm_up

```python
warm_up() -> None
```

Initializes the Presidio analyzer and anonymizer engines.

This method loads the underlying NLP models. In a Haystack Pipeline,
this is called automatically before the first run.

#### run

```python
run(documents: list[Document]) -> dict[str, list[Document]]
```

Anonymizes PII in the provided Documents.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – List of Documents whose text content will be anonymized.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with key `documents` containing the cleaned Documents.

## haystack_integrations.components.preprocessors.presidio.presidio_text_cleaner

### PresidioTextCleaner

Anonymizes PII in plain strings using [Microsoft Presidio](https://microsoft.github.io/presidio/).

Accepts a list of strings, detects personally identifiable information (PII), and returns
a new list of strings with PII replaced by entity type placeholders (e.g. `<PERSON>`).
Useful for sanitizing user queries before they are sent to an LLM.

The analyzer and anonymizer engines are loaded on the first call to `run()`,
or by calling `warm_up()` explicitly beforehand.

### Usage example

```python
from haystack_integrations.components.preprocessors.presidio import PresidioTextCleaner

cleaner = PresidioTextCleaner()
result = cleaner.run(texts=["Hi, I am John Smith, call me at 212-555-1234"])
print(result["texts"][0])
# Hi, I am <PERSON>, call me at <PHONE_NUMBER>
```

#### __init__

```python
__init__(
    *,
    language: str = "en",
    entities: list[str] | None = None,
    score_threshold: float = 0.35
) -> None
```

Initializes the PresidioTextCleaner.

**Parameters:**

- **language** (<code>str</code>) – Language code for PII detection. Defaults to `"en"`.
  See [Presidio supported languages](https://microsoft.github.io/presidio/analyzer/languages/).
- **entities** (<code>list\[str\] | None</code>) – List of PII entity types to detect and anonymize (e.g. `["PERSON", "PHONE_NUMBER"]`).
  If `None`, all supported entity types are used.
  See [Presidio supported entities](https://microsoft.github.io/presidio/supported_entities/).
- **score_threshold** (<code>float</code>) – Minimum confidence score (0-1) for a detected entity to be anonymized. Defaults to `0.35`.
  See [Presidio analyzer documentation](https://microsoft.github.io/presidio/analyzer/).

#### warm_up

```python
warm_up() -> None
```

Initializes the Presidio analyzer and anonymizer engines.

This method loads the underlying NLP models. In a Haystack Pipeline,
this is called automatically before the first run.

#### run

```python
run(texts: list[str]) -> dict[str, list[str]]
```

Anonymizes PII in the provided strings.

**Parameters:**

- **texts** (<code>list\[str\]</code>) – List of strings to anonymize.

**Returns:**

- <code>dict\[str, list\[str\]\]</code> – A dictionary with key `texts` containing the cleaned strings.
