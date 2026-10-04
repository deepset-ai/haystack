---
title: "Markitdown"
id: integrations-markitdown
description: "Markitdown integration for Haystack"
slug: "/integrations-markitdown"
---


## haystack_integrations.components.converters.markitdown.markitdown_converter

### MarkItDownConverter

Converts files to Haystack Documents using [MarkItDown](https://github.com/microsoft/markitdown).

MarkItDown is a Microsoft library that converts many file formats to Markdown,
including PDF, Word (.docx), PowerPoint (.pptx), Excel (.xlsx), HTML, images,
audio, and more. All processing is performed locally.

### Usage example

```python
from haystack_integrations.components.converters.markitdown import MarkItDownConverter

converter = MarkItDownConverter()
result = converter.run(sources=["document.pdf", "report.docx"])
documents = result["documents"]
```

#### __init__

```python
__init__(store_full_path: bool = False) -> None
```

Initializes the MarkItDownConverter.

**Parameters:**

- **store_full_path** (<code>bool</code>) – If `True`, the full file path is stored in the Document metadata.
  If `False`, only the file name is stored. Defaults to `False`.

#### run

```python
run(
    sources: list[str | Path | ByteStream],
    meta: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> dict[str, list[Document]]
```

Converts files to Documents using MarkItDown.

**Parameters:**

- **sources** (<code>list\[str | Path | ByteStream\]</code>) – List of file paths or ByteStream objects to convert.
- **meta** (<code>dict\[str, Any\] | list\[dict\[str, Any\]\] | None</code>) – Optional metadata to attach to the Documents. Can be a single dict
  applied to all Documents, or a list of dicts aligned with `sources`.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with key `documents` containing the converted Documents.
