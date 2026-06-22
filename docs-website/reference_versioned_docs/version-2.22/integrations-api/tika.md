---
title: "Tika"
id: integrations-tika
description: "Tika integration for Haystack"
slug: "/integrations-tika"
---


## haystack_integrations.components.converters.tika.converter

### XHTMLParser

Bases: <code>HTMLParser</code>

Custom parser to extract pages from Tika XHTML content.

#### __init__

```python
__init__() -> None
```

Initialize the XHTMLParser.

#### handle_starttag

```python
handle_starttag(tag: str, attrs: list[tuple[str, str | None]]) -> None
```

Identify the start of a page div.

**Parameters:**

- **tag** (<code>str</code>) – The HTML tag name.
- **attrs** (<code>list\[tuple\[str, str | None\]\]</code>) – The HTML tag attributes.

#### handle_endtag

```python
handle_endtag(tag: str) -> None
```

Identify the end of a page div.

**Parameters:**

- **tag** (<code>str</code>) – The HTML tag name.

#### handle_data

```python
handle_data(data: str) -> None
```

Populate the page content.

**Parameters:**

- **data** (<code>str</code>) – The text content of an HTML node.

### TikaDocumentConverter

Converts files of different types to Documents using Apache Tika.

This component uses [Apache Tika](https://tika.apache.org/) for parsing the files and, therefore,
requires a running Tika server.
For more options on running Tika,
see the [official documentation](https://github.com/apache/tika-docker/blob/main/README.md#usage).

Usage example:

```python
from haystack_integrations.components.converters.tika import TikaDocumentConverter
from datetime import datetime

converter = TikaDocumentConverter()
results = converter.run(
    sources=["sample.docx", "my_document.rtf", "archive.zip"],
    meta={"date_added": datetime.now().isoformat()}
)
documents = results["documents"]

print(documents[0].content)
# >> 'This is a text from the docx file.'
```

#### __init__

```python
__init__(
    tika_url: str = "http://localhost:9998/tika", store_full_path: bool = False
) -> None
```

Create a TikaDocumentConverter component.

**Parameters:**

- **tika_url** (<code>str</code>) – Tika server URL.
- **store_full_path** (<code>bool</code>) – If True, the full path of the file is stored in the metadata of the document.
  If False, only the file name is stored.

#### run

```python
run(
    sources: list[str | Path | ByteStream],
    meta: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> dict[str, list[Document]]
```

Convert files to Documents.

**Parameters:**

- **sources** (<code>list\[str | Path | ByteStream\]</code>) – List of file paths or ByteStream objects.
- **meta** (<code>dict\[str, Any\] | list\[dict\[str, Any\]\] | None</code>) – Optional metadata to attach to the Documents.
  This value can be either a list of dictionaries or a single dictionary.
  If it's a single dictionary, its content is added to the metadata of all produced Documents.
  If it's a list, the length of the list must match the number of sources, because the two lists will
  be zipped.
  If `sources` contains ByteStream objects, their `meta` will be added to the output Documents.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: Created Documents
