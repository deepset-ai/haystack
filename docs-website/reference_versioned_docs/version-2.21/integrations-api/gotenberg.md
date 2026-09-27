---
title: "Gotenberg"
id: integrations-gotenberg
description: "Gotenberg integration for Haystack"
slug: "/integrations-gotenberg"
---


## haystack_integrations.components.converters.gotenberg.converter

### GotenbergFileConverter

Automatically route local files, typed byte streams, and web URLs to Gotenberg and return ordered PDFs.

Local Markdown files use Gotenberg's Markdown route, local HTML files use its Chromium HTML route, and every
other supported local file uses its LibreOffice route. HTTP(S) strings use the Chromium URL route. `ByteStream`
sources are routed by their MIME type. Resources are validated for every batch but uploaded only with HTML and
Markdown sources. Output metadata preserves local source paths under `file_path` and metadata from `ByteStream`
sources; explicit metadata takes precedence.

### Usage example

```python
from pathlib import Path

from haystack_integrations.components.converters.gotenberg import GotenbergFileConverter

converter = GotenbergFileConverter()
result = converter.run(sources=[Path("report.docx"), "https://haystack.deepset.ai"])
pdfs = result["output"]
```

#### __init__

```python
__init__(
    url: str = "http://localhost:3000",
    timeout: float = 30.0,
    concurrency_limit: int = 5,
) -> None
```

Create a Gotenberg file converter.

**Parameters:**

- **url** (<code>str</code>) – The URL of the Gotenberg service.
- **timeout** (<code>float</code>) – The request timeout in seconds.
- **concurrency_limit** (<code>int</code>) – Maximum number of Gotenberg requests in flight during `run_async`. Has no
  effect on synchronous `run`, which converts one source at a time.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize this component to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> Self
```

Deserialize this component from a dictionary.

#### run

```python
run(
    sources: list[str | Path | ByteStream],
    meta: dict[str, Any] | list[dict[str, Any]] | None = None,
    *,
    resources: list[Path] | None = None
) -> dict[str, list[ByteStream]]
```

Convert automatically routed sources to PDF.

Each source is classified independently and converted using the corresponding Gotenberg route:

- A string containing `://` is treated as a URL. It must be an HTTP(S) URL, and Gotenberg's Chromium URL
  route navigates to that URL and prints the resulting page to PDF.
- A string without `://`, or a `Path`, is treated as a local file. Markdown files (`.md` and `.markdown`) use
  the Chromium Markdown route, HTML files (`.html`, `.htm`, and `.xhtml`) use the Chromium HTML route, and
  other supported file extensions use the LibreOffice route.
- A `ByteStream` is classified by its MIME type. HTML MIME types use the Chromium HTML route, Markdown MIME
  types use the Chromium Markdown route, and other supported MIME types use the LibreOffice route. For
  LibreOffice conversion, the MIME type determines the staged file extension. MIME parameters such as
  `charset=utf-8` are ignored when classifying the stream.

HTML and Markdown inputs are uploaded to Gotenberg together with `resources`, while URL inputs are fetched by
Gotenberg and LibreOffice inputs are uploaded as files. A mixed batch can contain any combination of these
source types. Routes are executed in input order, and one PDF is produced for each source. HTML and Markdown
`ByteStream` content must be UTF-8 text.

**Parameters:**

- **sources** (<code>list\[str | Path | ByteStream\]</code>) – Sources to convert. Strings, `Path` objects, and `ByteStream` objects are supported as
  described above.
- **meta** (<code>dict\[str, Any\] | list\[dict\[str, Any\]\] | None</code>) – Optional metadata to attach to the output PDFs. A single dictionary is applied to every output. A
  list of dictionaries must have the same length as `sources` and is applied to corresponding outputs.
  Metadata on a source `ByteStream` is preserved, with values from this parameter taking precedence.
- **resources** (<code>list\[Path\] | None</code>) – Optional local resources for HTML and Markdown conversion. Resources are validated once and
  uploaded only with those routes. Filenames must be unique and cannot be `index.html`; a resource also
  cannot have the same filename as a staged Markdown source.

**Returns:**

- <code>dict\[str, list\[ByteStream\]\]</code> – A dictionary containing an `"output"` list with one PDF `ByteStream` per source, in input order.

**Raises:**

- <code>TypeError</code> – If a source or resource has an unsupported type.
- <code>FileNotFoundError</code> – If a local source or resource does not exist or is not a file.
- <code>ValueError</code> – If sources, metadata, URLs, suffixes, MIME types, resources, or text contents are invalid.
- <code>RuntimeError</code> – If Gotenberg returns a ZIP archive instead of a PDF.

#### run_async

```python
run_async(
    sources: list[str | Path | ByteStream],
    meta: dict[str, Any] | list[dict[str, Any]] | None = None,
    *,
    resources: list[Path] | None = None
) -> dict[str, list[ByteStream]]
```

Asynchronously convert automatically routed sources to PDF.

This is the asynchronous equivalent of `run()` and uses the same source classification and routing rules. Each
source is classified independently and converted using the corresponding Gotenberg route:

- A string containing `://` is treated as a valid HTTP(S) URL and converted by Gotenberg's Chromium URL route.
- A string without `://`, or a `Path`, is treated as a local file. Markdown files (`.md` and `.markdown`) use
  the Chromium Markdown route, HTML files (`.html`, `.htm`, and `.xhtml`) use the Chromium HTML route, and
  other supported file extensions use the LibreOffice route.
- A `ByteStream` is classified by its MIME type: HTML MIME types use the Chromium HTML route, Markdown MIME
  types use the Chromium Markdown route, and other supported MIME types use the LibreOffice route. The MIME
  type determines the staged file extension for LibreOffice conversion, without its parameters (for example,
  `charset=utf-8`).

HTML and Markdown inputs are uploaded with `resources`, URL inputs are fetched by Gotenberg, and LibreOffice
inputs are uploaded as files. A mixed batch can contain any combination of these source types. Routes are
executed in input order, and one PDF is produced for each source. HTML and Markdown `ByteStream` content must
be UTF-8 text.

**Parameters:**

- **sources** (<code>list\[str | Path | ByteStream\]</code>) – Sources to convert. Strings, `Path` objects, and `ByteStream` objects are supported as
  described above.
- **meta** (<code>dict\[str, Any\] | list\[dict\[str, Any\]\] | None</code>) – Optional metadata to attach to the output PDFs. A single dictionary is applied to every output. A
  list of dictionaries must have the same length as `sources` and is applied to corresponding outputs.
  Metadata on a source `ByteStream` is preserved, with values from this parameter taking precedence.
- **resources** (<code>list\[Path\] | None</code>) – Optional local resources for HTML and Markdown conversion. Resources are validated once and
  uploaded only with those routes. Filenames must be unique and cannot be `index.html`; a resource also
  cannot have the same filename as a staged Markdown source.

**Returns:**

- <code>dict\[str, list\[ByteStream\]\]</code> – A dictionary containing an `"output"` list with one PDF `ByteStream` per source, in input order.

**Raises:**

- <code>TypeError</code> – If a source or resource has an unsupported type.
- <code>FileNotFoundError</code> – If a local source or resource does not exist or is not a file.
- <code>ValueError</code> – If sources, metadata, URLs, suffixes, MIME types, resources, or text contents are invalid.
- <code>RuntimeError</code> – If Gotenberg returns a ZIP archive instead of a PDF.
