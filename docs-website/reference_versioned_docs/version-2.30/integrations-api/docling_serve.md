---
title: "Docling Serve"
id: integrations-docling_serve
description: "Docling Serve integration for Haystack"
slug: "/integrations-docling_serve"
---


## haystack_integrations.components.converters.docling_serve.converter

### ExportType

Bases: <code>str</code>, <code>Enum</code>

Enumeration of export formats supported by DoclingServe.

- `MARKDOWN`: Converts documents to Markdown format.
- `TEXT`: Extracts plain text.
- `JSON`: Returns the full Docling document as a JSON string.

### DoclingServeConverter

Converts documents to Haystack Documents using a DoclingServe server.

See [DoclingServe](https://github.com/docling-project/docling-serve).

DoclingServe hosts Docling in a scalable HTTP server, supporting PDFs, Office documents, HTML, and many other
formats. Unlike the local `DoclingConverter`, this component has no heavy ML dependencies — all processing
happens on the remote server.

Local files and ByteStreams are uploaded via the `/v1/convert/file` endpoint. URL strings are sent to
`/v1/convert/source`.

Supports both synchronous (`run`) and asynchronous (`run_async`) execution.

### Usage example

```python
from haystack_integrations.components.converters.docling_serve import DoclingServeConverter

converter = DoclingServeConverter(base_url="http://localhost:5001")
result = converter.run(sources=["https://arxiv.org/pdf/2206.01062"])
print(result["documents"][0].content[:200])
```

#### __init__

```python
__init__(
    *,
    base_url: str = "http://localhost:5001",
    export_type: ExportType = ExportType.MARKDOWN,
    convert_options: dict[str, Any] | None = None,
    timeout: float = 120.0,
    api_key: Secret | None = Secret.from_env_var(
        "DOCLING_SERVE_API_KEY", strict=False
    )
) -> None
```

Initializes the DoclingServeConverter.

**Parameters:**

- **base_url** (<code>str</code>) – Base URL of the DoclingServe instance. Defaults to `"http://localhost:5001"`.
- **export_type** (<code>ExportType</code>) – The output format for converted documents. One of `ExportType.MARKDOWN` (default),
  `ExportType.TEXT`, or `ExportType.JSON`.
- **convert_options** (<code>dict\[str, Any\] | None</code>) – Optional dictionary of conversion options passed directly to the DoclingServe API
  (e.g. `{"do_ocr": True, "ocr_engine": "tesseract"}`).
  See [DoclingServe options](https://github.com/docling-project/docling-serve/blob/main/docs/usage.md).
  Note: `to_formats` is set automatically based on `export_type` and should not be included here.
- **timeout** (<code>float</code>) – HTTP request timeout in seconds. Defaults to `120.0`.
- **api_key** (<code>Secret | None</code>) – API key for authenticating with a secured DoclingServe instance. Reads from the
  `DOCLING_SERVE_API_KEY` environment variable by default. Set to `None` to disable
  authentication.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary representation of the component.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> DoclingServeConverter
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary representation of the component.

**Returns:**

- <code>DoclingServeConverter</code> – A new `DoclingServeConverter` instance.

#### run

```python
run(
    sources: list[str | Path | ByteStream],
    meta: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> dict[str, list[Document]]
```

Converts documents by sending them to DoclingServe and returns Haystack Documents.

**Parameters:**

- **sources** (<code>list\[str | Path | ByteStream\]</code>) – List of sources to convert. Each item can be a URL string, a local file path, or a
  `ByteStream`. URL strings are sent to `/v1/convert/source`; all other sources are
  uploaded to `/v1/convert/file`.
- **meta** (<code>dict\[str, Any\] | list\[dict\[str, Any\]\] | None</code>) – Optional metadata to attach to the output Documents. Can be a single dict applied to
  all documents, or a list of dicts with one entry per source.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with key `"documents"` containing the converted Haystack Documents.

#### run_async

```python
run_async(
    sources: list[str | Path | ByteStream],
    meta: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> dict[str, list[Document]]
```

Asynchronously converts documents by sending them to DoclingServe.

This is the async equivalent of `run()`, useful when DoclingServe requests should not
block the event loop.

**Parameters:**

- **sources** (<code>list\[str | Path | ByteStream\]</code>) – List of sources to convert. Each item can be a URL string, a local file path, or a
  `ByteStream`. URL strings are sent to `/v1/convert/source`; all other sources are
  uploaded to `/v1/convert/file`.
- **meta** (<code>dict\[str, Any\] | list\[dict\[str, Any\]\] | None</code>) – Optional metadata to attach to the output Documents.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with key `"documents"` containing the converted Haystack Documents.
