---
title: "Docling"
id: integrations-docling
description: "Docling integration for Haystack"
slug: "/integrations-docling"
---


## haystack_integrations.components.converters.docling.converter

Docling Haystack converter module.

### ExportType

Bases: <code>str</code>, <code>Enum</code>

Enumeration of available export types.

### BaseMetaExtractor

Bases: <code>ABC</code>

BaseMetaExtractor.

#### extract_chunk_meta

```python
extract_chunk_meta(chunk: BaseChunk) -> dict[str, Any]
```

Extract chunk meta.

#### extract_dl_doc_meta

```python
extract_dl_doc_meta(dl_doc: DoclingDocument) -> dict[str, Any]
```

Extract Docling document meta.

### MetaExtractor

Bases: <code>BaseMetaExtractor</code>

MetaExtractor.

#### extract_chunk_meta

```python
extract_chunk_meta(chunk: BaseChunk) -> dict[str, Any]
```

Extract chunk meta.

#### extract_dl_doc_meta

```python
extract_dl_doc_meta(dl_doc: DoclingDocument) -> dict[str, Any]
```

Extract Docling document meta.

### DoclingConverter

Docling Haystack converter.

#### __init__

```python
__init__(
    converter: DocumentConverter | None = None,
    convert_kwargs: dict[str, Any] | None = None,
    export_type: ExportType = ExportType.DOC_CHUNKS,
    md_export_kwargs: dict[str, Any] | None = None,
    chunker: BaseChunker | None = None,
    meta_extractor: BaseMetaExtractor | None = None,
) -> None
```

Create a Docling Haystack converter.

**Parameters:**

- **converter** (<code>DocumentConverter | None</code>) – The Docling `DocumentConverter` to use; if not set, a system
  default is used.
- **convert_kwargs** (<code>dict\[str, Any\] | None</code>) – Any parameters to pass to Docling conversion; if not set, a
  system default is used.
- **export_type** (<code>ExportType</code>) – The export mode to use:

* `ExportType.MARKDOWN` captures each input document as a single
  markdown `Document`.
* `ExportType.DOC_CHUNKS` (default) first chunks each input document
  and then returns one `Document` per chunk.
* `ExportType.JSON` serializes the full Docling document to a JSON string.

- **md_export_kwargs** (<code>dict\[str, Any\] | None</code>) – Any parameters to pass to Markdown export (applicable in
  case of `ExportType.MARKDOWN`).
- **chunker** (<code>BaseChunker | None</code>) – The Docling chunker instance to use; if not set, a system default
  is used.
- **meta_extractor** (<code>BaseMetaExtractor | None</code>) – The extractor instance to use for populating the output
  document metadata; if not set, a system default is used.

#### run

```python
run(
    paths: list[str | Path] | None = None,
    sources: list[str | Path | ByteStream] | None = None,
    meta: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> dict[str, list[Document]]
```

Run the DoclingConverter.

**Parameters:**

- **paths** (<code>list\[str | Path\] | None</code>) – Deprecated. Use `sources` instead.
- **sources** (<code>list\[str | Path | ByteStream\] | None</code>) – List of file paths, URLs, or ByteStream objects to convert.
- **meta** (<code>dict\[str, Any\] | list\[dict\[str, Any\]\] | None</code>) – Optional metadata to attach to the Documents.
  This value can be either a list of dictionaries or a single dictionary.
  If it's a single dictionary, its content is added to the metadata of all produced Documents.
  If it's a list, the length of the list must match the number of sources, because the two lists will
  be zipped.
  If a source is a ByteStream, its own metadata is also merged into the output.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with key `"documents"` containing the output Haystack Documents.

**Raises:**

- <code>ValueError</code> – If `meta` is a list whose length does not match the number of sources.
- <code>RuntimeError</code> – If an unexpected `export_type` is encountered.
