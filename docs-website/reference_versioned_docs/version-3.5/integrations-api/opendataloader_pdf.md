---
title: "Opendataloader Pdf"
id: integrations-opendataloader-pdf
description: "Opendataloader Pdf integration for Haystack"
slug: "/integrations-opendataloader-pdf"
---


## haystack_integrations.components.converters.opendataloader_pdf.converter

### OpenDataLoaderConverter

OpenDataLoader PDF converter component.

The component accepts PDF file paths and Haystack ByteStream objects, runs OpenDataLoader PDF extraction, and
returns Haystack Document objects. It can also extract images to a persistent directory and return one image
Document per extracted file.

With `output_format="json"`, the Document content is Markdown with pages separated by form feeds (`\f`), so
`DocumentSplitter` can compute page numbers, and the PDF metadata from the OpenDataLoader JSON output is added to
`Document.meta`. Set `raw_json=True` to get the raw JSON string as content instead.

Java 11 or newer must be installed and available on PATH.

### Usage example

```python
from haystack_integrations.components.converters.opendataloader_pdf import OpenDataLoaderConverter

converter = OpenDataLoaderConverter(
    output_format="markdown", extract_images=True, image_output_dir="extracted_images"
)
result = converter.run(sources=["report.pdf"], meta={"source": "annual-report"})

documents = result["documents"]
image_documents = result["image_documents"]
print(documents[0].content)
print(documents[0].meta["file_path"])
```

#### __init__

```python
__init__(
    *,
    output_format: OutputFormat = "markdown",
    convert_kwargs: dict[str, Any] | None = None,
    extract_images: bool = False,
    image_output_dir: str | Path | None = None,
    skip_failed_pdfs: bool = False,
    raw_json: bool = False,
    include_layout: bool = False
) -> None
```

Initialize the OpenDataLoader converter.

**Parameters:**

- **output_format** (<code>OutputFormat</code>) – Format OpenDataLoader should produce. With `"json"`, the Document content is Markdown
  with pages separated by `\f`, and `Document.meta` gets the PDF's `title`, `author`, `creation_date`,
  `modification_date` (ISO 8601 when the PDF date can be parsed, otherwise as found in the PDF), and
  `page_count`. Fields missing from the PDF are omitted. User-provided metadata takes precedence on key
  conflicts.
- **convert_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional arguments passed to `opendataloader_pdf.convert`. See the
  [OpenDataLoader PDF Python options](https://opendataloader.org/docs/quick-start-python#convert-options).
  The `image_output` and `image_dir` arguments are managed by this component; supplied values are ignored.
  With `output_format="json"`, passing `markdown_page_separator` replaces the default `\f` page separator.
- **extract_images** (<code>bool</code>) – Whether to extract images and return them through the `image_documents` output.
- **image_output_dir** (<code>str | Path | None</code>) – Persistent root directory for extracted image files. Each `run()` stores its images
  in a unique subdirectory of this directory. Required when `extract_images` is `True`.
- **skip_failed_pdfs** (<code>bool</code>) – Re-run failing batches in halves until bad PDFs are identified and skipped with a
  warning. Each bad PDF adds extra Java runs.
- **raw_json** (<code>bool</code>) – If `True` with `output_format="json"`, store the raw OpenDataLoader JSON string in
  `Document.content` and add no PDF metadata. Has no effect with other output formats.
- **include_layout** (<code>bool</code>) – If `True` with `output_format="json"`, add a `layout` list to `Document.meta` with one
  entry per positioned element in reading order: `type`, `page_number`, `bounding_box`, and
  `heading_level` for headings. The list covers the whole PDF and is copied to every chunk when the Document
  is split, so it can be large for long PDFs. Has no effect with other output formats or with `raw_json`.

**Raises:**

- <code>ValueError</code> – If image extraction is enabled without an output directory.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the component.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary representation of the converter.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> OpenDataLoaderConverter
```

Deserialize the component.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Serialized component dictionary.

**Returns:**

- <code>OpenDataLoaderConverter</code> – Reconstructed OpenDataLoaderConverter.

#### run

```python
run(
    sources: list[str | Path | ByteStream],
    meta: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> dict[str, list[Document]]
```

Convert PDF sources into Haystack Documents.

**Parameters:**

- **sources** (<code>list\[str | Path | ByteStream\]</code>) – PDF file paths or Haystack ByteStream objects.
- **meta** (<code>dict\[str, Any\] | list\[dict\[str, Any\]\] | None</code>) – Optional metadata attached to the generated Documents. A single dictionary is applied to every
  source. A list must contain one dictionary per source. ByteStream metadata is also preserved.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – Dictionary containing the converted text Documents and image Documents. Each image Document has the
  persistent extracted image path in its `file_path` metadata field. With `skip_failed_pdfs=True`, PDFs
  that fail individual conversion are omitted from both outputs.
