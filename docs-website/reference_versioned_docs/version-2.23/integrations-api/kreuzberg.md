---
title: "Kreuzberg"
id: integrations-kreuzberg
description: "Kreuzberg integration for Haystack"
slug: "/integrations-kreuzberg"
---


## haystack_integrations.components.converters.kreuzberg.converter

### KreuzbergConverter

Converts files to Documents using [Kreuzberg](https://docs.kreuzberg.dev/).

Kreuzberg is a document intelligence framework that extracts text from
PDFs, Office documents, images, and 75+ other formats. All processing
is performed locally with no external API calls.

**Usage Example:**

```python
from haystack_integrations.components.converters.kreuzberg import (
    KreuzbergConverter,
)

converter = KreuzbergConverter()
result = converter.run(sources=["document.pdf", "report.docx"])
documents = result["documents"]
```

You can also pass kreuzberg's `ExtractionConfig` to customize extraction:

```python
from kreuzberg import ExtractionConfig, OcrConfig

converter = KreuzbergConverter(
    config=ExtractionConfig(
        output_format="markdown",
        ocr=OcrConfig(backend="tesseract", language="eng"),
    ),
)
```

**Token reduction** can be configured via
`ExtractionConfig(token_reduction=TokenReductionConfig(mode="moderate"))`
to reduce output size for LLM consumption. Five levels are available:
`"off"`, `"light"`, `"moderate"`, `"aggressive"`, `"maximum"`.
The reduced text appears directly in `Document.content`.

**Image preprocessing for OCR** can be tuned via
`OcrConfig(tesseract_config=TesseractConfig(preprocessing=ImagePreprocessingConfig(...)))`
with options for target DPI, auto-rotate, deskew, denoise,
contrast enhancement, and binarization method.

#### __init__

```python
__init__(
    *,
    config: ExtractionConfig | None = None,
    config_path: str | Path | None = None,
    store_full_path: bool = False,
    batch: bool = True,
    easyocr_kwargs: dict[str, Any] | None = None
) -> None
```

Create a `KreuzbergConverter` component.

**Parameters:**

- **config** (<code>ExtractionConfig | None</code>) – An optional `kreuzberg.ExtractionConfig` object to customize
  extraction behavior. Use this to set output format, OCR backend
  and language, force-OCR mode, per-page extraction, chunking,
  keyword extraction, and other kreuzberg options. If not provided,
  kreuzberg's defaults are used.
  See the [kreuzberg API reference](https://docs.kreuzberg.dev/reference/api-python/)
  for the full list of configuration options.
- **config_path** (<code>str | Path | None</code>) – Path to a kreuzberg configuration file (`.toml`, `.yaml`, or
  `.json`). Cannot be used together with `config`.
- **store_full_path** (<code>bool</code>) – If `True`, the full file path is stored in the Document metadata.
  If `False`, only the file name is stored.
- **batch** (<code>bool</code>) – If `True`, use kreuzberg's batch extraction APIs, which leverage
  Rust's rayon thread pool for parallel processing. If `False`,
  sources are extracted one at a time.
- **easyocr_kwargs** (<code>dict\[str, Any\] | None</code>) – Optional keyword arguments to pass to EasyOCR when using the
  `"easyocr"` backend. Supports GPU, beam width, model storage,
  and other EasyOCR-specific options.
  See the [EasyOCR documentation](https://www.jaided.ai/easyocr/documentation/)
  for the full list of supported arguments.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize this component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> KreuzbergConverter
```

Deserialize this component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>KreuzbergConverter</code> – Deserialized component.

#### run

```python
run(
    sources: list[str | Path | ByteStream],
    meta: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> dict[str, list[Document]]
```

Convert files to Documents using Kreuzberg.

**Parameters:**

- **sources** (<code>list\[str | Path | ByteStream\]</code>) – List of file paths, directory paths, or ByteStream objects to
  convert. Directory paths are expanded to their direct file children
  (non-recursive, sorted alphabetically).
- **meta** (<code>dict\[str, Any\] | list\[dict\[str, Any\]\] | None</code>) – Optional metadata to attach to the Documents.
  This value can be either a list of dictionaries or a single
  dictionary. If it's a single dictionary, its content is added to
  the metadata of all produced Documents. If it's a list, the length
  of the list must match the number of sources, because the two
  lists will be zipped. If `sources` contains ByteStream objects,
  their `meta` will be added to the output Documents.

**Note:** When directories are present in `sources`, `meta` must
be a single dictionary (not a list), since the number of files in
a directory is not known in advance.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following key:

- `documents`: A list of created Documents.
