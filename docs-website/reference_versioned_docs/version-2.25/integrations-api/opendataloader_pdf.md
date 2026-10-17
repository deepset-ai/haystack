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
    image_output_dir: str | Path | None = None
) -> None
```

Initialize the OpenDataLoader converter.

**Parameters:**

- **output_format** (<code>OutputFormat</code>) – Format OpenDataLoader should produce.
- **convert_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional arguments passed to `opendataloader_pdf.convert`. See the
  [OpenDataLoader PDF Python options](https://opendataloader.org/docs/quick-start-python#convert-options).
  The `image_output` and `image_dir` arguments are managed by this component; supplied values are ignored.
- **extract_images** (<code>bool</code>) – Whether to extract images and return them through the `image_documents` output.
- **image_output_dir** (<code>str | Path | None</code>) – Persistent root directory for extracted image files. Each `run()` stores its images
  in a unique subdirectory of this directory. Required when `extract_images` is `True`.

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
  persistent extracted image path in its `file_path` metadata field.
