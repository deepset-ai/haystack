---
title: "LibreOffice"
id: integrations-libreoffice
description: "LibreOffice integration for Haystack"
slug: "/integrations-libreoffice"
---


## haystack_integrations.components.converters.libreoffice.converter

### LibreOfficeFileConverter

Component that uses libreoffice's command line utility (soffice) to convert files into various formats.

### Usage examples

**Simple conversion:**

```python
from pathlib import Path

from haystack_integrations.components.converters.libreoffice import LibreOfficeFileConverter

# Convert documents
converter = LibreOfficeFileConverter()
results = converter.run(sources=[Path("sample.doc")], output_file_type="docx")
print(results["output"])  # [ByteStream(data=b'...', meta={}, mime_type=None)]
```

**Conversion pipeline:**

```python
from pathlib import Path

from haystack import Pipeline
from haystack.components.converters import DOCXToDocument

from haystack_integrations.components.converters.libreoffice import LibreOfficeFileConverter

# Create pipeline with components
pipeline = Pipeline()
pipeline.add_component("libreoffice_converter", LibreOfficeFileConverter())
pipeline.add_component("docx_converter", DOCXToDocument())

pipeline.connect("libreoffice_converter.output", "docx_converter.sources")

# Run pipeline and convert legacy documents into Haystack documents
results = pipeline.run(
    {
        "libreoffice_converter": {
            "sources": [Path("sample_doc.doc")],
            "output_file_type": "docx",
        }
    }
)
print(results["docx_converter"]["documents"])
```

#### SUPPORTED_TYPES

```python
SUPPORTED_TYPES: dict[str, frozenset[str]] = {
    "doc": frozenset(["pdf", "docx", "odt", "rtf", "txt", "html", "epub"]),
    "docx": frozenset(["pdf", "doc", "odt", "rtf", "txt", "html", "epub"]),
    "odt": frozenset(["pdf", "docx", "doc", "rtf", "txt", "html", "epub"]),
    "rtf": frozenset(["pdf", "docx", "doc", "odt", "txt", "html"]),
    "txt": frozenset(["pdf", "docx", "doc", "odt", "rtf", "html"]),
    "html": frozenset(["pdf", "docx", "doc", "odt", "rtf", "txt"]),
    "xlsx": frozenset(["pdf", "xls", "ods", "csv", "html"]),
    "xls": frozenset(["pdf", "xlsx", "ods", "csv", "html"]),
    "ods": frozenset(["pdf", "xlsx", "xls", "csv", "html"]),
    "csv": frozenset(["pdf", "xlsx", "xls", "ods"]),
    "pptx": frozenset(["pdf", "ppt", "odp", "html", "png", "jpg"]),
    "ppt": frozenset(["pdf", "pptx", "odp", "html", "png", "jpg"]),
    "odp": frozenset(["pdf", "pptx", "ppt", "html", "png", "jpg"]),
}

```

A non-exhaustive mapping of supported conversion types by this component.
See https://help.libreoffice.org/latest/en-GB/text/shared/guide/convertfilters.html for more information.

#### __init__

```python
__init__(output_file_type: OUTPUT_FILE_TYPE | None = None) -> None
```

Check whether soffice is installed.

**Parameters:**

- **output_file_type** (<code>OUTPUT_FILE_TYPE | None</code>) – Target file format to convert to. Must be a valid conversion target for
  each source's input type — see :attr:`SUPPORTED_TYPES` for the full mapping.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> Self
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to deserialize from.

**Returns:**

- <code>Self</code> – The deserialized component.

#### run

```python
run(
    sources: Iterable[str | Path | ByteStream],
    output_file_type: OUTPUT_FILE_TYPE | None = None,
) -> LibreOfficeFileConverterOutput
```

Convert office files to the specified output format using LibreOffice.

**Parameters:**

- **sources** (<code>Iterable\[str | Path | ByteStream\]</code>) – List of sources to convert. Each source can be a file path (`str` or
  `Path`) or a `ByteStream`. For `ByteStream` sources, the input file
  type cannot be inferred from the filename, so only `output_file_type` is
  validated (not the source type).
- **output_file_type** (<code>OUTPUT_FILE_TYPE | None</code>) – Target file format to convert to. Must be a valid conversion target for
  each source's input type — see :attr:`SUPPORTED_TYPES` for the full mapping.
  If set, it will override the `output_file_type` parameter provided during initialization.

**Returns:**

- <code>LibreOfficeFileConverterOutput</code> – A dictionary with the following key:
- `output`: List of `ByteStream` objects containing the converted file
  data, in the same order as `sources`.

**Raises:**

- <code>FileNotFoundError</code> – If a source file path does not exist.
- <code>OSError</code> – If the internal temporary output directory is not writable.
- <code>ValueError</code> – If a source's file type is not in :attr:`SUPPORTED_TYPES`,
  or if `output_file_type` is not a valid conversion target for it,
  or if `output_file_type` has not been provided anywhere.

#### run_async

```python
run_async(
    sources: Iterable[str | Path | ByteStream],
    output_file_type: OUTPUT_FILE_TYPE | None = None,
) -> LibreOfficeFileConverterOutput
```

Asynchronously convert office files to the specified output format using LibreOffice.

This is the asynchronous version of the `run` method with the same parameters and return values.

**Parameters:**

- **sources** (<code>Iterable\[str | Path | ByteStream\]</code>) – List of sources to convert. Each source can be a file path (`str` or
  `Path`) or a `ByteStream`. For `ByteStream` sources, the input file
  type cannot be inferred from the filename, so only `output_file_type` is
  validated (not the source type).
- **output_file_type** (<code>OUTPUT_FILE_TYPE | None</code>) – Target file format to convert to. Must be a valid conversion target for
  each source's input type — see :attr:`SUPPORTED_TYPES` for the full mapping.
  If set, it will override the `output_file_type` parameter provided during initialization.

**Returns:**

- <code>LibreOfficeFileConverterOutput</code> – A dictionary with the following key:
- `output`: List of `ByteStream` objects containing the converted file
  data, in the same order as `sources`.

**Raises:**

- <code>FileNotFoundError</code> – If a source file path does not exist.
- <code>OSError</code> – If the internal temporary output directory is not writable.
- <code>ValueError</code> – If a source's file type is not in :attr:`SUPPORTED_TYPES`,
  or if `output_file_type` is not a valid conversion target for it,
  or if `output_file_type` has not been provided anywhere.
