---
title: "Azure Form Recognizer"
id: integrations-azure_form_recognizer
description: "Azure Form Recognizer integration for Haystack"
slug: "/integrations-azure_form_recognizer"
---


## haystack_integrations.components.converters.azure_form_recognizer.converter

### AzureOCRDocumentConverter

Converts files to documents using Azure's Document Intelligence service.

Supported file formats are: PDF, JPEG, PNG, BMP, TIFF, DOCX, XLSX, PPTX, and HTML.

To use this component, you need an active Azure account
and a Document Intelligence or Cognitive Services resource. For help with setting up your resource, see
[Azure documentation](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/quickstarts/get-started-sdks-rest-api).

### Usage example

```python
import os
from datetime import datetime
from haystack_integrations.components.converters.azure_form_recognizer import AzureOCRDocumentConverter
from haystack.utils import Secret

converter = AzureOCRDocumentConverter(
    endpoint=os.environ["CORE_AZURE_CS_ENDPOINT"],
    api_key=Secret.from_env_var("CORE_AZURE_CS_API_KEY"),
)
results = converter.run(
    sources=["test/test_files/pdf/react_paper.pdf"],
    meta={"date_added": datetime.now().isoformat()},
)
documents = results["documents"]
print(documents[0].content)
# 'This is a text from the PDF file.'
```

#### __init__

```python
__init__(
    endpoint: str,
    api_key: Secret = Secret.from_env_var("AZURE_AI_API_KEY"),
    model_id: str = "prebuilt-read",
    preceding_context_len: int = 3,
    following_context_len: int = 3,
    merge_multiple_column_headers: bool = True,
    page_layout: Literal["natural", "single_column"] = "natural",
    threshold_y: float | None = 0.05,
    store_full_path: bool = False,
) -> None
```

Creates an AzureOCRDocumentConverter component.

**Parameters:**

- **endpoint** (<code>str</code>) – The endpoint of your Azure resource.
- **api_key** (<code>Secret</code>) – The API key of your Azure resource.
- **model_id** (<code>str</code>) – The ID of the model you want to use. For a list of available models, see [Azure documentation]
  (https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/choose-model-feature).
- **preceding_context_len** (<code>int</code>) – Number of lines before a table to include as preceding context
  (this will be added to the metadata).
- **following_context_len** (<code>int</code>) – Number of lines after a table to include as subsequent context (
  this will be added to the metadata).
- **merge_multiple_column_headers** (<code>bool</code>) – If `True`, merges multiple column header rows into a single row.
- **page_layout** (<code>Literal['natural', 'single_column']</code>) – The type reading order to follow. Possible options:
- `natural`: Uses the natural reading order determined by Azure.
- `single_column`: Groups all lines with the same height on the page based on a threshold
  determined by `threshold_y`.
- **threshold_y** (<code>float | None</code>) – Only relevant if `single_column` is set to `page_layout`.
  The threshold, in inches, to determine if two recognized PDF elements are grouped into a
  single line. This is crucial for section headers or numbers which may be spatially separated
  from the remaining text on the horizontal axis.
- **store_full_path** (<code>bool</code>) – If True, the full path of the file is stored in the metadata of the document.
  If False, only the file name is stored.

#### run

```python
run(
    sources: list[str | Path | ByteStream],
    meta: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> dict[str, Any]
```

Convert a list of files to Documents using Azure's Document Intelligence service.

**Parameters:**

- **sources** (<code>list\[str | Path | ByteStream\]</code>) – List of file paths or ByteStream objects.
- **meta** (<code>dict\[str, Any\] | list\[dict\[str, Any\]\] | None</code>) – Optional metadata to attach to the Documents.
  This value can be either a list of dictionaries or a single dictionary.
  If it's a single dictionary, its content is added to the metadata of all produced Documents.
  If it's a list, the length of the list must match the number of sources, because the two lists will be
  zipped. If `sources` contains ByteStream objects, their `meta` will be added to the output Documents.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the following keys:
- `documents`: List of created Documents
- `raw_azure_response`: List of raw Azure responses used to create the Documents

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> AzureOCRDocumentConverter
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to deserialize from.

**Returns:**

- <code>AzureOCRDocumentConverter</code> – The deserialized component.
