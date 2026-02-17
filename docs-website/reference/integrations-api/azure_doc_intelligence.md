---
title: "Azure Document Intelligence"
id: integrations-azure_doc_intelligence
description: "Azure Document Intelligence integration for Haystack"
slug: "/integrations-azure_doc_intelligence"
---


## `haystack_integrations.components.converters.azure_doc_intelligence.converter`

### `AzureDocumentIntelligenceConverter`

Converts files to Documents using Azure's Document Intelligence service.

This component uses the azure-ai-documentintelligence package (v1.0.0+) and outputs
GitHub Flavored Markdown for better integration with LLM/RAG applications.

Supported file formats: PDF, JPEG, PNG, BMP, TIFF, DOCX, XLSX, PPTX, HTML.

Key features:

- Markdown output with preserved structure (headings, tables, lists)
- Inline table integration (tables rendered as markdown tables)
- Improved layout analysis and reading order
- Support for section headings

To use this component, you need an active Azure account
and a Document Intelligence or Cognitive Services resource. For setup instructions, see
[Azure documentation](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/quickstarts/get-started-sdks-rest-api).

### Usage example

```python
import os
from haystack_integrations.components.converters.azure_doc_intelligence import (
    AzureDocumentIntelligenceConverter,
)
from haystack.utils import Secret

converter = AzureDocumentIntelligenceConverter(
    endpoint=os.environ["AZURE_DI_ENDPOINT"],
    api_key=Secret.from_env_var("AZURE_DI_API_KEY"),
)

results = converter.run(sources=["invoice.pdf", "contract.docx"])
documents = results["documents"]

# Documents contain markdown with inline tables
print(documents[0].content)
```

#### `__init__`

```python
__init__(
    endpoint: str,
    *,
    api_key: Secret = Secret.from_env_var("AZURE_DI_API_KEY"),
    model_id: str = "prebuilt-document",
    store_full_path: bool = False
)
```

Creates an AzureDocumentIntelligenceConverter component.

**Parameters:**

- **endpoint** (<code>str</code>) – The endpoint URL of your Azure Document Intelligence resource.
  Example: "https://YOUR_RESOURCE.cognitiveservices.azure.com/"
- **api_key** (<code>Secret</code>) – API key for Azure authentication. Can use Secret.from_env_var()
  to load from AZURE_DI_API_KEY environment variable.
- **model_id** (<code>str</code>) – Azure model to use for analysis. Options:
- "prebuilt-document": General document analysis (default)
- "prebuilt-read": Fast OCR for text extraction
- "prebuilt-layout": Enhanced layout analysis with better table/structure detection
- Custom model IDs from your Azure resource
- **store_full_path** (<code>bool</code>) – If True, stores complete file path in metadata.
  If False, stores only the filename (default).

#### `warm_up`

```python
warm_up()
```

Initializes the Azure Document Intelligence client.

#### `run`

```python
run(
    sources: list[str | Path | ByteStream],
    meta: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> dict[str, list[Document] | list[dict]]
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

- <code>dict\[str, list\[Document\] | list\[dict\]\]</code> – A dictionary with the following keys:
- `documents`: List of created Documents
- `raw_azure_response`: List of raw Azure responses used to create the Documents

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> AzureDocumentIntelligenceConverter
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to deserialize from.

**Returns:**

- <code>AzureDocumentIntelligenceConverter</code> – The deserialized component.
