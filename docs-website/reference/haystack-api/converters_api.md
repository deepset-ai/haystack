---
title: "Converters"
id: converters-api
description: "Various converters to transform data from one format to another."
slug: "/converters-api"
---


## `haystack.components.converters.azure`

### `AzureOCRDocumentConverter`

Converts files to documents using Azure's Document Intelligence service.

Supported file formats are: PDF, JPEG, PNG, BMP, TIFF, DOCX, XLSX, PPTX, and HTML.

To use this component, you need an active Azure account
and a Document Intelligence or Cognitive Services resource. For help with setting up your resource, see
[Azure documentation](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/quickstarts/get-started-sdks-rest-api).

### Usage example

```python
import os
from datetime import datetime
from haystack.components.converters import AzureOCRDocumentConverter
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

#### `__init__`

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
)
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

#### `run`

```python
run(
    sources: list[str | Path | ByteStream],
    meta: dict[str, Any] | list[dict[str, Any]] | None = None,
)
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

- – A dictionary with the following keys:
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
from_dict(data: dict[str, Any]) -> AzureOCRDocumentConverter
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to deserialize from.

**Returns:**

- <code>AzureOCRDocumentConverter</code> – The deserialized component.

## `haystack.components.converters.csv`

### `CSVToDocument`

Converts CSV files to Documents.

By default, it uses UTF-8 encoding when converting files but
you can also set a custom encoding.
It can attach metadata to the resulting documents.

### Usage example

```python
from haystack.components.converters.csv import CSVToDocument
converter = CSVToDocument()
results = converter.run(sources=["sample.csv"], meta={"date_added": datetime.now().isoformat()})
documents = results["documents"]
print(documents[0].content)
# 'col1,col2\nrow1,row1\nrow2,row2\n'
```

#### `__init__`

```python
__init__(
    encoding: str = "utf-8",
    store_full_path: bool = False,
    *,
    conversion_mode: Literal["file", "row"] = "file",
    delimiter: str = ",",
    quotechar: str = '"'
)
```

Creates a CSVToDocument component.

**Parameters:**

- **encoding** (<code>str</code>) – The encoding of the csv files to convert.
  If the encoding is specified in the metadata of a source ByteStream,
  it overrides this value.
- **store_full_path** (<code>bool</code>) – If True, the full path of the file is stored in the metadata of the document.
  If False, only the file name is stored.
- **conversion_mode** (<code>Literal['file', 'row']</code>) – - "file" (default): one Document per CSV file whose content is the raw CSV text.
- "row": convert each CSV row to its own Document (requires `content_column` in `run()`).
- **delimiter** (<code>str</code>) – CSV delimiter used when parsing in row mode (passed to `csv.DictReader`).
- **quotechar** (<code>str</code>) – CSV quote character used when parsing in row mode (passed to `csv.DictReader`).

#### `run`

```python
run(
    sources: list[str | Path | ByteStream],
    *,
    content_column: str | None = None,
    meta: dict[str, Any] | list[dict[str, Any]] | None = None
)
```

Converts CSV files to a Document (file mode) or to one Document per row (row mode).

**Parameters:**

- **sources** (<code>list\[str | Path | ByteStream\]</code>) – List of file paths or ByteStream objects.
- **content_column** (<code>str | None</code>) – **Required when** `conversion_mode="row"`.
  The column name whose values become `Document.content` for each row.
  The column must exist in the CSV header.
- **meta** (<code>dict\[str, Any\] | list\[dict\[str, Any\]\] | None</code>) – Optional metadata to attach to the documents.
  This value can be either a list of dictionaries or a single dictionary.
  If it's a single dictionary, its content is added to the metadata of all produced documents.
  If it's a list, the length of the list must match the number of sources, because the two lists will
  be zipped.
  If `sources` contains ByteStream objects, their `meta` will be added to the output documents.

**Returns:**

- – A dictionary with the following keys:
- `documents`: Created documents

## `haystack.components.converters.docx`

### `DOCXMetadata`

Describes the metadata of Docx file.

**Parameters:**

- **author** (<code>str</code>) – The author
- **category** (<code>str</code>) – The category
- **comments** (<code>str</code>) – The comments
- **content_status** (<code>str</code>) – The content status
- **created** (<code>str | None</code>) – The creation date (ISO formatted string)
- **identifier** (<code>str</code>) – The identifier
- **keywords** (<code>str</code>) – Available keywords
- **language** (<code>str</code>) – The language of the document
- **last_modified_by** (<code>str</code>) – User who last modified the document
- **last_printed** (<code>str | None</code>) – The last printed date (ISO formatted string)
- **modified** (<code>str | None</code>) – The last modification date (ISO formatted string)
- **revision** (<code>int</code>) – The revision number
- **subject** (<code>str</code>) – The subject
- **title** (<code>str</code>) – The title
- **version** (<code>str</code>) – The version

### `DOCXTableFormat`

Bases: <code>Enum</code>

Supported formats for storing DOCX tabular data in a Document.

#### `from_str`

```python
from_str(string: str) -> DOCXTableFormat
```

Convert a string to a DOCXTableFormat enum.

### `DOCXLinkFormat`

Bases: <code>Enum</code>

Supported formats for storing DOCX link information in a Document.

#### `from_str`

```python
from_str(string: str) -> DOCXLinkFormat
```

Convert a string to a DOCXLinkFormat enum.

### `DOCXToDocument`

Converts DOCX files to Documents.

Uses `python-docx` library to convert the DOCX file to a document.
This component does not preserve page breaks in the original document.

Usage example:

```python
from haystack.components.converters.docx import DOCXToDocument, DOCXTableFormat, DOCXLinkFormat

converter = DOCXToDocument(table_format=DOCXTableFormat.CSV, link_format=DOCXLinkFormat.MARKDOWN)
results = converter.run(sources=["sample.docx"], meta={"date_added": datetime.now().isoformat()})
documents = results["documents"]
print(documents[0].content)
# 'This is a text from the DOCX file.'
```

#### `__init__`

```python
__init__(
    table_format: str | DOCXTableFormat = DOCXTableFormat.CSV,
    link_format: str | DOCXLinkFormat = DOCXLinkFormat.NONE,
    store_full_path: bool = False,
)
```

Create a DOCXToDocument component.

**Parameters:**

- **table_format** (<code>str | DOCXTableFormat</code>) – The format for table output. Can be either DOCXTableFormat.MARKDOWN,
  DOCXTableFormat.CSV, "markdown", or "csv".
- **link_format** (<code>str | DOCXLinkFormat</code>) – The format for link output. Can be either:
  DOCXLinkFormat.MARKDOWN or "markdown" to get `[text](address)`,
  DOCXLinkFormat.PLAIN or "plain" to get text (address),
  DOCXLinkFormat.NONE or "none" to get text without links.
- **store_full_path** (<code>bool</code>) – If True, the full path of the file is stored in the metadata of the document.
  If False, only the file name is stored.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> DOCXToDocument
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to deserialize from.

**Returns:**

- <code>DOCXToDocument</code> – The deserialized component.

#### `run`

```python
run(
    sources: list[str | Path | ByteStream],
    meta: dict[str, Any] | list[dict[str, Any]] | None = None,
)
```

Converts DOCX files to Documents.

**Parameters:**

- **sources** (<code>list\[str | Path | ByteStream\]</code>) – List of file paths or ByteStream objects.
- **meta** (<code>dict\[str, Any\] | list\[dict\[str, Any\]\] | None</code>) – Optional metadata to attach to the Documents.
  This value can be either a list of dictionaries or a single dictionary.
  If it's a single dictionary, its content is added to the metadata of all produced Documents.
  If it's a list, the length of the list must match the number of sources, because the two lists will
  be zipped.
  If `sources` contains ByteStream objects, their `meta` will be added to the output Documents.

**Returns:**

- – A dictionary with the following keys:
- `documents`: Created Documents

## `haystack.components.converters.html`

### `HTMLToDocument`

Converts an HTML file to a Document.

Usage example:

```python
from haystack.components.converters import HTMLToDocument

converter = HTMLToDocument()
results = converter.run(sources=["path/to/sample.html"])
documents = results["documents"]
print(documents[0].content)
# 'This is a text from the HTML file.'
```

#### `__init__`

```python
__init__(
    extraction_kwargs: dict[str, Any] | None = None,
    store_full_path: bool = False,
)
```

Create an HTMLToDocument component.

**Parameters:**

- **extraction_kwargs** (<code>dict\[str, Any\] | None</code>) – A dictionary containing keyword arguments to customize the extraction process. These
  are passed to the underlying Trafilatura `extract` function. For the full list of available arguments, see
  the [Trafilatura documentation](https://trafilatura.readthedocs.io/en/latest/corefunctions.html#extract).
- **store_full_path** (<code>bool</code>) – If True, the full path of the file is stored in the metadata of the document.
  If False, only the file name is stored.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> HTMLToDocument
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to deserialize from.

**Returns:**

- <code>HTMLToDocument</code> – The deserialized component.

#### `run`

```python
run(
    sources: list[str | Path | ByteStream],
    meta: dict[str, Any] | list[dict[str, Any]] | None = None,
    extraction_kwargs: dict[str, Any] | None = None,
)
```

Converts a list of HTML files to Documents.

**Parameters:**

- **sources** (<code>list\[str | Path | ByteStream\]</code>) – List of HTML file paths or ByteStream objects.
- **meta** (<code>dict\[str, Any\] | list\[dict\[str, Any\]\] | None</code>) – Optional metadata to attach to the Documents.
  This value can be either a list of dictionaries or a single dictionary.
  If it's a single dictionary, its content is added to the metadata of all produced Documents.
  If it's a list, the length of the list must match the number of sources, because the two lists will
  be zipped.
  If `sources` contains ByteStream objects, their `meta` will be added to the output Documents.
- **extraction_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments to customize the extraction process.

**Returns:**

- – A dictionary with the following keys:
- `documents`: Created Documents

## `haystack.components.converters.json`

### `JSONConverter`

Converts one or more JSON files into a text document.

### Usage examples

```python
import json

from haystack.components.converters import JSONConverter
from haystack.dataclasses import ByteStream

source = ByteStream.from_string(json.dumps({"text": "This is the content of my document"}))

converter = JSONConverter(content_key="text")
results = converter.run(sources=[source])
documents = results["documents"]
print(documents[0].content)
# 'This is the content of my document'
```

Optionally, you can also provide a `jq_schema` string to filter the JSON source files and `extra_meta_fields`
to extract from the filtered data:

```python
import json

from haystack.components.converters import JSONConverter
from haystack.dataclasses import ByteStream

data = {
    "laureates": [
        {
            "firstname": "Enrico",
            "surname": "Fermi",
            "motivation": "for his demonstrations of the existence of new radioactive elements produced "
            "by neutron irradiation, and for his related discovery of nuclear reactions brought about by"
            " slow neutrons",
        },
        {
            "firstname": "Rita",
            "surname": "Levi-Montalcini",
            "motivation": "for their discoveries of growth factors",
        },
    ],
}
source = ByteStream.from_string(json.dumps(data))
converter = JSONConverter(
    jq_schema=".laureates[]", content_key="motivation", extra_meta_fields={"firstname", "surname"}
)

results = converter.run(sources=[source])
documents = results["documents"]
print(documents[0].content)
# 'for his demonstrations of the existence of new radioactive elements produced by
# neutron irradiation, and for his related discovery of nuclear reactions brought
# about by slow neutrons'

print(documents[0].meta)
# {'firstname': 'Enrico', 'surname': 'Fermi'}

print(documents[1].content)
# 'for their discoveries of growth factors'

print(documents[1].meta)
# {'firstname': 'Rita', 'surname': 'Levi-Montalcini'}
```

#### `__init__`

```python
__init__(
    jq_schema: str | None = None,
    content_key: str | None = None,
    extra_meta_fields: set[str] | Literal["*"] | None = None,
    store_full_path: bool = False,
)
```

Creates a JSONConverter component.

An optional `jq_schema` can be provided to extract nested data in the JSON source files.
See the [official jq documentation](https://jqlang.github.io/jq/) for more info on the filters syntax.
If `jq_schema` is not set, whole JSON source files will be used to extract content.

Optionally, you can provide a `content_key` to specify which key in the extracted object must
be set as the document's content.

If both `jq_schema` and `content_key` are set, the component will search for the `content_key` in
the JSON object extracted by `jq_schema`. If the extracted data is not a JSON object, it will be skipped.

If only `jq_schema` is set, the extracted data must be a scalar value. If it's a JSON object or array,
it will be skipped.

If only `content_key` is set, the source JSON file must be a JSON object, else it will be skipped.

`extra_meta_fields` can either be set to a set of strings or a literal `"*"` string.
If it's a set of strings, it must specify fields in the extracted objects that must be set in
the extracted documents. If a field is not found, the meta value will be `None`.
If set to `"*"`, all fields that are not `content_key` found in the filtered JSON object will
be saved as metadata.

Initialization will fail if neither `jq_schema` nor `content_key` are set.

**Parameters:**

- **jq_schema** (<code>str | None</code>) – Optional jq filter string to extract content.
  If not specified, whole JSON object will be used to extract information.
- **content_key** (<code>str | None</code>) – Optional key to extract document content.
  If `jq_schema` is specified, the `content_key` will be extracted from that object.
- **extra_meta_fields** (<code>set\[str\] | Literal['\*'] | None</code>) – An optional set of meta keys to extract from the content.
  If `jq_schema` is specified, all keys will be extracted from that object.
- **store_full_path** (<code>bool</code>) – If True, the full path of the file is stored in the metadata of the document.
  If False, only the file name is stored.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> JSONConverter
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>JSONConverter</code> – Deserialized component.

#### `run`

```python
run(
    sources: list[str | Path | ByteStream],
    meta: dict[str, Any] | list[dict[str, Any]] | None = None,
)
```

Converts a list of JSON files to documents.

**Parameters:**

- **sources** (<code>list\[str | Path | ByteStream\]</code>) – A list of file paths or ByteStream objects.
- **meta** (<code>dict\[str, Any\] | list\[dict\[str, Any\]\] | None</code>) – Optional metadata to attach to the documents.
  This value can be either a list of dictionaries or a single dictionary.
  If it's a single dictionary, its content is added to the metadata of all produced documents.
  If it's a list, the length of the list must match the number of sources.
  If `sources` contain ByteStream objects, their `meta` will be added to the output documents.

**Returns:**

- – A dictionary with the following keys:
- `documents`: A list of created documents.

## `haystack.components.converters.markdown`

### `MarkdownToDocument`

Converts a Markdown file into a text Document.

Usage example:

```python
from haystack.components.converters import MarkdownToDocument
from datetime import datetime

converter = MarkdownToDocument()
results = converter.run(sources=["path/to/sample.md"], meta={"date_added": datetime.now().isoformat()})
documents = results["documents"]
print(documents[0].content)
# 'This is a text from the markdown file.'
```

#### `__init__`

```python
__init__(
    table_to_single_line: bool = False,
    progress_bar: bool = True,
    store_full_path: bool = False,
)
```

Create a MarkdownToDocument component.

**Parameters:**

- **table_to_single_line** (<code>bool</code>) – If True converts table contents into a single line.
- **progress_bar** (<code>bool</code>) – If True shows a progress bar when running.
- **store_full_path** (<code>bool</code>) – If True, the full path of the file is stored in the metadata of the document.
  If False, only the file name is stored.

#### `run`

```python
run(
    sources: list[str | Path | ByteStream],
    meta: dict[str, Any] | list[dict[str, Any]] | None = None,
)
```

Converts a list of Markdown files to Documents.

**Parameters:**

- **sources** (<code>list\[str | Path | ByteStream\]</code>) – List of file paths or ByteStream objects.
- **meta** (<code>dict\[str, Any\] | list\[dict\[str, Any\]\] | None</code>) – Optional metadata to attach to the Documents.
  This value can be either a list of dictionaries or a single dictionary.
  If it's a single dictionary, its content is added to the metadata of all produced Documents.
  If it's a list, the length of the list must match the number of sources, because the two lists will
  be zipped.
  If `sources` contains ByteStream objects, their `meta` will be added to the output Documents.

**Returns:**

- – A dictionary with the following keys:
- `documents`: List of created Documents

## `haystack.components.converters.msg`

### `MSGToDocument`

Converts Microsoft Outlook .msg files into Haystack Documents.

This component extracts email metadata (such as sender, recipients, CC, BCC, subject) and body content from .msg
files and converts them into structured Haystack Documents. Additionally, any file attachments within the .msg
file are extracted as ByteStream objects.

### Example Usage

```python
from haystack.components.converters.msg import MSGToDocument
from datetime import datetime

converter = MSGToDocument()
results = converter.run(sources=["sample.msg"], meta={"date_added": datetime.now().isoformat()})
documents = results["documents"]
attachments = results["attachments"]
print(documents[0].content)
```

#### `__init__`

```python
__init__(store_full_path: bool = False) -> None
```

Creates a MSGToDocument component.

**Parameters:**

- **store_full_path** (<code>bool</code>) – If True, the full path of the file is stored in the metadata of the document.
  If False, only the file name is stored.

#### `run`

```python
run(
    sources: list[str | Path | ByteStream],
    meta: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> dict[str, list[Document] | list[ByteStream]]
```

Converts MSG files to Documents.

**Parameters:**

- **sources** (<code>list\[str | Path | ByteStream\]</code>) – List of file paths or ByteStream objects.
- **meta** (<code>dict\[str, Any\] | list\[dict\[str, Any\]\] | None</code>) – Optional metadata to attach to the Documents.
  This value can be either a list of dictionaries or a single dictionary.
  If it's a single dictionary, its content is added to the metadata of all produced Documents.
  If it's a list, the length of the list must match the number of sources, because the two lists will
  be zipped.
  If `sources` contains ByteStream objects, their `meta` will be added to the output Documents.

**Returns:**

- <code>dict\[str, list\[Document\] | list\[ByteStream\]\]</code> – A dictionary with the following keys:
- `documents`: Created Documents.
- `attachments`: Created ByteStream objects from file attachments.

## `haystack.components.converters.multi_file_converter`

### `MultiFileConverter`

A file converter that handles conversion of multiple file types.

The MultiFileConverter handles the following file types:

- CSV
- DOCX
- HTML
- JSON
- MD
- TEXT
- PDF (no OCR)
- PPTX
- XLSX

Usage example:

```
from haystack.super_components.converters import MultiFileConverter

converter = MultiFileConverter()
converter.run(sources=["test.txt", "test.pdf"], meta={})
```

#### `__init__`

```python
__init__(encoding: str = 'utf-8', json_content_key: str = 'content') -> None
```

Initialize the MultiFileConverter.

**Parameters:**

- **encoding** (<code>str</code>) – The encoding to use when reading files.
- **json_content_key** (<code>str</code>) – The key to use in a content field in a document when converting JSON files.

## `haystack.components.converters.openapi_functions`

### `OpenAPIServiceToFunctions`

Converts OpenAPI service definitions to a format suitable for OpenAI function calling.

The definition must respect OpenAPI specification 3.0.0 or higher.
It can be specified in JSON or YAML format.
Each function must have:
\- unique operationId
\- description
\- requestBody and/or parameters
\- schema for the requestBody and/or parameters
For more details on OpenAPI specification see the [official documentation](https://github.com/OAI/OpenAPI-Specification).
For more details on OpenAI function calling see the [official documentation](https://platform.openai.com/docs/guides/function-calling).

Usage example:

```python
from haystack.components.converters import OpenAPIServiceToFunctions

converter = OpenAPIServiceToFunctions()
result = converter.run(sources=["path/to/openapi_definition.yaml"])
assert result["functions"]
```

#### `__init__`

```python
__init__()
```

Create an OpenAPIServiceToFunctions component.

#### `run`

```python
run(sources: list[str | Path | ByteStream]) -> dict[str, Any]
```

Converts OpenAPI definitions in OpenAI function calling format.

**Parameters:**

- **sources** (<code>list\[str | Path | ByteStream\]</code>) – File paths or ByteStream objects of OpenAPI definitions (in JSON or YAML format).

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the following keys:
- functions: Function definitions in JSON object format
- openapi_specs: OpenAPI specs in JSON/YAML object format with resolved references

**Raises:**

- <code>RuntimeError</code> – If the OpenAPI definitions cannot be downloaded or processed.
- <code>ValueError</code> – If the source type is not recognized or no functions are found in the OpenAPI definitions.

## `haystack.components.converters.output_adapter`

### `OutputAdaptationException`

Bases: <code>Exception</code>

Exception raised when there is an error during output adaptation.

### `OutputAdapter`

Adapts output of a Component using Jinja templates.

Usage example:

```python
from haystack import Document
from haystack.components.converters import OutputAdapter

adapter = OutputAdapter(template="{{ documents[0].content }}", output_type=str)
documents = [Document(content="Test content"]
result = adapter.run(documents=documents)

assert result["output"] == "Test content"
```

#### `__init__`

```python
__init__(
    template: str,
    output_type: TypeAlias,
    custom_filters: dict[str, Callable] | None = None,
    unsafe: bool = False,
) -> None
```

Create an OutputAdapter component.

**Parameters:**

- **template** (<code>str</code>) – A Jinja template that defines how to adapt the input data.
  The variables in the template define the input of this instance.
  e.g.
  With this template:

```
{{ documents[0].content }}
```

The Component input will be `documents`.

- **output_type** (<code>TypeAlias</code>) – The type of output this instance will return.
- **custom_filters** (<code>dict\[str, Callable\] | None</code>) – A dictionary of custom Jinja filters used in the template.
- **unsafe** (<code>bool</code>) – Enable execution of arbitrary code in the Jinja template.
  This should only be used if you trust the source of the template as it can be lead to remote code execution.

#### `run`

```python
run(**kwargs)
```

Renders the Jinja template with the provided inputs.

**Parameters:**

- **kwargs** – Must contain all variables used in the `template` string.

**Returns:**

- – A dictionary with the following keys:
- `output`: Rendered Jinja template.

**Raises:**

- <code>OutputAdaptationException</code> – If template rendering fails.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> OutputAdapter
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to deserialize from.

**Returns:**

- <code>OutputAdapter</code> – The deserialized component.

## `haystack.components.converters.pdfminer`

### `PDFMinerToDocument`

Converts PDF files to Documents.

Uses `pdfminer` compatible converters to convert PDF files to Documents. https://pdfminersix.readthedocs.io/en/latest/

Usage example:

```python
from haystack.components.converters.pdfminer import PDFMinerToDocument

converter = PDFMinerToDocument()
results = converter.run(sources=["sample.pdf"], meta={"date_added": datetime.now().isoformat()})
documents = results["documents"]
print(documents[0].content)
# 'This is a text from the PDF file.'
```

#### `__init__`

```python
__init__(
    line_overlap: float = 0.5,
    char_margin: float = 2.0,
    line_margin: float = 0.5,
    word_margin: float = 0.1,
    boxes_flow: float | None = 0.5,
    detect_vertical: bool = True,
    all_texts: bool = False,
    store_full_path: bool = False,
) -> None
```

Create a PDFMinerToDocument component.

**Parameters:**

- **line_overlap** (<code>float</code>) – This parameter determines whether two characters are considered to be on
  the same line based on the amount of overlap between them.
  The overlap is calculated relative to the minimum height of both characters.
- **char_margin** (<code>float</code>) – Determines whether two characters are part of the same line based on the distance between them.
  If the distance is less than the margin specified, the characters are considered to be on the same line.
  The margin is calculated relative to the width of the character.
- **word_margin** (<code>float</code>) – Determines whether two characters on the same line are part of the same word
  based on the distance between them. If the distance is greater than the margin specified,
  an intermediate space will be added between them to make the text more readable.
  The margin is calculated relative to the width of the character.
- **line_margin** (<code>float</code>) – This parameter determines whether two lines are part of the same paragraph based on
  the distance between them. If the distance is less than the margin specified,
  the lines are considered to be part of the same paragraph.
  The margin is calculated relative to the height of a line.
- **boxes_flow** (<code>float | None</code>) – This parameter determines the importance of horizontal and vertical position when
  determining the order of text boxes. A value between -1.0 and +1.0 can be set,
  with -1.0 indicating that only horizontal position matters and +1.0 indicating
  that only vertical position matters. Setting the value to 'None' will disable advanced
  layout analysis, and text boxes will be ordered based on the position of their bottom left corner.
- **detect_vertical** (<code>bool</code>) – This parameter determines whether vertical text should be considered during layout analysis.
- **all_texts** (<code>bool</code>) – If layout analysis should be performed on text in figures.
- **store_full_path** (<code>bool</code>) – If True, the full path of the file is stored in the metadata of the document.
  If False, only the file name is stored.

#### `detect_undecoded_cid_characters`

```python
detect_undecoded_cid_characters(text: str) -> dict[str, Any]
```

Look for character sequences of CID, i.e.: characters that haven't been properly decoded from their CID format.

This is useful to detect if the text extractor is not able to extract the text correctly, e.g. if the PDF uses
non-standard fonts.

A PDF font may include a ToUnicode map (mapping from character code to Unicode) to support operations like
searching strings or copy & paste in a PDF viewer. This map immediately provides the mapping the text extractor
needs. If that map is not available the text extractor cannot decode the CID characters and will return them
as is.

see: https://pdfminersix.readthedocs.io/en/latest/faq.html#why-are-there-cid-x-values-in-the-textual-output

**Parameters:**

- **text** (<code>str</code>) – The text to check for undecoded CID characters

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary containing detection results

#### `run`

```python
run(
    sources: list[str | Path | ByteStream],
    meta: dict[str, Any] | list[dict[str, Any]] | None = None,
)
```

Converts PDF files to Documents.

**Parameters:**

- **sources** (<code>list\[str | Path | ByteStream\]</code>) – List of PDF file paths or ByteStream objects.
- **meta** (<code>dict\[str, Any\] | list\[dict\[str, Any\]\] | None</code>) – Optional metadata to attach to the Documents.
  This value can be either a list of dictionaries or a single dictionary.
  If it's a single dictionary, its content is added to the metadata of all produced Documents.
  If it's a list, the length of the list must match the number of sources, because the two lists will
  be zipped.
  If `sources` contains ByteStream objects, their `meta` will be added to the output Documents.

**Returns:**

- – A dictionary with the following keys:
- `documents`: Created Documents

## `haystack.components.converters.pptx`

### `PPTXToDocument`

Converts PPTX files to Documents.

Usage example:

```python
from haystack.components.converters.pptx import PPTXToDocument

converter = PPTXToDocument()
results = converter.run(sources=["sample.pptx"], meta={"date_added": datetime.now().isoformat()})
documents = results["documents"]
print(documents[0].content)
# 'This is the text from the PPTX file.'
```

#### `__init__`

```python
__init__(store_full_path: bool = False)
```

Create an PPTXToDocument component.

**Parameters:**

- **store_full_path** (<code>bool</code>) – If True, the full path of the file is stored in the metadata of the document.
  If False, only the file name is stored.

#### `run`

```python
run(
    sources: list[str | Path | ByteStream],
    meta: dict[str, Any] | list[dict[str, Any]] | None = None,
)
```

Converts PPTX files to Documents.

**Parameters:**

- **sources** (<code>list\[str | Path | ByteStream\]</code>) – List of file paths or ByteStream objects.
- **meta** (<code>dict\[str, Any\] | list\[dict\[str, Any\]\] | None</code>) – Optional metadata to attach to the Documents.
  This value can be either a list of dictionaries or a single dictionary.
  If it's a single dictionary, its content is added to the metadata of all produced Documents.
  If it's a list, the length of the list must match the number of sources, because the two lists will
  be zipped.
  If `sources` contains ByteStream objects, their `meta` will be added to the output Documents.

**Returns:**

- – A dictionary with the following keys:
- `documents`: Created Documents

## `haystack.components.converters.pypdf`

### `PyPDFExtractionMode`

Bases: <code>Enum</code>

The mode to use for extracting text from a PDF.

#### `from_str`

```python
from_str(string: str) -> PyPDFExtractionMode
```

Convert a string to a PyPDFExtractionMode enum.

### `PyPDFToDocument`

Converts PDF files to documents your pipeline can query.

This component uses the PyPDF library.
You can attach metadata to the resulting documents.

### Usage example

```python
from haystack.components.converters.pypdf import PyPDFToDocument

converter = PyPDFToDocument()
results = converter.run(sources=["sample.pdf"], meta={"date_added": datetime.now().isoformat()})
documents = results["documents"]
print(documents[0].content)
# 'This is a text from the PDF file.'
```

#### `__init__`

```python
__init__(
    *,
    extraction_mode: str | PyPDFExtractionMode = PyPDFExtractionMode.PLAIN,
    plain_mode_orientations: tuple = (0, 90, 180, 270),
    plain_mode_space_width: float = 200.0,
    layout_mode_space_vertically: bool = True,
    layout_mode_scale_weight: float = 1.25,
    layout_mode_strip_rotated: bool = True,
    layout_mode_font_height_weight: float = 1.0,
    store_full_path: bool = False
)
```

Create an PyPDFToDocument component.

**Parameters:**

- **extraction_mode** (<code>str | PyPDFExtractionMode</code>) – The mode to use for extracting text from a PDF.
  Layout mode is an experimental mode that adheres to the rendered layout of the PDF.
- **plain_mode_orientations** (<code>tuple</code>) – Tuple of orientations to look for when extracting text from a PDF in plain mode.
  Ignored if `extraction_mode` is `PyPDFExtractionMode.LAYOUT`.
- **plain_mode_space_width** (<code>float</code>) – Forces default space width if not extracted from font.
  Ignored if `extraction_mode` is `PyPDFExtractionMode.LAYOUT`.
- **layout_mode_space_vertically** (<code>bool</code>) – Whether to include blank lines inferred from y distance + font height.
  Ignored if `extraction_mode` is `PyPDFExtractionMode.PLAIN`.
- **layout_mode_scale_weight** (<code>float</code>) – Multiplier for string length when calculating weighted average character width.
  Ignored if `extraction_mode` is `PyPDFExtractionMode.PLAIN`.
- **layout_mode_strip_rotated** (<code>bool</code>) – Layout mode does not support rotated text. Set to `False` to include rotated text anyway.
  If rotated text is discovered, layout will be degraded and a warning will be logged.
  Ignored if `extraction_mode` is `PyPDFExtractionMode.PLAIN`.
- **layout_mode_font_height_weight** (<code>float</code>) – Multiplier for font height when calculating blank line height.
  Ignored if `extraction_mode` is `PyPDFExtractionMode.PLAIN`.
- **store_full_path** (<code>bool</code>) – If True, the full path of the file is stored in the metadata of the document.
  If False, only the file name is stored.

#### `to_dict`

```python
to_dict()
```

Serializes the component to a dictionary.

**Returns:**

- – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data)
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** – Dictionary with serialized data.

**Returns:**

- – Deserialized component.

#### `run`

```python
run(
    sources: list[str | Path | ByteStream],
    meta: dict[str, Any] | list[dict[str, Any]] | None = None,
)
```

Converts PDF files to documents.

**Parameters:**

- **sources** (<code>list\[str | Path | ByteStream\]</code>) – List of file paths or ByteStream objects to convert.
- **meta** (<code>dict\[str, Any\] | list\[dict\[str, Any\]\] | None</code>) – Optional metadata to attach to the documents.
  This value can be a list of dictionaries or a single dictionary.
  If it's a single dictionary, its content is added to the metadata of all produced documents.
  If it's a list, its length must match the number of sources, as they are zipped together.
  For ByteStream objects, their `meta` is added to the output documents.

**Returns:**

- – A dictionary with the following keys:
- `documents`: A list of converted documents.

## `haystack.components.converters.tika`

### `XHTMLParser`

Bases: <code>HTMLParser</code>

Custom parser to extract pages from Tika XHTML content.

#### `handle_starttag`

```python
handle_starttag(tag: str, attrs: list[tuple])
```

Identify the start of a page div.

#### `handle_endtag`

```python
handle_endtag(tag: str)
```

Identify the end of a page div.

#### `handle_data`

```python
handle_data(data: str)
```

Populate the page content.

### `TikaDocumentConverter`

Converts files of different types to Documents.

This component uses [Apache Tika](https://tika.apache.org/) for parsing the files and, therefore,
requires a running Tika server.
For more options on running Tika,
see the [official documentation](https://github.com/apache/tika-docker/blob/main/README.md#usage).

Usage example:

```python
from haystack.components.converters.tika import TikaDocumentConverter

converter = TikaDocumentConverter()
results = converter.run(
    sources=["sample.docx", "my_document.rtf", "archive.zip"],
    meta={"date_added": datetime.now().isoformat()}
)
documents = results["documents"]
print(documents[0].content)
# 'This is a text from the docx file.'
```

#### `__init__`

```python
__init__(
    tika_url: str = "http://localhost:9998/tika", store_full_path: bool = False
)
```

Create a TikaDocumentConverter component.

**Parameters:**

- **tika_url** (<code>str</code>) – Tika server URL.
- **store_full_path** (<code>bool</code>) – If True, the full path of the file is stored in the metadata of the document.
  If False, only the file name is stored.

#### `run`

```python
run(
    sources: list[str | Path | ByteStream],
    meta: dict[str, Any] | list[dict[str, Any]] | None = None,
)
```

Converts files to Documents.

**Parameters:**

- **sources** (<code>list\[str | Path | ByteStream\]</code>) – List of HTML file paths or ByteStream objects.
- **meta** (<code>dict\[str, Any\] | list\[dict\[str, Any\]\] | None</code>) – Optional metadata to attach to the Documents.
  This value can be either a list of dictionaries or a single dictionary.
  If it's a single dictionary, its content is added to the metadata of all produced Documents.
  If it's a list, the length of the list must match the number of sources, because the two lists will
  be zipped.
  If `sources` contains ByteStream objects, their `meta` will be added to the output Documents.

**Returns:**

- – A dictionary with the following keys:
- `documents`: Created Documents

## `haystack.components.converters.txt`

### `TextFileToDocument`

Converts text files to documents your pipeline can query.

By default, it uses UTF-8 encoding when converting files but
you can also set custom encoding.
It can attach metadata to the resulting documents.

### Usage example

```python
from haystack.components.converters.txt import TextFileToDocument

converter = TextFileToDocument()
results = converter.run(sources=["sample.txt"])
documents = results["documents"]
print(documents[0].content)
# 'This is the content from the txt file.'
```

#### `__init__`

```python
__init__(encoding: str = 'utf-8', store_full_path: bool = False)
```

Creates a TextFileToDocument component.

**Parameters:**

- **encoding** (<code>str</code>) – The encoding of the text files to convert.
  If the encoding is specified in the metadata of a source ByteStream,
  it overrides this value.
- **store_full_path** (<code>bool</code>) – If True, the full path of the file is stored in the metadata of the document.
  If False, only the file name is stored.

#### `run`

```python
run(
    sources: list[str | Path | ByteStream],
    meta: dict[str, Any] | list[dict[str, Any]] | None = None,
)
```

Converts text files to documents.

**Parameters:**

- **sources** (<code>list\[str | Path | ByteStream\]</code>) – List of text file paths or ByteStream objects to convert.
- **meta** (<code>dict\[str, Any\] | list\[dict\[str, Any\]\] | None</code>) – Optional metadata to attach to the documents.
  This value can be a list of dictionaries or a single dictionary.
  If it's a single dictionary, its content is added to the metadata of all produced documents.
  If it's a list, its length must match the number of sources as they're zipped together.
  For ByteStream objects, their `meta` is added to the output documents.

**Returns:**

- – A dictionary with the following keys:
- `documents`: A list of converted documents.

## `haystack.components.converters.xlsx`

### `XLSXToDocument`

````
Converts XLSX (Excel) files into Documents.

Supports reading data from specific sheets or all sheets in the Excel file. If all sheets are read, a Document is
created for each sheet. The content of the Document is the table which can be saved in CSV or Markdown format.

### Usage example

```python
from haystack.components.converters.xlsx import XLSXToDocument

converter = XLSXToDocument()
results = converter.run(sources=["sample.xlsx"], meta={"date_added": datetime.now().isoformat()})
documents = results["documents"]
print(documents[0].content)
# ",A,B
````

1,col_a,col_b
2,1.5,test
"
\`\`\`

#### `__init__`

```python
__init__(
    table_format: Literal["csv", "markdown"] = "csv",
    sheet_name: str | int | list[str | int] | None = None,
    read_excel_kwargs: dict[str, Any] | None = None,
    table_format_kwargs: dict[str, Any] | None = None,
    *,
    store_full_path: bool = False
)
```

Creates a XLSXToDocument component.

**Parameters:**

- **table_format** (<code>Literal['csv', 'markdown']</code>) – The format to convert the Excel file to.
- **sheet_name** (<code>str | int | list\[str | int\] | None</code>) – The name of the sheet to read. If None, all sheets are read.
- **read_excel_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional arguments to pass to `pandas.read_excel`.
  See https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html#pandas-read-excel
- **table_format_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments to pass to the table format function.
- If `table_format` is "csv", these arguments are passed to `pandas.DataFrame.to_csv`.
  See https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html#pandas-dataframe-to-csv
- If `table_format` is "markdown", these arguments are passed to `pandas.DataFrame.to_markdown`.
  See https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_markdown.html#pandas-dataframe-to-markdown
- **store_full_path** (<code>bool</code>) – If True, the full path of the file is stored in the metadata of the document.
  If False, only the file name is stored.

#### `run`

```python
run(
    sources: list[str | Path | ByteStream],
    meta: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> dict[str, list[Document]]
```

Converts a XLSX file to a Document.

**Parameters:**

- **sources** (<code>list\[str | Path | ByteStream\]</code>) – List of file paths or ByteStream objects.
- **meta** (<code>dict\[str, Any\] | list\[dict\[str, Any\]\] | None</code>) – Optional metadata to attach to the documents.
  This value can be either a list of dictionaries or a single dictionary.
  If it's a single dictionary, its content is added to the metadata of all produced documents.
  If it's a list, the length of the list must match the number of sources, because the two lists will
  be zipped.
  If `sources` contains ByteStream objects, their `meta` will be added to the output documents.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: Created documents
