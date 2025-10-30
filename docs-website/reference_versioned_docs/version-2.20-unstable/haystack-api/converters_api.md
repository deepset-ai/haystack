---
title: "Converters"
id: converters-api
description: "Various converters to transform data from one format to another."
slug: "/converters-api"
---

<a id="azure"></a>

## Module azure

<a id="azure.AzureOCRDocumentConverter"></a>

### AzureOCRDocumentConverter

Converts files to documents using Azure's Document Intelligence service.

Supported file formats are: PDF, JPEG, PNG, BMP, TIFF, DOCX, XLSX, PPTX, and HTML.

To use this component, you need an active Azure account
and a Document Intelligence or Cognitive Services resource. For help with setting up your resource, see
[Azure documentation](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/quickstarts/get-started-sdks-rest-api).

### Usage example

```python
from haystack.components.converters import AzureOCRDocumentConverter
from haystack.utils import Secret

converter = AzureOCRDocumentConverter(endpoint="<url>", api_key=Secret.from_token("<your-api-key>"))
results = converter.run(sources=["path/to/doc_with_images.pdf"], meta={"date_added": datetime.now().isoformat()})
documents = results["documents"]
print(documents[0].content)
# 'This is a text from the PDF file.'
```

<a id="azure.AzureOCRDocumentConverter.__init__"></a>

#### AzureOCRDocumentConverter.\_\_init\_\_

```python
def __init__(endpoint: str,
             api_key: Secret = Secret.from_env_var("AZURE_AI_API_KEY"),
             model_id: str = "prebuilt-read",
             preceding_context_len: int = 3,
             following_context_len: int = 3,
             merge_multiple_column_headers: bool = True,
             page_layout: Literal["natural", "single_column"] = "natural",
             threshold_y: Optional[float] = 0.05,
             store_full_path: bool = False)
```

Creates an AzureOCRDocumentConverter component.

**Arguments**:

- `endpoint`: The endpoint of your Azure resource.
- `api_key`: The API key of your Azure resource.
- `model_id`: The ID of the model you want to use. For a list of available models, see [Azure documentation]
(https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/choose-model-feature).
- `preceding_context_len`: Number of lines before a table to include as preceding context
(this will be added to the metadata).
- `following_context_len`: Number of lines after a table to include as subsequent context (
this will be added to the metadata).
- `merge_multiple_column_headers`: If `True`, merges multiple column header rows into a single row.
- `page_layout`: The type reading order to follow. Possible options:
- `natural`: Uses the natural reading order determined by Azure.
- `single_column`: Groups all lines with the same height on the page based on a threshold
determined by `threshold_y`.
- `threshold_y`: Only relevant if `single_column` is set to `page_layout`.
The threshold, in inches, to determine if two recognized PDF elements are grouped into a
single line. This is crucial for section headers or numbers which may be spatially separated
from the remaining text on the horizontal axis.
- `store_full_path`: If True, the full path of the file is stored in the metadata of the document.
If False, only the file name is stored.

<a id="azure.AzureOCRDocumentConverter.run"></a>

#### AzureOCRDocumentConverter.run

```python
@component.output_types(documents=list[Document],
                        raw_azure_response=list[dict])
def run(sources: list[Union[str, Path, ByteStream]],
        meta: Optional[Union[dict[str, Any], list[dict[str, Any]]]] = None)
```

Convert a list of files to Documents using Azure's Document Intelligence service.

**Arguments**:

- `sources`: List of file paths or ByteStream objects.
- `meta`: Optional metadata to attach to the Documents.
This value can be either a list of dictionaries or a single dictionary.
If it's a single dictionary, its content is added to the metadata of all produced Documents.
If it's a list, the length of the list must match the number of sources, because the two lists will be
zipped. If `sources` contains ByteStream objects, their `meta` will be added to the output Documents.

**Returns**:

A dictionary with the following keys:
- `documents`: List of created Documents
- `raw_azure_response`: List of raw Azure responses used to create the Documents

<a id="azure.AzureOCRDocumentConverter.to_dict"></a>

#### AzureOCRDocumentConverter.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="azure.AzureOCRDocumentConverter.from_dict"></a>

#### AzureOCRDocumentConverter.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "AzureOCRDocumentConverter"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: The dictionary to deserialize from.

**Returns**:

The deserialized component.

<a id="csv"></a>

## Module csv

<a id="csv.CSVToDocument"></a>

### CSVToDocument

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

<a id="csv.CSVToDocument.__init__"></a>

#### CSVToDocument.\_\_init\_\_

```python
def __init__(encoding: str = "utf-8",
             store_full_path: bool = False,
             *,
             conversion_mode: Literal["file", "row"] = "file",
             delimiter: str = ",",
             quotechar: str = '"')
```

Creates a CSVToDocument component.

**Arguments**:

- `encoding`: The encoding of the csv files to convert.
If the encoding is specified in the metadata of a source ByteStream,
it overrides this value.
- `store_full_path`: If True, the full path of the file is stored in the metadata of the document.
If False, only the file name is stored.
- `conversion_mode`: - "file" (default): one Document per CSV file whose content is the raw CSV text.
- "row": convert each CSV row to its own Document (requires `content_column` in `run()`).
- `delimiter`: CSV delimiter used when parsing in row mode (passed to ``csv.DictReader``).
- `quotechar`: CSV quote character used when parsing in row mode (passed to ``csv.DictReader``).

<a id="csv.CSVToDocument.run"></a>

#### CSVToDocument.run

```python
@component.output_types(documents=list[Document])
def run(sources: list[Union[str, Path, ByteStream]],
        *,
        content_column: Optional[str] = None,
        meta: Optional[Union[dict[str, Any], list[dict[str, Any]]]] = None)
```

Converts CSV files to a Document (file mode) or to one Document per row (row mode).

**Arguments**:

- `sources`: List of file paths or ByteStream objects.
- `content_column`: **Required when** ``conversion_mode="row"``.
The column name whose values become ``Document.content`` for each row.
The column must exist in the CSV header.
- `meta`: Optional metadata to attach to the documents.
This value can be either a list of dictionaries or a single dictionary.
If it's a single dictionary, its content is added to the metadata of all produced documents.
If it's a list, the length of the list must match the number of sources, because the two lists will
be zipped.
If `sources` contains ByteStream objects, their `meta` will be added to the output documents.

**Returns**:

A dictionary with the following keys:
- `documents`: Created documents

<a id="docx"></a>

## Module docx

<a id="docx.DOCXMetadata"></a>

### DOCXMetadata

Describes the metadata of Docx file.

**Arguments**:

- `author`: The author
- `category`: The category
- `comments`: The comments
- `content_status`: The content status
- `created`: The creation date (ISO formatted string)
- `identifier`: The identifier
- `keywords`: Available keywords
- `language`: The language of the document
- `last_modified_by`: User who last modified the document
- `last_printed`: The last printed date (ISO formatted string)
- `modified`: The last modification date (ISO formatted string)
- `revision`: The revision number
- `subject`: The subject
- `title`: The title
- `version`: The version

<a id="docx.DOCXTableFormat"></a>

### DOCXTableFormat

Supported formats for storing DOCX tabular data in a Document.

<a id="docx.DOCXTableFormat.from_str"></a>

#### DOCXTableFormat.from\_str

```python
@staticmethod
def from_str(string: str) -> "DOCXTableFormat"
```

Convert a string to a DOCXTableFormat enum.

<a id="docx.DOCXLinkFormat"></a>

### DOCXLinkFormat

Supported formats for storing DOCX link information in a Document.

<a id="docx.DOCXLinkFormat.from_str"></a>

#### DOCXLinkFormat.from\_str

```python
@staticmethod
def from_str(string: str) -> "DOCXLinkFormat"
```

Convert a string to a DOCXLinkFormat enum.

<a id="docx.DOCXToDocument"></a>

### DOCXToDocument

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

<a id="docx.DOCXToDocument.__init__"></a>

#### DOCXToDocument.\_\_init\_\_

```python
def __init__(table_format: Union[str, DOCXTableFormat] = DOCXTableFormat.CSV,
             link_format: Union[str, DOCXLinkFormat] = DOCXLinkFormat.NONE,
             store_full_path: bool = False)
```

Create a DOCXToDocument component.

**Arguments**:

- `table_format`: The format for table output. Can be either DOCXTableFormat.MARKDOWN,
DOCXTableFormat.CSV, "markdown", or "csv".
- `link_format`: The format for link output. Can be either:
DOCXLinkFormat.MARKDOWN or "markdown" to get [text](address),
DOCXLinkFormat.PLAIN or "plain" to get text (address),
DOCXLinkFormat.NONE or "none" to get text without links.
- `store_full_path`: If True, the full path of the file is stored in the metadata of the document.
If False, only the file name is stored.

<a id="docx.DOCXToDocument.to_dict"></a>

#### DOCXToDocument.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="docx.DOCXToDocument.from_dict"></a>

#### DOCXToDocument.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "DOCXToDocument"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: The dictionary to deserialize from.

**Returns**:

The deserialized component.

<a id="docx.DOCXToDocument.run"></a>

#### DOCXToDocument.run

```python
@component.output_types(documents=list[Document])
def run(sources: list[Union[str, Path, ByteStream]],
        meta: Optional[Union[dict[str, Any], list[dict[str, Any]]]] = None)
```

Converts DOCX files to Documents.

**Arguments**:

- `sources`: List of file paths or ByteStream objects.
- `meta`: Optional metadata to attach to the Documents.
This value can be either a list of dictionaries or a single dictionary.
If it's a single dictionary, its content is added to the metadata of all produced Documents.
If it's a list, the length of the list must match the number of sources, because the two lists will
be zipped.
If `sources` contains ByteStream objects, their `meta` will be added to the output Documents.

**Returns**:

A dictionary with the following keys:
- `documents`: Created Documents

<a id="html"></a>

## Module html

<a id="html.HTMLToDocument"></a>

### HTMLToDocument

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

<a id="html.HTMLToDocument.__init__"></a>

#### HTMLToDocument.\_\_init\_\_

```python
def __init__(extraction_kwargs: Optional[dict[str, Any]] = None,
             store_full_path: bool = False)
```

Create an HTMLToDocument component.

**Arguments**:

- `extraction_kwargs`: A dictionary containing keyword arguments to customize the extraction process. These
are passed to the underlying Trafilatura `extract` function. For the full list of available arguments, see
the [Trafilatura documentation](https://trafilatura.readthedocs.io/en/latest/corefunctions.html#extract).
- `store_full_path`: If True, the full path of the file is stored in the metadata of the document.
If False, only the file name is stored.

<a id="html.HTMLToDocument.to_dict"></a>

#### HTMLToDocument.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="html.HTMLToDocument.from_dict"></a>

#### HTMLToDocument.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "HTMLToDocument"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: The dictionary to deserialize from.

**Returns**:

The deserialized component.

<a id="html.HTMLToDocument.run"></a>

#### HTMLToDocument.run

```python
@component.output_types(documents=list[Document])
def run(sources: list[Union[str, Path, ByteStream]],
        meta: Optional[Union[dict[str, Any], list[dict[str, Any]]]] = None,
        extraction_kwargs: Optional[dict[str, Any]] = None)
```

Converts a list of HTML files to Documents.

**Arguments**:

- `sources`: List of HTML file paths or ByteStream objects.
- `meta`: Optional metadata to attach to the Documents.
This value can be either a list of dictionaries or a single dictionary.
If it's a single dictionary, its content is added to the metadata of all produced Documents.
If it's a list, the length of the list must match the number of sources, because the two lists will
be zipped.
If `sources` contains ByteStream objects, their `meta` will be added to the output Documents.
- `extraction_kwargs`: Additional keyword arguments to customize the extraction process.

**Returns**:

A dictionary with the following keys:
- `documents`: Created Documents

<a id="json"></a>

## Module json

<a id="json.JSONConverter"></a>

### JSONConverter

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

<a id="json.JSONConverter.__init__"></a>

#### JSONConverter.\_\_init\_\_

```python
def __init__(jq_schema: Optional[str] = None,
             content_key: Optional[str] = None,
             extra_meta_fields: Optional[Union[set[str], Literal["*"]]] = None,
             store_full_path: bool = False)
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

**Arguments**:

- `jq_schema`: Optional jq filter string to extract content.
If not specified, whole JSON object will be used to extract information.
- `content_key`: Optional key to extract document content.
If `jq_schema` is specified, the `content_key` will be extracted from that object.
- `extra_meta_fields`: An optional set of meta keys to extract from the content.
If `jq_schema` is specified, all keys will be extracted from that object.
- `store_full_path`: If True, the full path of the file is stored in the metadata of the document.
If False, only the file name is stored.

<a id="json.JSONConverter.to_dict"></a>

#### JSONConverter.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="json.JSONConverter.from_dict"></a>

#### JSONConverter.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "JSONConverter"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="json.JSONConverter.run"></a>

#### JSONConverter.run

```python
@component.output_types(documents=list[Document])
def run(sources: list[Union[str, Path, ByteStream]],
        meta: Optional[Union[dict[str, Any], list[dict[str, Any]]]] = None)
```

Converts a list of JSON files to documents.

**Arguments**:

- `sources`: A list of file paths or ByteStream objects.
- `meta`: Optional metadata to attach to the documents.
This value can be either a list of dictionaries or a single dictionary.
If it's a single dictionary, its content is added to the metadata of all produced documents.
If it's a list, the length of the list must match the number of sources.
If `sources` contain ByteStream objects, their `meta` will be added to the output documents.

**Returns**:

A dictionary with the following keys:
- `documents`: A list of created documents.

<a id="markdown"></a>

## Module markdown

<a id="markdown.MarkdownToDocument"></a>

### MarkdownToDocument

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

<a id="markdown.MarkdownToDocument.__init__"></a>

#### MarkdownToDocument.\_\_init\_\_

```python
def __init__(table_to_single_line: bool = False,
             progress_bar: bool = True,
             store_full_path: bool = False)
```

Create a MarkdownToDocument component.

**Arguments**:

- `table_to_single_line`: If True converts table contents into a single line.
- `progress_bar`: If True shows a progress bar when running.
- `store_full_path`: If True, the full path of the file is stored in the metadata of the document.
If False, only the file name is stored.

<a id="markdown.MarkdownToDocument.run"></a>

#### MarkdownToDocument.run

```python
@component.output_types(documents=list[Document])
def run(sources: list[Union[str, Path, ByteStream]],
        meta: Optional[Union[dict[str, Any], list[dict[str, Any]]]] = None)
```

Converts a list of Markdown files to Documents.

**Arguments**:

- `sources`: List of file paths or ByteStream objects.
- `meta`: Optional metadata to attach to the Documents.
This value can be either a list of dictionaries or a single dictionary.
If it's a single dictionary, its content is added to the metadata of all produced Documents.
If it's a list, the length of the list must match the number of sources, because the two lists will
be zipped.
If `sources` contains ByteStream objects, their `meta` will be added to the output Documents.

**Returns**:

A dictionary with the following keys:
- `documents`: List of created Documents

<a id="msg"></a>

## Module msg

<a id="msg.MSGToDocument"></a>

### MSGToDocument

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

<a id="msg.MSGToDocument.__init__"></a>

#### MSGToDocument.\_\_init\_\_

```python
def __init__(store_full_path: bool = False) -> None
```

Creates a MSGToDocument component.

**Arguments**:

- `store_full_path`: If True, the full path of the file is stored in the metadata of the document.
If False, only the file name is stored.

<a id="msg.MSGToDocument.run"></a>

#### MSGToDocument.run

```python
@component.output_types(documents=list[Document], attachments=list[ByteStream])
def run(
    sources: list[Union[str, Path, ByteStream]],
    meta: Optional[Union[dict[str, Any], list[dict[str, Any]]]] = None
) -> dict[str, Union[list[Document], list[ByteStream]]]
```

Converts MSG files to Documents.

**Arguments**:

- `sources`: List of file paths or ByteStream objects.
- `meta`: Optional metadata to attach to the Documents.
This value can be either a list of dictionaries or a single dictionary.
If it's a single dictionary, its content is added to the metadata of all produced Documents.
If it's a list, the length of the list must match the number of sources, because the two lists will
be zipped.
If `sources` contains ByteStream objects, their `meta` will be added to the output Documents.

**Returns**:

A dictionary with the following keys:
- `documents`: Created Documents.
- `attachments`: Created ByteStream objects from file attachments.

<a id="multi_file_converter"></a>

## Module multi\_file\_converter

<a id="multi_file_converter.MultiFileConverter"></a>

### MultiFileConverter

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

<a id="multi_file_converter.MultiFileConverter.__init__"></a>

#### MultiFileConverter.\_\_init\_\_

```python
def __init__(encoding: str = "utf-8",
             json_content_key: str = "content") -> None
```

Initialize the MultiFileConverter.

**Arguments**:

- `encoding`: The encoding to use when reading files.
- `json_content_key`: The key to use in a content field in a document when converting JSON files.

<a id="openapi_functions"></a>

## Module openapi\_functions

<a id="openapi_functions.OpenAPIServiceToFunctions"></a>

### OpenAPIServiceToFunctions

Converts OpenAPI service definitions to a format suitable for OpenAI function calling.

The definition must respect OpenAPI specification 3.0.0 or higher.
It can be specified in JSON or YAML format.
Each function must have:
    - unique operationId
    - description
    - requestBody and/or parameters
    - schema for the requestBody and/or parameters
For more details on OpenAPI specification see the [official documentation](https://github.com/OAI/OpenAPI-Specification).
For more details on OpenAI function calling see the [official documentation](https://platform.openai.com/docs/guides/function-calling).

Usage example:
```python
from haystack.components.converters import OpenAPIServiceToFunctions

converter = OpenAPIServiceToFunctions()
result = converter.run(sources=["path/to/openapi_definition.yaml"])
assert result["functions"]
```

<a id="openapi_functions.OpenAPIServiceToFunctions.__init__"></a>

#### OpenAPIServiceToFunctions.\_\_init\_\_

```python
def __init__()
```

Create an OpenAPIServiceToFunctions component.

<a id="openapi_functions.OpenAPIServiceToFunctions.run"></a>

#### OpenAPIServiceToFunctions.run

```python
@component.output_types(functions=list[dict[str, Any]],
                        openapi_specs=list[dict[str, Any]])
def run(sources: list[Union[str, Path, ByteStream]]) -> dict[str, Any]
```

Converts OpenAPI definitions in OpenAI function calling format.

**Arguments**:

- `sources`: File paths or ByteStream objects of OpenAPI definitions (in JSON or YAML format).

**Raises**:

- `RuntimeError`: If the OpenAPI definitions cannot be downloaded or processed.
- `ValueError`: If the source type is not recognized or no functions are found in the OpenAPI definitions.

**Returns**:

A dictionary with the following keys:
- functions: Function definitions in JSON object format
- openapi_specs: OpenAPI specs in JSON/YAML object format with resolved references

<a id="output_adapter"></a>

## Module output\_adapter

<a id="output_adapter.OutputAdaptationException"></a>

### OutputAdaptationException

Exception raised when there is an error during output adaptation.

<a id="output_adapter.OutputAdapter"></a>

### OutputAdapter

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

<a id="output_adapter.OutputAdapter.__init__"></a>

#### OutputAdapter.\_\_init\_\_

```python
def __init__(template: str,
             output_type: TypeAlias,
             custom_filters: Optional[dict[str, Callable]] = None,
             unsafe: bool = False)
```

Create an OutputAdapter component.

**Arguments**:

- `template`: A Jinja template that defines how to adapt the input data.
The variables in the template define the input of this instance.
e.g.
With this template:
```
{{ documents[0].content }}
```
The Component input will be `documents`.
- `output_type`: The type of output this instance will return.
- `custom_filters`: A dictionary of custom Jinja filters used in the template.
- `unsafe`: Enable execution of arbitrary code in the Jinja template.
This should only be used if you trust the source of the template as it can be lead to remote code execution.

<a id="output_adapter.OutputAdapter.run"></a>

#### OutputAdapter.run

```python
def run(**kwargs)
```

Renders the Jinja template with the provided inputs.

**Arguments**:

- `kwargs`: Must contain all variables used in the `template` string.

**Raises**:

- `OutputAdaptationException`: If template rendering fails.

**Returns**:

A dictionary with the following keys:
- `output`: Rendered Jinja template.

<a id="output_adapter.OutputAdapter.to_dict"></a>

#### OutputAdapter.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="output_adapter.OutputAdapter.from_dict"></a>

#### OutputAdapter.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "OutputAdapter"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: The dictionary to deserialize from.

**Returns**:

The deserialized component.

<a id="pdfminer"></a>

## Module pdfminer

<a id="pdfminer.CID_PATTERN"></a>

#### CID\_PATTERN

regex pattern to detect CID characters

<a id="pdfminer.PDFMinerToDocument"></a>

### PDFMinerToDocument

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

<a id="pdfminer.PDFMinerToDocument.__init__"></a>

#### PDFMinerToDocument.\_\_init\_\_

```python
def __init__(line_overlap: float = 0.5,
             char_margin: float = 2.0,
             line_margin: float = 0.5,
             word_margin: float = 0.1,
             boxes_flow: Optional[float] = 0.5,
             detect_vertical: bool = True,
             all_texts: bool = False,
             store_full_path: bool = False) -> None
```

Create a PDFMinerToDocument component.

**Arguments**:

- `line_overlap`: This parameter determines whether two characters are considered to be on
the same line based on the amount of overlap between them.
The overlap is calculated relative to the minimum height of both characters.
- `char_margin`: Determines whether two characters are part of the same line based on the distance between them.
If the distance is less than the margin specified, the characters are considered to be on the same line.
The margin is calculated relative to the width of the character.
- `word_margin`: Determines whether two characters on the same line are part of the same word
based on the distance between them. If the distance is greater than the margin specified,
an intermediate space will be added between them to make the text more readable.
The margin is calculated relative to the width of the character.
- `line_margin`: This parameter determines whether two lines are part of the same paragraph based on
the distance between them. If the distance is less than the margin specified,
the lines are considered to be part of the same paragraph.
The margin is calculated relative to the height of a line.
- `boxes_flow`: This parameter determines the importance of horizontal and vertical position when
determining the order of text boxes. A value between -1.0 and +1.0 can be set,
with -1.0 indicating that only horizontal position matters and +1.0 indicating
that only vertical position matters. Setting the value to 'None' will disable advanced
layout analysis, and text boxes will be ordered based on the position of their bottom left corner.
- `detect_vertical`: This parameter determines whether vertical text should be considered during layout analysis.
- `all_texts`: If layout analysis should be performed on text in figures.
- `store_full_path`: If True, the full path of the file is stored in the metadata of the document.
If False, only the file name is stored.

<a id="pdfminer.PDFMinerToDocument.detect_undecoded_cid_characters"></a>

#### PDFMinerToDocument.detect\_undecoded\_cid\_characters

```python
def detect_undecoded_cid_characters(text: str) -> dict[str, Any]
```

Look for character sequences of CID, i.e.: characters that haven't been properly decoded from their CID format.

This is useful to detect if the text extractor is not able to extract the text correctly, e.g. if the PDF uses
non-standard fonts.

A PDF font may include a ToUnicode map (mapping from character code to Unicode) to support operations like
searching strings or copy & paste in a PDF viewer. This map immediately provides the mapping the text extractor
needs. If that map is not available the text extractor cannot decode the CID characters and will return them
as is.

see: https://pdfminersix.readthedocs.io/en/latest/faq.html#why-are-there-cid-x-values-in-the-textual-output

:param: text: The text to check for undecoded CID characters
:returns:
    A dictionary containing detection results


<a id="pdfminer.PDFMinerToDocument.run"></a>

#### PDFMinerToDocument.run

```python
@component.output_types(documents=list[Document])
def run(sources: list[Union[str, Path, ByteStream]],
        meta: Optional[Union[dict[str, Any], list[dict[str, Any]]]] = None)
```

Converts PDF files to Documents.

**Arguments**:

- `sources`: List of PDF file paths or ByteStream objects.
- `meta`: Optional metadata to attach to the Documents.
This value can be either a list of dictionaries or a single dictionary.
If it's a single dictionary, its content is added to the metadata of all produced Documents.
If it's a list, the length of the list must match the number of sources, because the two lists will
be zipped.
If `sources` contains ByteStream objects, their `meta` will be added to the output Documents.

**Returns**:

A dictionary with the following keys:
- `documents`: Created Documents

<a id="pptx"></a>

## Module pptx

<a id="pptx.PPTXToDocument"></a>

### PPTXToDocument

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

<a id="pptx.PPTXToDocument.__init__"></a>

#### PPTXToDocument.\_\_init\_\_

```python
def __init__(store_full_path: bool = False)
```

Create an PPTXToDocument component.

**Arguments**:

- `store_full_path`: If True, the full path of the file is stored in the metadata of the document.
If False, only the file name is stored.

<a id="pptx.PPTXToDocument.run"></a>

#### PPTXToDocument.run

```python
@component.output_types(documents=list[Document])
def run(sources: list[Union[str, Path, ByteStream]],
        meta: Optional[Union[dict[str, Any], list[dict[str, Any]]]] = None)
```

Converts PPTX files to Documents.

**Arguments**:

- `sources`: List of file paths or ByteStream objects.
- `meta`: Optional metadata to attach to the Documents.
This value can be either a list of dictionaries or a single dictionary.
If it's a single dictionary, its content is added to the metadata of all produced Documents.
If it's a list, the length of the list must match the number of sources, because the two lists will
be zipped.
If `sources` contains ByteStream objects, their `meta` will be added to the output Documents.

**Returns**:

A dictionary with the following keys:
- `documents`: Created Documents

<a id="pypdf"></a>

## Module pypdf

<a id="pypdf.PyPDFExtractionMode"></a>

### PyPDFExtractionMode

The mode to use for extracting text from a PDF.

<a id="pypdf.PyPDFExtractionMode.__str__"></a>

#### PyPDFExtractionMode.\_\_str\_\_

```python
def __str__() -> str
```

Convert a PyPDFExtractionMode enum to a string.

<a id="pypdf.PyPDFExtractionMode.from_str"></a>

#### PyPDFExtractionMode.from\_str

```python
@staticmethod
def from_str(string: str) -> "PyPDFExtractionMode"
```

Convert a string to a PyPDFExtractionMode enum.

<a id="pypdf.PyPDFToDocument"></a>

### PyPDFToDocument

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

<a id="pypdf.PyPDFToDocument.__init__"></a>

#### PyPDFToDocument.\_\_init\_\_

```python
def __init__(*,
             extraction_mode: Union[
                 str, PyPDFExtractionMode] = PyPDFExtractionMode.PLAIN,
             plain_mode_orientations: tuple = (0, 90, 180, 270),
             plain_mode_space_width: float = 200.0,
             layout_mode_space_vertically: bool = True,
             layout_mode_scale_weight: float = 1.25,
             layout_mode_strip_rotated: bool = True,
             layout_mode_font_height_weight: float = 1.0,
             store_full_path: bool = False)
```

Create an PyPDFToDocument component.

**Arguments**:

- `extraction_mode`: The mode to use for extracting text from a PDF.
Layout mode is an experimental mode that adheres to the rendered layout of the PDF.
- `plain_mode_orientations`: Tuple of orientations to look for when extracting text from a PDF in plain mode.
Ignored if `extraction_mode` is `PyPDFExtractionMode.LAYOUT`.
- `plain_mode_space_width`: Forces default space width if not extracted from font.
Ignored if `extraction_mode` is `PyPDFExtractionMode.LAYOUT`.
- `layout_mode_space_vertically`: Whether to include blank lines inferred from y distance + font height.
Ignored if `extraction_mode` is `PyPDFExtractionMode.PLAIN`.
- `layout_mode_scale_weight`: Multiplier for string length when calculating weighted average character width.
Ignored if `extraction_mode` is `PyPDFExtractionMode.PLAIN`.
- `layout_mode_strip_rotated`: Layout mode does not support rotated text. Set to `False` to include rotated text anyway.
If rotated text is discovered, layout will be degraded and a warning will be logged.
Ignored if `extraction_mode` is `PyPDFExtractionMode.PLAIN`.
- `layout_mode_font_height_weight`: Multiplier for font height when calculating blank line height.
Ignored if `extraction_mode` is `PyPDFExtractionMode.PLAIN`.
- `store_full_path`: If True, the full path of the file is stored in the metadata of the document.
If False, only the file name is stored.

<a id="pypdf.PyPDFToDocument.to_dict"></a>

#### PyPDFToDocument.to\_dict

```python
def to_dict()
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="pypdf.PyPDFToDocument.from_dict"></a>

#### PyPDFToDocument.from\_dict

```python
@classmethod
def from_dict(cls, data)
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary with serialized data.

**Returns**:

Deserialized component.

<a id="pypdf.PyPDFToDocument.run"></a>

#### PyPDFToDocument.run

```python
@component.output_types(documents=list[Document])
def run(sources: list[Union[str, Path, ByteStream]],
        meta: Optional[Union[dict[str, Any], list[dict[str, Any]]]] = None)
```

Converts PDF files to documents.

**Arguments**:

- `sources`: List of file paths or ByteStream objects to convert.
- `meta`: Optional metadata to attach to the documents.
This value can be a list of dictionaries or a single dictionary.
If it's a single dictionary, its content is added to the metadata of all produced documents.
If it's a list, its length must match the number of sources, as they are zipped together.
For ByteStream objects, their `meta` is added to the output documents.

**Returns**:

A dictionary with the following keys:
- `documents`: A list of converted documents.

<a id="tika"></a>

## Module tika

<a id="tika.XHTMLParser"></a>

### XHTMLParser

Custom parser to extract pages from Tika XHTML content.

<a id="tika.XHTMLParser.handle_starttag"></a>

#### XHTMLParser.handle\_starttag

```python
def handle_starttag(tag: str, attrs: list[tuple])
```

Identify the start of a page div.

<a id="tika.XHTMLParser.handle_endtag"></a>

#### XHTMLParser.handle\_endtag

```python
def handle_endtag(tag: str)
```

Identify the end of a page div.

<a id="tika.XHTMLParser.handle_data"></a>

#### XHTMLParser.handle\_data

```python
def handle_data(data: str)
```

Populate the page content.

<a id="tika.TikaDocumentConverter"></a>

### TikaDocumentConverter

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

<a id="tika.TikaDocumentConverter.__init__"></a>

#### TikaDocumentConverter.\_\_init\_\_

```python
def __init__(tika_url: str = "http://localhost:9998/tika",
             store_full_path: bool = False)
```

Create a TikaDocumentConverter component.

**Arguments**:

- `tika_url`: Tika server URL.
- `store_full_path`: If True, the full path of the file is stored in the metadata of the document.
If False, only the file name is stored.

<a id="tika.TikaDocumentConverter.run"></a>

#### TikaDocumentConverter.run

```python
@component.output_types(documents=list[Document])
def run(sources: list[Union[str, Path, ByteStream]],
        meta: Optional[Union[dict[str, Any], list[dict[str, Any]]]] = None)
```

Converts files to Documents.

**Arguments**:

- `sources`: List of HTML file paths or ByteStream objects.
- `meta`: Optional metadata to attach to the Documents.
This value can be either a list of dictionaries or a single dictionary.
If it's a single dictionary, its content is added to the metadata of all produced Documents.
If it's a list, the length of the list must match the number of sources, because the two lists will
be zipped.
If `sources` contains ByteStream objects, their `meta` will be added to the output Documents.

**Returns**:

A dictionary with the following keys:
- `documents`: Created Documents

<a id="txt"></a>

## Module txt

<a id="txt.TextFileToDocument"></a>

### TextFileToDocument

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

<a id="txt.TextFileToDocument.__init__"></a>

#### TextFileToDocument.\_\_init\_\_

```python
def __init__(encoding: str = "utf-8", store_full_path: bool = False)
```

Creates a TextFileToDocument component.

**Arguments**:

- `encoding`: The encoding of the text files to convert.
If the encoding is specified in the metadata of a source ByteStream,
it overrides this value.
- `store_full_path`: If True, the full path of the file is stored in the metadata of the document.
If False, only the file name is stored.

<a id="txt.TextFileToDocument.run"></a>

#### TextFileToDocument.run

```python
@component.output_types(documents=list[Document])
def run(sources: list[Union[str, Path, ByteStream]],
        meta: Optional[Union[dict[str, Any], list[dict[str, Any]]]] = None)
```

Converts text files to documents.

**Arguments**:

- `sources`: List of text file paths or ByteStream objects to convert.
- `meta`: Optional metadata to attach to the documents.
This value can be a list of dictionaries or a single dictionary.
If it's a single dictionary, its content is added to the metadata of all produced documents.
If it's a list, its length must match the number of sources as they're zipped together.
For ByteStream objects, their `meta` is added to the output documents.

**Returns**:

A dictionary with the following keys:
- `documents`: A list of converted documents.

<a id="xlsx"></a>

## Module xlsx

<a id="xlsx.XLSXToDocument"></a>

### XLSXToDocument

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
1,col_a,col_b
2,1.5,test
"
    ```

<a id="xlsx.XLSXToDocument.__init__"></a>

#### XLSXToDocument.\_\_init\_\_

```python
def __init__(table_format: Literal["csv", "markdown"] = "csv",
             sheet_name: Union[str, int, list[Union[str, int]], None] = None,
             read_excel_kwargs: Optional[dict[str, Any]] = None,
             table_format_kwargs: Optional[dict[str, Any]] = None,
             *,
             store_full_path: bool = False)
```

Creates a XLSXToDocument component.

**Arguments**:

- `table_format`: The format to convert the Excel file to.
- `sheet_name`: The name of the sheet to read. If None, all sheets are read.
- `read_excel_kwargs`: Additional arguments to pass to `pandas.read_excel`.
See https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html#pandas-read-excel
- `table_format_kwargs`: Additional keyword arguments to pass to the table format function.
- If `table_format` is "csv", these arguments are passed to `pandas.DataFrame.to_csv`.
  See https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html#pandas-dataframe-to-csv
- If `table_format` is "markdown", these arguments are passed to `pandas.DataFrame.to_markdown`.
  See https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_markdown.html#pandas-dataframe-to-markdown
- `store_full_path`: If True, the full path of the file is stored in the metadata of the document.
If False, only the file name is stored.

<a id="xlsx.XLSXToDocument.run"></a>

#### XLSXToDocument.run

```python
@component.output_types(documents=list[Document])
def run(
    sources: list[Union[str, Path, ByteStream]],
    meta: Optional[Union[dict[str, Any], list[dict[str, Any]]]] = None
) -> dict[str, list[Document]]
```

Converts a XLSX file to a Document.

**Arguments**:

- `sources`: List of file paths or ByteStream objects.
- `meta`: Optional metadata to attach to the documents.
This value can be either a list of dictionaries or a single dictionary.
If it's a single dictionary, its content is added to the metadata of all produced documents.
If it's a list, the length of the list must match the number of sources, because the two lists will
be zipped.
If `sources` contains ByteStream objects, their `meta` will be added to the output documents.

**Returns**:

A dictionary with the following keys:
- `documents`: Created documents
