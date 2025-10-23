---
title: "Mistral"
id: integrations-mistral
description: "Mistral integration for Haystack"
slug: "/integrations-mistral"
---

<a id="haystack_integrations.components.embedders.mistral.document_embedder"></a>

## Module haystack\_integrations.components.embedders.mistral.document\_embedder

<a id="haystack_integrations.components.embedders.mistral.document_embedder.MistralDocumentEmbedder"></a>

### MistralDocumentEmbedder

A component for computing Document embeddings using Mistral models.
The embedding of each Document is stored in the `embedding` field of the Document.

Usage example:
```python
from haystack import Document
from haystack_integrations.components.embedders.mistral import MistralDocumentEmbedder

doc = Document(content="I love pizza!")

document_embedder = MistralDocumentEmbedder()

result = document_embedder.run([doc])
print(result['documents'][0].embedding)

# [0.017020374536514282, -0.023255806416273117, ...]
```

<a id="haystack_integrations.components.embedders.mistral.document_embedder.MistralDocumentEmbedder.__init__"></a>

#### MistralDocumentEmbedder.\_\_init\_\_

```python
def __init__(api_key: Secret = Secret.from_env_var("MISTRAL_API_KEY"),
             model: str = "mistral-embed",
             api_base_url: Optional[str] = "https://api.mistral.ai/v1",
             prefix: str = "",
             suffix: str = "",
             batch_size: int = 32,
             progress_bar: bool = True,
             meta_fields_to_embed: Optional[List[str]] = None,
             embedding_separator: str = "\n",
             *,
             timeout: Optional[float] = None,
             max_retries: Optional[int] = None,
             http_client_kwargs: Optional[Dict[str, Any]] = None)
```

Creates a MistralDocumentEmbedder component.

**Arguments**:

- `api_key`: The Mistral API key.
- `model`: The name of the model to use.
- `api_base_url`: The Mistral API Base url. For more details, see Mistral [docs](https://docs.mistral.ai/api/).
- `prefix`: A string to add to the beginning of each text.
- `suffix`: A string to add to the end of each text.
- `batch_size`: Number of Documents to encode at once.
- `progress_bar`: Whether to show a progress bar or not. Can be helpful to disable in production deployments to keep
the logs clean.
- `meta_fields_to_embed`: List of meta fields that should be embedded along with the Document text.
- `embedding_separator`: Separator used to concatenate the meta fields to the Document text.
- `timeout`: Timeout for Mistral client calls. If not set, it defaults to either the `OPENAI_TIMEOUT` environment
variable, or 30 seconds.
- `max_retries`: Maximum number of retries to contact Mistral after an internal error.
If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
- `http_client_kwargs`: A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/`client`).

<a id="haystack_integrations.components.embedders.mistral.document_embedder.MistralDocumentEmbedder.to_dict"></a>

#### MistralDocumentEmbedder.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.embedders.mistral.text_embedder"></a>

## Module haystack\_integrations.components.embedders.mistral.text\_embedder

<a id="haystack_integrations.components.embedders.mistral.text_embedder.MistralTextEmbedder"></a>

### MistralTextEmbedder

A component for embedding strings using Mistral models.

Usage example:
 ```python
from haystack_integrations.components.embedders.mistral.text_embedder import MistralTextEmbedder

text_to_embed = "I love pizza!"
text_embedder = MistralTextEmbedder()
print(text_embedder.run(text_to_embed))

__output:__

__{'embedding': [0.017020374536514282, -0.023255806416273117, ...],__

__'meta': {'model': 'mistral-embed',__

__         'usage': {'prompt_tokens': 4, 'total_tokens': 4}}}__

```

<a id="haystack_integrations.components.embedders.mistral.text_embedder.MistralTextEmbedder.__init__"></a>

#### MistralTextEmbedder.\_\_init\_\_

```python
def __init__(api_key: Secret = Secret.from_env_var("MISTRAL_API_KEY"),
             model: str = "mistral-embed",
             api_base_url: Optional[str] = "https://api.mistral.ai/v1",
             prefix: str = "",
             suffix: str = "",
             *,
             timeout: Optional[float] = None,
             max_retries: Optional[int] = None,
             http_client_kwargs: Optional[Dict[str, Any]] = None)
```

Creates an MistralTextEmbedder component.

**Arguments**:

- `api_key`: The Mistral API key.
- `model`: The name of the Mistral embedding model to be used.
- `api_base_url`: The Mistral API Base url.
For more details, see Mistral [docs](https://docs.mistral.ai/api/).
- `prefix`: A string to add to the beginning of each text.
- `suffix`: A string to add to the end of each text.
- `timeout`: Timeout for Mistral client calls. If not set, it defaults to either the `OPENAI_TIMEOUT` environment
variable, or 30 seconds.
- `max_retries`: Maximum number of retries to contact Mistral after an internal error.
If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
- `http_client_kwargs`: A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/`client`).

<a id="haystack_integrations.components.embedders.mistral.text_embedder.MistralTextEmbedder.to_dict"></a>

#### MistralTextEmbedder.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.generators.mistral.chat.chat_generator"></a>

## Module haystack\_integrations.components.generators.mistral.chat.chat\_generator

<a id="haystack_integrations.components.generators.mistral.chat.chat_generator.MistralChatGenerator"></a>

### MistralChatGenerator

Enables text generation using Mistral AI generative models.
For supported models, see [Mistral AI docs](https://docs.mistral.ai/platform/endpoints/`operation`/listModels).

Users can pass any text generation parameters valid for the Mistral Chat Completion API
directly to this component via the `generation_kwargs` parameter in `__init__` or the `generation_kwargs`
parameter in `run` method.

Key Features and Compatibility:
- **Primary Compatibility**: Designed to work seamlessly with the Mistral API Chat Completion endpoint.
- **Streaming Support**: Supports streaming responses from the Mistral API Chat Completion endpoint.
- **Customizability**: Supports all parameters supported by the Mistral API Chat Completion endpoint.

This component uses the ChatMessage format for structuring both input and output,
ensuring coherent and contextually relevant responses in chat-based text generation scenarios.
Details on the ChatMessage format can be found in the
[Haystack docs](https://docs.haystack.deepset.ai/v2.0/docs/data-classes#chatmessage)

For more details on the parameters supported by the Mistral API, refer to the
[Mistral API Docs](https://docs.mistral.ai/api/).

Usage example:
```python
from haystack_integrations.components.generators.mistral import MistralChatGenerator
from haystack.dataclasses import ChatMessage

messages = [ChatMessage.from_user("What's Natural Language Processing?")]

client = MistralChatGenerator()
response = client.run(messages)
print(response)

>>{'replies': [ChatMessage(_role=<ChatRole.ASSISTANT: 'assistant'>, _content=[TextContent(text=
>> "Natural Language Processing (NLP) is a branch of artificial intelligence
>> that focuses on enabling computers to understand, interpret, and generate human language in a way that is
>> meaningful and useful.")], _name=None,
>> _meta={'model': 'mistral-small-latest', 'index': 0, 'finish_reason': 'stop',
>> 'usage': {'prompt_tokens': 15, 'completion_tokens': 36, 'total_tokens': 51}})]}
```

<a id="haystack_integrations.components.generators.mistral.chat.chat_generator.MistralChatGenerator.__init__"></a>

#### MistralChatGenerator.\_\_init\_\_

```python
def __init__(api_key: Secret = Secret.from_env_var("MISTRAL_API_KEY"),
             model: str = "mistral-small-latest",
             streaming_callback: Optional[StreamingCallbackT] = None,
             api_base_url: Optional[str] = "https://api.mistral.ai/v1",
             generation_kwargs: Optional[Dict[str, Any]] = None,
             tools: Optional[ToolsType] = None,
             *,
             timeout: Optional[float] = None,
             max_retries: Optional[int] = None,
             http_client_kwargs: Optional[Dict[str, Any]] = None)
```

Creates an instance of MistralChatGenerator. Unless specified otherwise in the `model`, this is for Mistral's

`mistral-small-latest` model.

**Arguments**:

- `api_key`: The Mistral API key.
- `model`: The name of the Mistral chat completion model to use.
- `streaming_callback`: A callback function that is called when a new token is received from the stream.
The callback function accepts StreamingChunk as an argument.
- `api_base_url`: The Mistral API Base url.
For more details, see Mistral [docs](https://docs.mistral.ai/api/).
- `generation_kwargs`: Other parameters to use for the model. These parameters are all sent directly to
the Mistral endpoint. See [Mistral API docs](https://docs.mistral.ai/api/) for more details.
Some of the supported parameters:
- `max_tokens`: The maximum number of tokens the output text can have.
- `temperature`: What sampling temperature to use. Higher values mean the model will take more risks.
    Try 0.9 for more creative applications and 0 (argmax sampling) for ones with a well-defined answer.
- `top_p`: An alternative to sampling with temperature, called nucleus sampling, where the model
    considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens
    comprising the top 10% probability mass are considered.
- `stream`: Whether to stream back partial progress. If set, tokens will be sent as data-only server-sent
    events as they become available, with the stream terminated by a data: [DONE] message.
- `safe_prompt`: Whether to inject a safety prompt before all conversations.
- `random_seed`: The seed to use for random sampling.
 - `response_format`: A JSON schema or a Pydantic model that enforces the structure of the model's response.
    If provided, the output will always be validated against this
    format (unless the model returns a tool call).
    For details, see the [OpenAI Structured Outputs documentation](https://platform.openai.com/docs/guides/structured-outputs).
    Notes:
    - For structured outputs with streaming,
      the `response_format` must be a JSON schema and not a Pydantic model.
- `tools`: A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
Each tool should have a unique name.
- `timeout`: The timeout for the Mistral API call. If not set, it defaults to either the `OPENAI_TIMEOUT`
environment variable, or 30 seconds.
- `max_retries`: Maximum number of retries to contact OpenAI after an internal error.
If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
- `http_client_kwargs`: A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/`client`).

<a id="haystack_integrations.components.generators.mistral.chat.chat_generator.MistralChatGenerator.to_dict"></a>

#### MistralChatGenerator.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serialize this component to a dictionary.

**Returns**:

The serialized component as a dictionary.

<a id="haystack_integrations.components.converters.mistral.ocr_document_converter"></a>

## Module haystack\_integrations.components.converters.mistral.ocr\_document\_converter

<a id="haystack_integrations.components.converters.mistral.ocr_document_converter.MistralOCRDocumentConverter"></a>

### MistralOCRDocumentConverter

This component extracts text from documents using Mistral's OCR API, with optional structured
annotations for both individual image regions (bounding boxes) and full documents.

Accepts document sources in various formats (str/Path for local files, ByteStream for in-memory data,
DocumentURLChunk for document URLs, ImageURLChunk for image URLs, or FileChunk for Mistral file IDs)
and retrieves the recognized text via Mistral's OCR service. Local files are automatically uploaded
to Mistral's storage.
Returns Haystack Documents (one per source) containing all pages concatenated with form feed characters (\f),
ensuring compatibility with Haystack's DocumentSplitter for accurate page-wise splitting and overlap handling.

**How Annotations Work:**
When annotation schemas (`bbox_annotation_schema` or `document_annotation_schema`) are provided,
the OCR model first extracts text and structure from the document. Then, a Vision LLM is called
to analyze the content and generate structured annotations according to your defined schemas.
For more details, see: https://docs.mistral.ai/capabilities/document_ai/annotations/`how`-it-works

**Usage Example:**

**Structured Output Example:**
```python
from haystack.utils import Secret
from haystack_integrations.mistral import MistralOCRDocumentConverter
from mistralai.models import DocumentURLChunk, ImageURLChunk, FileChunk

converter = MistralOCRDocumentConverter(
    api_key=Secret.from_env_var("MISTRAL_API_KEY"),
    model="mistral-ocr-2505"
)

# Process multiple sources
sources = [
    DocumentURLChunk(document_url="https://example.com/document.pdf"),
    ImageURLChunk(image_url="https://example.com/receipt.jpg"),
    FileChunk(file_id="file-abc123"),
]
result = converter.run(sources=sources)

documents = result["documents"]  # List of 3 Documents
raw_responses = result["raw_mistral_response"]  # List of 3 raw responses
```
```python
from pydantic import BaseModel, Field
from haystack_integrations.mistral import MistralOCRDocumentConverter

# Define schema for structured image annotations
class ImageAnnotation(BaseModel):
    image_type: str = Field(..., description="The type of image content")
    short_description: str = Field(..., description="Short natural-language description")
    summary: str = Field(..., description="Detailed summary of the image content")

# Define schema for structured document annotations
class DocumentAnnotation(BaseModel):
    language: str = Field(..., description="Primary language of the document")
    chapter_titles: List[str] = Field(..., description="Detected chapter or section titles")
    urls: List[str] = Field(..., description="URLs found in the text")

converter = MistralOCRDocumentConverter(
    model="mistral-ocr-2505",
)

sources = [DocumentURLChunk(document_url="https://example.com/report.pdf")]
result = converter.run(
    sources=sources,
    bbox_annotation_schema=ImageAnnotation,
    document_annotation_schema=DocumentAnnotation,
)

documents = result["documents"]
raw_responses = result["raw_mistral_response"]
```

<a id="haystack_integrations.components.converters.mistral.ocr_document_converter.MistralOCRDocumentConverter.__init__"></a>

#### MistralOCRDocumentConverter.\_\_init\_\_

```python
def __init__(api_key: Secret = Secret.from_env_var("MISTRAL_API_KEY"),
             model: str = "mistral-ocr-2505",
             include_image_base64: bool = False,
             pages: Optional[List[int]] = None,
             image_limit: Optional[int] = None,
             image_min_size: Optional[int] = None,
             cleanup_uploaded_files: bool = True)
```

Creates a MistralOCRDocumentConverter component.

**Arguments**:

- `api_key`: The Mistral API key. Defaults to the MISTRAL_API_KEY environment variable.
- `model`: The OCR model to use. Default is "mistral-ocr-2505".
See more: https://docs.mistral.ai/getting-started/models/models_overview/
- `include_image_base64`: If True, includes base64 encoded images in the response.
This may significantly increase response size and processing time.
- `pages`: Specific page numbers to process (0-indexed). If None, processes all pages.
- `image_limit`: Maximum number of images to extract from the document.
- `image_min_size`: Minimum height and width (in pixels) for images to be extracted.
- `cleanup_uploaded_files`: If True, automatically deletes files uploaded to Mistral after processing.
Only affects files uploaded from local sources (str, Path, ByteStream).
Files provided as FileChunk are not deleted. Default is True.

<a id="haystack_integrations.components.converters.mistral.ocr_document_converter.MistralOCRDocumentConverter.to_dict"></a>

#### MistralOCRDocumentConverter.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.converters.mistral.ocr_document_converter.MistralOCRDocumentConverter.from_dict"></a>

#### MistralOCRDocumentConverter.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "MistralOCRDocumentConverter"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.converters.mistral.ocr_document_converter.MistralOCRDocumentConverter.run"></a>

#### MistralOCRDocumentConverter.run

```python
@component.output_types(documents=List[Document],
                        raw_mistral_response=List[Dict[str, Any]])
def run(
    sources: List[Union[str, Path, ByteStream, DocumentURLChunk, FileChunk,
                        ImageURLChunk]],
    bbox_annotation_schema: Optional[Type[BaseModel]] = None,
    document_annotation_schema: Optional[Type[BaseModel]] = None
) -> Dict[str, Any]
```

Extract text from documents using Mistral OCR.

**Arguments**:

- `sources`: List of document sources to process. Each source can be one of:
- str: File path to a local document
- Path: Path object to a local document
- ByteStream: Haystack ByteStream object containing document data
- DocumentURLChunk: Mistral chunk for document URLs (signed or public URLs to PDFs, etc.)
- ImageURLChunk: Mistral chunk for image URLs (signed or public URLs to images)
- FileChunk: Mistral chunk for file IDs (files previously uploaded to Mistral)
- `bbox_annotation_schema`: Optional Pydantic model for structured annotations per bounding box.
When provided, a Vision LLM analyzes each image region and returns structured data.
- `document_annotation_schema`: Optional Pydantic model for structured annotations for the full document.
When provided, a Vision LLM analyzes the entire document and returns structured data.
Note: Document annotation is limited to a maximum of 8 pages. Documents exceeding
this limit will not be processed for document annotation.

**Returns**:

A dictionary with the following keys:
- `documents`: List of Haystack Documents (one per source). Each Document has the following structure:
    - `content`: All pages joined with form feed (\f) separators in markdown format.
      When using bbox_annotation_schema, image tags will be enriched with your defined descriptions.
    - `meta`: Aggregated metadata dictionary with structure:
      `{"source_page_count": int, "source_total_images": int, "source_*": any}`.
      If document_annotation_schema was provided, all annotation fields are unpacked
      with 'source_' prefix (e.g., source_language, source_chapter_titles, source_urls).
- `raw_mistral_response`:
    List of dictionaries containing raw OCR responses from Mistral API (one per source).
    Each response includes per-page details, images, annotations, and usage info.

