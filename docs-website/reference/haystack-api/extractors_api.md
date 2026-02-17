---
title: "Extractors"
id: extractors-api
description: "Components to extract specific elements from textual data."
slug: "/extractors-api"
---


## `haystack.components.extractors.image.llm_document_content_extractor`

### `LLMDocumentContentExtractor`

Extracts textual content and optionally metadata from image-based documents using a vision-enabled LLM.

One prompt and one LLM call per document. The component converts each document to an image via
DocumentToImageContent and sends it to the ChatGenerator. The prompt must not contain Jinja variables.

Response handling:

- If the LLM returns a **plain string** (non-JSON or not a JSON object), it is written to the document's content.
- If the LLM returns a **JSON object with only the key** `document_content`, that value is written to content.
- If the LLM returns a **JSON object with multiple keys**, the value of `document_content` (if present) is
  written to content and all other keys are merged into the document's metadata.

The ChatGenerator can be configured to return JSON (e.g. `response_format={"type": "json_object"}`
in `generation_kwargs`).

Documents that fail extraction are returned in `failed_documents` with `content_extraction_error` in metadata.

### Usage example

```python
from haystack import Document
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.extractors.image import LLMDocumentContentExtractor

prompt = """
Extract the content from the provided image.
Format everything as markdown. Return only the extracted content as a JSON object with the key 'document_content'.
No markdown, no code fence, only raw JSON.

Extract metadata about the image like source of the image, date of creation, etc. if you can.
Return this metadata as additional key-value pairs in the same JSON object.
"""

chat_generator = OpenAIChatGenerator()
extractor = LLMDocumentContentExtractor(
    chat_generator=chat_generator,
    generation_kwargs={
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "entity_extraction",
                "schema": {
                    "type": "object",
                    "properties": {
                        "document_content": {"type": "string"},
                        "author": {"type": "string"},
                        "date": {"type": "string"},
                        "document_type": {"type": "string"},
                        "title": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            },
        }
    }
)
documents = [
    Document(content="", meta={"file_path": "image.jpg"}),
    Document(content="", meta={"file_path": "document.pdf", "page_number": 1})
]
result = extractor.run(documents=documents)
updated_documents = result["documents"]
```

#### `__init__`

```python
__init__(
    *,
    chat_generator: ChatGenerator,
    prompt: str = DEFAULT_PROMPT_TEMPLATE,
    file_path_meta_field: str = "file_path",
    root_path: str | None = None,
    detail: Literal["auto", "high", "low"] | None = None,
    size: tuple[int, int] | None = None,
    raise_on_failure: bool = False,
    max_workers: int = 3
)
```

Initialize the LLMDocumentContentExtractor component.

**Parameters:**

- **chat_generator** (<code>ChatGenerator</code>) – A ChatGenerator that supports vision input. Optionally configured for JSON
  (e.g. `response_format={"type": "json_object"}` in `generation_kwargs`).
- **prompt** (<code>str</code>) – Prompt for extraction. Must not contain Jinja variables.
- **file_path_meta_field** (<code>str</code>) – The metadata field in the Document that contains the file path to the image or PDF.
- **root_path** (<code>str | None</code>) – The root directory path where document files are located. If provided, file paths in
  document metadata will be resolved relative to this path. If None, file paths are treated as absolute paths.
- **detail** (<code>Literal['auto', 'high', 'low'] | None</code>) – Optional detail level of the image (only supported by OpenAI). Can be "auto", "high", or "low".
- **size** (<code>tuple\[int, int\] | None</code>) – If provided, resizes the image to fit within (width, height) while keeping aspect ratio.
- **raise_on_failure** (<code>bool</code>) – If True, exceptions from the LLM are raised. If False, failed documents are returned.
- **max_workers** (<code>int</code>) – Maximum number of threads for parallel LLM calls.

#### `warm_up`

```python
warm_up()
```

Warm up the ChatGenerator if it has a warm_up method.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> LLMDocumentContentExtractor
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary with serialized data.

**Returns:**

- <code>LLMDocumentContentExtractor</code> – An instance of the component.

#### `run`

```python
run(documents: list[Document]) -> dict[str, list[Document]]
```

Run extraction on image-based documents. One LLM call per document.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – A list of image-based documents to process. Each must have a valid file path in its metadata.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with "documents" (successfully processed) and "failed_documents" (with failure metadata).

## `haystack.components.extractors.llm_metadata_extractor`

### `LLMMetadataExtractor`

Extracts metadata from documents using a Large Language Model (LLM).

The metadata is extracted by providing a prompt to an LLM that generates the metadata.

This component expects as input a list of documents and a prompt. The prompt should have a variable called
`document` that will point to a single document in the list of documents. So to access the content of the document,
you can use `{{ document.content }}` in the prompt.

The component will run the LLM on each document in the list and extract metadata from the document. The metadata
will be added to the document's metadata field. If the LLM fails to extract metadata from a document, the document
will be added to the `failed_documents` list. The failed documents will have the keys `metadata_extraction_error` and
`metadata_extraction_response` in their metadata. These documents can be re-run with another extractor to
extract metadata by using the `metadata_extraction_response` and `metadata_extraction_error` in the prompt.

```python
from haystack import Document
from haystack.components.extractors.llm_metadata_extractor import LLMMetadataExtractor
from haystack.components.generators.chat import OpenAIChatGenerator

NER_PROMPT = '''
-Goal-
Given text and a list of entity types, identify all entities of those types from the text.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity: Name of the entity
- entity_type: One of the following types: [organization, product, service, industry]
Format each entity as a JSON like: {"entity": <entity_name>, "entity_type": <entity_type>}

2. Return output in a single list with all the entities identified in steps 1.

-Examples-
######################
Example 1:
entity_types: [organization, person, partnership, financial metric, product, service, industry, investment strategy, market trend]
text: Another area of strength is our co-brand issuance. Visa is the primary network partner for eight of the top
10 co-brand partnerships in the US today and we are pleased that Visa has finalized a multi-year extension of
our successful credit co-branded partnership with Alaska Airlines, a portfolio that benefits from a loyal customer
base and high cross-border usage.
We have also had significant co-brand momentum in CEMEA. First, we launched a new co-brand card in partnership
with Qatar Airways, British Airways and the National Bank of Kuwait. Second, we expanded our strong global
Marriott relationship to launch Qatar's first hospitality co-branded card with Qatar Islamic Bank. Across the
United Arab Emirates, we now have exclusive agreements with all the leading airlines marked by a recent
agreement with Emirates Skywards.
And we also signed an inaugural Airline co-brand agreement in Morocco with Royal Air Maroc. Now newer digital
issuers are equally
------------------------
output:
{"entities": [{"entity": "Visa", "entity_type": "company"}, {"entity": "Alaska Airlines", "entity_type": "company"}, {"entity": "Qatar Airways", "entity_type": "company"}, {"entity": "British Airways", "entity_type": "company"}, {"entity": "National Bank of Kuwait", "entity_type": "company"}, {"entity": "Marriott", "entity_type": "company"}, {"entity": "Qatar Islamic Bank", "entity_type": "company"}, {"entity": "Emirates Skywards", "entity_type": "company"}, {"entity": "Royal Air Maroc", "entity_type": "company"}]}
#############################
-Real Data-
######################
entity_types: [company, organization, person, country, product, service]
text: {{ document.content }}
######################
output:
'''

docs = [
    Document(content="deepset was founded in 2018 in Berlin, and is known for its Haystack framework"),
    Document(content="Hugging Face is a company that was founded in New York, USA and is known for its Transformers library")
]

chat_generator = OpenAIChatGenerator(
    generation_kwargs={
        "max_completion_tokens": 500,
        "temperature": 0.0,
        "seed": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "entity_extraction",
                "schema": {
                    "type": "object",
                    "properties": {
                        "entities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "entity": {"type": "string"},
                                    "entity_type": {"type": "string"}
                                },
                                "required": ["entity", "entity_type"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["entities"],
                    "additionalProperties": False
                }
            }
        },
    },
    max_retries=1,
    timeout=60.0,
)

extractor = LLMMetadataExtractor(
    prompt=NER_PROMPT,
    chat_generator=generator,
    expected_keys=["entities"],
    raise_on_failure=False,
)

extractor.warm_up()
extractor.run(documents=docs)
>> {'documents': [
    Document(id=.., content: 'deepset was founded in 2018 in Berlin, and is known for its Haystack framework',
    meta: {'entities': [{'entity': 'deepset', 'entity_type': 'company'}, {'entity': 'Berlin', 'entity_type': 'city'},
          {'entity': 'Haystack', 'entity_type': 'product'}]}),
    Document(id=.., content: 'Hugging Face is a company that was founded in New York, USA and is known for its Transformers library',
    meta: {'entities': [
            {'entity': 'Hugging Face', 'entity_type': 'company'}, {'entity': 'New York', 'entity_type': 'city'},
            {'entity': 'USA', 'entity_type': 'country'}, {'entity': 'Transformers', 'entity_type': 'product'}
            ]})
       ]
    'failed_documents': []
   }
>>
```

#### `__init__`

```python
__init__(
    prompt: str,
    chat_generator: ChatGenerator,
    expected_keys: list[str] | None = None,
    page_range: list[str | int] | None = None,
    raise_on_failure: bool = False,
    max_workers: int = 3,
)
```

Initializes the LLMMetadataExtractor.

**Parameters:**

- **prompt** (<code>str</code>) – The prompt to be used for the LLM.
- **chat_generator** (<code>ChatGenerator</code>) – a ChatGenerator instance which represents the LLM. In order for the component to work,
  the LLM should be configured to return a JSON object. For example, when using the OpenAIChatGenerator, you
  should pass `{"response_format": {"type": "json_object"}}` in the `generation_kwargs`.
- **expected_keys** (<code>list\[str\] | None</code>) – The keys expected in the JSON output from the LLM.
- **page_range** (<code>list\[str | int\] | None</code>) – A range of pages to extract metadata from. For example, page_range=['1', '3'] will extract
  metadata from the first and third pages of each document. It also accepts printable range strings, e.g.:
  ['1-3', '5', '8', '10-12'] will extract metadata from pages 1, 2, 3, 5, 8, 10,11, 12.
  If None, metadata will be extracted from the entire document for each document in the documents list.
  This parameter is optional and can be overridden in the `run` method.
- **raise_on_failure** (<code>bool</code>) – Whether to raise an error on failure during the execution of the Generator or
  validation of the JSON output.
- **max_workers** (<code>int</code>) – The maximum number of workers to use in the thread pool executor.

#### `warm_up`

```python
warm_up()
```

Warm up the LLM provider component.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> LLMMetadataExtractor
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary with serialized data.

**Returns:**

- <code>LLMMetadataExtractor</code> – An instance of the component.

#### `run`

```python
run(documents: list[Document], page_range: list[str | int] | None = None)
```

Extract metadata from documents using a Large Language Model.

If `page_range` is provided, the metadata will be extracted from the specified range of pages. This component
will split the documents into pages and extract metadata from the specified range of pages. The metadata will be
extracted from the entire document if `page_range` is not provided.

The original documents will be returned updated with the extracted metadata.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – List of documents to extract metadata from.
- **page_range** (<code>list\[str | int\] | None</code>) – A range of pages to extract metadata from. For example, page_range=['1', '3'] will extract
  metadata from the first and third pages of each document. It also accepts printable range
  strings, e.g.: ['1-3', '5', '8', '10-12'] will extract metadata from pages 1, 2, 3, 5, 8, 10,
  11, 12.
  If None, metadata will be extracted from the entire document for each document in the
  documents list.

**Returns:**

- – A dictionary with the keys:
- "documents": A list of documents that were successfully updated with the extracted metadata.
- "failed_documents": A list of documents that failed to extract metadata. These documents will have
  "metadata_extraction_error" and "metadata_extraction_response" in their metadata. These documents can be
  re-run with the extractor to extract metadata.

## `haystack.components.extractors.named_entity_extractor`

### `NamedEntityExtractorBackend`

Bases: <code>Enum</code>

NLP backend to use for Named Entity Recognition.

#### `from_str`

```python
from_str(string: str) -> NamedEntityExtractorBackend
```

Convert a string to a NamedEntityExtractorBackend enum.

### `NamedEntityAnnotation`

Describes a single NER annotation.

**Parameters:**

- **entity** (<code>str</code>) – Entity label.
- **start** (<code>int</code>) – Start index of the entity in the document.
- **end** (<code>int</code>) – End index of the entity in the document.
- **score** (<code>float | None</code>) – Score calculated by the model.

### `NamedEntityExtractor`

Annotates named entities in a collection of documents.

The component supports two backends: Hugging Face and spaCy. The
former can be used with any sequence classification model from the
[Hugging Face model hub](https://huggingface.co/models), while the
latter can be used with any [spaCy model](https://spacy.io/models)
that contains an NER component. Annotations are stored as metadata
in the documents.

Usage example:

```python
from haystack import Document
from haystack.components.extractors.named_entity_extractor import NamedEntityExtractor

documents = [
    Document(content="I'm Merlin, the happy pig!"),
    Document(content="My name is Clara and I live in Berkeley, California."),
]
extractor = NamedEntityExtractor(backend="hugging_face", model="dslim/bert-base-NER")
extractor.warm_up()
results = extractor.run(documents=documents)["documents"]
annotations = [NamedEntityExtractor.get_stored_annotations(doc) for doc in results]
print(annotations)
```

#### `__init__`

```python
__init__(
    *,
    backend: str | NamedEntityExtractorBackend,
    model: str,
    pipeline_kwargs: dict[str, Any] | None = None,
    device: ComponentDevice | None = None,
    token: Secret | None = Secret.from_env_var(
        ["HF_API_TOKEN", "HF_TOKEN"], strict=False
    )
) -> None
```

Create a Named Entity extractor component.

**Parameters:**

- **backend** (<code>str | NamedEntityExtractorBackend</code>) – Backend to use for NER.
- **model** (<code>str</code>) – Name of the model or a path to the model on
  the local disk. Dependent on the backend.
- **pipeline_kwargs** (<code>dict\[str, Any\] | None</code>) – Keyword arguments passed to the pipeline. The
  pipeline can override these arguments. Dependent on the backend.
- **device** (<code>ComponentDevice | None</code>) – The device on which the model is loaded. If `None`,
  the default device is automatically selected. If a
  device/device map is specified in `pipeline_kwargs`,
  it overrides this parameter (only applicable to the
  HuggingFace backend).
- **token** (<code>Secret | None</code>) – The API token to download private models from Hugging Face.

#### `warm_up`

```python
warm_up()
```

Initialize the component.

**Raises:**

- <code>ComponentError</code> – If the backend fails to initialize successfully.

#### `run`

```python
run(documents: list[Document], batch_size: int = 1) -> dict[str, Any]
```

Annotate named entities in each document and store the annotations in the document's metadata.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – Documents to process.
- **batch_size** (<code>int</code>) – Batch size used for processing the documents.

**Returns:**

- <code>dict\[str, Any\]</code> – Processed documents.

**Raises:**

- <code>ComponentError</code> – If the backend fails to process a document.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> NamedEntityExtractor
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>NamedEntityExtractor</code> – Deserialized component.

#### `initialized`

```python
initialized: bool
```

Returns if the extractor is ready to annotate text.

#### `get_stored_annotations`

```python
get_stored_annotations(
    document: Document,
) -> list[NamedEntityAnnotation] | None
```

Returns the document's named entity annotations stored in its metadata, if any.

**Parameters:**

- **document** (<code>Document</code>) – Document whose annotations are to be fetched.

**Returns:**

- <code>list\[NamedEntityAnnotation\] | None</code> – The stored annotations.

## `haystack.components.extractors.regex_text_extractor`

### `RegexTextExtractor`

Extracts text from chat message or string input using a regex pattern.

RegexTextExtractor parses input text or ChatMessages using a provided regular expression pattern.
It can be configured to search through all messages or only the last message in a list of ChatMessages.

### Usage example

```python
from haystack.components.extractors import RegexTextExtractor
from haystack.dataclasses import ChatMessage

# Using with a string
parser = RegexTextExtractor(regex_pattern='<issue url="(.+)">')
result = parser.run(text_or_messages='<issue url="github.com/hahahaha">hahahah</issue>')
# result: {"captured_text": "github.com/hahahaha"}

# Using with ChatMessages
messages = [ChatMessage.from_user('<issue url="github.com/hahahaha">hahahah</issue>')]
result = parser.run(text_or_messages=messages)
# result: {"captured_text": "github.com/hahahaha"}
```

#### `__init__`

```python
__init__(regex_pattern: str)
```

Creates an instance of the RegexTextExtractor component.

**Parameters:**

- **regex_pattern** (<code>str</code>) – The regular expression pattern used to extract text.
  The pattern should include a capture group to extract the desired text.
  Example: `'<issue url="(.+)">'` captures `'github.com/hahahaha'` from `'<issue url="github.com/hahahaha">'`.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> RegexTextExtractor
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to deserialize from.

**Returns:**

- <code>RegexTextExtractor</code> – The deserialized component.

#### `run`

```python
run(text_or_messages: str | list[ChatMessage]) -> dict[str, str]
```

Extracts text from input using the configured regex pattern.

**Parameters:**

- **text_or_messages** (<code>str | list\[ChatMessage\]</code>) – Either a string or a list of ChatMessage objects to search through.

**Returns:**

- <code>dict\[str, str\]</code> – - `{"captured_text": "matched text"}` if a match is found
- `{"captured_text": ""}` if no match is found

**Raises:**

- <code>ValueError</code> – if receiving a list the last element is not a ChatMessage instance.
