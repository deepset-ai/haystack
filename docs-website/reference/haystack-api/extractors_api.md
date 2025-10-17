---
title: Extractors
excerpt: Components to extract specific elements from textual data.
category: placeholder-haystack-api
slug: extractors-api
parentDoc: 
order: 65
hidden: false
---

<a id="named_entity_extractor"></a>

# Module named\_entity\_extractor

<a id="named_entity_extractor.NamedEntityExtractorBackend"></a>

## NamedEntityExtractorBackend

NLP backend to use for Named Entity Recognition.

<a id="named_entity_extractor.NamedEntityExtractorBackend.HUGGING_FACE"></a>

#### HUGGING\_FACE

Uses an Hugging Face model and pipeline.

<a id="named_entity_extractor.NamedEntityExtractorBackend.SPACY"></a>

#### SPACY

Uses a spaCy model and pipeline.

<a id="named_entity_extractor.NamedEntityExtractorBackend.from_str"></a>

#### NamedEntityExtractorBackend.from\_str

```python
@staticmethod
def from_str(string: str) -> "NamedEntityExtractorBackend"
```

Convert a string to a NamedEntityExtractorBackend enum.

<a id="named_entity_extractor.NamedEntityAnnotation"></a>

## NamedEntityAnnotation

Describes a single NER annotation.

**Arguments**:

- `entity`: Entity label.
- `start`: Start index of the entity in the document.
- `end`: End index of the entity in the document.
- `score`: Score calculated by the model.

<a id="named_entity_extractor.NamedEntityExtractor"></a>

## NamedEntityExtractor

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

<a id="named_entity_extractor.NamedEntityExtractor.__init__"></a>

#### NamedEntityExtractor.\_\_init\_\_

```python
def __init__(
    *,
    backend: Union[str, NamedEntityExtractorBackend],
    model: str,
    pipeline_kwargs: Optional[dict[str, Any]] = None,
    device: Optional[ComponentDevice] = None,
    token: Optional[Secret] = Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"],
                                                  strict=False)
) -> None
```

Create a Named Entity extractor component.

**Arguments**:

- `backend`: Backend to use for NER.
- `model`: Name of the model or a path to the model on
the local disk. Dependent on the backend.
- `pipeline_kwargs`: Keyword arguments passed to the pipeline. The
pipeline can override these arguments. Dependent on the backend.
- `device`: The device on which the model is loaded. If `None`,
the default device is automatically selected. If a
device/device map is specified in `pipeline_kwargs`,
it overrides this parameter (only applicable to the
HuggingFace backend).
- `token`: The API token to download private models from Hugging Face.

<a id="named_entity_extractor.NamedEntityExtractor.warm_up"></a>

#### NamedEntityExtractor.warm\_up

```python
def warm_up()
```

Initialize the component.

**Raises**:

- `ComponentError`: If the backend fails to initialize successfully.

<a id="named_entity_extractor.NamedEntityExtractor.run"></a>

#### NamedEntityExtractor.run

```python
@component.output_types(documents=list[Document])
def run(documents: list[Document], batch_size: int = 1) -> dict[str, Any]
```

Annotate named entities in each document and store the annotations in the document's metadata.

**Arguments**:

- `documents`: Documents to process.
- `batch_size`: Batch size used for processing the documents.

**Raises**:

- `ComponentError`: If the backend fails to process a document.

**Returns**:

Processed documents.

<a id="named_entity_extractor.NamedEntityExtractor.to_dict"></a>

#### NamedEntityExtractor.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="named_entity_extractor.NamedEntityExtractor.from_dict"></a>

#### NamedEntityExtractor.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "NamedEntityExtractor"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="named_entity_extractor.NamedEntityExtractor.initialized"></a>

#### NamedEntityExtractor.initialized

```python
@property
def initialized() -> bool
```

Returns if the extractor is ready to annotate text.

<a id="named_entity_extractor.NamedEntityExtractor.get_stored_annotations"></a>

#### NamedEntityExtractor.get\_stored\_annotations

```python
@classmethod
def get_stored_annotations(
        cls, document: Document) -> Optional[list[NamedEntityAnnotation]]
```

Returns the document's named entity annotations stored in its metadata, if any.

**Arguments**:

- `document`: Document whose annotations are to be fetched.

**Returns**:

The stored annotations.

<a id="llm_metadata_extractor"></a>

# Module llm\_metadata\_extractor

<a id="llm_metadata_extractor.LLMMetadataExtractor"></a>

## LLMMetadataExtractor

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
        "response_format": {"type": "json_object"},
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

<a id="llm_metadata_extractor.LLMMetadataExtractor.__init__"></a>

#### LLMMetadataExtractor.\_\_init\_\_

```python
def __init__(prompt: str,
             chat_generator: ChatGenerator,
             expected_keys: Optional[list[str]] = None,
             page_range: Optional[list[Union[str, int]]] = None,
             raise_on_failure: bool = False,
             max_workers: int = 3)
```

Initializes the LLMMetadataExtractor.

**Arguments**:

- `prompt`: The prompt to be used for the LLM.
- `chat_generator`: a ChatGenerator instance which represents the LLM. In order for the component to work,
the LLM should be configured to return a JSON object. For example, when using the OpenAIChatGenerator, you
should pass `{"response_format": {"type": "json_object"}}` in the `generation_kwargs`.
- `expected_keys`: The keys expected in the JSON output from the LLM.
- `page_range`: A range of pages to extract metadata from. For example, page_range=['1', '3'] will extract
metadata from the first and third pages of each document. It also accepts printable range strings, e.g.:
['1-3', '5', '8', '10-12'] will extract metadata from pages 1, 2, 3, 5, 8, 10,11, 12.
If None, metadata will be extracted from the entire document for each document in the documents list.
This parameter is optional and can be overridden in the `run` method.
- `raise_on_failure`: Whether to raise an error on failure during the execution of the Generator or
validation of the JSON output.
- `max_workers`: The maximum number of workers to use in the thread pool executor.

<a id="llm_metadata_extractor.LLMMetadataExtractor.warm_up"></a>

#### LLMMetadataExtractor.warm\_up

```python
def warm_up()
```

Warm up the LLM provider component.

<a id="llm_metadata_extractor.LLMMetadataExtractor.to_dict"></a>

#### LLMMetadataExtractor.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="llm_metadata_extractor.LLMMetadataExtractor.from_dict"></a>

#### LLMMetadataExtractor.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "LLMMetadataExtractor"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary with serialized data.

**Returns**:

An instance of the component.

<a id="llm_metadata_extractor.LLMMetadataExtractor.run"></a>

#### LLMMetadataExtractor.run

```python
@component.output_types(documents=list[Document],
                        failed_documents=list[Document])
def run(documents: list[Document],
        page_range: Optional[list[Union[str, int]]] = None)
```

Extract metadata from documents using a Large Language Model.

If `page_range` is provided, the metadata will be extracted from the specified range of pages. This component
will split the documents into pages and extract metadata from the specified range of pages. The metadata will be
extracted from the entire document if `page_range` is not provided.

The original documents will be returned  updated with the extracted metadata.

**Arguments**:

- `documents`: List of documents to extract metadata from.
- `page_range`: A range of pages to extract metadata from. For example, page_range=['1', '3'] will extract
metadata from the first and third pages of each document. It also accepts printable range
strings, e.g.: ['1-3', '5', '8', '10-12'] will extract metadata from pages 1, 2, 3, 5, 8, 10,
11, 12.
If None, metadata will be extracted from the entire document for each document in the
documents list.

**Returns**:

A dictionary with the keys:
- "documents": A list of documents that were successfully updated with the extracted metadata.
- "failed_documents": A list of documents that failed to extract metadata. These documents will have
"metadata_extraction_error" and "metadata_extraction_response" in their metadata. These documents can be
re-run with the extractor to extract metadata.

<a id="image/llm_document_content_extractor"></a>

# Module image/llm\_document\_content\_extractor

<a id="image/llm_document_content_extractor.LLMDocumentContentExtractor"></a>

## LLMDocumentContentExtractor

Extracts textual content from image-based documents using a vision-enabled LLM (Large Language Model).

This component converts each input document into an image using the DocumentToImageContent component,
uses a prompt to instruct the LLM on how to extract content, and uses a ChatGenerator to extract structured
textual content based on the provided prompt.

The prompt must not contain variables; it should only include instructions for the LLM. Image data and the prompt
are passed together to the LLM as a chat message.

Documents for which the LLM fails to extract content are returned in a separate `failed_documents` list. These
failed documents will have a `content_extraction_error` entry in their metadata. This metadata can be used for
debugging or for reprocessing the documents later.

### Usage example
```python
from haystack import Document
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.extractors.image import LLMDocumentContentExtractor
chat_generator = OpenAIChatGenerator()
extractor = LLMDocumentContentExtractor(chat_generator=chat_generator)
documents = [
    Document(content="", meta={"file_path": "image.jpg"}),
    Document(content="", meta={"file_path": "document.pdf", "page_number": 1}),
]
updated_documents = extractor.run(documents=documents)["documents"]
print(updated_documents)
# [Document(content='Extracted text from image.jpg',
#           meta={'file_path': 'image.jpg'}),
#  ...]
```

<a id="image/llm_document_content_extractor.LLMDocumentContentExtractor.__init__"></a>

#### LLMDocumentContentExtractor.\_\_init\_\_

```python
def __init__(*,
             chat_generator: ChatGenerator,
             prompt: str = DEFAULT_PROMPT_TEMPLATE,
             file_path_meta_field: str = "file_path",
             root_path: Optional[str] = None,
             detail: Optional[Literal["auto", "high", "low"]] = None,
             size: Optional[tuple[int, int]] = None,
             raise_on_failure: bool = False,
             max_workers: int = 3)
```

Initialize the LLMDocumentContentExtractor component.

**Arguments**:

- `chat_generator`: A ChatGenerator instance representing the LLM used to extract text. This generator must
support vision-based input and return a plain text response.
- `prompt`: Instructional text provided to the LLM. It must not contain Jinja variables.
The prompt should only contain instructions on how to extract the content of the image-based document.
- `file_path_meta_field`: The metadata field in the Document that contains the file path to the image or PDF.
- `root_path`: The root directory path where document files are located. If provided, file paths in
document metadata will be resolved relative to this path. If None, file paths are treated as absolute paths.
- `detail`: Optional detail level of the image (only supported by OpenAI). Can be "auto", "high", or "low".
This will be passed to chat_generator when processing the images.
- `size`: If provided, resizes the image to fit within the specified dimensions (width, height) while
maintaining aspect ratio. This reduces file size, memory usage, and processing time, which is beneficial
when working with models that have resolution constraints or when transmitting images to remote services.
- `raise_on_failure`: If True, exceptions from the LLM are raised. If False, failed documents are logged
and returned.
- `max_workers`: Maximum number of threads used to parallelize LLM calls across documents using a
ThreadPoolExecutor.

<a id="image/llm_document_content_extractor.LLMDocumentContentExtractor.warm_up"></a>

#### LLMDocumentContentExtractor.warm\_up

```python
def warm_up()
```

Warm up the ChatGenerator if it has a warm_up method.

<a id="image/llm_document_content_extractor.LLMDocumentContentExtractor.to_dict"></a>

#### LLMDocumentContentExtractor.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="image/llm_document_content_extractor.LLMDocumentContentExtractor.from_dict"></a>

#### LLMDocumentContentExtractor.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "LLMDocumentContentExtractor"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary with serialized data.

**Returns**:

An instance of the component.

<a id="image/llm_document_content_extractor.LLMDocumentContentExtractor.run"></a>

#### LLMDocumentContentExtractor.run

```python
@component.output_types(documents=list[Document],
                        failed_documents=list[Document])
def run(documents: list[Document]) -> dict[str, list[Document]]
```

Run content extraction on a list of image-based documents using a vision-capable LLM.

Each document is passed to the LLM along with a predefined prompt. The response is used to update the document's
content. If the extraction fails, the document is returned in the `failed_documents` list with metadata
describing the failure.

**Arguments**:

- `documents`: A list of image-based documents to process. Each must have a valid file path in its metadata.

**Returns**:

A dictionary with:
- "documents": Successfully processed documents, updated with extracted content.
- "failed_documents": Documents that failed processing, annotated with failure metadata.

<a id="regex_text_extractor"></a>

# Module regex\_text\_extractor

<a id="regex_text_extractor.RegexTextExtractor"></a>

## RegexTextExtractor

Extracts text from chat message or string input using a regex pattern.

RegexTextExtractor parses input text or ChatMessages using a provided regular expression pattern.
It can be configured to search through all messages or only the last message in a list of ChatMessages.

### Usage example

```python
from haystack_experimental.components.extractors import RegexTextExtractor
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

<a id="regex_text_extractor.RegexTextExtractor.__init__"></a>

#### RegexTextExtractor.\_\_init\_\_

```python
def __init__(regex_pattern: str)
```

Creates an instance of the RegexTextExtractor component.

**Arguments**:

- `regex_pattern`: The regular expression pattern used to extract text.
The pattern should include a capture group to extract the desired text.
Example: '<issue url="(.+)">' captures 'github.com/hahahaha' from '<issue url="github.com/hahahaha">'.

<a id="regex_text_extractor.RegexTextExtractor.run"></a>

#### RegexTextExtractor.run

```python
@component.output_types(captured_text=str, captured_texts=list[str])
def run(text_or_messages: Union[str, list[ChatMessage]]) -> dict
```

Extracts text from input using the configured regex pattern.

**Arguments**:

- `text_or_messages`: Either a string or a list of ChatMessage objects to search through.

**Raises**:

- `None`: - ValueError: if receiving a list the last element is not a ChatMessage instance.

**Returns**:

- If match found: {"captured_text": "matched text"}
- If no match and return_empty_on_no_match=True: {}

