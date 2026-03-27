---
title: Routers
id: routers-api
description: Routers is a group of components that route queries or Documents to other components that can handle them best.
slug: "/routers-api"
---

<a id="conditional_router"></a>

# Module conditional\_router

<a id="conditional_router.NoRouteSelectedException"></a>

## NoRouteSelectedException

Exception raised when no route is selected in ConditionalRouter.

<a id="conditional_router.RouteConditionException"></a>

## RouteConditionException

Exception raised when there is an error parsing or evaluating the condition expression in ConditionalRouter.

<a id="conditional_router.ConditionalRouter"></a>

## ConditionalRouter

Routes data based on specific conditions.

You define these conditions in a list of dictionaries called `routes`.
Each dictionary in this list represents a single route. Each route has these four elements:
- `condition`: A Jinja2 string expression that determines if the route is selected.
- `output`: A Jinja2 expression defining the route's output value.
- `output_type`: The type of the output data (for example, `str`, `list[int]`).
- `output_name`: The name you want to use to publish `output`. This name is used to connect
the router to other components in the pipeline.

### Usage example

```python
from haystack.components.routers import ConditionalRouter

routes = [
    {
        "condition": "{{streams|length > 2}}",
        "output": "{{streams}}",
        "output_name": "enough_streams",
        "output_type": list[int],
    },
    {
        "condition": "{{streams|length <= 2}}",
        "output": "{{streams}}",
        "output_name": "insufficient_streams",
        "output_type": list[int],
    },
]
router = ConditionalRouter(routes)
# When 'streams' has more than 2 items, 'enough_streams' output will activate, emitting the list [1, 2, 3]
kwargs = {"streams": [1, 2, 3], "query": "Haystack"}
result = router.run(**kwargs)
assert result == {"enough_streams": [1, 2, 3]}
```

In this example, we configure two routes. The first route sends the 'streams' value to 'enough_streams' if the
stream count exceeds two. The second route directs 'streams' to 'insufficient_streams' if there
are two or fewer streams.

In the pipeline setup, the Router connects to other components using the output names. For example,
'enough_streams' might connect to a component that processes streams, while
'insufficient_streams' might connect to a component that fetches more streams.


Here is a pipeline that uses `ConditionalRouter` and routes the fetched `ByteStreams` to
different components depending on the number of streams fetched:

```python
from haystack import Pipeline
from haystack.dataclasses import ByteStream
from haystack.components.routers import ConditionalRouter

routes = [
    {
        "condition": "{{streams|length > 2}}",
        "output": "{{streams}}",
        "output_name": "enough_streams",
        "output_type": list[ByteStream],
    },
    {
        "condition": "{{streams|length <= 2}}",
        "output": "{{streams}}",
        "output_name": "insufficient_streams",
        "output_type": list[ByteStream],
    },
]

pipe = Pipeline()
pipe.add_component("router", router)
...
pipe.connect("router.enough_streams", "some_component_a.streams")
pipe.connect("router.insufficient_streams", "some_component_b.streams_or_some_other_input")
...
```

<a id="conditional_router.ConditionalRouter.__init__"></a>

#### ConditionalRouter.\_\_init\_\_

```python
def __init__(routes: list[Route],
             custom_filters: Optional[dict[str, Callable]] = None,
             unsafe: bool = False,
             validate_output_type: bool = False,
             optional_variables: Optional[list[str]] = None)
```

Initializes the `ConditionalRouter` with a list of routes detailing the conditions for routing.

**Arguments**:

- `routes`: A list of dictionaries, each defining a route.
Each route has these four elements:
- `condition`: A Jinja2 string expression that determines if the route is selected.
- `output`: A Jinja2 expression defining the route's output value.
- `output_type`: The type of the output data (for example, `str`, `list[int]`).
- `output_name`: The name you want to use to publish `output`. This name is used to connect
the router to other components in the pipeline.
- `custom_filters`: A dictionary of custom Jinja2 filters used in the condition expressions.
For example, passing `{"my_filter": my_filter_fcn}` where:
- `my_filter` is the name of the custom filter.
- `my_filter_fcn` is a callable that takes `my_var:str` and returns `my_var[:3]`.
  `{{ my_var|my_filter }}` can then be used inside a route condition expression:
    `"condition": "{{ my_var|my_filter == 'foo' }}"`.
- `unsafe`: Enable execution of arbitrary code in the Jinja template.
This should only be used if you trust the source of the template as it can be lead to remote code execution.
- `validate_output_type`: Enable validation of routes' output.
If a route output doesn't match the declared type a ValueError is raised running.
- `optional_variables`: A list of variable names that are optional in your route conditions and outputs.
If these variables are not provided at runtime, they will be set to `None`.
This allows you to write routes that can handle missing inputs gracefully without raising errors.

Example usage with a default fallback route in a Pipeline:
```python
from haystack import Pipeline
from haystack.components.routers import ConditionalRouter

routes = [
    {
        "condition": '{{ path == "rag" }}',
        "output": "{{ question }}",
        "output_name": "rag_route",
        "output_type": str
    },
    {
        "condition": "{{ True }}",  # fallback route
        "output": "{{ question }}",
        "output_name": "default_route",
        "output_type": str
    }
]

router = ConditionalRouter(routes, optional_variables=["path"])
pipe = Pipeline()
pipe.add_component("router", router)

# When 'path' is provided in the pipeline:
result = pipe.run(data={"router": {"question": "What?", "path": "rag"}})
assert result["router"] == {"rag_route": "What?"}

# When 'path' is not provided, fallback route is taken:
result = pipe.run(data={"router": {"question": "What?"}})
assert result["router"] == {"default_route": "What?"}
```

This pattern is particularly useful when:
- You want to provide default/fallback behavior when certain inputs are missing
- Some variables are only needed for specific routing conditions
- You're building flexible pipelines where not all inputs are guaranteed to be present

<a id="conditional_router.ConditionalRouter.to_dict"></a>

#### ConditionalRouter.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="conditional_router.ConditionalRouter.from_dict"></a>

#### ConditionalRouter.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "ConditionalRouter"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: The dictionary to deserialize from.

**Returns**:

The deserialized component.

<a id="conditional_router.ConditionalRouter.run"></a>

#### ConditionalRouter.run

```python
def run(**kwargs)
```

Executes the routing logic.

Executes the routing logic by evaluating the specified boolean condition expressions for each route in the
order they are listed. The method directs the flow of data to the output specified in the first route whose
`condition` is True.

**Arguments**:

- `kwargs`: All variables used in the `condition` expressed in the routes. When the component is used in a
pipeline, these variables are passed from the previous component's output.

**Raises**:

- `NoRouteSelectedException`: If no `condition' in the routes is `True`.
- `RouteConditionException`: If there is an error parsing or evaluating the `condition` expression in the routes.
- `ValueError`: If type validation is enabled and route type doesn't match actual value type.

**Returns**:

A dictionary where the key is the `output_name` of the selected route and the value is the `output`
of the selected route.

<a id="document_length_router"></a>

# Module document\_length\_router

<a id="document_length_router.DocumentLengthRouter"></a>

## DocumentLengthRouter

Categorizes documents based on the length of the `content` field and routes them to the appropriate output.

A common use case for DocumentLengthRouter is handling documents obtained from PDFs that contain non-text
content, such as scanned pages or images. This component can detect empty or low-content documents and route them to
components that perform OCR, generate captions, or compute image embeddings.

### Usage example

```python
from haystack.components.routers import DocumentLengthRouter
from haystack.dataclasses import Document

docs = [
    Document(content="Short"),
    Document(content="Long document "*20),
]

router = DocumentLengthRouter(threshold=10)

result = router.run(documents=docs)
print(result)

# {
#     "short_documents": [Document(content="Short", ...)],
#     "long_documents": [Document(content="Long document ...", ...)],
# }
```

<a id="document_length_router.DocumentLengthRouter.__init__"></a>

#### DocumentLengthRouter.\_\_init\_\_

```python
def __init__(*, threshold: int = 10) -> None
```

Initialize the DocumentLengthRouter component.

**Arguments**:

- `threshold`: The threshold for the number of characters in the document `content` field. Documents where `content` is
None or whose character count is less than or equal to the threshold will be routed to the `short_documents`
output. Otherwise, they will be routed to the `long_documents` output.
To route only documents with None content to `short_documents`, set the threshold to a negative number.

<a id="document_length_router.DocumentLengthRouter.run"></a>

#### DocumentLengthRouter.run

```python
@component.output_types(short_documents=list[Document],
                        long_documents=list[Document])
def run(documents: list[Document]) -> dict[str, list[Document]]
```

Categorize input documents into groups based on the length of the `content` field.

**Arguments**:

- `documents`: A list of documents to be categorized.

**Returns**:

A dictionary with the following keys:
- `short_documents`: A list of documents where `content` is None or the length of `content` is less than or
   equal to the threshold.
- `long_documents`: A list of documents where the length of `content` is greater than the threshold.

<a id="document_type_router"></a>

# Module document\_type\_router

<a id="document_type_router.DocumentTypeRouter"></a>

## DocumentTypeRouter

Routes documents by their MIME types.

DocumentTypeRouter is used to dynamically route documents within a pipeline based on their MIME types.
It supports exact MIME type matches and regex patterns.

MIME types can be extracted directly from document metadata or inferred from file paths using standard or
user-supplied MIME type mappings.

### Usage example

```python
from haystack.components.routers import DocumentTypeRouter
from haystack.dataclasses import Document

docs = [
    Document(content="Example text", meta={"file_path": "example.txt"}),
    Document(content="Another document", meta={"mime_type": "application/pdf"}),
    Document(content="Unknown type")
]

router = DocumentTypeRouter(
    mime_type_meta_field="mime_type",
    file_path_meta_field="file_path",
    mime_types=["text/plain", "application/pdf"]
)

result = router.run(documents=docs)
print(result)
```

Expected output:
```python
{
    "text/plain": [Document(...)],
    "application/pdf": [Document(...)],
    "unclassified": [Document(...)]
}
```

<a id="document_type_router.DocumentTypeRouter.__init__"></a>

#### DocumentTypeRouter.\_\_init\_\_

```python
def __init__(*,
             mime_types: list[str],
             mime_type_meta_field: Optional[str] = None,
             file_path_meta_field: Optional[str] = None,
             additional_mimetypes: Optional[dict[str, str]] = None) -> None
```

Initialize the DocumentTypeRouter component.

**Arguments**:

- `mime_types`: A list of MIME types or regex patterns to classify the input documents.
(for example: `["text/plain", "audio/x-wav", "image/jpeg"]`).
- `mime_type_meta_field`: Optional name of the metadata field that holds the MIME type.
- `file_path_meta_field`: Optional name of the metadata field that holds the file path. Used to infer the MIME type if
`mime_type_meta_field` is not provided or missing in a document.
- `additional_mimetypes`: Optional dictionary mapping MIME types to file extensions to enhance or override the standard
`mimetypes` module. Useful when working with uncommon or custom file types.
For example: `{"application/vnd.custom-type": ".custom"}`.

**Raises**:

- `ValueError`: If `mime_types` is empty or if both `mime_type_meta_field` and `file_path_meta_field` are
not provided.

<a id="document_type_router.DocumentTypeRouter.run"></a>

#### DocumentTypeRouter.run

```python
def run(documents: list[Document]) -> dict[str, list[Document]]
```

Categorize input documents into groups based on their MIME type.

MIME types can either be directly available in document metadata or derived from file paths using the
standard Python `mimetypes` module and custom mappings.

**Arguments**:

- `documents`: A list of documents to be categorized.

**Returns**:

A dictionary where the keys are MIME types (or `"unclassified"`) and the values are lists of documents.

<a id="file_type_router"></a>

# Module file\_type\_router

<a id="file_type_router.FileTypeRouter"></a>

## FileTypeRouter

Categorizes files or byte streams by their MIME types, helping in context-based routing.

FileTypeRouter supports both exact MIME type matching and regex patterns.

For file paths, MIME types come from extensions, while byte streams use metadata.
You can use regex patterns in the `mime_types` parameter to set broad categories
(such as 'audio/*' or 'text/*') or specific types.
MIME types without regex patterns are treated as exact matches.

### Usage example

```python
from haystack.components.routers import FileTypeRouter
from pathlib import Path

# For exact MIME type matching
router = FileTypeRouter(mime_types=["text/plain", "application/pdf"])

# For flexible matching using regex, to handle all audio types
router_with_regex = FileTypeRouter(mime_types=[r"audio/.*", r"text/plain"])

sources = [Path("file.txt"), Path("document.pdf"), Path("song.mp3")]
print(router.run(sources=sources))
print(router_with_regex.run(sources=sources))

# Expected output:
# {'text/plain': [
#   PosixPath('file.txt')], 'application/pdf': [PosixPath('document.pdf')], 'unclassified': [PosixPath('song.mp3')
# ]}
# {'audio/.*': [
#   PosixPath('song.mp3')], 'text/plain': [PosixPath('file.txt')], 'unclassified': [PosixPath('document.pdf')
# ]}
```

<a id="file_type_router.FileTypeRouter.__init__"></a>

#### FileTypeRouter.\_\_init\_\_

```python
def __init__(mime_types: list[str],
             additional_mimetypes: Optional[dict[str, str]] = None,
             raise_on_failure: bool = False)
```

Initialize the FileTypeRouter component.

**Arguments**:

- `mime_types`: A list of MIME types or regex patterns to classify the input files or byte streams.
(for example: `["text/plain", "audio/x-wav", "image/jpeg"]`).
- `additional_mimetypes`: A dictionary containing the MIME type to add to the mimetypes package to prevent unsupported or non-native
packages from being unclassified.
(for example: `{"application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx"}`).
- `raise_on_failure`: If True, raises FileNotFoundError when a file path doesn't exist.
If False (default), only emits a warning when a file path doesn't exist.

<a id="file_type_router.FileTypeRouter.to_dict"></a>

#### FileTypeRouter.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="file_type_router.FileTypeRouter.from_dict"></a>

#### FileTypeRouter.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "FileTypeRouter"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: The dictionary to deserialize from.

**Returns**:

The deserialized component.

<a id="file_type_router.FileTypeRouter.run"></a>

#### FileTypeRouter.run

```python
def run(
    sources: list[Union[str, Path, ByteStream]],
    meta: Optional[Union[dict[str, Any], list[dict[str, Any]]]] = None
) -> dict[str, list[Union[ByteStream, Path]]]
```

Categorize files or byte streams according to their MIME types.

**Arguments**:

- `sources`: A list of file paths or byte streams to categorize.
- `meta`: Optional metadata to attach to the sources.
When provided, the sources are internally converted to ByteStream objects and the metadata is added.
This value can be a list of dictionaries or a single dictionary.
If it's a single dictionary, its content is added to the metadata of all ByteStream objects.
If it's a list, its length must match the number of sources, as they are zipped together.

**Returns**:

A dictionary where the keys are MIME types and the values are lists of data sources.
Two extra keys may be returned: `"unclassified"` when a source's MIME type doesn't match any pattern
and `"failed"` when a source cannot be processed (for example, a file path that doesn't exist).

<a id="llm_messages_router"></a>

# Module llm\_messages\_router

<a id="llm_messages_router.LLMMessagesRouter"></a>

## LLMMessagesRouter

Routes Chat Messages to different connections using a generative Language Model to perform classification.

    This component can be used with general-purpose LLMs and with specialized LLMs for moderation like Llama Guard.

    ### Usage example
    ```python
    from haystack.components.generators.chat import HuggingFaceAPIChatGenerator
    from haystack.components.routers.llm_messages_router import LLMMessagesRouter
    from haystack.dataclasses import ChatMessage

    # initialize a Chat Generator with a generative model for moderation
    chat_generator = HuggingFaceAPIChatGenerator(
        api_type="serverless_inference_api",
        api_params={"model": "meta-llama/Llama-Guard-4-12B", "provider": "groq"},
    )

    router = LLMMessagesRouter(chat_generator=chat_generator,
                                output_names=["unsafe", "safe"],
                                output_patterns=["unsafe", "safe"])


    print(router.run([ChatMessage.from_user("How to rob a bank?")]))

    # {
    #     'chat_generator_text': 'unsafe
S2',
    #     'unsafe': [
    #         ChatMessage(
    #             _role=<ChatRole.USER: 'user'>,
    #             _content=[TextContent(text='How to rob a bank?')],
    #             _name=None,
    #             _meta={}
    #         )
    #     ]
    # }
    ```

<a id="llm_messages_router.LLMMessagesRouter.__init__"></a>

#### LLMMessagesRouter.\_\_init\_\_

```python
def __init__(chat_generator: ChatGenerator,
             output_names: list[str],
             output_patterns: list[str],
             system_prompt: Optional[str] = None)
```

Initialize the LLMMessagesRouter component.

**Arguments**:

- `chat_generator`: A ChatGenerator instance which represents the LLM.
- `output_names`: A list of output connection names. These can be used to connect the router to other
components.
- `output_patterns`: A list of regular expressions to be matched against the output of the LLM. Each pattern
corresponds to an output name. Patterns are evaluated in order.
When using moderation models, refer to the model card to understand the expected outputs.
- `system_prompt`: An optional system prompt to customize the behavior of the LLM.
For moderation models, refer to the model card for supported customization options.

**Raises**:

- `ValueError`: If output_names and output_patterns are not non-empty lists of the same length.

<a id="llm_messages_router.LLMMessagesRouter.warm_up"></a>

#### LLMMessagesRouter.warm\_up

```python
def warm_up()
```

Warm up the underlying LLM.

<a id="llm_messages_router.LLMMessagesRouter.run"></a>

#### LLMMessagesRouter.run

```python
def run(messages: list[ChatMessage]
        ) -> dict[str, Union[str, list[ChatMessage]]]
```

Classify the messages based on LLM output and route them to the appropriate output connection.

**Arguments**:

- `messages`: A list of ChatMessages to be routed. Only user and assistant messages are supported.

**Raises**:

- `ValueError`: If messages is an empty list or contains messages with unsupported roles.
- `RuntimeError`: If the component is not warmed up and the ChatGenerator has a warm_up method.

**Returns**:

A dictionary with the following keys:
- "chat_generator_text": The text output of the LLM, useful for debugging.
- "output_names": Each contains the list of messages that matched the corresponding pattern.
- "unmatched": The messages that did not match any of the output patterns.

<a id="llm_messages_router.LLMMessagesRouter.to_dict"></a>

#### LLMMessagesRouter.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serialize this component to a dictionary.

**Returns**:

The serialized component as a dictionary.

<a id="llm_messages_router.LLMMessagesRouter.from_dict"></a>

#### LLMMessagesRouter.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "LLMMessagesRouter"
```

Deserialize this component from a dictionary.

**Arguments**:

- `data`: The dictionary representation of this component.

**Returns**:

The deserialized component instance.

<a id="metadata_router"></a>

# Module metadata\_router

<a id="metadata_router.MetadataRouter"></a>

## MetadataRouter

Routes documents or byte streams to different connections based on their metadata fields.

Specify the routing rules in the `init` method.
If a document or byte stream does not match any of the rules, it's routed to a connection named "unmatched".


### Usage examples

**Routing Documents by metadata:**
```python
from haystack import Document
from haystack.components.routers import MetadataRouter

docs = [Document(content="Paris is the capital of France.", meta={"language": "en"}),
        Document(content="Berlin ist die Haupststadt von Deutschland.", meta={"language": "de"})]

router = MetadataRouter(rules={"en": {"field": "meta.language", "operator": "==", "value": "en"}})

print(router.run(documents=docs))
# {'en': [Document(id=..., content: 'Paris is the capital of France.', meta: {'language': 'en'})],
# 'unmatched': [Document(id=..., content: 'Berlin ist die Haupststadt von Deutschland.', meta: {'language': 'de'})]}
```

**Routing ByteStreams by metadata:**
```python
from haystack.dataclasses import ByteStream
from haystack.components.routers import MetadataRouter

streams = [
    ByteStream.from_string("Hello world", meta={"language": "en"}),
    ByteStream.from_string("Bonjour le monde", meta={"language": "fr"})
]

router = MetadataRouter(
    rules={"english": {"field": "meta.language", "operator": "==", "value": "en"}},
    output_type=list[ByteStream]
)

result = router.run(documents=streams)
# {'english': [ByteStream(...)], 'unmatched': [ByteStream(...)]}
```

<a id="metadata_router.MetadataRouter.__init__"></a>

#### MetadataRouter.\_\_init\_\_

```python
def __init__(rules: dict[str, dict],
             output_type: type = list[Document]) -> None
```

Initializes the MetadataRouter component.

**Arguments**:

- `rules`: A dictionary defining how to route documents or byte streams to output connections based on their
metadata. Keys are output connection names, and values are dictionaries of
[filtering expressions](https://docs.haystack.deepset.ai/docs/metadata-filtering) in Haystack.
For example:
```python
{
"edge_1": {
    "operator": "AND",
    "conditions": [
        {"field": "meta.created_at", "operator": ">=", "value": "2023-01-01"},
        {"field": "meta.created_at", "operator": "<", "value": "2023-04-01"},
    ],
},
"edge_2": {
    "operator": "AND",
    "conditions": [
        {"field": "meta.created_at", "operator": ">=", "value": "2023-04-01"},
        {"field": "meta.created_at", "operator": "<", "value": "2023-07-01"},
    ],
},
"edge_3": {
    "operator": "AND",
    "conditions": [
        {"field": "meta.created_at", "operator": ">=", "value": "2023-07-01"},
        {"field": "meta.created_at", "operator": "<", "value": "2023-10-01"},
    ],
},
"edge_4": {
    "operator": "AND",
    "conditions": [
        {"field": "meta.created_at", "operator": ">=", "value": "2023-10-01"},
        {"field": "meta.created_at", "operator": "<", "value": "2024-01-01"},
    ],
},
}
```
:param output_type: The type of the output produced. Lists of Documents or ByteStreams can be specified.

<a id="metadata_router.MetadataRouter.run"></a>

#### MetadataRouter.run

```python
def run(documents: Union[list[Document], list[ByteStream]])
```

Routes documents or byte streams to different connections based on their metadata fields.

If a document or byte stream does not match any of the rules, it's routed to a connection named "unmatched".

**Arguments**:

- `documents`: A list of `Document` or `ByteStream` objects to be routed based on their metadata.

**Returns**:

A dictionary where the keys are the names of the output connections (including `"unmatched"`)
and the values are lists of `Document` or `ByteStream` objects that matched the corresponding rules.

<a id="metadata_router.MetadataRouter.to_dict"></a>

#### MetadataRouter.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serialize this component to a dictionary.

**Returns**:

The serialized component as a dictionary.

<a id="metadata_router.MetadataRouter.from_dict"></a>

#### MetadataRouter.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "MetadataRouter"
```

Deserialize this component from a dictionary.

**Arguments**:

- `data`: The dictionary representation of this component.

**Returns**:

The deserialized component instance.

<a id="text_language_router"></a>

# Module text\_language\_router

<a id="text_language_router.TextLanguageRouter"></a>

## TextLanguageRouter

Routes text strings to different output connections based on their language.

Provide a list of languages during initialization. If the document's text doesn't match any of the
specified languages, the metadata value is set to "unmatched".
For routing documents based on their language, use the DocumentLanguageClassifier component,
followed by the MetaDataRouter.

### Usage example

```python
from haystack import Pipeline, Document
from haystack.components.routers import TextLanguageRouter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever

document_store = InMemoryDocumentStore()
document_store.write_documents([Document(content="Elvis Presley was an American singer and actor.")])

p = Pipeline()
p.add_component(instance=TextLanguageRouter(languages=["en"]), name="text_language_router")
p.add_component(instance=InMemoryBM25Retriever(document_store=document_store), name="retriever")
p.connect("text_language_router.en", "retriever.query")

result = p.run({"text_language_router": {"text": "Who was Elvis Presley?"}})
assert result["retriever"]["documents"][0].content == "Elvis Presley was an American singer and actor."

result = p.run({"text_language_router": {"text": "ένα ελληνικό κείμενο"}})
assert result["text_language_router"]["unmatched"] == "ένα ελληνικό κείμενο"
```

<a id="text_language_router.TextLanguageRouter.__init__"></a>

#### TextLanguageRouter.\_\_init\_\_

```python
def __init__(languages: Optional[list[str]] = None)
```

Initialize the TextLanguageRouter component.

**Arguments**:

- `languages`: A list of ISO language codes.
See the supported languages in [`langdetect` documentation](https://github.com/Mimino666/langdetect#languages).
If not specified, defaults to ["en"].

<a id="text_language_router.TextLanguageRouter.run"></a>

#### TextLanguageRouter.run

```python
def run(text: str) -> dict[str, str]
```

Routes the text strings to different output connections based on their language.

If the document's text doesn't match any of the specified languages, the metadata value is set to "unmatched".

**Arguments**:

- `text`: A text string to route.

**Raises**:

- `TypeError`: If the input is not a string.

**Returns**:

A dictionary in which the key is the language (or `"unmatched"`),
and the value is the text.

<a id="transformers_text_router"></a>

# Module transformers\_text\_router

<a id="transformers_text_router.TransformersTextRouter"></a>

## TransformersTextRouter

Routes the text strings to different connections based on a category label.

The labels are specific to each model and can be found it its description on Hugging Face.

### Usage example

```python
from haystack.core.pipeline import Pipeline
from haystack.components.routers import TransformersTextRouter
from haystack.components.builders import PromptBuilder
from haystack.components.generators import HuggingFaceLocalGenerator

p = Pipeline()
p.add_component(
    instance=TransformersTextRouter(model="papluca/xlm-roberta-base-language-detection"),
    name="text_router"
)
p.add_component(
    instance=PromptBuilder(template="Answer the question: {{query}}\nAnswer:"),
    name="english_prompt_builder"
)
p.add_component(
    instance=PromptBuilder(template="Beantworte die Frage: {{query}}\nAntwort:"),
    name="german_prompt_builder"
)

p.add_component(
    instance=HuggingFaceLocalGenerator(model="DiscoResearch/Llama3-DiscoLeo-Instruct-8B-v0.1"),
    name="german_llm"
)
p.add_component(
    instance=HuggingFaceLocalGenerator(model="microsoft/Phi-3-mini-4k-instruct"),
    name="english_llm"
)

p.connect("text_router.en", "english_prompt_builder.query")
p.connect("text_router.de", "german_prompt_builder.query")
p.connect("english_prompt_builder.prompt", "english_llm.prompt")
p.connect("german_prompt_builder.prompt", "german_llm.prompt")

# English Example
print(p.run({"text_router": {"text": "What is the capital of Germany?"}}))

# German Example
print(p.run({"text_router": {"text": "Was ist die Hauptstadt von Deutschland?"}}))
```

<a id="transformers_text_router.TransformersTextRouter.__init__"></a>

#### TransformersTextRouter.\_\_init\_\_

```python
def __init__(model: str,
             labels: Optional[list[str]] = None,
             device: Optional[ComponentDevice] = None,
             token: Optional[Secret] = Secret.from_env_var(
                 ["HF_API_TOKEN", "HF_TOKEN"], strict=False),
             huggingface_pipeline_kwargs: Optional[dict[str, Any]] = None)
```

Initializes the TransformersTextRouter component.

**Arguments**:

- `model`: The name or path of a Hugging Face model for text classification.
- `labels`: The list of labels. If not provided, the component fetches the labels
from the model configuration file hosted on the Hugging Face Hub using
`transformers.AutoConfig.from_pretrained`.
- `device`: The device for loading the model. If `None`, automatically selects the default device.
If a device or device map is specified in `huggingface_pipeline_kwargs`, it overrides this parameter.
- `token`: The API token used to download private models from Hugging Face.
If `True`, uses either `HF_API_TOKEN` or `HF_TOKEN` environment variables.
To generate these tokens, run `transformers-cli login`.
- `huggingface_pipeline_kwargs`: A dictionary of keyword arguments for initializing the Hugging Face
text classification pipeline.

<a id="transformers_text_router.TransformersTextRouter.warm_up"></a>

#### TransformersTextRouter.warm\_up

```python
def warm_up()
```

Initializes the component.

<a id="transformers_text_router.TransformersTextRouter.to_dict"></a>

#### TransformersTextRouter.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="transformers_text_router.TransformersTextRouter.from_dict"></a>

#### TransformersTextRouter.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "TransformersTextRouter"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="transformers_text_router.TransformersTextRouter.run"></a>

#### TransformersTextRouter.run

```python
def run(text: str) -> dict[str, str]
```

Routes the text strings to different connections based on a category label.

**Arguments**:

- `text`: A string of text to route.

**Raises**:

- `TypeError`: If the input is not a str.
- `RuntimeError`: If the pipeline has not been loaded because warm_up() was not called before.

**Returns**:

A dictionary with the label as key and the text as value.

<a id="zero_shot_text_router"></a>

# Module zero\_shot\_text\_router

<a id="zero_shot_text_router.TransformersZeroShotTextRouter"></a>

## TransformersZeroShotTextRouter

Routes the text strings to different connections based on a category label.

Specify the set of labels for categorization when initializing the component.

### Usage example

```python
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.core.pipeline import Pipeline
from haystack.components.routers import TransformersZeroShotTextRouter
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever

document_store = InMemoryDocumentStore()
doc_embedder = SentenceTransformersDocumentEmbedder(model="intfloat/e5-base-v2")
doc_embedder.warm_up()
docs = [
    Document(
        content="Germany, officially the Federal Republic of Germany, is a country in the western region of "
        "Central Europe. The nation's capital and most populous city is Berlin and its main financial centre "
        "is Frankfurt; the largest urban area is the Ruhr."
    ),
    Document(
        content="France, officially the French Republic, is a country located primarily in Western Europe. "
        "France is a unitary semi-presidential republic with its capital in Paris, the country's largest city "
        "and main cultural and commercial centre; other major urban areas include Marseille, Lyon, Toulouse, "
        "Lille, Bordeaux, Strasbourg, Nantes and Nice."
    )
]
docs_with_embeddings = doc_embedder.run(docs)
document_store.write_documents(docs_with_embeddings["documents"])

p = Pipeline()
p.add_component(instance=TransformersZeroShotTextRouter(labels=["passage", "query"]), name="text_router")
p.add_component(
    instance=SentenceTransformersTextEmbedder(model="intfloat/e5-base-v2", prefix="passage: "),
    name="passage_embedder"
)
p.add_component(
    instance=SentenceTransformersTextEmbedder(model="intfloat/e5-base-v2", prefix="query: "),
    name="query_embedder"
)
p.add_component(
    instance=InMemoryEmbeddingRetriever(document_store=document_store),
    name="query_retriever"
)
p.add_component(
    instance=InMemoryEmbeddingRetriever(document_store=document_store),
    name="passage_retriever"
)

p.connect("text_router.passage", "passage_embedder.text")
p.connect("passage_embedder.embedding", "passage_retriever.query_embedding")
p.connect("text_router.query", "query_embedder.text")
p.connect("query_embedder.embedding", "query_retriever.query_embedding")

# Query Example
p.run({"text_router": {"text": "What is the capital of Germany?"}})

# Passage Example
p.run({
    "text_router":{
        "text": "The United Kingdom of Great Britain and Northern Ireland, commonly known as the "            "United Kingdom (UK) or Britain, is a country in Northwestern Europe, off the north-western coast of "            "the continental mainland."
    }
})
```

<a id="zero_shot_text_router.TransformersZeroShotTextRouter.__init__"></a>

#### TransformersZeroShotTextRouter.\_\_init\_\_

```python
def __init__(labels: list[str],
             multi_label: bool = False,
             model: str = "MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33",
             device: Optional[ComponentDevice] = None,
             token: Optional[Secret] = Secret.from_env_var(
                 ["HF_API_TOKEN", "HF_TOKEN"], strict=False),
             huggingface_pipeline_kwargs: Optional[dict[str, Any]] = None)
```

Initializes the TransformersZeroShotTextRouter component.

**Arguments**:

- `labels`: The set of labels to use for classification. Can be a single label,
a string of comma-separated labels, or a list of labels.
- `multi_label`: Indicates if multiple labels can be true.
If `False`, label scores are normalized so their sum equals 1 for each sequence.
If `True`, the labels are considered independent and probabilities are normalized for each candidate by
doing a softmax of the entailment score vs. the contradiction score.
- `model`: The name or path of a Hugging Face model for zero-shot text classification.
- `device`: The device for loading the model. If `None`, automatically selects the default device.
If a device or device map is specified in `huggingface_pipeline_kwargs`, it overrides this parameter.
- `token`: The API token used to download private models from Hugging Face.
If `True`, uses either `HF_API_TOKEN` or `HF_TOKEN` environment variables.
To generate these tokens, run `transformers-cli login`.
- `huggingface_pipeline_kwargs`: A dictionary of keyword arguments for initializing the Hugging Face
zero shot text classification.

<a id="zero_shot_text_router.TransformersZeroShotTextRouter.warm_up"></a>

#### TransformersZeroShotTextRouter.warm\_up

```python
def warm_up()
```

Initializes the component.

<a id="zero_shot_text_router.TransformersZeroShotTextRouter.to_dict"></a>

#### TransformersZeroShotTextRouter.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="zero_shot_text_router.TransformersZeroShotTextRouter.from_dict"></a>

#### TransformersZeroShotTextRouter.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "TransformersZeroShotTextRouter"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="zero_shot_text_router.TransformersZeroShotTextRouter.run"></a>

#### TransformersZeroShotTextRouter.run

```python
def run(text: str) -> dict[str, str]
```

Routes the text strings to different connections based on a category label.

**Arguments**:

- `text`: A string of text to route.

**Raises**:

- `TypeError`: If the input is not a str.
- `RuntimeError`: If the pipeline has not been loaded because warm_up() was not called before.

**Returns**:

A dictionary with the label as key and the text as value.
