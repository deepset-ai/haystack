---
title: "Cohere"
id: integrations-cohere
description: "Cohere integration for Haystack"
slug: "/integrations-cohere"
---

<a id="haystack_integrations.components.embedders.cohere.document_embedder"></a>

# Module haystack\_integrations.components.embedders.cohere.document\_embedder

<a id="haystack_integrations.components.embedders.cohere.document_embedder.CohereDocumentEmbedder"></a>

## CohereDocumentEmbedder

A component for computing Document embeddings using Cohere models.

The embedding of each Document is stored in the `embedding` field of the Document.

Usage example:
```python
from haystack import Document
from haystack_integrations.components.embedders.cohere import CohereDocumentEmbedder

doc = Document(content="I love pizza!")

document_embedder = CohereDocumentEmbedder()

result = document_embedder.run([doc])
print(result['documents'][0].embedding)

# [-0.453125, 1.2236328, 2.0058594, ...]
```

<a id="haystack_integrations.components.embedders.cohere.document_embedder.CohereDocumentEmbedder.__init__"></a>

#### CohereDocumentEmbedder.\_\_init\_\_

```python
def __init__(api_key: Secret = Secret.from_env_var(
    ["COHERE_API_KEY", "CO_API_KEY"]),
             model: str = "embed-english-v2.0",
             input_type: str = "search_document",
             api_base_url: str = "https://api.cohere.com",
             truncate: str = "END",
             timeout: float = 120.0,
             batch_size: int = 32,
             progress_bar: bool = True,
             meta_fields_to_embed: Optional[List[str]] = None,
             embedding_separator: str = "\n",
             embedding_type: Optional[EmbeddingTypes] = None)
```

**Arguments**:

- `api_key`: the Cohere API key.
- `model`: the name of the model to use. Supported Models are:
`"embed-english-v3.0"`, `"embed-english-light-v3.0"`, `"embed-multilingual-v3.0"`,
`"embed-multilingual-light-v3.0"`, `"embed-english-v2.0"`, `"embed-english-light-v2.0"`,
`"embed-multilingual-v2.0"`. This list of all supported models can be found in the
[model documentation](https://docs.cohere.com/docs/models#representation).
- `input_type`: specifies the type of input you're giving to the model. Supported values are
"search_document", "search_query", "classification" and "clustering". Not
required for older versions of the embedding models (meaning anything lower than v3), but is required for
more recent versions (meaning anything bigger than v2).
- `api_base_url`: the Cohere API Base url.
- `truncate`: truncate embeddings that are too long from start or end, ("NONE"|"START"|"END").
Passing "START" will discard the start of the input. "END" will discard the end of the input. In both
cases, input is discarded until the remaining input is exactly the maximum input token length for the model.
If "NONE" is selected, when the input exceeds the maximum input token length an error will be returned.
- `timeout`: request timeout in seconds.
- `batch_size`: number of Documents to encode at once.
- `progress_bar`: whether to show a progress bar or not. Can be helpful to disable in production deployments
to keep the logs clean.
- `meta_fields_to_embed`: list of meta fields that should be embedded along with the Document text.
- `embedding_separator`: separator used to concatenate the meta fields to the Document text.
- `embedding_type`: the type of embeddings to return. Defaults to float embeddings.
Note that int8, uint8, binary, and ubinary are only valid for v3 models.

<a id="haystack_integrations.components.embedders.cohere.document_embedder.CohereDocumentEmbedder.to_dict"></a>

#### CohereDocumentEmbedder.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.embedders.cohere.document_embedder.CohereDocumentEmbedder.from_dict"></a>

#### CohereDocumentEmbedder.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "CohereDocumentEmbedder"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.embedders.cohere.document_embedder.CohereDocumentEmbedder.run"></a>

#### CohereDocumentEmbedder.run

```python
@component.output_types(documents=List[Document], meta=Dict[str, Any])
def run(
    documents: List[Document]
) -> Dict[str, Union[List[Document], Dict[str, Any]]]
```

Embed a list of `Documents`.

**Arguments**:

- `documents`: documents to embed.

**Raises**:

- `TypeError`: if the input is not a list of `Documents`.

**Returns**:

A dictionary with the following keys:
- `documents`: documents with the `embedding` field set.
- `meta`: metadata about the embedding process.

<a id="haystack_integrations.components.embedders.cohere.document_embedder.CohereDocumentEmbedder.run_async"></a>

#### CohereDocumentEmbedder.run\_async

```python
@component.output_types(documents=List[Document], meta=Dict[str, Any])
async def run_async(
    documents: List[Document]
) -> Dict[str, Union[List[Document], Dict[str, Any]]]
```

Embed a list of `Documents` asynchronously.

**Arguments**:

- `documents`: documents to embed.

**Raises**:

- `TypeError`: if the input is not a list of `Documents`.

**Returns**:

A dictionary with the following keys:
- `documents`: documents with the `embedding` field set.
- `meta`: metadata about the embedding process.

<a id="haystack_integrations.components.embedders.cohere.document_image_embedder"></a>

# Module haystack\_integrations.components.embedders.cohere.document\_image\_embedder

<a id="haystack_integrations.components.embedders.cohere.document_image_embedder.CohereDocumentImageEmbedder"></a>

## CohereDocumentImageEmbedder

A component for computing Document embeddings based on images using Cohere models.

The embedding of each Document is stored in the `embedding` field of the Document.

### Usage example
```python
from haystack import Document
from haystack_integrations.components.embedders.cohere import CohereDocumentImageEmbedder

embedder = CohereDocumentImageEmbedder(model="embed-v4.0")

documents = [
    Document(content="A photo of a cat", meta={"file_path": "cat.jpg"}),
    Document(content="A photo of a dog", meta={"file_path": "dog.jpg"}),
]

result = embedder.run(documents=documents)
documents_with_embeddings = result["documents"]
print(documents_with_embeddings)

# [Document(id=...,
#           content='A photo of a cat',
#           meta={'file_path': 'cat.jpg',
#                 'embedding_source': {'type': 'image', 'file_path_meta_field': 'file_path'}},
#           embedding=vector of size 1536),
#  ...]
```

<a id="haystack_integrations.components.embedders.cohere.document_image_embedder.CohereDocumentImageEmbedder.__init__"></a>

#### CohereDocumentImageEmbedder.\_\_init\_\_

```python
def __init__(*,
             file_path_meta_field: str = "file_path",
             root_path: Optional[str] = None,
             image_size: Optional[Tuple[int, int]] = None,
             api_key: Secret = Secret.from_env_var(
                 ["COHERE_API_KEY", "CO_API_KEY"]),
             model: str = "embed-v4.0",
             api_base_url: str = "https://api.cohere.com",
             timeout: float = 120.0,
             embedding_dimension: Optional[int] = None,
             embedding_type: EmbeddingTypes = EmbeddingTypes.FLOAT,
             progress_bar: bool = True) -> None
```

Creates a CohereDocumentImageEmbedder component.

**Arguments**:

- `file_path_meta_field`: The metadata field in the Document that contains the file path to the image or PDF.
- `root_path`: The root directory path where document files are located. If provided, file paths in
document metadata will be resolved relative to this path. If None, file paths are treated as absolute paths.
- `image_size`: If provided, resizes the image to fit within the specified dimensions (width, height) while
maintaining aspect ratio. This reduces file size, memory usage, and processing time, which is beneficial
when working with models that have resolution constraints or when transmitting images to remote services.
- `api_key`: The Cohere API key.
- `model`: The Cohere model to use for calculating embeddings.
Read [Cohere documentation](https://docs.cohere.com/docs/models#embed) for a list of all supported models.
- `api_base_url`: The Cohere API base URL.
- `timeout`: Request timeout in seconds.
- `embedding_dimension`: The dimension of the embeddings to return. Only valid for v4 and newer models.
Read [Cohere API reference](https://docs.cohere.com/reference/embed) for a list possible values and
supported models.
- `embedding_type`: The type of embeddings to return. Defaults to float embeddings.
Specifying a type different from float is only supported for Embed v3.0 and newer models.
- `progress_bar`: Whether to show a progress bar or not. Can be helpful to disable in production deployments
to keep the logs clean.

<a id="haystack_integrations.components.embedders.cohere.document_image_embedder.CohereDocumentImageEmbedder.to_dict"></a>

#### CohereDocumentImageEmbedder.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.embedders.cohere.document_image_embedder.CohereDocumentImageEmbedder.from_dict"></a>

#### CohereDocumentImageEmbedder.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "CohereDocumentImageEmbedder"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.embedders.cohere.document_image_embedder.CohereDocumentImageEmbedder.run"></a>

#### CohereDocumentImageEmbedder.run

```python
@component.output_types(documents=list[Document])
def run(documents: list[Document]) -> dict[str, list[Document]]
```

Embed a list of image documents.

**Arguments**:

- `documents`: Documents to embed.

**Returns**:

A dictionary with the following keys:
- `documents`: Documents with embeddings.

<a id="haystack_integrations.components.embedders.cohere.document_image_embedder.CohereDocumentImageEmbedder.run_async"></a>

#### CohereDocumentImageEmbedder.run\_async

```python
@component.output_types(documents=list[Document])
async def run_async(documents: list[Document]) -> dict[str, list[Document]]
```

Asynchronously embed a list of image documents.

**Arguments**:

- `documents`: Documents to embed.

**Returns**:

A dictionary with the following keys:
- `documents`: Documents with embeddings.

<a id="haystack_integrations.components.embedders.cohere.text_embedder"></a>

# Module haystack\_integrations.components.embedders.cohere.text\_embedder

<a id="haystack_integrations.components.embedders.cohere.text_embedder.CohereTextEmbedder"></a>

## CohereTextEmbedder

A component for embedding strings using Cohere models.

Usage example:
```python
from haystack_integrations.components.embedders.cohere import CohereTextEmbedder

text_to_embed = "I love pizza!"

text_embedder = CohereTextEmbedder()

print(text_embedder.run(text_to_embed))

# {'embedding': [-0.453125, 1.2236328, 2.0058594, ...]
# 'meta': {'api_version': {'version': '1'}, 'billed_units': {'input_tokens': 4}}}
```

<a id="haystack_integrations.components.embedders.cohere.text_embedder.CohereTextEmbedder.__init__"></a>

#### CohereTextEmbedder.\_\_init\_\_

```python
def __init__(api_key: Secret = Secret.from_env_var(
    ["COHERE_API_KEY", "CO_API_KEY"]),
             model: str = "embed-english-v2.0",
             input_type: str = "search_query",
             api_base_url: str = "https://api.cohere.com",
             truncate: str = "END",
             timeout: float = 120.0,
             embedding_type: Optional[EmbeddingTypes] = None)
```

**Arguments**:

- `api_key`: the Cohere API key.
- `model`: the name of the model to use. Supported Models are:
`"embed-english-v3.0"`, `"embed-english-light-v3.0"`, `"embed-multilingual-v3.0"`,
`"embed-multilingual-light-v3.0"`, `"embed-english-v2.0"`, `"embed-english-light-v2.0"`,
`"embed-multilingual-v2.0"`. This list of all supported models can be found in the
[model documentation](https://docs.cohere.com/docs/models#representation).
- `input_type`: specifies the type of input you're giving to the model. Supported values are
"search_document", "search_query", "classification" and "clustering". Not
required for older versions of the embedding models (meaning anything lower than v3), but is required for
more recent versions (meaning anything bigger than v2).
- `api_base_url`: the Cohere API Base url.
- `truncate`: truncate embeddings that are too long from start or end, ("NONE"|"START"|"END").
Passing "START" will discard the start of the input. "END" will discard the end of the input. In both
cases, input is discarded until the remaining input is exactly the maximum input token length for the model.
If "NONE" is selected, when the input exceeds the maximum input token length an error will be returned.
- `timeout`: request timeout in seconds.
- `embedding_type`: the type of embeddings to return. Defaults to float embeddings.
Note that int8, uint8, binary, and ubinary are only valid for v3 models.

<a id="haystack_integrations.components.embedders.cohere.text_embedder.CohereTextEmbedder.to_dict"></a>

#### CohereTextEmbedder.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.embedders.cohere.text_embedder.CohereTextEmbedder.from_dict"></a>

#### CohereTextEmbedder.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "CohereTextEmbedder"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.embedders.cohere.text_embedder.CohereTextEmbedder.run"></a>

#### CohereTextEmbedder.run

```python
@component.output_types(embedding=List[float], meta=Dict[str, Any])
def run(text: str) -> Dict[str, Union[List[float], Dict[str, Any]]]
```

Embed text.

**Arguments**:

- `text`: the text to embed.

**Raises**:

- `TypeError`: If the input is not a string.

**Returns**:

A dictionary with the following keys:
- `embedding`: the embedding of the text.
- `meta`: metadata about the request.

<a id="haystack_integrations.components.embedders.cohere.text_embedder.CohereTextEmbedder.run_async"></a>

#### CohereTextEmbedder.run\_async

```python
@component.output_types(embedding=List[float], meta=Dict[str, Any])
async def run_async(
        text: str) -> Dict[str, Union[List[float], Dict[str, Any]]]
```

Asynchronously embed text.

This is the asynchronous version of the `run` method. It has the same parameters and return values
but can be used with `await` in async code.

 :param text:
    Text to embed.

**Raises**:

- `TypeError`: If the input is not a string.

**Returns**:

A dictionary with the following keys:
- `embedding`: the embedding of the text.
- `meta`: metadata about the request.

<a id="haystack_integrations.components.embedders.cohere.utils"></a>

# Module haystack\_integrations.components.embedders.cohere.utils

<a id="haystack_integrations.components.embedders.cohere.utils.get_async_response"></a>

#### get\_async\_response

```python
async def get_async_response(
    cohere_async_client: AsyncClientV2,
    texts: List[str],
    model_name: str,
    input_type: str,
    truncate: str,
    embedding_type: Optional[EmbeddingTypes] = None
) -> Tuple[List[List[float]], Dict[str, Any]]
```

Embeds a list of texts asynchronously using the Cohere API.

**Arguments**:

- `cohere_async_client`: the Cohere `AsyncClient`
- `texts`: the texts to embed
- `model_name`: the name of the model to use
- `input_type`: one of "classification", "clustering", "search_document", "search_query".
The type of input text provided to embed.
- `truncate`: one of "NONE", "START", "END". How the API handles text longer than the maximum token length.
- `embedding_type`: the type of embeddings to return. Defaults to float embeddings.

**Raises**:

- `ValueError`: If an error occurs while querying the Cohere API.

**Returns**:

A tuple of the embeddings and metadata.

<a id="haystack_integrations.components.embedders.cohere.utils.get_response"></a>

#### get\_response

```python
def get_response(
    cohere_client: ClientV2,
    texts: List[str],
    model_name: str,
    input_type: str,
    truncate: str,
    batch_size: int = 32,
    progress_bar: bool = False,
    embedding_type: Optional[EmbeddingTypes] = None
) -> Tuple[List[List[float]], Dict[str, Any]]
```

Embeds a list of texts using the Cohere API.

**Arguments**:

- `cohere_client`: the Cohere `Client`
- `texts`: the texts to embed
- `model_name`: the name of the model to use
- `input_type`: one of "classification", "clustering", "search_document", "search_query".
The type of input text provided to embed.
- `truncate`: one of "NONE", "START", "END". How the API handles text longer than the maximum token length.
- `batch_size`: the batch size to use
- `progress_bar`: if `True`, show a progress bar
- `embedding_type`: the type of embeddings to return. Defaults to float embeddings.

**Raises**:

- `ValueError`: If an error occurs while querying the Cohere API.

**Returns**:

A tuple of the embeddings and metadata.

<a id="haystack_integrations.components.generators.cohere.generator"></a>

# Module haystack\_integrations.components.generators.cohere.generator

<a id="haystack_integrations.components.generators.cohere.generator.CohereGenerator"></a>

## CohereGenerator

Generates text using Cohere's models through Cohere's `generate` endpoint.

NOTE: Cohere discontinued the `generate` API, so this generator is a mere wrapper
around `CohereChatGenerator` provided for backward compatibility.

### Usage example

```python
from haystack_integrations.components.generators.cohere import CohereGenerator

generator = CohereGenerator(api_key="test-api-key")
generator.run(prompt="What's the capital of France?")
```

<a id="haystack_integrations.components.generators.cohere.generator.CohereGenerator.__init__"></a>

#### CohereGenerator.\_\_init\_\_

```python
def __init__(api_key: Secret = Secret.from_env_var(
    ["COHERE_API_KEY", "CO_API_KEY"]),
             model: str = "command-r-08-2024",
             streaming_callback: Optional[Callable] = None,
             api_base_url: Optional[str] = None,
             **kwargs: Any)
```

Instantiates a `CohereGenerator` component.

**Arguments**:

- `api_key`: Cohere API key.
- `model`: Cohere model to use for generation.
- `streaming_callback`: Callback function that is called when a new token is received from the stream.
The callback function accepts [StreamingChunk](https://docs.haystack.deepset.ai/docs/data-classes#streamingchunk)
as an argument.
- `api_base_url`: Cohere base URL.
- `**kwargs`: Additional arguments passed to the model. These arguments are specific to the model.
You can check them in model's documentation.

<a id="haystack_integrations.components.generators.cohere.generator.CohereGenerator.run"></a>

#### CohereGenerator.run

```python
@component.output_types(replies=List[str], meta=List[Dict[str, Any]])
def run(prompt: str) -> Dict[str, Union[List[str], List[Dict[str, Any]]]]
```

Queries the LLM with the prompts to produce replies.

**Arguments**:

- `prompt`: the prompt to be sent to the generative model.

**Returns**:

A dictionary with the following keys:
- `replies`: A list of replies generated by the model.
- `meta`: Information about the request.

<a id="haystack_integrations.components.generators.cohere.generator.CohereGenerator.run_async"></a>

#### CohereGenerator.run\_async

```python
@component.output_types(replies=List[str], meta=List[Dict[str, Any]])
async def run_async(
        prompt: str) -> Dict[str, Union[List[str], List[Dict[str, Any]]]]
```

Queries the LLM asynchronously with the prompts to produce replies.

**Arguments**:

- `prompt`: the prompt to be sent to the generative model.

**Returns**:

A dictionary with the following keys:
- `replies`: A list of replies generated by the model.
- `meta`: Information about the request.

<a id="haystack_integrations.components.generators.cohere.chat.chat_generator"></a>

# Module haystack\_integrations.components.generators.cohere.chat.chat\_generator

<a id="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator"></a>

## CohereChatGenerator

Completes chats using Cohere's models using cohere.ClientV2 `chat` endpoint.

This component supports both text-only and multimodal (text + image) conversations
using Cohere's vision models like Command A Vision.

Supported image formats: PNG, JPEG, WEBP, GIF (non-animated).
Maximum 20 images per request with 20MB total limit.

You can customize how the chat response is generated by passing parameters to the
Cohere API through the `**generation_kwargs` parameter. You can do this when
initializing or running the component. Any parameter that works with
 `cohere.ClientV2.chat` will work here too.
For details, see [Cohere API](https://docs.cohere.com/reference/chat).

Below is an example of how to use the component:

### Simple example
```python
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from haystack_integrations.components.generators.cohere import CohereChatGenerator

client = CohereChatGenerator(model="command-r-08-2024", api_key=Secret.from_env_var("COHERE_API_KEY"))
messages = [ChatMessage.from_user("What's Natural Language Processing?")]
client.run(messages)

# Output: {'replies': [ChatMessage(_role=<ChatRole.ASSISTANT: 'assistant'>,
# _content=[TextContent(text='Natural Language Processing (NLP) is an interdisciplinary...
```

### Multimodal example
```python
from haystack.dataclasses import ChatMessage, ImageContent
from haystack.utils import Secret
from haystack_integrations.components.generators.cohere import CohereChatGenerator

# Create an image from file path or base64
image_content = ImageContent.from_file_path("path/to/your/image.jpg")

# Create a multimodal message with both text and image
messages = [ChatMessage.from_user(content_parts=["What's in this image?", image_content])]

# Use a multimodal model like Command A Vision
client = CohereChatGenerator(model="command-a-vision-07-2025", api_key=Secret.from_env_var("COHERE_API_KEY"))
response = client.run(messages)
print(response)
```

### Advanced example

CohereChatGenerator can be integrated into pipelines and supports Haystack's tooling
architecture, enabling tools to be invoked seamlessly across various generators.

```python
from haystack import Pipeline
from haystack.dataclasses import ChatMessage
from haystack.components.tools import ToolInvoker
from haystack.tools import Tool
from haystack_integrations.components.generators.cohere import CohereChatGenerator

# Create a weather tool
def weather(city: str) -> str:
    return f"The weather in {city} is sunny and 32°C"

weather_tool = Tool(
    name="weather",
    description="useful to determine the weather in a given location",
    parameters={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The name of the city to get weather for, e.g. Paris, London",
            }
        },
        "required": ["city"],
    },
    function=weather,
)

# Create and set up the pipeline
pipeline = Pipeline()
pipeline.add_component("generator", CohereChatGenerator(model="command-r-08-2024", tools=[weather_tool]))
pipeline.add_component("tool_invoker", ToolInvoker(tools=[weather_tool]))
pipeline.connect("generator", "tool_invoker")

# Run the pipeline with a weather query
results = pipeline.run(
    data={"generator": {"messages": [ChatMessage.from_user("What's the weather like in Paris?")]}}
)

# The tool result will be available in the pipeline output
print(results["tool_invoker"]["tool_messages"][0].tool_call_result.result)
# Output: "The weather in Paris is sunny and 32°C"
```

<a id="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator.__init__"></a>

#### CohereChatGenerator.\_\_init\_\_

```python
def __init__(api_key: Secret = Secret.from_env_var(
    ["COHERE_API_KEY", "CO_API_KEY"]),
             model: str = "command-r-08-2024",
             streaming_callback: Optional[StreamingCallbackT] = None,
             api_base_url: Optional[str] = None,
             generation_kwargs: Optional[Dict[str, Any]] = None,
             tools: Optional[Union[List[Tool], Toolset]] = None,
             **kwargs: Any)
```

Initialize the CohereChatGenerator instance.

**Arguments**:

- `api_key`: The API key for the Cohere API.
- `model`: The name of the model to use. You can use models from the `command` family.
- `streaming_callback`: A callback function that is called when a new token is received from the stream.
The callback function accepts [StreamingChunk](https://docs.haystack.deepset.ai/docs/data-classes#streamingchunk)
as an argument.
- `api_base_url`: The base URL of the Cohere API.
- `generation_kwargs`: Other parameters to use for the model during generation. For a list of parameters,
see [Cohere Chat endpoint](https://docs.cohere.com/reference/chat).
Some of the parameters are:
- 'messages': A list of messages between the user and the model, meant to give the model
  conversational context for responding to the user's message.
- 'system_message': When specified, adds a system message at the beginning of the conversation.
- 'citation_quality': Defaults to `accurate`. Dictates the approach taken to generating citations
  as part of the RAG flow by allowing the user to specify whether they want
  `accurate` results or `fast` results.
- 'temperature': A non-negative float that tunes the degree of randomness in generation. Lower temperatures
  mean less random generations.
- `tools`: A list of Tool objects or a Toolset that the model can use. Each tool should have a unique name.

<a id="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator.to_dict"></a>

#### CohereChatGenerator.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator.from_dict"></a>

#### CohereChatGenerator.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "CohereChatGenerator"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator.run"></a>

#### CohereChatGenerator.run

```python
@component.output_types(replies=List[ChatMessage])
def run(
    messages: List[ChatMessage],
    generation_kwargs: Optional[Dict[str, Any]] = None,
    tools: Optional[Union[List[Tool], Toolset]] = None,
    streaming_callback: Optional[StreamingCallbackT] = None
) -> Dict[str, List[ChatMessage]]
```

Invoke the chat endpoint based on the provided messages and generation parameters.

**Arguments**:

- `messages`: list of `ChatMessage` instances representing the input messages.
- `generation_kwargs`: additional keyword arguments for chat generation. These parameters will
potentially override the parameters passed in the __init__ method.
For more details on the parameters supported by the Cohere API, refer to the
Cohere [documentation](https://docs.cohere.com/reference/chat).
- `tools`: A list of tools or a Toolset for which the model can prepare calls. If set, it will override
the `tools` parameter set during component initialization.
- `streaming_callback`: A callback function that is called when a new token is received from the stream.
The callback function accepts StreamingChunk as an argument.

**Returns**:

A dictionary with the following keys:
- `replies`: a list of `ChatMessage` instances representing the generated responses.

<a id="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator.run_async"></a>

#### CohereChatGenerator.run\_async

```python
@component.output_types(replies=List[ChatMessage])
async def run_async(
    messages: List[ChatMessage],
    generation_kwargs: Optional[Dict[str, Any]] = None,
    tools: Optional[Union[List[Tool], Toolset]] = None,
    streaming_callback: Optional[StreamingCallbackT] = None
) -> Dict[str, List[ChatMessage]]
```

Asynchronously invoke the chat endpoint based on the provided messages and generation parameters.

**Arguments**:

- `messages`: list of `ChatMessage` instances representing the input messages.
- `generation_kwargs`: additional keyword arguments for chat generation. These parameters will
potentially override the parameters passed in the __init__ method.
For more details on the parameters supported by the Cohere API, refer to the
Cohere [documentation](https://docs.cohere.com/reference/chat).
- `tools`: A list of tools for which the model can prepare calls. If set, it will override
the `tools` parameter set during component initialization.
- `streaming_callback`: A callback function that is called when a new token is received from the stream.

**Returns**:

A dictionary with the following keys:
- `replies`: a list of `ChatMessage` instances representing the generated responses.

<a id="haystack_integrations.components.rankers.cohere.ranker"></a>

# Module haystack\_integrations.components.rankers.cohere.ranker

<a id="haystack_integrations.components.rankers.cohere.ranker.CohereRanker"></a>

## CohereRanker

Ranks Documents based on their similarity to the query using [Cohere models](https://docs.cohere.com/reference/rerank-1).

Documents are indexed from most to least semantically relevant to the query.

Usage example:
```python
from haystack import Document
from haystack_integrations.components.rankers.cohere import CohereRanker

ranker = CohereRanker(model="rerank-v3.5", top_k=2)

docs = [Document(content="Paris"), Document(content="Berlin")]
query = "What is the capital of germany?"
output = ranker.run(query=query, documents=docs)
docs = output["documents"]
```

<a id="haystack_integrations.components.rankers.cohere.ranker.CohereRanker.__init__"></a>

#### CohereRanker.\_\_init\_\_

```python
def __init__(model: str = "rerank-v3.5",
             top_k: int = 10,
             api_key: Secret = Secret.from_env_var(
                 ["COHERE_API_KEY", "CO_API_KEY"]),
             api_base_url: str = "https://api.cohere.com",
             meta_fields_to_embed: Optional[List[str]] = None,
             meta_data_separator: str = "\n",
             max_tokens_per_doc: int = 4096)
```

Creates an instance of the 'CohereRanker'.

**Arguments**:

- `model`: Cohere model name. Check the list of supported models in the [Cohere documentation](https://docs.cohere.com/docs/models).
- `top_k`: The maximum number of documents to return.
- `api_key`: Cohere API key.
- `api_base_url`: the base URL of the Cohere API.
- `meta_fields_to_embed`: List of meta fields that should be concatenated
with the document content for reranking.
- `meta_data_separator`: Separator used to concatenate the meta fields
to the Document content.
- `max_tokens_per_doc`: The maximum number of tokens to embed for each document defaults to 4096.

<a id="haystack_integrations.components.rankers.cohere.ranker.CohereRanker.to_dict"></a>

#### CohereRanker.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.rankers.cohere.ranker.CohereRanker.from_dict"></a>

#### CohereRanker.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "CohereRanker"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: The dictionary to deserialize from.

**Returns**:

The deserialized component.

<a id="haystack_integrations.components.rankers.cohere.ranker.CohereRanker.run"></a>

#### CohereRanker.run

```python
@component.output_types(documents=List[Document])
def run(query: str,
        documents: List[Document],
        top_k: Optional[int] = None) -> Dict[str, List[Document]]
```

Use the Cohere Reranker to re-rank the list of documents based on the query.

**Arguments**:

- `query`: Query string.
- `documents`: List of Documents.
- `top_k`: The maximum number of Documents you want the Ranker to return.

**Raises**:

- `ValueError`: If `top_k` is not > 0.

**Returns**:

A dictionary with the following keys:
- `documents`: List of Documents most similar to the given query in descending order of similarity.
