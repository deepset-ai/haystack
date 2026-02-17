---
title: "Cohere"
id: integrations-cohere
description: "Cohere integration for Haystack"
slug: "/integrations-cohere"
---


## `haystack_integrations.components.embedders.cohere.document_embedder`

### `CohereDocumentEmbedder`

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

#### `__init__`

```python
__init__(
    api_key: Secret = Secret.from_env_var(["COHERE_API_KEY", "CO_API_KEY"]),
    model: str = "embed-english-v2.0",
    input_type: str = "search_document",
    api_base_url: str = "https://api.cohere.com",
    truncate: str = "END",
    timeout: float = 120.0,
    batch_size: int = 32,
    progress_bar: bool = True,
    meta_fields_to_embed: list[str] | None = None,
    embedding_separator: str = "\n",
    embedding_type: EmbeddingTypes | None = None,
)
```

**Parameters:**

- **api_key** (<code>Secret</code>) – the Cohere API key.
- **model** (<code>str</code>) – the name of the model to use. Supported Models are:
  `"embed-english-v3.0"`, `"embed-english-light-v3.0"`, `"embed-multilingual-v3.0"`,
  `"embed-multilingual-light-v3.0"`, `"embed-english-v2.0"`, `"embed-english-light-v2.0"`,
  `"embed-multilingual-v2.0"`. This list of all supported models can be found in the
  [model documentation](https://docs.cohere.com/docs/models#representation).
- **input_type** (<code>str</code>) – specifies the type of input you're giving to the model. Supported values are
  "search_document", "search_query", "classification" and "clustering". Not
  required for older versions of the embedding models (meaning anything lower than v3), but is required for
  more recent versions (meaning anything bigger than v2).
- **api_base_url** (<code>str</code>) – the Cohere API Base url.
- **truncate** (<code>str</code>) – truncate embeddings that are too long from start or end, ("NONE"|"START"|"END").
  Passing "START" will discard the start of the input. "END" will discard the end of the input. In both
  cases, input is discarded until the remaining input is exactly the maximum input token length for the model.
  If "NONE" is selected, when the input exceeds the maximum input token length an error will be returned.
- **timeout** (<code>float</code>) – request timeout in seconds.
- **batch_size** (<code>int</code>) – number of Documents to encode at once.
- **progress_bar** (<code>bool</code>) – whether to show a progress bar or not. Can be helpful to disable in production deployments
  to keep the logs clean.
- **meta_fields_to_embed** (<code>list\[str\] | None</code>) – list of meta fields that should be embedded along with the Document text.
- **embedding_separator** (<code>str</code>) – separator used to concatenate the meta fields to the Document text.
- **embedding_type** (<code>EmbeddingTypes | None</code>) – the type of embeddings to return. Defaults to float embeddings.
  Note that int8, uint8, binary, and ubinary are only valid for v3 models.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> CohereDocumentEmbedder
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>CohereDocumentEmbedder</code> – Deserialized component.

#### `run`

```python
run(documents: list[Document]) -> dict[str, list[Document] | dict[str, Any]]
```

Embed a list of `Documents`.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – documents to embed.

**Returns:**

- <code>dict\[str, list\[Document\] | dict\[str, Any\]\]</code> – A dictionary with the following keys:
- `documents`: documents with the `embedding` field set.
- `meta`: metadata about the embedding process.

**Raises:**

- <code>TypeError</code> – if the input is not a list of `Documents`.

#### `run_async`

```python
run_async(
    documents: list[Document],
) -> dict[str, list[Document] | dict[str, Any]]
```

Embed a list of `Documents` asynchronously.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – documents to embed.

**Returns:**

- <code>dict\[str, list\[Document\] | dict\[str, Any\]\]</code> – A dictionary with the following keys:
- `documents`: documents with the `embedding` field set.
- `meta`: metadata about the embedding process.

**Raises:**

- <code>TypeError</code> – if the input is not a list of `Documents`.

## `haystack_integrations.components.embedders.cohere.document_image_embedder`

### `CohereDocumentImageEmbedder`

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

#### `__init__`

```python
__init__(
    *,
    file_path_meta_field: str = "file_path",
    root_path: str | None = None,
    image_size: tuple[int, int] | None = None,
    api_key: Secret = Secret.from_env_var(["COHERE_API_KEY", "CO_API_KEY"]),
    model: str = "embed-v4.0",
    api_base_url: str = "https://api.cohere.com",
    timeout: float = 120.0,
    embedding_dimension: int | None = None,
    embedding_type: EmbeddingTypes = EmbeddingTypes.FLOAT,
    progress_bar: bool = True
) -> None
```

Creates a CohereDocumentImageEmbedder component.

**Parameters:**

- **file_path_meta_field** (<code>str</code>) – The metadata field in the Document that contains the file path to the image or PDF.
- **root_path** (<code>str | None</code>) – The root directory path where document files are located. If provided, file paths in
  document metadata will be resolved relative to this path. If None, file paths are treated as absolute paths.
- **image_size** (<code>tuple\[int, int\] | None</code>) – If provided, resizes the image to fit within the specified dimensions (width, height) while
  maintaining aspect ratio. This reduces file size, memory usage, and processing time, which is beneficial
  when working with models that have resolution constraints or when transmitting images to remote services.
- **api_key** (<code>Secret</code>) – The Cohere API key.
- **model** (<code>str</code>) – The Cohere model to use for calculating embeddings.
  Read [Cohere documentation](https://docs.cohere.com/docs/models#embed) for a list of all supported models.
- **api_base_url** (<code>str</code>) – The Cohere API base URL.
- **timeout** (<code>float</code>) – Request timeout in seconds.
- **embedding_dimension** (<code>int | None</code>) – The dimension of the embeddings to return. Only valid for v4 and newer models.
  Read [Cohere API reference](https://docs.cohere.com/reference/embed) for a list possible values and
  supported models.
- **embedding_type** (<code>EmbeddingTypes</code>) – The type of embeddings to return. Defaults to float embeddings.
  Specifying a type different from float is only supported for Embed v3.0 and newer models.
- **progress_bar** (<code>bool</code>) – Whether to show a progress bar or not. Can be helpful to disable in production deployments
  to keep the logs clean.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> CohereDocumentImageEmbedder
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>CohereDocumentImageEmbedder</code> – Deserialized component.

#### `run`

```python
run(documents: list[Document]) -> dict[str, list[Document]]
```

Embed a list of image documents.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – Documents to embed.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: Documents with embeddings.

#### `run_async`

```python
run_async(documents: list[Document]) -> dict[str, list[Document]]
```

Asynchronously embed a list of image documents.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – Documents to embed.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: Documents with embeddings.

## `haystack_integrations.components.embedders.cohere.text_embedder`

### `CohereTextEmbedder`

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

#### `__init__`

```python
__init__(
    api_key: Secret = Secret.from_env_var(["COHERE_API_KEY", "CO_API_KEY"]),
    model: str = "embed-english-v2.0",
    input_type: str = "search_query",
    api_base_url: str = "https://api.cohere.com",
    truncate: str = "END",
    timeout: float = 120.0,
    embedding_type: EmbeddingTypes | None = None,
)
```

**Parameters:**

- **api_key** (<code>Secret</code>) – the Cohere API key.
- **model** (<code>str</code>) – the name of the model to use. Supported Models are:
  `"embed-english-v3.0"`, `"embed-english-light-v3.0"`, `"embed-multilingual-v3.0"`,
  `"embed-multilingual-light-v3.0"`, `"embed-english-v2.0"`, `"embed-english-light-v2.0"`,
  `"embed-multilingual-v2.0"`. This list of all supported models can be found in the
  [model documentation](https://docs.cohere.com/docs/models#representation).
- **input_type** (<code>str</code>) – specifies the type of input you're giving to the model. Supported values are
  "search_document", "search_query", "classification" and "clustering". Not
  required for older versions of the embedding models (meaning anything lower than v3), but is required for
  more recent versions (meaning anything bigger than v2).
- **api_base_url** (<code>str</code>) – the Cohere API Base url.
- **truncate** (<code>str</code>) – truncate embeddings that are too long from start or end, ("NONE"|"START"|"END").
  Passing "START" will discard the start of the input. "END" will discard the end of the input. In both
  cases, input is discarded until the remaining input is exactly the maximum input token length for the model.
  If "NONE" is selected, when the input exceeds the maximum input token length an error will be returned.
- **timeout** (<code>float</code>) – request timeout in seconds.
- **embedding_type** (<code>EmbeddingTypes | None</code>) – the type of embeddings to return. Defaults to float embeddings.
  Note that int8, uint8, binary, and ubinary are only valid for v3 models.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> CohereTextEmbedder
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>CohereTextEmbedder</code> – Deserialized component.

#### `run`

```python
run(text: str) -> dict[str, list[float] | dict[str, Any]]
```

Embed text.

**Parameters:**

- **text** (<code>str</code>) – the text to embed.

**Returns:**

- <code>dict\[str, list\[float\] | dict\[str, Any\]\]</code> – A dictionary with the following keys:
  - `embedding`: the embedding of the text.
  - `meta`: metadata about the request.

**Raises:**

- <code>TypeError</code> – If the input is not a string.

#### `run_async`

```python
run_async(text: str) -> dict[str, list[float] | dict[str, Any]]
```

Asynchronously embed text.

This is the asynchronous version of the `run` method. It has the same parameters and return values
but can be used with `await` in async code.

:param text:
Text to embed.

**Returns:**

- <code>dict\[str, list\[float\] | dict\[str, Any\]\]</code> – A dictionary with the following keys:
- `embedding`: the embedding of the text.
- `meta`: metadata about the request.

**Raises:**

- <code>TypeError</code> – If the input is not a string.

## `haystack_integrations.components.embedders.cohere.utils`

### `get_async_response`

```python
get_async_response(
    cohere_async_client: AsyncClientV2,
    texts: list[str],
    model_name: str,
    input_type: str,
    truncate: str,
    embedding_type: EmbeddingTypes | None = None,
) -> tuple[list[list[float]], dict[str, Any]]
```

Embeds a list of texts asynchronously using the Cohere API.

**Parameters:**

- **cohere_async_client** (<code>AsyncClientV2</code>) – the Cohere `AsyncClient`
- **texts** (<code>list\[str\]</code>) – the texts to embed
- **model_name** (<code>str</code>) – the name of the model to use
- **input_type** (<code>str</code>) – one of "classification", "clustering", "search_document", "search_query".
  The type of input text provided to embed.
- **truncate** (<code>str</code>) – one of "NONE", "START", "END". How the API handles text longer than the maximum token length.
- **embedding_type** (<code>EmbeddingTypes | None</code>) – the type of embeddings to return. Defaults to float embeddings.

**Returns:**

- <code>tuple\[list\[list\[float\]\], dict\[str, Any\]\]</code> – A tuple of the embeddings and metadata.

**Raises:**

- <code>ValueError</code> – If an error occurs while querying the Cohere API.

### `get_response`

```python
get_response(
    cohere_client: ClientV2,
    texts: list[str],
    model_name: str,
    input_type: str,
    truncate: str,
    batch_size: int = 32,
    progress_bar: bool = False,
    embedding_type: EmbeddingTypes | None = None,
) -> tuple[list[list[float]], dict[str, Any]]
```

Embeds a list of texts using the Cohere API.

**Parameters:**

- **cohere_client** (<code>ClientV2</code>) – the Cohere `Client`
- **texts** (<code>list\[str\]</code>) – the texts to embed
- **model_name** (<code>str</code>) – the name of the model to use
- **input_type** (<code>str</code>) – one of "classification", "clustering", "search_document", "search_query".
  The type of input text provided to embed.
- **truncate** (<code>str</code>) – one of "NONE", "START", "END". How the API handles text longer than the maximum token length.
- **batch_size** (<code>int</code>) – the batch size to use
- **progress_bar** (<code>bool</code>) – if `True`, show a progress bar
- **embedding_type** (<code>EmbeddingTypes | None</code>) – the type of embeddings to return. Defaults to float embeddings.

**Returns:**

- <code>tuple\[list\[list\[float\]\], dict\[str, Any\]\]</code> – A tuple of the embeddings and metadata.

**Raises:**

- <code>ValueError</code> – If an error occurs while querying the Cohere API.

## `haystack_integrations.components.generators.cohere.chat.chat_generator`

### `CohereChatGenerator`

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

client = CohereChatGenerator(api_key=Secret.from_env_var("COHERE_API_KEY"))
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
pipeline.add_component("generator", CohereChatGenerator(tools=[weather_tool]))
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

#### `__init__`

```python
__init__(
    api_key: Secret = Secret.from_env_var(["COHERE_API_KEY", "CO_API_KEY"]),
    model: str = "command-a-03-2025",
    streaming_callback: StreamingCallbackT | None = None,
    api_base_url: str | None = None,
    generation_kwargs: dict[str, Any] | None = None,
    tools: ToolsType | None = None,
    **kwargs: Any
)
```

Initialize the CohereChatGenerator instance.

**Parameters:**

- **api_key** (<code>Secret</code>) – The API key for the Cohere API.
- **model** (<code>str</code>) – The name of the model to use. You can use models from the `command` family.
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – A callback function that is called when a new token is received from the stream.
  The callback function accepts [StreamingChunk](https://docs.haystack.deepset.ai/docs/data-classes#streamingchunk)
  as an argument.
- **api_base_url** (<code>str | None</code>) – The base URL of the Cohere API.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Other parameters to use for the model during generation. For a list of parameters,
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
- **tools** (<code>ToolsType | None</code>) – A list of Tool and/or Toolset objects, or a single Toolset that the model can use.
  Each tool should have a unique name.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> CohereChatGenerator
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>CohereChatGenerator</code> – Deserialized component.

#### `run`

```python
run(
    messages: list[ChatMessage],
    generation_kwargs: dict[str, Any] | None = None,
    tools: ToolsType | None = None,
    streaming_callback: StreamingCallbackT | None = None,
) -> dict[str, list[ChatMessage]]
```

Invoke the chat endpoint based on the provided messages and generation parameters.

**Parameters:**

- **messages** (<code>list\[ChatMessage\]</code>) – list of `ChatMessage` instances representing the input messages.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – additional keyword arguments for chat generation. These parameters will
  potentially override the parameters passed in the __init__ method.
  For more details on the parameters supported by the Cohere API, refer to the
  Cohere [documentation](https://docs.cohere.com/reference/chat).
- **tools** (<code>ToolsType | None</code>) – A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
  If set, it will override the `tools` parameter set during component initialization.
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – A callback function that is called when a new token is received from the stream.
  The callback function accepts StreamingChunk as an argument.

**Returns:**

- <code>dict\[str, list\[ChatMessage\]\]</code> – A dictionary with the following keys:
- `replies`: a list of `ChatMessage` instances representing the generated responses.

#### `run_async`

```python
run_async(
    messages: list[ChatMessage],
    generation_kwargs: dict[str, Any] | None = None,
    tools: ToolsType | None = None,
    streaming_callback: StreamingCallbackT | None = None,
) -> dict[str, list[ChatMessage]]
```

Asynchronously invoke the chat endpoint based on the provided messages and generation parameters.

**Parameters:**

- **messages** (<code>list\[ChatMessage\]</code>) – list of `ChatMessage` instances representing the input messages.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – additional keyword arguments for chat generation. These parameters will
  potentially override the parameters passed in the __init__ method.
  For more details on the parameters supported by the Cohere API, refer to the
  Cohere [documentation](https://docs.cohere.com/reference/chat).
- **tools** (<code>ToolsType | None</code>) – A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
  If set, it will override the `tools` parameter set during component initialization.
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – A callback function that is called when a new token is received from the stream.

**Returns:**

- <code>dict\[str, list\[ChatMessage\]\]</code> – A dictionary with the following keys:
- `replies`: a list of `ChatMessage` instances representing the generated responses.

## `haystack_integrations.components.generators.cohere.generator`

### `CohereGenerator`

Bases: <code>CohereChatGenerator</code>

Generates text using Cohere's models through Cohere's `generate` endpoint.

NOTE: Cohere discontinued the `generate` API, so this generator is a mere wrapper
around `CohereChatGenerator` provided for backward compatibility.

### Usage example

```python
from haystack_integrations.components.generators.cohere import CohereGenerator

generator = CohereGenerator(api_key="test-api-key")
generator.run(prompt="What's the capital of France?")
```

#### `__init__`

```python
__init__(
    api_key: Secret = Secret.from_env_var(["COHERE_API_KEY", "CO_API_KEY"]),
    model: str = "command-a-03-2025",
    streaming_callback: Callable | None = None,
    api_base_url: str | None = None,
    **kwargs: Any
)
```

Instantiates a `CohereGenerator` component.

**Parameters:**

- **api_key** (<code>Secret</code>) – Cohere API key.
- **model** (<code>str</code>) – Cohere model to use for generation.
- **streaming_callback** (<code>Callable | None</code>) – Callback function that is called when a new token is received from the stream.
  The callback function accepts [StreamingChunk](https://docs.haystack.deepset.ai/docs/data-classes#streamingchunk)
  as an argument.
- **api_base_url** (<code>str | None</code>) – Cohere base URL.
- \*\***kwargs** (<code>Any</code>) – Additional arguments passed to the model. These arguments are specific to the model.
  You can check them in model's documentation.

#### `run`

```python
run(prompt: str) -> dict[str, list[str] | list[dict[str, Any]]]
```

Queries the LLM with the prompts to produce replies.

**Parameters:**

- **prompt** (<code>str</code>) – the prompt to be sent to the generative model.

**Returns:**

- <code>dict\[str, list\[str\] | list\[dict\[str, Any\]\]\]</code> – A dictionary with the following keys:
- `replies`: A list of replies generated by the model.
- `meta`: Information about the request.

#### `run_async`

```python
run_async(prompt: str) -> dict[str, list[str] | list[dict[str, Any]]]
```

Queries the LLM asynchronously with the prompts to produce replies.

**Parameters:**

- **prompt** (<code>str</code>) – the prompt to be sent to the generative model.

**Returns:**

- <code>dict\[str, list\[str\] | list\[dict\[str, Any\]\]\]</code> – A dictionary with the following keys:
- `replies`: A list of replies generated by the model.
- `meta`: Information about the request.

## `haystack_integrations.components.rankers.cohere.ranker`

### `CohereRanker`

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

#### `__init__`

```python
__init__(
    model: str = "rerank-v3.5",
    top_k: int = 10,
    api_key: Secret = Secret.from_env_var(["COHERE_API_KEY", "CO_API_KEY"]),
    api_base_url: str = "https://api.cohere.com",
    meta_fields_to_embed: list[str] | None = None,
    meta_data_separator: str = "\n",
    max_tokens_per_doc: int = 4096,
)
```

Creates an instance of the 'CohereRanker'.

**Parameters:**

- **model** (<code>str</code>) – Cohere model name. Check the list of supported models in the [Cohere documentation](https://docs.cohere.com/docs/models).
- **top_k** (<code>int</code>) – The maximum number of documents to return.
- **api_key** (<code>Secret</code>) – Cohere API key.
- **api_base_url** (<code>str</code>) – the base URL of the Cohere API.
- **meta_fields_to_embed** (<code>list\[str\] | None</code>) – List of meta fields that should be concatenated
  with the document content for reranking.
- **meta_data_separator** (<code>str</code>) – Separator used to concatenate the meta fields
  to the Document content.
- **max_tokens_per_doc** (<code>int</code>) – The maximum number of tokens to embed for each document defaults to 4096.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> CohereRanker
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to deserialize from.

**Returns:**

- <code>CohereRanker</code> – The deserialized component.

#### `run`

```python
run(
    query: str, documents: list[Document], top_k: int | None = None
) -> dict[str, list[Document]]
```

Use the Cohere Reranker to re-rank the list of documents based on the query.

**Parameters:**

- **query** (<code>str</code>) – Query string.
- **documents** (<code>list\[Document\]</code>) – List of Documents.
- **top_k** (<code>int | None</code>) – The maximum number of Documents you want the Ranker to return.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: List of Documents most similar to the given query in descending order of similarity.

**Raises:**

- <code>ValueError</code> – If `top_k` is not > 0.
