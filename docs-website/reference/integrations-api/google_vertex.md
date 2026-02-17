---
title: "Google Vertex"
id: integrations-google-vertex
description: "Google Vertex integration for Haystack"
slug: "/integrations-google-vertex"
---


## `haystack_integrations.components.embedders.google_vertex.document_embedder`

### `VertexAIDocumentEmbedder`

Embed text using Vertex AI Embeddings API.

See available models in the official
[Google documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api#syntax).

Usage example:

```python
from haystack import Document
from haystack_integrations.components.embedders.google_vertex import VertexAIDocumentEmbedder

doc = Document(content="I love pizza!")

document_embedder = VertexAIDocumentEmbedder(model="text-embedding-005")

result = document_embedder.run([doc])
print(result['documents'][0].embedding)
# [-0.044606007635593414, 0.02857724390923977, -0.03549133986234665,
```

#### `__init__`

```python
__init__(
    model: Literal[
        "text-embedding-004",
        "text-embedding-005",
        "textembedding-gecko-multilingual@001",
        "text-multilingual-embedding-002",
        "text-embedding-large-exp-03-07",
    ],
    task_type: Literal[
        "RETRIEVAL_DOCUMENT",
        "RETRIEVAL_QUERY",
        "SEMANTIC_SIMILARITY",
        "CLASSIFICATION",
        "CLUSTERING",
        "QUESTION_ANSWERING",
        "FACT_VERIFICATION",
        "CODE_RETRIEVAL_QUERY",
    ] = "RETRIEVAL_DOCUMENT",
    gcp_region_name: Optional[Secret] = Secret.from_env_var(
        "GCP_DEFAULT_REGION", strict=False
    ),
    gcp_project_id: Optional[Secret] = Secret.from_env_var(
        "GCP_PROJECT_ID", strict=False
    ),
    batch_size: int = 32,
    max_tokens_total: int = 20000,
    time_sleep: int = 30,
    retries: int = 3,
    progress_bar: bool = True,
    truncate_dim: Optional[int] = None,
    meta_fields_to_embed: Optional[list[str]] = None,
    embedding_separator: str = "\n",
) -> None
```

Generate Document Embedder using a Google Vertex AI model.

Authenticates using Google Cloud Application Default Credentials (ADCs).
For more information see the official [Google documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc).

**Parameters:**

- **model** (<code>Literal['text-embedding-004', 'text-embedding-005', 'textembedding-gecko-multilingual@001', 'text-multilingual-embedding-002', 'text-embedding-large-exp-03-07']</code>) – Name of the model to use.
- **task_type** (<code>Literal['RETRIEVAL_DOCUMENT', 'RETRIEVAL_QUERY', 'SEMANTIC_SIMILARITY', 'CLASSIFICATION', 'CLUSTERING', 'QUESTION_ANSWERING', 'FACT_VERIFICATION', 'CODE_RETRIEVAL_QUERY']</code>) – The type of task for which the embeddings are being generated.
  For more information see the official [Google documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api#tasktype).
- **gcp_region_name** (<code>Optional\[Secret\]</code>) – The default location to use when making API calls, if not set uses us-central-1.
- **gcp_project_id** (<code>Optional\[Secret\]</code>) – ID of the GCP project to use. By default, it is set during Google Cloud authentication.
- **batch_size** (<code>int</code>) – The number of documents to process in a single batch.
- **max_tokens_total** (<code>int</code>) – The maximum number of tokens to process in total.
- **time_sleep** (<code>int</code>) – The time to sleep between retries in seconds.
- **retries** (<code>int</code>) – The number of retries in case of failure.
- **progress_bar** (<code>bool</code>) – Whether to display a progress bar during processing.
- **truncate_dim** (<code>Optional\[int\]</code>) – The dimension to truncate the embeddings to, if specified.
- **meta_fields_to_embed** (<code>Optional\[list\[str\]\]</code>) – A list of metadata fields to include in the embeddings.
- **embedding_separator** (<code>str</code>) – The separator to use between different embeddings.

**Raises:**

- <code>ValueError</code> – If the provided model is not in the list of supported models.

#### `get_text_embedding_input`

```python
get_text_embedding_input(batch: list[Document]) -> list[TextEmbeddingInput]
```

Converts a batch of Document objects into a list of TextEmbeddingInput objects.

Args:
batch (List[Document]): A list of Document objects to be converted.

Returns:
List\[TextEmbeddingInput\]: A list of TextEmbeddingInput objects created from the input documents.

#### `embed_batch_by_smaller_batches`

```python
embed_batch_by_smaller_batches(
    batch: list[str], subbatch: list[str] = 1
) -> list[list[float]]
```

Embeds a batch of text strings by dividing them into smaller sub-batches.
Args:
batch (List[str]): A list of text strings to be embedded.
subbatch (int, optional): The size of the smaller sub-batches. Defaults to 1.
Returns:
List\[List[float]\]: A list of embeddings, where each embedding is a list of floats.
Raises:
Exception: If embedding fails at the item level, an exception is raised with the error details.

#### `embed_batch`

```python
embed_batch(batch: list[str]) -> list[list[float]]
```

Generate embeddings for a batch of text strings.

Args:
batch (List[str]): A list of text strings to be embedded.

Returns:
List\[List[float]\]: A list of embeddings, where each embedding is a list of floats.

#### `run`

```python
run(documents: list[Document])
```

Processes all documents in batches while adhering to the API's token limit per request.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – A list of documents to embed.

**Returns:**

- – A dictionary with the following keys:
- `documents`: A list of documents with embeddings.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> VertexAIDocumentEmbedder
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>VertexAIDocumentEmbedder</code> – Deserialized component.

## `haystack_integrations.components.embedders.google_vertex.text_embedder`

### `VertexAITextEmbedder`

Embed text using VertexAI Text Embeddings API.

See available models in the official
[Google documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api#syntax).

Usage example:

```python
from haystack_integrations.components.embedders.google_vertex import VertexAITextEmbedder

text_to_embed = "I love pizza!"

text_embedder = VertexAITextEmbedder(model="text-embedding-005")

print(text_embedder.run(text_to_embed))
# {'embedding': [-0.08127457648515701, 0.03399784862995148, -0.05116401985287666, ...]
```

#### `__init__`

```python
__init__(
    model: Literal[
        "text-embedding-004",
        "text-embedding-005",
        "textembedding-gecko-multilingual@001",
        "text-multilingual-embedding-002",
        "text-embedding-large-exp-03-07",
    ],
    task_type: Literal[
        "RETRIEVAL_DOCUMENT",
        "RETRIEVAL_QUERY",
        "SEMANTIC_SIMILARITY",
        "CLASSIFICATION",
        "CLUSTERING",
        "QUESTION_ANSWERING",
        "FACT_VERIFICATION",
        "CODE_RETRIEVAL_QUERY",
    ] = "RETRIEVAL_QUERY",
    gcp_region_name: Optional[Secret] = Secret.from_env_var(
        "GCP_DEFAULT_REGION", strict=False
    ),
    gcp_project_id: Optional[Secret] = Secret.from_env_var(
        "GCP_PROJECT_ID", strict=False
    ),
    progress_bar: bool = True,
    truncate_dim: Optional[int] = None,
) -> None
```

Initializes the TextEmbedder with the specified model, task type, and GCP configuration.

**Parameters:**

- **model** (<code>Literal['text-embedding-004', 'text-embedding-005', 'textembedding-gecko-multilingual@001', 'text-multilingual-embedding-002', 'text-embedding-large-exp-03-07']</code>) – Name of the model to use.
- **task_type** (<code>Literal['RETRIEVAL_DOCUMENT', 'RETRIEVAL_QUERY', 'SEMANTIC_SIMILARITY', 'CLASSIFICATION', 'CLUSTERING', 'QUESTION_ANSWERING', 'FACT_VERIFICATION', 'CODE_RETRIEVAL_QUERY']</code>) – The type of task for which the embeddings are being generated.
  For more information see the official [Google documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api#tasktype).
- **gcp_region_name** (<code>Optional\[Secret\]</code>) – The default location to use when making API calls, if not set uses us-central-1.
- **gcp_project_id** (<code>Optional\[Secret\]</code>) – ID of the GCP project to use. By default, it is set during Google Cloud authentication.
- **progress_bar** (<code>bool</code>) – Whether to display a progress bar during processing.
- **truncate_dim** (<code>Optional\[int\]</code>) – The dimension to truncate the embeddings to, if specified.

#### `run`

```python
run(text: Union[list[Document], list[str], str])
```

Processes text in batches while adhering to the API's token limit per request.

**Parameters:**

- **text** (<code>Union\[list\[Document\], list\[str\], str\]</code>) – The text to embed.

**Returns:**

- – A dictionary with the following keys:
- `embedding`: The embedding of the input text.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> VertexAITextEmbedder
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>VertexAITextEmbedder</code> – Deserialized component.

## `haystack_integrations.components.generators.google_vertex.captioner`

### `VertexAIImageCaptioner`

`VertexAIImageCaptioner` enables text generation using Google Vertex AI imagetext generative model.

Authenticates using Google Cloud Application Default Credentials (ADCs).
For more information see the official [Google documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc).

Usage example:

```python
import requests

from haystack.dataclasses.byte_stream import ByteStream
from haystack_integrations.components.generators.google_vertex import VertexAIImageCaptioner

captioner = VertexAIImageCaptioner()

image = ByteStream(
    data=requests.get(
        "https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/main/integrations/google_vertex/example_assets/robot1.jpg"
    ).content
)
result = captioner.run(image=image)

for caption in result["captions"]:
    print(caption)

>>> two gold robots are standing next to each other in the desert
```

#### `__init__`

```python
__init__(
    *,
    model: str = "imagetext",
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    **kwargs: Optional[str]
)
```

Generate image captions using a Google Vertex AI model.

Authenticates using Google Cloud Application Default Credentials (ADCs).
For more information see the official [Google documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc).

**Parameters:**

- **project_id** (<code>Optional\[str\]</code>) – ID of the GCP project to use. By default, it is set during Google Cloud authentication.
- **model** (<code>str</code>) – Name of the model to use.
- **location** (<code>Optional\[str\]</code>) – The default location to use when making API calls, if not set uses us-central-1.
  Defaults to None.
- **kwargs** – Additional keyword arguments to pass to the model.
  For a list of supported arguments see the `ImageTextModel.get_captions()` documentation.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> VertexAIImageCaptioner
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>VertexAIImageCaptioner</code> – Deserialized component.

#### `run`

```python
run(image: ByteStream)
```

Prompts the model to generate captions for the given image.

**Parameters:**

- **image** (<code>ByteStream</code>) – The image to generate captions for.

**Returns:**

- – A dictionary with the following keys:
- `captions`: A list of captions generated by the model.

## `haystack_integrations.components.generators.google_vertex.chat.gemini`

### `VertexAIGeminiChatGenerator`

`VertexAIGeminiChatGenerator` enables chat completion using Google Gemini models.

Authenticates using Google Cloud Application Default Credentials (ADCs).
For more information see the official [Google documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc).

### Usage example

````python
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.google_vertex import VertexAIGeminiChatGenerator

gemini_chat = VertexAIGeminiChatGenerator()

messages = [ChatMessage.from_user("Tell me the name of a movie")]
res = gemini_chat.run(messages)

print(res["replies"][0].text)
>>> The Shawshank Redemption

#### With Tool calling:

```python
from typing import Annotated
from haystack.utils import Secret
from haystack.dataclasses.chat_message import ChatMessage
from haystack.components.tools import ToolInvoker
from haystack.tools import create_tool_from_function

from haystack_integrations.components.generators.google_vertex import VertexAIGeminiChatGenerator

# example function to get the current weather
def get_current_weather(
    location: Annotated[str, "The city for which to get the weather, e.g. 'San Francisco'"] = "Munich",
    unit: Annotated[str, "The unit for the temperature, e.g. 'celsius'"] = "celsius",
) -> str:
    return f"The weather in {location} is sunny. The temperature is 20 {unit}."

tool = create_tool_from_function(get_current_weather)
tool_invoker = ToolInvoker(tools=[tool])

gemini_chat = VertexAIGeminiChatGenerator(
    model="gemini-2.0-flash-exp",
    tools=[tool],
)
user_message = [ChatMessage.from_user("What is the temperature in celsius in Berlin?")]
replies = gemini_chat.run(messages=user_message)["replies"]
print(replies[0].tool_calls)

# actually invoke the tool
tool_messages = tool_invoker.run(messages=replies)["tool_messages"]
messages = user_message + replies + tool_messages

# transform the tool call result into a human readable message
final_replies = gemini_chat.run(messages=messages)["replies"]
print(final_replies[0].text)
````

#### `__init__`

```python
__init__(
    *,
    model: str = "gemini-1.5-flash",
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    generation_config: Optional[Union[GenerationConfig, dict[str, Any]]] = None,
    safety_settings: Optional[dict[HarmCategory, HarmBlockThreshold]] = None,
    tools: Optional[list[Tool]] = None,
    tool_config: Optional[ToolConfig] = None,
    streaming_callback: Optional[StreamingCallbackT] = None
)
```

`VertexAIGeminiChatGenerator` enables chat completion using Google Gemini models.

Authenticates using Google Cloud Application Default Credentials (ADCs).
For more information see the official [Google documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc).

**Parameters:**

- **model** (<code>str</code>) – Name of the model to use. For available models, see https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models.
- **project_id** (<code>Optional\[str\]</code>) – ID of the GCP project to use. By default, it is set during Google Cloud authentication.
- **location** (<code>Optional\[str\]</code>) – The default location to use when making API calls, if not set uses us-central-1.
  Defaults to None.
- **generation_config** (<code>Optional\[Union\[GenerationConfig, dict\[str, Any\]\]\]</code>) – Configuration for the generation process.
  See the \[GenerationConfig documentation\](https://cloud.google.com/python/docs/reference/aiplatform/latest/vertexai.generative_models.GenerationConfig
  for a list of supported arguments.
- **safety_settings** (<code>Optional\[dict\[HarmCategory, HarmBlockThreshold\]\]</code>) – Safety settings to use when generating content. See the documentation
  for [HarmBlockThreshold](https://cloud.google.com/python/docs/reference/aiplatform/latest/vertexai.generative_models.HarmBlockThreshold)
  and [HarmCategory](https://cloud.google.com/python/docs/reference/aiplatform/latest/vertexai.generative_models.HarmCategory)
  for more details.
- **tools** (<code>Optional\[list\[Tool\]\]</code>) – A list of tools for which the model can prepare calls.
- **tool_config** (<code>Optional\[ToolConfig\]</code>) – The tool config to use. See the documentation for [ToolConfig]
  (https://cloud.google.com/vertex-ai/generative-ai/docs/reference/python/latest/vertexai.generative_models.ToolConfig)
- **streaming_callback** (<code>Optional\[StreamingCallbackT\]</code>) – A callback function that is called when a new token is received from
  the stream. The callback function accepts StreamingChunk as an argument.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> VertexAIGeminiChatGenerator
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>VertexAIGeminiChatGenerator</code> – Deserialized component.

#### `run`

```python
run(
    messages: list[ChatMessage],
    streaming_callback: Optional[StreamingCallbackT] = None,
    *,
    tools: Optional[list[Tool]] = None
)
```

**Parameters:**

- **messages** (<code>list\[ChatMessage\]</code>) – A list of `ChatMessage` instances, representing the input messages.
- **streaming_callback** (<code>Optional\[StreamingCallbackT\]</code>) – A callback function that is called when a new token is received from the stream.
- **tools** (<code>Optional\[list\[Tool\]\]</code>) – A list of tools for which the model can prepare calls. If set, it will override the `tools` parameter set
  during component initialization.

**Returns:**

- – A dictionary containing the following key:
- `replies`: A list containing the generated responses as `ChatMessage` instances.

#### `run_async`

```python
run_async(
    messages: list[ChatMessage],
    streaming_callback: Optional[StreamingCallbackT] = None,
    *,
    tools: Optional[list[Tool]] = None
)
```

Async version of the run method. Generates text based on the provided messages.

**Parameters:**

- **messages** (<code>list\[ChatMessage\]</code>) – A list of `ChatMessage` instances, representing the input messages.
- **streaming_callback** (<code>Optional\[StreamingCallbackT\]</code>) – A callback function that is called when a new token is received from the stream.
- **tools** (<code>Optional\[list\[Tool\]\]</code>) – A list of tools for which the model can prepare calls. If set, it will override the `tools` parameter set
  during component initialization.

**Returns:**

- – A dictionary containing the following key:
- `replies`: A list containing the generated responses as `ChatMessage` instances.

## `haystack_integrations.components.generators.google_vertex.code_generator`

### `VertexAICodeGenerator`

This component enables code generation using Google Vertex AI generative model.

`VertexAICodeGenerator` supports `code-bison`, `code-bison-32k`, and `code-gecko`.

Usage example:

````python
    from haystack_integrations.components.generators.google_vertex import VertexAICodeGenerator

    generator = VertexAICodeGenerator()

    result = generator.run(prefix="def to_json(data):")

    for answer in result["replies"]:
        print(answer)

    >>> ```python
    >>> import json
    >>>
    >>> def to_json(data):
    >>>   """Converts a Python object to a JSON string.
    >>>
    >>>   Args:
    >>>     data: The Python object to convert.
    >>>
    >>>   Returns:
    >>>     A JSON string representing the Python object.
    >>>   """
    >>>
    >>>   return json.dumps(data)
    >>> ```
````

#### `__init__`

```python
__init__(
    *,
    model: str = "code-bison",
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    **kwargs: Optional[str]
)
```

Generate code using a Google Vertex AI model.

Authenticates using Google Cloud Application Default Credentials (ADCs).
For more information see the official [Google documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc).

**Parameters:**

- **project_id** (<code>Optional\[str\]</code>) – ID of the GCP project to use. By default, it is set during Google Cloud authentication.
- **model** (<code>str</code>) – Name of the model to use.
- **location** (<code>Optional\[str\]</code>) – The default location to use when making API calls, if not set uses us-central-1.
- **kwargs** – Additional keyword arguments to pass to the model.
  For a list of supported arguments see the `TextGenerationModel.predict()` documentation.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> VertexAICodeGenerator
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>VertexAICodeGenerator</code> – Deserialized component.

#### `run`

```python
run(prefix: str, suffix: Optional[str] = None)
```

Generate code using a Google Vertex AI model.

**Parameters:**

- **prefix** (<code>str</code>) – Code before the current point.
- **suffix** (<code>Optional\[str\]</code>) – Code after the current point.

**Returns:**

- – A dictionary with the following keys:
- `replies`: A list of generated code snippets.

## `haystack_integrations.components.generators.google_vertex.gemini`

### `VertexAIGeminiGenerator`

`VertexAIGeminiGenerator` enables text generation using Google Gemini models.

Usage example:

```python
from haystack_integrations.components.generators.google_vertex import VertexAIGeminiGenerator


gemini = VertexAIGeminiGenerator()
result = gemini.run(parts = ["What is the most interesting thing you know?"])
for answer in result["replies"]:
    print(answer)

>>> 1. **The Origin of Life:** How and where did life begin? The answers to this ...
>>> 2. **The Unseen Universe:** The vast majority of the universe is ...
>>> 3. **Quantum Entanglement:** This eerie phenomenon in quantum mechanics allows ...
>>> 4. **Time Dilation:** Einstein's theory of relativity revealed that time can ...
>>> 5. **The Fermi Paradox:** Despite the vastness of the universe and the ...
>>> 6. **Biological Evolution:** The idea that life evolves over time through natural ...
>>> 7. **Neuroplasticity:** The brain's ability to adapt and change throughout life, ...
>>> 8. **The Goldilocks Zone:** The concept of the habitable zone, or the Goldilocks zone, ...
>>> 9. **String Theory:** This theoretical framework in physics aims to unify all ...
>>> 10. **Consciousness:** The nature of human consciousness and how it arises ...
```

#### `__init__`

```python
__init__(
    *,
    model: str = "gemini-2.0-flash",
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    generation_config: Optional[Union[GenerationConfig, dict[str, Any]]] = None,
    safety_settings: Optional[dict[HarmCategory, HarmBlockThreshold]] = None,
    system_instruction: Optional[Union[str, ByteStream, Part]] = None,
    streaming_callback: Optional[Callable[[StreamingChunk], None]] = None
)
```

Multi-modal generator using Gemini model via Google Vertex AI.

Authenticates using Google Cloud Application Default Credentials (ADCs).
For more information see the official [Google documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc).

**Parameters:**

- **project_id** (<code>Optional\[str\]</code>) – ID of the GCP project to use. By default, it is set during Google Cloud authentication.
- **model** (<code>str</code>) – Name of the model to use. For available models, see https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models.
- **location** (<code>Optional\[str\]</code>) – The default location to use when making API calls, if not set uses us-central-1.
- **generation_config** (<code>Optional\[Union\[GenerationConfig, dict\[str, Any\]\]\]</code>) – The generation config to use.
  Can either be a [`GenerationConfig`](https://cloud.google.com/python/docs/reference/aiplatform/latest/vertexai.generative_models.GenerationConfig)
  object or a dictionary of parameters.
  Accepted fields are:
  - temperature
  - top_p
  - top_k
  - candidate_count
  - max_output_tokens
  - stop_sequences
- **safety_settings** (<code>Optional\[dict\[HarmCategory, HarmBlockThreshold\]\]</code>) – The safety settings to use. See the documentation
  for [HarmBlockThreshold](https://cloud.google.com/python/docs/reference/aiplatform/latest/vertexai.generative_models.HarmBlockThreshold)
  and [HarmCategory](https://cloud.google.com/python/docs/reference/aiplatform/latest/vertexai.generative_models.HarmCategory)
  for more details.
- **system_instruction** (<code>Optional\[Union\[str, ByteStream, Part\]\]</code>) – Default system instruction to use for generating content.
- **streaming_callback** (<code>Optional\[Callable\\[[StreamingChunk\], None\]\]</code>) – A callback function that is called when a new token is received from the stream.
  The callback function accepts StreamingChunk as an argument.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> VertexAIGeminiGenerator
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>VertexAIGeminiGenerator</code> – Deserialized component.

#### `run`

```python
run(
    parts: Variadic[Union[str, ByteStream, Part]],
    streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
)
```

Generates content using the Gemini model.

**Parameters:**

- **parts** (<code>Variadic\[Union\[str, ByteStream, Part\]\]</code>) – Prompt for the model.
- **streaming_callback** (<code>Optional\[Callable\\[[StreamingChunk\], None\]\]</code>) – A callback function that is called when a new token is received from the stream.

**Returns:**

- – A dictionary with the following keys:
- `replies`: A list of generated content.

## `haystack_integrations.components.generators.google_vertex.image_generator`

### `VertexAIImageGenerator`

This component enables image generation using Google Vertex AI generative model.

Authenticates using Google Cloud Application Default Credentials (ADCs).
For more information see the official [Google documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc).

Usage example:

```python
from pathlib import Path

from haystack_integrations.components.generators.google_vertex import VertexAIImageGenerator

generator = VertexAIImageGenerator()
result = generator.run(prompt="Generate an image of a cute cat")
result["images"][0].to_file(Path("my_image.png"))
```

#### `__init__`

```python
__init__(
    *,
    model: str = "imagegeneration",
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    **kwargs: Optional[str]
)
```

Generates images using a Google Vertex AI model.

Authenticates using Google Cloud Application Default Credentials (ADCs).
For more information see the official [Google documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc).

**Parameters:**

- **project_id** (<code>Optional\[str\]</code>) – ID of the GCP project to use. By default, it is set during Google Cloud authentication.
- **model** (<code>str</code>) – Name of the model to use.
- **location** (<code>Optional\[str\]</code>) – The default location to use when making API calls, if not set uses us-central-1.
- **kwargs** – Additional keyword arguments to pass to the model.
  For a list of supported arguments see the `ImageGenerationModel.generate_images()` documentation.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> VertexAIImageGenerator
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>VertexAIImageGenerator</code> – Deserialized component.

#### `run`

```python
run(prompt: str, negative_prompt: Optional[str] = None)
```

Produces images based on the given prompt.

**Parameters:**

- **prompt** (<code>str</code>) – The prompt to generate images from.
- **negative_prompt** (<code>Optional\[str\]</code>) – A description of what you want to omit in
  the generated images.

**Returns:**

- – A dictionary with the following keys:
- `images`: A list of ByteStream objects, each containing an image.

## `haystack_integrations.components.generators.google_vertex.question_answering`

### `VertexAIImageQA`

This component enables text generation (image captioning) using Google Vertex AI generative models.

Authenticates using Google Cloud Application Default Credentials (ADCs).
For more information see the official [Google documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc).

Usage example:

```python
from haystack.dataclasses.byte_stream import ByteStream
from haystack_integrations.components.generators.google_vertex import VertexAIImageQA

qa = VertexAIImageQA()

image = ByteStream.from_file_path("dog.jpg")

res = qa.run(image=image, question="What color is this dog")

print(res["replies"][0])

>>> white
```

#### `__init__`

```python
__init__(
    *,
    model: str = "imagetext",
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    **kwargs: Optional[str]
)
```

Answers questions about an image using a Google Vertex AI model.

Authenticates using Google Cloud Application Default Credentials (ADCs).
For more information see the official [Google documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc).

**Parameters:**

- **project_id** (<code>Optional\[str\]</code>) – ID of the GCP project to use. By default, it is set during Google Cloud authentication.
- **model** (<code>str</code>) – Name of the model to use.
- **location** (<code>Optional\[str\]</code>) – The default location to use when making API calls, if not set uses us-central-1.
- **kwargs** – Additional keyword arguments to pass to the model.
  For a list of supported arguments see the `ImageTextModel.ask_question()` documentation.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> VertexAIImageQA
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>VertexAIImageQA</code> – Deserialized component.

#### `run`

```python
run(image: ByteStream, question: str)
```

Prompts model to answer a question about an image.

**Parameters:**

- **image** (<code>ByteStream</code>) – The image to ask the question about.
- **question** (<code>str</code>) – The question to ask.

**Returns:**

- – A dictionary with the following keys:
- `replies`: A list of answers to the question.

## `haystack_integrations.components.generators.google_vertex.text_generator`

### `VertexAITextGenerator`

This component enables text generation using Google Vertex AI generative models.

`VertexAITextGenerator` supports `text-bison`, `text-unicorn` and `text-bison-32k` models.

Authenticates using Google Cloud Application Default Credentials (ADCs).
For more information see the official [Google documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc).

Usage example:

````python
    from haystack_integrations.components.generators.google_vertex import VertexAITextGenerator

    generator = VertexAITextGenerator()
    res = generator.run("Tell me a good interview question for a software engineer.")

    print(res["replies"][0])

    >>> **Question:**
    >>> You are given a list of integers and a target sum.
    >>> Find all unique combinations of numbers in the list that add up to the target sum.
    >>>
    >>> **Example:**
    >>>
    >>> ```
    >>> Input: [1, 2, 3, 4, 5], target = 7
    >>> Output: [[1, 2, 4], [3, 4]]
    >>> ```
    >>>
    >>> **Follow-up:** What if the list contains duplicate numbers?
````

#### `__init__`

```python
__init__(
    *,
    model: str = "text-bison",
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    **kwargs: Optional[str]
)
```

Generate text using a Google Vertex AI model.

Authenticates using Google Cloud Application Default Credentials (ADCs).
For more information see the official [Google documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc).

**Parameters:**

- **project_id** (<code>Optional\[str\]</code>) – ID of the GCP project to use. By default, it is set during Google Cloud authentication.
- **model** (<code>str</code>) – Name of the model to use.
- **location** (<code>Optional\[str\]</code>) – The default location to use when making API calls, if not set uses us-central-1.
- **kwargs** – Additional keyword arguments to pass to the model.
  For a list of supported arguments see the `TextGenerationModel.predict()` documentation.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> VertexAITextGenerator
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>VertexAITextGenerator</code> – Deserialized component.

#### `run`

```python
run(prompt: str)
```

Prompts the model to generate text.

**Parameters:**

- **prompt** (<code>str</code>) – The prompt to use for text generation.

**Returns:**

- – A dictionary with the following keys:
- `replies`: A list of generated replies.
- `safety_attributes`: A dictionary with the [safety scores](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/responsible-ai#safety_attribute_descriptions)
  of each answer.
- `citations`: A list of citations for each answer.
