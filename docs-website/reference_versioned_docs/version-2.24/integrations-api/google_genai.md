---
title: "Google GenAI"
id: integrations-google-genai
description: "Google GenAI integration for Haystack"
slug: "/integrations-google-genai"
---


## haystack_integrations.components.embedders.google_genai.document_embedder

### GoogleGenAIDocumentEmbedder

Computes document embeddings using Google AI models.

### Authentication examples

**1. Gemini Developer API (API Key Authentication)**

````python
from haystack_integrations.components.embedders.google_genai import GoogleGenAIDocumentEmbedder

# export the environment variable (GOOGLE_API_KEY or GEMINI_API_KEY)
document_embedder = GoogleGenAIDocumentEmbedder(model="gemini-embedding-001")

**2. Vertex AI (Application Default Credentials)**
```python
from haystack_integrations.components.embedders.google_genai import GoogleGenAIDocumentEmbedder

# Using Application Default Credentials (requires gcloud auth setup)
document_embedder = GoogleGenAIDocumentEmbedder(
    api="vertex",
    vertex_ai_project="my-project",
    vertex_ai_location="us-central1",
    model="gemini-embedding-001"
)
````

**3. Vertex AI (API Key Authentication)**

```python
from haystack_integrations.components.embedders.google_genai import GoogleGenAIDocumentEmbedder

# export the environment variable (GOOGLE_API_KEY or GEMINI_API_KEY)
document_embedder = GoogleGenAIDocumentEmbedder(
    api="vertex",
    model="gemini-embedding-001"
)
```

### Usage example

```python
from haystack import Document
from haystack_integrations.components.embedders.google_genai import GoogleGenAIDocumentEmbedder

doc = Document(content="I love pizza!")

document_embedder = GoogleGenAIDocumentEmbedder()

result = document_embedder.run([doc])
print(result['documents'][0].embedding)

# [0.017020374536514282, -0.023255806416273117, ...]
```

#### __init__

```python
__init__(
    *,
    api_key: Secret = Secret.from_env_var(
        ["GOOGLE_API_KEY", "GEMINI_API_KEY"], strict=False
    ),
    api: Literal["gemini", "vertex"] = "gemini",
    vertex_ai_project: str | None = None,
    vertex_ai_location: str | None = None,
    model: str = "gemini-embedding-001",
    prefix: str = "",
    suffix: str = "",
    batch_size: int = 32,
    progress_bar: bool = True,
    meta_fields_to_embed: list[str] | None = None,
    embedding_separator: str = "\n",
    config: dict[str, Any] | None = None
) -> None
```

Creates an GoogleGenAIDocumentEmbedder component.

**Parameters:**

- **api_key** (<code>Secret</code>) – Google API key, defaults to the `GOOGLE_API_KEY` and `GEMINI_API_KEY` environment variables.
  Not needed if using Vertex AI with Application Default Credentials.
  Go to https://aistudio.google.com/app/apikey for a Gemini API key.
  Go to https://cloud.google.com/vertex-ai/generative-ai/docs/start/api-keys for a Vertex AI API key.
- **api** (<code>Literal['gemini', 'vertex']</code>) – Which API to use. Either "gemini" for the Gemini Developer API or "vertex" for Vertex AI.
- **vertex_ai_project** (<code>str | None</code>) – Google Cloud project ID for Vertex AI. Required when using Vertex AI with
  Application Default Credentials.
- **vertex_ai_location** (<code>str | None</code>) – Google Cloud location for Vertex AI (e.g., "us-central1", "europe-west1").
  Required when using Vertex AI with Application Default Credentials.
- **model** (<code>str</code>) – The name of the model to use for calculating embeddings.
  The default model is `text-embedding-ada-002`.
- **prefix** (<code>str</code>) – A string to add at the beginning of each text.
- **suffix** (<code>str</code>) – A string to add at the end of each text.
- **batch_size** (<code>int</code>) – Number of documents to embed at once.
- **progress_bar** (<code>bool</code>) – If `True`, shows a progress bar when running.
- **meta_fields_to_embed** (<code>list\[str\] | None</code>) – List of metadata fields to embed along with the document text.
- **embedding_separator** (<code>str</code>) – Separator used to concatenate the metadata fields to the document text.
- **config** (<code>dict\[str, Any\] | None</code>) – A dictionary of keyword arguments to configure embedding content configuration `types.EmbedContentConfig`.
  If not specified, it defaults to `{"task_type": "SEMANTIC_SIMILARITY"}`.
  For more information, see the [Google AI Task types](https://ai.google.dev/gemini-api/docs/embeddings#task-types).

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> GoogleGenAIDocumentEmbedder
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>GoogleGenAIDocumentEmbedder</code> – Deserialized component.

#### run

```python
run(documents: list[Document]) -> dict[str, list[Document]] | dict[str, Any]
```

Embeds a list of documents.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – A list of documents to embed.

**Returns:**

- <code>dict\[str, list\[Document\]\] | dict\[str, Any\]</code> – A dictionary with the following keys:
- `documents`: A list of documents with embeddings.
- `meta`: Information about the usage of the model.

#### run_async

```python
run_async(
    documents: list[Document],
) -> dict[str, list[Document]] | dict[str, Any]
```

Embeds a list of documents asynchronously.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – A list of documents to embed.

**Returns:**

- <code>dict\[str, list\[Document\]\] | dict\[str, Any\]</code> – A dictionary with the following keys:
- `documents`: A list of documents with embeddings.
- `meta`: Information about the usage of the model.

## haystack_integrations.components.embedders.google_genai.text_embedder

### GoogleGenAITextEmbedder

Embeds strings using Google AI models.

You can use it to embed user query and send it to an embedding Retriever.

### Authentication examples

**1. Gemini Developer API (API Key Authentication)**

````python
from haystack_integrations.components.embedders.google_genai import GoogleGenAITextEmbedder

# export the environment variable (GOOGLE_API_KEY or GEMINI_API_KEY)
text_embedder = GoogleGenAITextEmbedder(model="gemini-embedding-001")

**2. Vertex AI (Application Default Credentials)**
```python
from haystack_integrations.components.embedders.google_genai import GoogleGenAITextEmbedder

# Using Application Default Credentials (requires gcloud auth setup)
text_embedder = GoogleGenAITextEmbedder(
    api="vertex",
    vertex_ai_project="my-project",
    vertex_ai_location="us-central1",
    model="gemini-embedding-001"
)
````

**3. Vertex AI (API Key Authentication)**

```python
from haystack_integrations.components.embedders.google_genai import GoogleGenAITextEmbedder

# export the environment variable (GOOGLE_API_KEY or GEMINI_API_KEY)
text_embedder = GoogleGenAITextEmbedder(
    api="vertex",
    model="gemini-embedding-001"
)
```

### Usage example

```python
from haystack_integrations.components.embedders.google_genai import GoogleGenAITextEmbedder

text_to_embed = "I love pizza!"

text_embedder = GoogleGenAITextEmbedder()

print(text_embedder.run(text_to_embed))

# {'embedding': [0.017020374536514282, -0.023255806416273117, ...],
# 'meta': {'model': 'gemini-embedding-001-v2',
#          'usage': {'prompt_tokens': 4, 'total_tokens': 4}}}
```

#### __init__

```python
__init__(
    *,
    api_key: Secret = Secret.from_env_var(
        ["GOOGLE_API_KEY", "GEMINI_API_KEY"], strict=False
    ),
    api: Literal["gemini", "vertex"] = "gemini",
    vertex_ai_project: str | None = None,
    vertex_ai_location: str | None = None,
    model: str = "gemini-embedding-001",
    prefix: str = "",
    suffix: str = "",
    config: dict[str, Any] | None = None
) -> None
```

Creates an GoogleGenAITextEmbedder component.

**Parameters:**

- **api_key** (<code>Secret</code>) – Google API key, defaults to the `GOOGLE_API_KEY` and `GEMINI_API_KEY` environment variables.
  Not needed if using Vertex AI with Application Default Credentials.
  Go to https://aistudio.google.com/app/apikey for a Gemini API key.
  Go to https://cloud.google.com/vertex-ai/generative-ai/docs/start/api-keys for a Vertex AI API key.
- **api** (<code>Literal['gemini', 'vertex']</code>) – Which API to use. Either "gemini" for the Gemini Developer API or "vertex" for Vertex AI.
- **vertex_ai_project** (<code>str | None</code>) – Google Cloud project ID for Vertex AI. Required when using Vertex AI with
  Application Default Credentials.
- **vertex_ai_location** (<code>str | None</code>) – Google Cloud location for Vertex AI (e.g., "us-central1", "europe-west1").
  Required when using Vertex AI with Application Default Credentials.
- **model** (<code>str</code>) – The name of the model to use for calculating embeddings.
  The default model is `gemini-embedding-001`.
- **prefix** (<code>str</code>) – A string to add at the beginning of each text to embed.
- **suffix** (<code>str</code>) – A string to add at the end of each text to embed.
- **config** (<code>dict\[str, Any\] | None</code>) – A dictionary of keyword arguments to configure embedding content configuration `types.EmbedContentConfig`.
  If not specified, it defaults to `{"task_type": "SEMANTIC_SIMILARITY"}`.
  For more information, see the [Google AI Task types](https://ai.google.dev/gemini-api/docs/embeddings#task-types).

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> GoogleGenAITextEmbedder
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>GoogleGenAITextEmbedder</code> – Deserialized component.

#### run

```python
run(text: str) -> dict[str, list[float]] | dict[str, Any]
```

Embeds a single string.

**Parameters:**

- **text** (<code>str</code>) – Text to embed.

**Returns:**

- <code>dict\[str, list\[float\]\] | dict\[str, Any\]</code> – A dictionary with the following keys:
- `embedding`: The embedding of the input text.
- `meta`: Information about the usage of the model.

#### run_async

```python
run_async(text: str) -> dict[str, list[float]] | dict[str, Any]
```

Asynchronously embed a single string.

This is the asynchronous version of the `run` method. It has the same parameters and return values
but can be used with `await` in async code.

**Parameters:**

- **text** (<code>str</code>) – Text to embed.

**Returns:**

- <code>dict\[str, list\[float\]\] | dict\[str, Any\]</code> – A dictionary with the following keys:
- `embedding`: The embedding of the input text.
- `meta`: Information about the usage of the model.

## haystack_integrations.components.generators.google_genai.chat.chat_generator

### GoogleGenAIChatGenerator

A component for generating chat completions using Google's Gemini models via the Google Gen AI SDK.

Supports models like gemini-2.5-flash and other Gemini variants. For Gemini 2.5 series models,
enables thinking features via `generation_kwargs={"thinking_budget": value}`.

### Thinking Support (Gemini 2.5 Series)

- **Reasoning transparency**: Models can show their reasoning process
- **Thought signatures**: Maintains thought context across multi-turn conversations with tools
- **Configurable thinking budgets**: Control token allocation for reasoning

Configure thinking behavior:

- `thinking_budget: -1`: Dynamic allocation (default)
- `thinking_budget: 0`: Disable thinking (Flash/Flash-Lite only)
- `thinking_budget: N`: Set explicit token budget

### Multi-Turn Thinking with Thought Signatures

Gemini uses **thought signatures** when tools are present - encrypted "save states" that maintain
context across turns. Include previous assistant responses in chat history for context preservation.

### Authentication

**Gemini Developer API**: Set `GOOGLE_API_KEY` or `GEMINI_API_KEY` environment variable
**Vertex AI**: Use `api="vertex"` with Application Default Credentials or API key

### Authentication Examples

**1. Gemini Developer API (API Key Authentication)**

```python
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator

# export the environment variable (GOOGLE_API_KEY or GEMINI_API_KEY)
chat_generator = GoogleGenAIChatGenerator(model="gemini-2.5-flash")
```

**2. Vertex AI (Application Default Credentials)**

```python
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator

# Using Application Default Credentials (requires gcloud auth setup)
chat_generator = GoogleGenAIChatGenerator(
    api="vertex",
    vertex_ai_project="my-project",
    vertex_ai_location="us-central1",
    model="gemini-2.5-flash",
)
```

**3. Vertex AI (API Key Authentication)**

```python
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator

# export the environment variable (GOOGLE_API_KEY or GEMINI_API_KEY)
chat_generator = GoogleGenAIChatGenerator(
    api="vertex",
    model="gemini-2.5-flash",
)
```

### Usage example

```python
from haystack.dataclasses.chat_message import ChatMessage
from haystack.tools import Tool, Toolset
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator

# Initialize the chat generator with thinking support
chat_generator = GoogleGenAIChatGenerator(
    model="gemini-2.5-flash",
    generation_kwargs={"thinking_budget": 1024}  # Enable thinking with 1024 token budget
)

# Generate a response
messages = [ChatMessage.from_user("Tell me about the future of AI")]
response = chat_generator.run(messages=messages)
print(response["replies"][0].text)

# Access reasoning content if available
message = response["replies"][0]
if message.reasonings:
    for reasoning in message.reasonings:
        print("Reasoning:", reasoning.reasoning_text)

# Tool usage example with thinking
def weather_function(city: str):
    return f"The weather in {city} is sunny and 25°C"

weather_tool = Tool(
    name="weather",
    description="Get weather information for a city",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    function=weather_function
)

# Can use either List[Tool] or Toolset
chat_generator_with_tools = GoogleGenAIChatGenerator(
    model="gemini-2.5-flash",
    tools=[weather_tool],  # or tools=Toolset([weather_tool])
    generation_kwargs={"thinking_budget": -1}  # Dynamic thinking allocation
)

messages = [ChatMessage.from_user("What's the weather in Paris?")]
response = chat_generator_with_tools.run(messages=messages)
```

#### __init__

```python
__init__(
    *,
    api_key: Secret = Secret.from_env_var(
        ["GOOGLE_API_KEY", "GEMINI_API_KEY"], strict=False
    ),
    api: Literal["gemini", "vertex"] = "gemini",
    vertex_ai_project: str | None = None,
    vertex_ai_location: str | None = None,
    model: str = "gemini-2.5-flash",
    generation_kwargs: dict[str, Any] | None = None,
    safety_settings: list[dict[str, Any]] | None = None,
    streaming_callback: StreamingCallbackT | None = None,
    tools: ToolsType | None = None
)
```

Initialize a GoogleGenAIChatGenerator instance.

**Parameters:**

- **api_key** (<code>Secret</code>) – Google API key, defaults to the `GOOGLE_API_KEY` and `GEMINI_API_KEY` environment variables.
  Not needed if using Vertex AI with Application Default Credentials.
  Go to https://aistudio.google.com/app/apikey for a Gemini API key.
  Go to https://cloud.google.com/vertex-ai/generative-ai/docs/start/api-keys for a Vertex AI API key.
- **api** (<code>Literal['gemini', 'vertex']</code>) – Which API to use. Either "gemini" for the Gemini Developer API or "vertex" for Vertex AI.
- **vertex_ai_project** (<code>str | None</code>) – Google Cloud project ID for Vertex AI. Required when using Vertex AI with
  Application Default Credentials.
- **vertex_ai_location** (<code>str | None</code>) – Google Cloud location for Vertex AI (e.g., "us-central1", "europe-west1").
  Required when using Vertex AI with Application Default Credentials.
- **model** (<code>str</code>) – Name of the model to use (e.g., "gemini-2.5-flash")
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Configuration for generation (temperature, max_tokens, etc.).
  For Gemini 2.5 series, supports `thinking_budget` to configure thinking behavior:
- `thinking_budget`: int, controls thinking token allocation
  - `-1`: Dynamic (default for most models)
  - `0`: Disable thinking (Flash/Flash-Lite only)
  - Positive integer: Set explicit budget
    For Gemini 3 series and newer, supports `thinking_level` to configure thinking depth:
- `thinking_level`: str, controls thinking (https://ai.google.dev/gemini-api/docs/thinking#levels-budgets)
  - `minimal`: Matches the "no thinking" setting for most queries. The model may think very minimally for
    complex coding tasks. Minimizes latency for chat or high throughput applications.
  - `low`: Minimizes latency and cost. Best for simple instruction following, chat, or high-throughput
    applications.
  - `medium`: Balanced thinking for most tasks.
  - `high`: (Default, dynamic): Maximizes reasoning depth. The model may take significantly longer to reach
    a first token, but the output will be more carefully reasoned.
- **safety_settings** (<code>list\[dict\[str, Any\]\] | None</code>) – Safety settings for content filtering
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – A callback function that is called when a new token is received from the stream.
- **tools** (<code>ToolsType | None</code>) – A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
  Each tool should have a unique name.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> GoogleGenAIChatGenerator
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>GoogleGenAIChatGenerator</code> – Deserialized component.

#### run

```python
run(
    messages: list[ChatMessage],
    generation_kwargs: dict[str, Any] | None = None,
    safety_settings: list[dict[str, Any]] | None = None,
    streaming_callback: StreamingCallbackT | None = None,
    tools: ToolsType | None = None,
) -> dict[str, Any]
```

Run the Google Gen AI chat generator on the given input data.

**Parameters:**

- **messages** (<code>list\[ChatMessage\]</code>) – A list of ChatMessage instances representing the input messages.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Configuration for generation. If provided, it will override
  the default config. Supports `thinking_budget` for Gemini 2.5 series thinking configuration.
- **safety_settings** (<code>list\[dict\[str, Any\]\] | None</code>) – Safety settings for content filtering. If provided, it will override the
  default settings.
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – A callback function that is called when a new token is
  received from the stream.
- **tools** (<code>ToolsType | None</code>) – A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
  If provided, it will override the tools set during initialization.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the following keys:
- `replies`: A list containing the generated ChatMessage responses.

**Raises:**

- <code>RuntimeError</code> – If there is an error in the Google Gen AI chat generation.
- <code>ValueError</code> – If a ChatMessage does not contain at least one of TextContent, ToolCall, or
  ToolCallResult or if the role in ChatMessage is different from User, System, Assistant.

#### run_async

```python
run_async(
    messages: list[ChatMessage],
    generation_kwargs: dict[str, Any] | None = None,
    safety_settings: list[dict[str, Any]] | None = None,
    streaming_callback: StreamingCallbackT | None = None,
    tools: ToolsType | None = None,
) -> dict[str, Any]
```

Async version of the run method. Run the Google Gen AI chat generator on the given input data.

**Parameters:**

- **messages** (<code>list\[ChatMessage\]</code>) – A list of ChatMessage instances representing the input messages.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Configuration for generation. If provided, it will override
  the default config. Supports `thinking_budget` for Gemini 2.5 series thinking configuration.
  See https://ai.google.dev/gemini-api/docs/thinking for possible values.
- **safety_settings** (<code>list\[dict\[str, Any\]\] | None</code>) – Safety settings for content filtering. If provided, it will override the
  default settings.
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – A callback function that is called when a new token is
  received from the stream.
- **tools** (<code>ToolsType | None</code>) – A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
  If provided, it will override the tools set during initialization.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the following keys:
- `replies`: A list containing the generated ChatMessage responses.

**Raises:**

- <code>RuntimeError</code> – If there is an error in the async Google Gen AI chat generation.
- <code>ValueError</code> – If a ChatMessage does not contain at least one of TextContent, ToolCall, or
  ToolCallResult or if the role in ChatMessage is different from User, System, Assistant.
