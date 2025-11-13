---
title: "Google GenAI"
id: integrations-google-genai
description: "Google GenAI integration for Haystack"
slug: "/integrations-google-genai"
---

<a id="haystack_integrations.components.generators.google_genai.chat.chat_generator"></a>

## Module haystack\_integrations.components.generators.google\_genai.chat.chat\_generator

<a id="haystack_integrations.components.generators.google_genai.chat.chat_generator.GoogleGenAIChatGenerator"></a>

### GoogleGenAIChatGenerator

A component for generating chat completions using Google's Gemini models via the Google Gen AI SDK.

Supports models like gemini-2.0-flash and other Gemini variants. For Gemini 2.5 series models,
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
chat_generator = GoogleGenAIChatGenerator(model="gemini-2.0-flash")
```

**2. Vertex AI (Application Default Credentials)**
```python
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator

# Using Application Default Credentials (requires gcloud auth setup)
chat_generator = GoogleGenAIChatGenerator(
    api="vertex",
    vertex_ai_project="my-project",
    vertex_ai_location="us-central1",
    model="gemini-2.0-flash"
)
```

**3. Vertex AI (API Key Authentication)**
```python
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator

# export the environment variable (GOOGLE_API_KEY or GEMINI_API_KEY)
chat_generator = GoogleGenAIChatGenerator(
    api="vertex",
    model="gemini-2.0-flash"
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

<a id="haystack_integrations.components.generators.google_genai.chat.chat_generator.GoogleGenAIChatGenerator.__init__"></a>

#### GoogleGenAIChatGenerator.\_\_init\_\_

```python
def __init__(*,
             api_key: Secret = Secret.from_env_var(
                 ["GOOGLE_API_KEY", "GEMINI_API_KEY"], strict=False),
             api: Literal["gemini", "vertex"] = "gemini",
             vertex_ai_project: Optional[str] = None,
             vertex_ai_location: Optional[str] = None,
             model: str = "gemini-2.0-flash",
             generation_kwargs: Optional[Dict[str, Any]] = None,
             safety_settings: Optional[List[Dict[str, Any]]] = None,
             streaming_callback: Optional[StreamingCallbackT] = None,
             tools: Optional[ToolsType] = None)
```

Initialize a GoogleGenAIChatGenerator instance.

**Arguments**:

- `api_key`: Google API key, defaults to the `GOOGLE_API_KEY` and `GEMINI_API_KEY` environment variables.
Not needed if using Vertex AI with Application Default Credentials.
Go to https://aistudio.google.com/app/apikey for a Gemini API key.
Go to https://cloud.google.com/vertex-ai/generative-ai/docs/start/api-keys for a Vertex AI API key.
- `api`: Which API to use. Either "gemini" for the Gemini Developer API or "vertex" for Vertex AI.
- `vertex_ai_project`: Google Cloud project ID for Vertex AI. Required when using Vertex AI with
Application Default Credentials.
- `vertex_ai_location`: Google Cloud location for Vertex AI (e.g., "us-central1", "europe-west1").
Required when using Vertex AI with Application Default Credentials.
- `model`: Name of the model to use (e.g., "gemini-2.0-flash")
- `generation_kwargs`: Configuration for generation (temperature, max_tokens, etc.).
For Gemini 2.5 series, supports `thinking_budget` to configure thinking behavior:
- `thinking_budget`: int, controls thinking token allocation
  - `-1`: Dynamic (default for most models)
  - `0`: Disable thinking (Flash/Flash-Lite only)
  - Positive integer: Set explicit budget
- `safety_settings`: Safety settings for content filtering
- `streaming_callback`: A callback function that is called when a new token is received from the stream.
- `tools`: A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
Each tool should have a unique name.

<a id="haystack_integrations.components.generators.google_genai.chat.chat_generator.GoogleGenAIChatGenerator.to_dict"></a>

#### GoogleGenAIChatGenerator.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.generators.google_genai.chat.chat_generator.GoogleGenAIChatGenerator.from_dict"></a>

#### GoogleGenAIChatGenerator.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "GoogleGenAIChatGenerator"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.generators.google_genai.chat.chat_generator.GoogleGenAIChatGenerator.run"></a>

#### GoogleGenAIChatGenerator.run

```python
@component.output_types(replies=List[ChatMessage])
def run(messages: List[ChatMessage],
        generation_kwargs: Optional[Dict[str, Any]] = None,
        safety_settings: Optional[List[Dict[str, Any]]] = None,
        streaming_callback: Optional[StreamingCallbackT] = None,
        tools: Optional[ToolsType] = None) -> Dict[str, Any]
```

Run the Google Gen AI chat generator on the given input data.

**Arguments**:

- `messages`: A list of ChatMessage instances representing the input messages.
- `generation_kwargs`: Configuration for generation. If provided, it will override
the default config. Supports `thinking_budget` for Gemini 2.5 series thinking configuration.
- `safety_settings`: Safety settings for content filtering. If provided, it will override the
default settings.
- `streaming_callback`: A callback function that is called when a new token is
received from the stream.
- `tools`: A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
If provided, it will override the tools set during initialization.

**Raises**:

- `RuntimeError`: If there is an error in the Google Gen AI chat generation.
- `ValueError`: If a ChatMessage does not contain at least one of TextContent, ToolCall, or
ToolCallResult or if the role in ChatMessage is different from User, System, Assistant.

**Returns**:

A dictionary with the following keys:
- `replies`: A list containing the generated ChatMessage responses.

<a id="haystack_integrations.components.generators.google_genai.chat.chat_generator.GoogleGenAIChatGenerator.run_async"></a>

#### GoogleGenAIChatGenerator.run\_async

```python
@component.output_types(replies=List[ChatMessage])
async def run_async(messages: List[ChatMessage],
                    generation_kwargs: Optional[Dict[str, Any]] = None,
                    safety_settings: Optional[List[Dict[str, Any]]] = None,
                    streaming_callback: Optional[StreamingCallbackT] = None,
                    tools: Optional[ToolsType] = None) -> Dict[str, Any]
```

Async version of the run method. Run the Google Gen AI chat generator on the given input data.

**Arguments**:

- `messages`: A list of ChatMessage instances representing the input messages.
- `generation_kwargs`: Configuration for generation. If provided, it will override
the default config. Supports `thinking_budget` for Gemini 2.5 series thinking configuration.
See https://ai.google.dev/gemini-api/docs/thinking for possible values.
- `safety_settings`: Safety settings for content filtering. If provided, it will override the
default settings.
- `streaming_callback`: A callback function that is called when a new token is
received from the stream.
- `tools`: A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
If provided, it will override the tools set during initialization.

**Raises**:

- `RuntimeError`: If there is an error in the async Google Gen AI chat generation.
- `ValueError`: If a ChatMessage does not contain at least one of TextContent, ToolCall, or
ToolCallResult or if the role in ChatMessage is different from User, System, Assistant.

**Returns**:

A dictionary with the following keys:
- `replies`: A list containing the generated ChatMessage responses.

<a id="haystack_integrations.components.embedders.google_genai.document_embedder"></a>

## Module haystack\_integrations.components.embedders.google\_genai.document\_embedder

<a id="haystack_integrations.components.embedders.google_genai.document_embedder.GoogleGenAIDocumentEmbedder"></a>

### GoogleGenAIDocumentEmbedder

Computes document embeddings using Google AI models.

### Authentication examples

**1. Gemini Developer API (API Key Authentication)**
```python
from haystack_integrations.components.embedders.google_genai import GoogleGenAIDocumentEmbedder

# export the environment variable (GOOGLE_API_KEY or GEMINI_API_KEY)
document_embedder = GoogleGenAIDocumentEmbedder(model="text-embedding-004")

**2. Vertex AI (Application Default Credentials)**
```python
from haystack_integrations.components.embedders.google_genai import GoogleGenAIDocumentEmbedder

__Using Application Default Credentials (requires gcloud auth setup)__

document_embedder = GoogleGenAIDocumentEmbedder(
    api="vertex",
    vertex_ai_project="my-project",
    vertex_ai_location="us-central1",
    model="text-embedding-004"
)
```

**3. Vertex AI (API Key Authentication)**
```python
from haystack_integrations.components.embedders.google_genai import GoogleGenAIDocumentEmbedder

__export the environment variable (GOOGLE_API_KEY or GEMINI_API_KEY)__

document_embedder = GoogleGenAIDocumentEmbedder(
    api="vertex",
    model="text-embedding-004"
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

__[0.017020374536514282, -0.023255806416273117, ...]__

```

<a id="haystack_integrations.components.embedders.google_genai.document_embedder.GoogleGenAIDocumentEmbedder.__init__"></a>

#### GoogleGenAIDocumentEmbedder.\_\_init\_\_

```python
def __init__(*,
             api_key: Secret = Secret.from_env_var(
                 ["GOOGLE_API_KEY", "GEMINI_API_KEY"], strict=False),
             api: Literal["gemini", "vertex"] = "gemini",
             vertex_ai_project: Optional[str] = None,
             vertex_ai_location: Optional[str] = None,
             model: str = "text-embedding-004",
             prefix: str = "",
             suffix: str = "",
             batch_size: int = 32,
             progress_bar: bool = True,
             meta_fields_to_embed: Optional[List[str]] = None,
             embedding_separator: str = "\n",
             config: Optional[Dict[str, Any]] = None) -> None
```

Creates an GoogleGenAIDocumentEmbedder component.

**Arguments**:

- `api_key`: Google API key, defaults to the `GOOGLE_API_KEY` and `GEMINI_API_KEY` environment variables.
Not needed if using Vertex AI with Application Default Credentials.
Go to https://aistudio.google.com/app/apikey for a Gemini API key.
Go to https://cloud.google.com/vertex-ai/generative-ai/docs/start/api-keys for a Vertex AI API key.
- `api`: Which API to use. Either "gemini" for the Gemini Developer API or "vertex" for Vertex AI.
- `vertex_ai_project`: Google Cloud project ID for Vertex AI. Required when using Vertex AI with
Application Default Credentials.
- `vertex_ai_location`: Google Cloud location for Vertex AI (e.g., "us-central1", "europe-west1").
Required when using Vertex AI with Application Default Credentials.
- `model`: The name of the model to use for calculating embeddings.
The default model is `text-embedding-ada-002`.
- `prefix`: A string to add at the beginning of each text.
- `suffix`: A string to add at the end of each text.
- `batch_size`: Number of documents to embed at once.
- `progress_bar`: If `True`, shows a progress bar when running.
- `meta_fields_to_embed`: List of metadata fields to embed along with the document text.
- `embedding_separator`: Separator used to concatenate the metadata fields to the document text.
- `config`: A dictionary of keyword arguments to configure embedding content configuration `types.EmbedContentConfig`.
If not specified, it defaults to `{"task_type": "SEMANTIC_SIMILARITY"}`.
For more information, see the [Google AI Task types](https://ai.google.dev/gemini-api/docs/embeddings#task-types).

<a id="haystack_integrations.components.embedders.google_genai.document_embedder.GoogleGenAIDocumentEmbedder.to_dict"></a>

#### GoogleGenAIDocumentEmbedder.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.embedders.google_genai.document_embedder.GoogleGenAIDocumentEmbedder.from_dict"></a>

#### GoogleGenAIDocumentEmbedder.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "GoogleGenAIDocumentEmbedder"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.embedders.google_genai.document_embedder.GoogleGenAIDocumentEmbedder.run"></a>

#### GoogleGenAIDocumentEmbedder.run

```python
@component.output_types(documents=List[Document], meta=Dict[str, Any])
def run(
    documents: List[Document]
) -> Union[Dict[str, List[Document]], Dict[str, Any]]
```

Embeds a list of documents.

**Arguments**:

- `documents`: A list of documents to embed.

**Returns**:

A dictionary with the following keys:
- `documents`: A list of documents with embeddings.
- `meta`: Information about the usage of the model.

<a id="haystack_integrations.components.embedders.google_genai.document_embedder.GoogleGenAIDocumentEmbedder.run_async"></a>

#### GoogleGenAIDocumentEmbedder.run\_async

```python
@component.output_types(documents=List[Document], meta=Dict[str, Any])
async def run_async(
    documents: List[Document]
) -> Union[Dict[str, List[Document]], Dict[str, Any]]
```

Embeds a list of documents asynchronously.

**Arguments**:

- `documents`: A list of documents to embed.

**Returns**:

A dictionary with the following keys:
- `documents`: A list of documents with embeddings.
- `meta`: Information about the usage of the model.

<a id="haystack_integrations.components.embedders.google_genai.text_embedder"></a>

## Module haystack\_integrations.components.embedders.google\_genai.text\_embedder

<a id="haystack_integrations.components.embedders.google_genai.text_embedder.GoogleGenAITextEmbedder"></a>

### GoogleGenAITextEmbedder

Embeds strings using Google AI models.

You can use it to embed user query and send it to an embedding Retriever.

### Authentication examples

**1. Gemini Developer API (API Key Authentication)**
```python
from haystack_integrations.components.embedders.google_genai import GoogleGenAITextEmbedder

# export the environment variable (GOOGLE_API_KEY or GEMINI_API_KEY)
text_embedder = GoogleGenAITextEmbedder(model="text-embedding-004")

**2. Vertex AI (Application Default Credentials)**
```python
from haystack_integrations.components.embedders.google_genai import GoogleGenAITextEmbedder

__Using Application Default Credentials (requires gcloud auth setup)__

text_embedder = GoogleGenAITextEmbedder(
    api="vertex",
    vertex_ai_project="my-project",
    vertex_ai_location="us-central1",
    model="text-embedding-004"
)
```

**3. Vertex AI (API Key Authentication)**
```python
from haystack_integrations.components.embedders.google_genai import GoogleGenAITextEmbedder

__export the environment variable (GOOGLE_API_KEY or GEMINI_API_KEY)__

text_embedder = GoogleGenAITextEmbedder(
    api="vertex",
    model="text-embedding-004"
)
```


### Usage example

```python
from haystack_integrations.components.embedders.google_genai import GoogleGenAITextEmbedder

text_to_embed = "I love pizza!"

text_embedder = GoogleGenAITextEmbedder()

print(text_embedder.run(text_to_embed))

__{'embedding': [0.017020374536514282, -0.023255806416273117, ...],__

__'meta': {'model': 'text-embedding-004-v2',__

__         'usage': {'prompt_tokens': 4, 'total_tokens': 4}}}__

```

<a id="haystack_integrations.components.embedders.google_genai.text_embedder.GoogleGenAITextEmbedder.__init__"></a>

#### GoogleGenAITextEmbedder.\_\_init\_\_

```python
def __init__(*,
             api_key: Secret = Secret.from_env_var(
                 ["GOOGLE_API_KEY", "GEMINI_API_KEY"], strict=False),
             api: Literal["gemini", "vertex"] = "gemini",
             vertex_ai_project: Optional[str] = None,
             vertex_ai_location: Optional[str] = None,
             model: str = "text-embedding-004",
             prefix: str = "",
             suffix: str = "",
             config: Optional[Dict[str, Any]] = None) -> None
```

Creates an GoogleGenAITextEmbedder component.

**Arguments**:

- `api_key`: Google API key, defaults to the `GOOGLE_API_KEY` and `GEMINI_API_KEY` environment variables.
Not needed if using Vertex AI with Application Default Credentials.
Go to https://aistudio.google.com/app/apikey for a Gemini API key.
Go to https://cloud.google.com/vertex-ai/generative-ai/docs/start/api-keys for a Vertex AI API key.
- `api`: Which API to use. Either "gemini" for the Gemini Developer API or "vertex" for Vertex AI.
- `vertex_ai_project`: Google Cloud project ID for Vertex AI. Required when using Vertex AI with
Application Default Credentials.
- `vertex_ai_location`: Google Cloud location for Vertex AI (e.g., "us-central1", "europe-west1").
Required when using Vertex AI with Application Default Credentials.
- `model`: The name of the model to use for calculating embeddings.
The default model is `text-embedding-004`.
- `prefix`: A string to add at the beginning of each text to embed.
- `suffix`: A string to add at the end of each text to embed.
- `config`: A dictionary of keyword arguments to configure embedding content configuration `types.EmbedContentConfig`.
If not specified, it defaults to `{"task_type": "SEMANTIC_SIMILARITY"}`.
For more information, see the [Google AI Task types](https://ai.google.dev/gemini-api/docs/embeddings#task-types).

<a id="haystack_integrations.components.embedders.google_genai.text_embedder.GoogleGenAITextEmbedder.to_dict"></a>

#### GoogleGenAITextEmbedder.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.embedders.google_genai.text_embedder.GoogleGenAITextEmbedder.from_dict"></a>

#### GoogleGenAITextEmbedder.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "GoogleGenAITextEmbedder"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.embedders.google_genai.text_embedder.GoogleGenAITextEmbedder.run"></a>

#### GoogleGenAITextEmbedder.run

```python
@component.output_types(embedding=List[float], meta=Dict[str, Any])
def run(text: str) -> Union[Dict[str, List[float]], Dict[str, Any]]
```

Embeds a single string.

**Arguments**:

- `text`: Text to embed.

**Returns**:

A dictionary with the following keys:
- `embedding`: The embedding of the input text.
- `meta`: Information about the usage of the model.

<a id="haystack_integrations.components.embedders.google_genai.text_embedder.GoogleGenAITextEmbedder.run_async"></a>

#### GoogleGenAITextEmbedder.run\_async

```python
@component.output_types(embedding=List[float], meta=Dict[str, Any])
async def run_async(
        text: str) -> Union[Dict[str, List[float]], Dict[str, Any]]
```

Asynchronously embed a single string.

This is the asynchronous version of the `run` method. It has the same parameters and return values
but can be used with `await` in async code.

**Arguments**:

- `text`: Text to embed.

**Returns**:

A dictionary with the following keys:
- `embedding`: The embedding of the input text.
- `meta`: Information about the usage of the model.
