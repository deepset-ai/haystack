---
title: "IBM watsonx.ai"
id: integrations-watsonx
description: "IBM watsonx.ai integration for Haystack"
slug: "/integrations-watsonx"
---

<a id="haystack_integrations.components.generators.watsonx.generator"></a>

## Module haystack\_integrations.components.generators.watsonx.generator

<a id="haystack_integrations.components.generators.watsonx.generator.WatsonxGenerator"></a>

### WatsonxGenerator

Enables text completions using IBM's watsonx.ai foundation models.

This component extends WatsonxChatGenerator to provide the standard Generator interface that works with prompt
strings instead of ChatMessage objects.

The generator works with IBM's foundation models that are listed
[here](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-models.html?context=wx&audience=wdp).

You can customize the generation behavior by passing parameters to the watsonx.ai API through the
`generation_kwargs` argument. These parameters are passed directly to the watsonx.ai inference endpoint.

For details on watsonx.ai API parameters, see
[IBM watsonx.ai documentation](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-parameters.html).

### Usage example

```python
from haystack_integrations.components.generators.watsonx.generator import WatsonxGenerator
from haystack.utils import Secret

generator = WatsonxGenerator(
    api_key=Secret.from_env_var("WATSONX_API_KEY"),
    model="ibm/granite-4-h-small",
    project_id=Secret.from_env_var("WATSONX_PROJECT_ID"),
)

response = generator.run(
    prompt="Explain quantum computing in simple terms",
    system_prompt="You are a helpful physics teacher.",
)
print(response)
```
Output:
```
{
    "replies": ["Quantum computing uses quantum-mechanical phenomena like...."],
    "meta": [
        {
            "model": "ibm/granite-4-h-small",
            "project_id": "your-project-id",
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 45,
                "total_tokens": 57,
            },
        }
    ],
}
```

<a id="haystack_integrations.components.generators.watsonx.generator.WatsonxGenerator.__init__"></a>

#### WatsonxGenerator.\_\_init\_\_

```python
def __init__(*,
             api_key: Secret = Secret.from_env_var("WATSONX_API_KEY"),
             model: str = "ibm/granite-4-h-small",
             project_id: Secret = Secret.from_env_var("WATSONX_PROJECT_ID"),
             api_base_url: str = "https://us-south.ml.cloud.ibm.com",
             system_prompt: str | None = None,
             generation_kwargs: dict[str, Any] | None = None,
             timeout: float | None = None,
             max_retries: int | None = None,
             verify: bool | str | None = None,
             streaming_callback: StreamingCallbackT | None = None) -> None
```

Creates an instance of WatsonxGenerator.

Before initializing the component, you can set environment variables:
- `WATSONX_TIMEOUT` to override the default timeout
- `WATSONX_MAX_RETRIES` to override the default retry count

**Arguments**:

- `api_key`: IBM Cloud API key for watsonx.ai access.
Can be set via `WATSONX_API_KEY` environment variable or passed directly.
- `model`: The model ID to use for completions. Defaults to "ibm/granite-4-h-small".
Available models can be found in your IBM Cloud account.
- `project_id`: IBM Cloud project ID
- `api_base_url`: Custom base URL for the API endpoint.
Defaults to "https://us-south.ml.cloud.ibm.com".
- `system_prompt`: The system prompt to use for text generation.
- `generation_kwargs`: Additional parameters to control text generation.
These parameters are passed directly to the watsonx.ai inference endpoint.
Supported parameters include:
- `temperature`: Controls randomness (lower = more deterministic)
- `max_new_tokens`: Maximum number of tokens to generate
- `min_new_tokens`: Minimum number of tokens to generate
- `top_p`: Nucleus sampling probability threshold
- `top_k`: Number of highest probability tokens to consider
- `repetition_penalty`: Penalty for repeated tokens
- `length_penalty`: Penalty based on output length
- `stop_sequences`: List of sequences where generation should stop
- `random_seed`: Seed for reproducible results
- `timeout`: Timeout in seconds for API requests.
Defaults to environment variable `WATSONX_TIMEOUT` or 30 seconds.
- `max_retries`: Maximum number of retry attempts for failed requests.
Defaults to environment variable `WATSONX_MAX_RETRIES` or 5.
- `verify`: SSL verification setting. Can be:
- True: Verify SSL certificates (default)
- False: Skip verification (insecure)
- Path to CA bundle for custom certificates
- `streaming_callback`: A callback function for streaming responses.

<a id="haystack_integrations.components.generators.watsonx.generator.WatsonxGenerator.to_dict"></a>

#### WatsonxGenerator.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serialize the component to a dictionary.

**Returns**:

The serialized component as a dictionary.

<a id="haystack_integrations.components.generators.watsonx.generator.WatsonxGenerator.from_dict"></a>

#### WatsonxGenerator.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "WatsonxGenerator"
```

Deserialize this component from a dictionary.

**Arguments**:

- `data`: The dictionary representation of this component.

**Returns**:

The deserialized component instance.

<a id="haystack_integrations.components.generators.watsonx.generator.WatsonxGenerator.run"></a>

#### WatsonxGenerator.run

```python
@component.output_types(replies=list[str], meta=list[dict[str, Any]])
def run(*,
        prompt: str,
        system_prompt: str | None = None,
        streaming_callback: StreamingCallbackT | None = None,
        generation_kwargs: dict[str, Any] | None = None) -> dict[str, Any]
```

Generate text completions synchronously.

**Arguments**:

- `prompt`: The input prompt string for text generation.
- `system_prompt`: An optional system prompt to provide context or instructions for the generation.
If not provided, the system prompt set in the `__init__` method will be used.
- `streaming_callback`: A callback function that is called when a new token is received from the stream.
If provided, this will override the `streaming_callback` set in the `__init__` method.
- `generation_kwargs`: Additional keyword arguments for text generation. These parameters will potentially override the parameters
passed in the `__init__` method. Supported parameters include temperature, max_new_tokens, top_p, etc.

**Returns**:

A dictionary with the following keys:
- `replies`: A list of generated text completions as strings.
- `meta`: A list of metadata dictionaries containing information about each generation,
including model name, finish reason, and token usage statistics.

<a id="haystack_integrations.components.generators.watsonx.generator.WatsonxGenerator.run_async"></a>

#### WatsonxGenerator.run\_async

```python
@component.output_types(replies=list[str], meta=list[dict[str, Any]])
async def run_async(
        *,
        prompt: str,
        system_prompt: str | None = None,
        streaming_callback: StreamingCallbackT | None = None,
        generation_kwargs: dict[str, Any] | None = None) -> dict[str, Any]
```

Generate text completions asynchronously.

**Arguments**:

- `prompt`: The input prompt string for text generation.
- `system_prompt`: An optional system prompt to provide context or instructions for the generation.
- `streaming_callback`: A callback function that is called when a new token is received from the stream.
If provided, this will override the `streaming_callback` set in the `__init__` method.
- `generation_kwargs`: Additional keyword arguments for text generation. These parameters will potentially override the parameters
passed in the `__init__` method. Supported parameters include temperature, max_new_tokens, top_p, etc.

**Returns**:

A dictionary with the following keys:
- `replies`: A list of generated text completions as strings.
- `meta`: A list of metadata dictionaries containing information about each generation,
including model name, finish reason, and token usage statistics.

<a id="haystack_integrations.components.generators.watsonx.chat.chat_generator"></a>

## Module haystack\_integrations.components.generators.watsonx.chat.chat\_generator

<a id="haystack_integrations.components.generators.watsonx.chat.chat_generator.WatsonxChatGenerator"></a>

### WatsonxChatGenerator

Enables chat completions using IBM's watsonx.ai foundation models.

This component interacts with IBM's watsonx.ai platform to generate chat responses using various foundation
models. It supports the [ChatMessage](https://docs.haystack.deepset.ai/docs/chatmessage) format for both input
and output, including multimodal inputs with text and images.

The generator works with IBM's foundation models that are listed
[here](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-models.html?context=wx&audience=wdp).

You can customize the generation behavior by passing parameters to the watsonx.ai API through the
`generation_kwargs` argument. These parameters are passed directly to the watsonx.ai inference endpoint.

For details on watsonx.ai API parameters, see
[IBM watsonx.ai documentation](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-parameters.html).

### Usage example

```python
from haystack_integrations.components.generators.watsonx.chat.chat_generator import WatsonxChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

messages = [ChatMessage.from_user("Explain quantum computing in simple terms")]

client = WatsonxChatGenerator(
    api_key=Secret.from_env_var("WATSONX_API_KEY"),
    model="ibm/granite-4-h-small",
    project_id=Secret.from_env_var("WATSONX_PROJECT_ID"),
)
response = client.run(messages)
print(response)
```

### Multimodal usage example

```python
from haystack.dataclasses import ChatMessage, ImageContent

# Create an image from file path or base64
image_content = ImageContent.from_file_path("path/to/your/image.jpg")

# Create a multimodal message with both text and image
messages = [ChatMessage.from_user(content_parts=["What's in this image?", image_content])]

# Use a multimodal model
client = WatsonxChatGenerator(
    api_key=Secret.from_env_var("WATSONX_API_KEY"),
    model="meta-llama/llama-3-2-11b-vision-instruct",
    project_id=Secret.from_env_var("WATSONX_PROJECT_ID"),
)
response = client.run(messages)
print(response)
```

<a id="haystack_integrations.components.generators.watsonx.chat.chat_generator.WatsonxChatGenerator.__init__"></a>

#### WatsonxChatGenerator.\_\_init\_\_

```python
def __init__(*,
             api_key: Secret = Secret.from_env_var("WATSONX_API_KEY"),
             model: str = "ibm/granite-4-h-small",
             project_id: Secret = Secret.from_env_var("WATSONX_PROJECT_ID"),
             api_base_url: str = "https://us-south.ml.cloud.ibm.com",
             generation_kwargs: dict[str, Any] | None = None,
             timeout: float | None = None,
             max_retries: int | None = None,
             verify: bool | str | None = None,
             streaming_callback: StreamingCallbackT | None = None) -> None
```

Creates an instance of WatsonxChatGenerator.

Before initializing the component, you can set environment variables:
- `WATSONX_TIMEOUT` to override the default timeout
- `WATSONX_MAX_RETRIES` to override the default retry count

**Arguments**:

- `api_key`: IBM Cloud API key for watsonx.ai access.
Can be set via `WATSONX_API_KEY` environment variable or passed directly.
- `model`: The model ID to use for completions. Defaults to "ibm/granite-4-h-small".
Available models can be found in your IBM Cloud account.
- `project_id`: IBM Cloud project ID
- `api_base_url`: Custom base URL for the API endpoint.
Defaults to "https://us-south.ml.cloud.ibm.com".
- `generation_kwargs`: Additional parameters to control text generation.
These parameters are passed directly to the watsonx.ai inference endpoint.
Supported parameters include:
- `temperature`: Controls randomness (lower = more deterministic)
- `max_new_tokens`: Maximum number of tokens to generate
- `min_new_tokens`: Minimum number of tokens to generate
- `top_p`: Nucleus sampling probability threshold
- `top_k`: Number of highest probability tokens to consider
- `repetition_penalty`: Penalty for repeated tokens
- `length_penalty`: Penalty based on output length
- `stop_sequences`: List of sequences where generation should stop
- `random_seed`: Seed for reproducible results
- `timeout`: Timeout in seconds for API requests.
Defaults to environment variable `WATSONX_TIMEOUT` or 30 seconds.
- `max_retries`: Maximum number of retry attempts for failed requests.
Defaults to environment variable `WATSONX_MAX_RETRIES` or 5.
- `verify`: SSL verification setting. Can be:
- True: Verify SSL certificates (default)
- False: Skip verification (insecure)
- Path to CA bundle for custom certificates
- `streaming_callback`: A callback function for streaming responses.

<a id="haystack_integrations.components.generators.watsonx.chat.chat_generator.WatsonxChatGenerator.to_dict"></a>

#### WatsonxChatGenerator.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serialize the component to a dictionary.

**Returns**:

The serialized component as a dictionary.

<a id="haystack_integrations.components.generators.watsonx.chat.chat_generator.WatsonxChatGenerator.from_dict"></a>

#### WatsonxChatGenerator.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "WatsonxChatGenerator"
```

Deserialize this component from a dictionary.

**Arguments**:

- `data`: The dictionary representation of this component.

**Returns**:

The deserialized component instance.

<a id="haystack_integrations.components.generators.watsonx.chat.chat_generator.WatsonxChatGenerator.run"></a>

#### WatsonxChatGenerator.run

```python
@component.output_types(replies=list[ChatMessage])
def run(
    *,
    messages: list[ChatMessage],
    generation_kwargs: dict[str, Any] | None = None,
    streaming_callback: StreamingCallbackT | None = None
) -> dict[str, list[ChatMessage]]
```

Generate chat completions synchronously.

**Arguments**:

- `messages`: A list of ChatMessage instances representing the input messages.
- `generation_kwargs`: Additional keyword arguments for text generation. These parameters will potentially override the parameters
passed in the `__init__` method.
- `streaming_callback`: A callback function that is called when a new token is received from the stream.
If provided this will override the `streaming_callback` set in the `__init__` method.

**Returns**:

A dictionary with the following key:
- `replies`: A list containing the generated responses as ChatMessage instances.

<a id="haystack_integrations.components.generators.watsonx.chat.chat_generator.WatsonxChatGenerator.run_async"></a>

#### WatsonxChatGenerator.run\_async

```python
@component.output_types(replies=list[ChatMessage])
async def run_async(
    *,
    messages: list[ChatMessage],
    generation_kwargs: dict[str, Any] | None = None,
    streaming_callback: StreamingCallbackT | None = None
) -> dict[str, list[ChatMessage]]
```

Generate chat completions asynchronously.

**Arguments**:

- `messages`: A list of ChatMessage instances representing the input messages.
- `generation_kwargs`: Additional keyword arguments for text generation. These parameters will potentially override the parameters
passed in the `__init__` method.
- `streaming_callback`: A callback function that is called when a new token is received from the stream.
If provided this will override the `streaming_callback` set in the `__init__` method.

**Returns**:

A dictionary with the following key:
- `replies`: A list containing the generated responses as ChatMessage instances.

<a id="haystack_integrations.components.embedders.watsonx.document_embedder"></a>

## Module haystack\_integrations.components.embedders.watsonx.document\_embedder

<a id="haystack_integrations.components.embedders.watsonx.document_embedder.WatsonxDocumentEmbedder"></a>

### WatsonxDocumentEmbedder

Computes document embeddings using IBM watsonx.ai models.

### Usage example

```python
from haystack import Document
from haystack_integrations.components.embedders.watsonx.document_embedder import WatsonxDocumentEmbedder

documents = [
    Document(content="I love pizza!"),
    Document(content="Pasta is great too"),
]

document_embedder = WatsonxDocumentEmbedder(
    model="ibm/slate-30m-english-rtrvr-v2",
    api_key=Secret.from_env_var("WATSONX_API_KEY"),
    api_base_url="https://us-south.ml.cloud.ibm.com",
    project_id=Secret.from_env_var("WATSONX_PROJECT_ID"),
)

result = document_embedder.run(documents=documents)
print(result["documents"][0].embedding)

# [0.017020374536514282, -0.023255806416273117, ...]
```

<a id="haystack_integrations.components.embedders.watsonx.document_embedder.WatsonxDocumentEmbedder.__init__"></a>

#### WatsonxDocumentEmbedder.\_\_init\_\_

```python
def __init__(*,
             model: str = "ibm/slate-30m-english-rtrvr-v2",
             api_key: Secret = Secret.from_env_var("WATSONX_API_KEY"),
             api_base_url: str = "https://us-south.ml.cloud.ibm.com",
             project_id: Secret = Secret.from_env_var("WATSONX_PROJECT_ID"),
             truncate_input_tokens: int | None = None,
             prefix: str = "",
             suffix: str = "",
             batch_size: int = 1000,
             concurrency_limit: int = 5,
             timeout: float | None = None,
             max_retries: int | None = None,
             meta_fields_to_embed: list[str] | None = None,
             embedding_separator: str = "\n")
```

Creates a WatsonxDocumentEmbedder component.

**Arguments**:

- `model`: The name of the model to use for calculating embeddings.
Default is "ibm/slate-30m-english-rtrvr-v2".
- `api_key`: The WATSONX API key. Can be set via environment variable WATSONX_API_KEY.
- `api_base_url`: The WATSONX URL for the watsonx.ai service.
Default is "https://us-south.ml.cloud.ibm.com".
- `project_id`: The ID of the Watson Studio project.
Can be set via environment variable WATSONX_PROJECT_ID.
- `truncate_input_tokens`: Maximum number of tokens to use from the input text.
If set to `None` (or not provided), the full input text is used, up to the model's maximum token limit.
- `prefix`: A string to add at the beginning of each text.
- `suffix`: A string to add at the end of each text.
- `batch_size`: Number of documents to embed in one API call. Default is 1000.
- `concurrency_limit`: Number of parallel requests to make. Default is 5.
- `timeout`: Timeout for API requests in seconds.
- `max_retries`: Maximum number of retries for API requests.

<a id="haystack_integrations.components.embedders.watsonx.document_embedder.WatsonxDocumentEmbedder.to_dict"></a>

#### WatsonxDocumentEmbedder.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serialize the component to a dictionary.

**Returns**:

The serialized component as a dictionary.

<a id="haystack_integrations.components.embedders.watsonx.document_embedder.WatsonxDocumentEmbedder.from_dict"></a>

#### WatsonxDocumentEmbedder.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "WatsonxDocumentEmbedder"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: The dictionary representation of this component.

**Returns**:

The deserialized component instance.

<a id="haystack_integrations.components.embedders.watsonx.document_embedder.WatsonxDocumentEmbedder.run"></a>

#### WatsonxDocumentEmbedder.run

```python
@component.output_types(documents=list[Document], meta=dict[str, Any])
def run(documents: list[Document]
        ) -> dict[str, list[Document] | dict[str, Any]]
```

Embeds a list of documents.

**Arguments**:

- `documents`: A list of documents to embed.

**Returns**:

A dictionary with:
- 'documents': List of Documents with embeddings added
- 'meta': Information about the model usage

<a id="haystack_integrations.components.embedders.watsonx.text_embedder"></a>

## Module haystack\_integrations.components.embedders.watsonx.text\_embedder

<a id="haystack_integrations.components.embedders.watsonx.text_embedder.WatsonxTextEmbedder"></a>

### WatsonxTextEmbedder

Embeds strings using IBM watsonx.ai foundation models.

You can use it to embed user query and send it to an embedding Retriever.

### Usage example

```python
from haystack_integrations.components.embedders.watsonx.text_embedder import WatsonxTextEmbedder

text_to_embed = "I love pizza!"

text_embedder = WatsonxTextEmbedder(
    model="ibm/slate-30m-english-rtrvr-v2",
    api_key=Secret.from_env_var("WATSONX_API_KEY"),
    api_base_url="https://us-south.ml.cloud.ibm.com",
    project_id=Secret.from_env_var("WATSONX_PROJECT_ID"),
)

print(text_embedder.run(text_to_embed))

# {'embedding': [0.017020374536514282, -0.023255806416273117, ...],
#  'meta': {'model': 'ibm/slate-30m-english-rtrvr-v2',
#           'truncated_input_tokens': 3}}
```

<a id="haystack_integrations.components.embedders.watsonx.text_embedder.WatsonxTextEmbedder.__init__"></a>

#### WatsonxTextEmbedder.\_\_init\_\_

```python
def __init__(*,
             model: str = "ibm/slate-30m-english-rtrvr-v2",
             api_key: Secret = Secret.from_env_var("WATSONX_API_KEY"),
             api_base_url: str = "https://us-south.ml.cloud.ibm.com",
             project_id: Secret = Secret.from_env_var("WATSONX_PROJECT_ID"),
             truncate_input_tokens: int | None = None,
             prefix: str = "",
             suffix: str = "",
             timeout: float | None = None,
             max_retries: int | None = None)
```

Creates an WatsonxTextEmbedder component.

**Arguments**:

- `model`: The name of the IBM watsonx model to use for calculating embeddings.
Default is "ibm/slate-30m-english-rtrvr-v2".
- `api_key`: The WATSONX API key. Can be set via environment variable WATSONX_API_KEY.
- `api_base_url`: The WATSONX URL for the watsonx.ai service.
Default is "https://us-south.ml.cloud.ibm.com".
- `project_id`: The ID of the Watson Studio project.
Can be set via environment variable WATSONX_PROJECT_ID.
- `truncate_input_tokens`: Maximum number of tokens to use from the input text.
If set to `None` (or not provided), the full input text is used, up to the model's maximum token limit.
- `prefix`: A string to add at the beginning of each text to embed.
- `suffix`: A string to add at the end of each text to embed.
- `timeout`: Timeout for API requests in seconds.
- `max_retries`: Maximum number of retries for API requests.

<a id="haystack_integrations.components.embedders.watsonx.text_embedder.WatsonxTextEmbedder.to_dict"></a>

#### WatsonxTextEmbedder.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serialize the component to a dictionary.

**Returns**:

The serialized component as a dictionary.

<a id="haystack_integrations.components.embedders.watsonx.text_embedder.WatsonxTextEmbedder.from_dict"></a>

#### WatsonxTextEmbedder.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "WatsonxTextEmbedder"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: The dictionary representation of this component.

**Returns**:

The deserialized component instance.

<a id="haystack_integrations.components.embedders.watsonx.text_embedder.WatsonxTextEmbedder.run"></a>

#### WatsonxTextEmbedder.run

```python
@component.output_types(embedding=list[float], meta=dict[str, Any])
def run(text: str) -> dict[str, list[float] | dict[str, Any]]
```

Embeds a single string.

**Arguments**:

- `text`: Text to embed.

**Returns**:

A dictionary with:
- 'embedding': The embedding of the input text
- 'meta': Information about the model usage

