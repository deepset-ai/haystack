---
title: "STACKIT"
id: integrations-stackit
description: "STACKIT integration for Haystack"
slug: "/integrations-stackit"
---


## `haystack_integrations.components.embedders.stackit.document_embedder`

### `STACKITDocumentEmbedder`

Bases: <code>OpenAIDocumentEmbedder</code>

A component for computing Document embeddings using STACKIT as model provider.
The embedding of each Document is stored in the `embedding` field of the Document.

Usage example:

```python
from haystack import Document
from haystack_integrations.components.embedders.stackit import STACKITDocumentEmbedder

doc = Document(content="I love pizza!")

document_embedder = STACKITDocumentEmbedder()

result = document_embedder.run([doc])
print(result['documents'][0].embedding)

# [0.017020374536514282, -0.023255806416273117, ...]
```

#### `__init__`

```python
__init__(
    model: str,
    api_key: Secret = Secret.from_env_var("STACKIT_API_KEY"),
    api_base_url: (
        str | None
    ) = "https://api.openai-compat.model-serving.eu01.onstackit.cloud/v1",
    prefix: str = "",
    suffix: str = "",
    batch_size: int = 32,
    progress_bar: bool = True,
    meta_fields_to_embed: list[str] | None = None,
    embedding_separator: str = "\n",
    *,
    timeout: float | None = None,
    max_retries: int | None = None,
    http_client_kwargs: dict[str, Any] | None = None
)
```

Creates a STACKITDocumentEmbedder component.

**Parameters:**

- **api_key** (<code>Secret</code>) – The STACKIT API key.
- **model** (<code>str</code>) – The name of the model to use.
- **api_base_url** (<code>str | None</code>) – The STACKIT API Base url.
  For more details, see STACKIT [docs](https://docs.stackit.cloud/stackit/en/basic-concepts-stackit-model-serving-319914567.html).
- **prefix** (<code>str</code>) – A string to add to the beginning of each text.
- **suffix** (<code>str</code>) – A string to add to the end of each text.
- **batch_size** (<code>int</code>) – Number of Documents to encode at once.
- **progress_bar** (<code>bool</code>) – Whether to show a progress bar or not. Can be helpful to disable in production deployments to keep
  the logs clean.
- **meta_fields_to_embed** (<code>list\[str\] | None</code>) – List of meta fields that should be embedded along with the Document text.
- **embedding_separator** (<code>str</code>) – Separator used to concatenate the meta fields to the Document text.
- **timeout** (<code>float | None</code>) – Timeout for STACKIT client calls. If not set, it defaults to either the `OPENAI_TIMEOUT` environment
  variable, or 30 seconds.
- **max_retries** (<code>int | None</code>) – Maximum number of retries to contact STACKIT after an internal error.
  If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
- **http_client_kwargs** (<code>dict\[str, Any\] | None</code>) – A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
  For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

## `haystack_integrations.components.embedders.stackit.text_embedder`

### `STACKITTextEmbedder`

Bases: <code>OpenAITextEmbedder</code>

A component for embedding strings using STACKIT as model provider.

Usage example:

```python
from haystack_integrations.components.embedders.stackit import STACKITTextEmbedder

text_to_embed = "I love pizza!"
text_embedder = STACKITTextEmbedder()
print(text_embedder.run(text_to_embed))
```

#### `__init__`

```python
__init__(
    model: str,
    api_key: Secret = Secret.from_env_var("STACKIT_API_KEY"),
    api_base_url: (
        str | None
    ) = "https://api.openai-compat.model-serving.eu01.onstackit.cloud/v1",
    prefix: str = "",
    suffix: str = "",
    *,
    timeout: float | None = None,
    max_retries: int | None = None,
    http_client_kwargs: dict[str, Any] | None = None
)
```

Creates a STACKITTextEmbedder component.

**Parameters:**

- **api_key** (<code>Secret</code>) – The STACKIT API key.
- **model** (<code>str</code>) – The name of the STACKIT embedding model to be used.
- **api_base_url** (<code>str | None</code>) – The STACKIT API Base url.
  For more details, see STACKIT [docs](https://docs.stackit.cloud/stackit/en/basic-concepts-stackit-model-serving-319914567.html).
- **prefix** (<code>str</code>) – A string to add to the beginning of each text.
- **suffix** (<code>str</code>) – A string to add to the end of each text.
- **timeout** (<code>float | None</code>) – Timeout for STACKIT client calls. If not set, it defaults to either the `OPENAI_TIMEOUT` environment
  variable, or 30 seconds.
- **max_retries** (<code>int | None</code>) – Maximum number of retries to contact STACKIT after an internal error.
  If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
- **http_client_kwargs** (<code>dict\[str, Any\] | None</code>) – A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
  For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

## `haystack_integrations.components.generators.stackit.chat.chat_generator`

### `STACKITChatGenerator`

Bases: <code>OpenAIChatGenerator</code>

Enables text generation using STACKIT generative models through their model serving service.

Users can pass any text generation parameters valid for the STACKIT Chat Completion API
directly to this component using the `generation_kwargs` parameter in `__init__` or the `generation_kwargs`
parameter in `run` method.

This component uses the ChatMessage format for structuring both input and output,
ensuring coherent and contextually relevant responses in chat-based text generation scenarios.
Details on the ChatMessage format can be found in the
[Haystack docs](https://docs.haystack.deepset.ai/docs/chatmessage)

### Usage example

```python
from haystack_integrations.components.generators.stackit import STACKITChatGenerator
from haystack.dataclasses import ChatMessage

generator = STACKITChatGenerator(model="neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8")

result = generator.run([ChatMessage.from_user("Tell me a joke.")])
print(result)
```

#### `__init__`

```python
__init__(
    model: str,
    api_key: Secret = Secret.from_env_var("STACKIT_API_KEY"),
    streaming_callback: StreamingCallbackT | None = None,
    api_base_url: (
        str | None
    ) = "https://api.openai-compat.model-serving.eu01.onstackit.cloud/v1",
    generation_kwargs: dict[str, Any] | None = None,
    *,
    timeout: float | None = None,
    max_retries: int | None = None,
    http_client_kwargs: dict[str, Any] | None = None
)
```

Creates an instance of STACKITChatGenerator class.

**Parameters:**

- **model** (<code>str</code>) – The name of the chat completion model to use.
- **api_key** (<code>Secret</code>) – The STACKIT API key.
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – A callback function that is called when a new token is received from the stream.
  The callback function accepts StreamingChunk as an argument.
- **api_base_url** (<code>str | None</code>) – The STACKIT API Base url.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Other parameters to use for the model. These parameters are all sent directly to
  the STACKIT endpoint.
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
- **timeout** (<code>float | None</code>) – Timeout for STACKIT client calls. If not set, it defaults to either the `OPENAI_TIMEOUT` environment
  variable, or 30 seconds.
- **max_retries** (<code>int | None</code>) – Maximum number of retries to contact STACKIT after an internal error.
  If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
- **http_client_kwargs** (<code>dict\[str, Any\] | None</code>) – A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
  For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serialize this component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – The serialized component as a dictionary.
