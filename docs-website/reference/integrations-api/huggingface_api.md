---
title: "Hugging Face API"
id: integrations-huggingface-api
description: "Hugging Face API integration for Haystack"
slug: "/integrations-huggingface-api"
---


## haystack_integrations.components.embedders.huggingface_api.document_embedder

### HuggingFaceAPIDocumentEmbedder

Embeds documents using Hugging Face APIs.

Use it with the following Hugging Face APIs:

- [Free Serverless Inference API](https://huggingface.co/inference-api)
- [Paid Inference Endpoints](https://huggingface.co/inference-endpoints)
- [Self-hosted Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference)

### Usage examples

#### With free serverless inference API

```python
from haystack_integrations.components.embedders.huggingface_api import HuggingFaceAPIDocumentEmbedder
from haystack.utils import Secret
from haystack.dataclasses import Document

doc = Document(content="I love pizza!")

doc_embedder = HuggingFaceAPIDocumentEmbedder(api_type="serverless_inference_api",
                                              api_params={"model": "BAAI/bge-small-en-v1.5"},
                                              token=Secret.from_token("<your-api-key>"))

result = document_embedder.run([doc])
print(result["documents"][0].embedding)

# [0.017020374536514282, -0.023255806416273117, ...]
```

#### With paid inference endpoints

```python
from haystack_integrations.components.embedders.huggingface_api import HuggingFaceAPIDocumentEmbedder
from haystack.utils import Secret
from haystack.dataclasses import Document

doc = Document(content="I love pizza!")

doc_embedder = HuggingFaceAPIDocumentEmbedder(api_type="inference_endpoints",
                                              api_params={"url": "<your-inference-endpoint-url>"},
                                              token=Secret.from_token("<your-api-key>"))

result = document_embedder.run([doc])
print(result["documents"][0].embedding)

# [0.017020374536514282, -0.023255806416273117, ...]
```

#### With self-hosted text embeddings inference

```python
from haystack_integrations.components.embedders.huggingface_api import HuggingFaceAPIDocumentEmbedder
from haystack.dataclasses import Document

doc = Document(content="I love pizza!")

doc_embedder = HuggingFaceAPIDocumentEmbedder(api_type="text_embeddings_inference",
                                              api_params={"url": "http://localhost:8080"})

result = document_embedder.run([doc])
print(result["documents"][0].embedding)

# [0.017020374536514282, -0.023255806416273117, ...]
```

#### __init__

```python
__init__(
    api_type: HFEmbeddingAPIType | str,
    api_params: dict[str, str],
    token: Secret | None = Secret.from_env_var(
        ["HF_API_TOKEN", "HF_TOKEN"], strict=False
    ),
    prefix: str = "",
    suffix: str = "",
    truncate: bool | None = True,
    normalize: bool | None = False,
    batch_size: int = 32,
    progress_bar: bool = True,
    meta_fields_to_embed: list[str] | None = None,
    embedding_separator: str = "\n",
    concurrency_limit: int = 4,
) -> None
```

Creates a HuggingFaceAPIDocumentEmbedder component.

**Parameters:**

- **api_type** (<code>HFEmbeddingAPIType | str</code>) – The type of Hugging Face API to use.
- **api_params** (<code>dict\[str, str\]</code>) – A dictionary with the following keys:
- `model`: Hugging Face model ID. Required when `api_type` is `SERVERLESS_INFERENCE_API`.
- `url`: URL of the inference endpoint. Required when `api_type` is `INFERENCE_ENDPOINTS` or
  `TEXT_EMBEDDINGS_INFERENCE`.
- **token** (<code>Secret | None</code>) – The Hugging Face token to use as HTTP bearer authorization.
  Check your HF token in your [account settings](https://huggingface.co/settings/tokens).
- **prefix** (<code>str</code>) – A string to add at the beginning of each text.
- **suffix** (<code>str</code>) – A string to add at the end of each text.
- **truncate** (<code>bool | None</code>) – Truncates the input text to the maximum length supported by the model.
  Applicable when `api_type` is `TEXT_EMBEDDINGS_INFERENCE`, or `INFERENCE_ENDPOINTS`
  if the backend uses Text Embeddings Inference.
  If `api_type` is `SERVERLESS_INFERENCE_API`, this parameter is ignored.
- **normalize** (<code>bool | None</code>) – Normalizes the embeddings to unit length.
  Applicable when `api_type` is `TEXT_EMBEDDINGS_INFERENCE`, or `INFERENCE_ENDPOINTS`
  if the backend uses Text Embeddings Inference.
  If `api_type` is `SERVERLESS_INFERENCE_API`, this parameter is ignored.
- **batch_size** (<code>int</code>) – Number of documents to process at once.
- **progress_bar** (<code>bool</code>) – If `True`, shows a progress bar when running.
- **meta_fields_to_embed** (<code>list\[str\] | None</code>) – List of metadata fields to embed along with the document text.
- **embedding_separator** (<code>str</code>) – Separator used to concatenate the metadata fields to the document text.
- **concurrency_limit** (<code>int</code>) – The maximum number of requests that should be allowed to run concurrently.
  This parameter is only used in the `run_async` method.

**Raises:**

- <code>ValueError</code> – If the required `model` or `url` is missing from `api_params`, the `url` is invalid,
  or the `api_type` is unknown.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> HuggingFaceAPIDocumentEmbedder
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>HuggingFaceAPIDocumentEmbedder</code> – Deserialized component.

#### run

```python
run(documents: list[Document]) -> dict[str, list[Document]]
```

Embeds a list of documents.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – Documents to embed.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: A list of documents with embeddings.

**Raises:**

- <code>TypeError</code> – If `documents` is not a list of Documents.
- <code>ValueError</code> – If the embeddings returned by the API have an unexpected shape.

#### run_async

```python
run_async(documents: list[Document]) -> dict[str, list[Document]]
```

Embeds a list of documents asynchronously.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – Documents to embed.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: A list of documents with embeddings.

**Raises:**

- <code>TypeError</code> – If `documents` is not a list of Documents.
- <code>ValueError</code> – If the embeddings returned by the API have an unexpected shape.

## haystack_integrations.components.embedders.huggingface_api.text_embedder

### HuggingFaceAPITextEmbedder

Embeds strings using Hugging Face APIs.

Use it with the following Hugging Face APIs:

- [Free Serverless Inference API](https://huggingface.co/inference-api)
- [Paid Inference Endpoints](https://huggingface.co/inference-endpoints)
- [Self-hosted Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference)

### Usage examples

#### With free serverless inference API

```python
from haystack_integrations.components.embedders.huggingface_api import HuggingFaceAPITextEmbedder
from haystack.utils import Secret

text_embedder = HuggingFaceAPITextEmbedder(api_type="serverless_inference_api",
                                           api_params={"model": "BAAI/bge-small-en-v1.5"},
                                           token=Secret.from_token("<your-api-key>"))

print(text_embedder.run("I love pizza!"))

# {'embedding': [0.017020374536514282, -0.023255806416273117, ...],
```

#### With paid inference endpoints

```python
from haystack_integrations.components.embedders.huggingface_api import HuggingFaceAPITextEmbedder
from haystack.utils import Secret
text_embedder = HuggingFaceAPITextEmbedder(api_type="inference_endpoints",
                                           api_params={"model": "BAAI/bge-small-en-v1.5"},
                                           token=Secret.from_token("<your-api-key>"))

print(text_embedder.run("I love pizza!"))

# {'embedding': [0.017020374536514282, -0.023255806416273117, ...],
```

#### With self-hosted text embeddings inference

```python
from haystack_integrations.components.embedders.huggingface_api import HuggingFaceAPITextEmbedder
from haystack.utils import Secret

text_embedder = HuggingFaceAPITextEmbedder(api_type="text_embeddings_inference",
                                           api_params={"url": "http://localhost:8080"})

print(text_embedder.run("I love pizza!"))

# {'embedding': [0.017020374536514282, -0.023255806416273117, ...],
```

#### __init__

```python
__init__(
    api_type: HFEmbeddingAPIType | str,
    api_params: dict[str, str],
    token: Secret | None = Secret.from_env_var(
        ["HF_API_TOKEN", "HF_TOKEN"], strict=False
    ),
    prefix: str = "",
    suffix: str = "",
    truncate: bool | None = True,
    normalize: bool | None = False,
) -> None
```

Creates a HuggingFaceAPITextEmbedder component.

**Parameters:**

- **api_type** (<code>HFEmbeddingAPIType | str</code>) – The type of Hugging Face API to use.
- **api_params** (<code>dict\[str, str\]</code>) – A dictionary with the following keys:
- `model`: Hugging Face model ID. Required when `api_type` is `SERVERLESS_INFERENCE_API`.
- `url`: URL of the inference endpoint. Required when `api_type` is `INFERENCE_ENDPOINTS` or
  `TEXT_EMBEDDINGS_INFERENCE`.
- **token** (<code>Secret | None</code>) – The Hugging Face token to use as HTTP bearer authorization.
  Check your HF token in your [account settings](https://huggingface.co/settings/tokens).
- **prefix** (<code>str</code>) – A string to add at the beginning of each text.
- **suffix** (<code>str</code>) – A string to add at the end of each text.
- **truncate** (<code>bool | None</code>) – Truncates the input text to the maximum length supported by the model.
  Applicable when `api_type` is `TEXT_EMBEDDINGS_INFERENCE`, or `INFERENCE_ENDPOINTS`
  if the backend uses Text Embeddings Inference.
  If `api_type` is `SERVERLESS_INFERENCE_API`, this parameter is ignored.
- **normalize** (<code>bool | None</code>) – Normalizes the embeddings to unit length.
  Applicable when `api_type` is `TEXT_EMBEDDINGS_INFERENCE`, or `INFERENCE_ENDPOINTS`
  if the backend uses Text Embeddings Inference.
  If `api_type` is `SERVERLESS_INFERENCE_API`, this parameter is ignored.

**Raises:**

- <code>ValueError</code> – If the required `model` or `url` is missing from `api_params`, the `url` is invalid,
  or the `api_type` is unknown.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> HuggingFaceAPITextEmbedder
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>HuggingFaceAPITextEmbedder</code> – Deserialized component.

#### run

```python
run(text: str) -> dict[str, Any]
```

Embeds a single string.

**Parameters:**

- **text** (<code>str</code>) – Text to embed.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the following keys:
- `embedding`: The embedding of the input text.

**Raises:**

- <code>TypeError</code> – If `text` is not a string.
- <code>ValueError</code> – If the embedding returned by the API has an unexpected shape.

#### run_async

```python
run_async(text: str) -> dict[str, Any]
```

Embeds a single string asynchronously.

**Parameters:**

- **text** (<code>str</code>) – Text to embed.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the following keys:
- `embedding`: The embedding of the input text.

**Raises:**

- <code>TypeError</code> – If `text` is not a string.
- <code>ValueError</code> – If the embedding returned by the API has an unexpected shape.

## haystack_integrations.components.generators.huggingface_api.chat.chat_generator

### HuggingFaceAPIChatGenerator

Completes chats using Hugging Face APIs.

HuggingFaceAPIChatGenerator uses the [ChatMessage](https://docs.haystack.deepset.ai/docs/chatmessage)
format for input and output. Use it to generate text with Hugging Face APIs:

- [Serverless Inference API (Inference Providers)](https://huggingface.co/docs/inference-providers)
- [Paid Inference Endpoints](https://huggingface.co/inference-endpoints)
- [Self-hosted Text Generation Inference](https://github.com/huggingface/text-generation-inference)

### Usage examples

#### With the serverless inference API (Inference Providers) - free tier available

```python
from haystack_integrations.components.generators.huggingface_api import HuggingFaceAPIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from haystack_integrations.components.common.huggingface_api.utils import HFGenerationAPIType

messages = [ChatMessage.from_system("\nYou are a helpful, respectful and honest assistant"),
            ChatMessage.from_user("What's Natural Language Processing?")]

# the api_type can be expressed using the HFGenerationAPIType enum or as a string
api_type = HFGenerationAPIType.SERVERLESS_INFERENCE_API
api_type = "serverless_inference_api" # this is equivalent to the above

generator = HuggingFaceAPIChatGenerator(api_type=api_type,
                                        api_params={"model": "Qwen/Qwen2.5-7B-Instruct",
                                                    "provider": "together"},
                                        token=Secret.from_token("<your-api-key>"))

result = generator.run(messages)
print(result)
```

#### With the serverless inference API (Inference Providers) and text+image input

```python
from haystack_integrations.components.generators.huggingface_api import HuggingFaceAPIChatGenerator
from haystack.dataclasses import ChatMessage, ImageContent
from haystack.utils import Secret
from haystack_integrations.components.common.huggingface_api.utils import HFGenerationAPIType

# Create an image from file path, URL, or base64
image = ImageContent.from_file_path("path/to/your/image.jpg")

# Create a multimodal message with both text and image
messages = [ChatMessage.from_user(content_parts=["Describe this image in detail", image])]

generator = HuggingFaceAPIChatGenerator(
    api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
    api_params={
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",  # Vision Language Model
        "provider": "hyperbolic"
    },
    token=Secret.from_token("<your-api-key>")
)

result = generator.run(messages)
print(result)
```

#### With paid inference endpoints

```python
from haystack_integrations.components.generators.huggingface_api import HuggingFaceAPIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

messages = [ChatMessage.from_system("\nYou are a helpful, respectful and honest assistant"),
            ChatMessage.from_user("What's Natural Language Processing?")]

generator = HuggingFaceAPIChatGenerator(api_type="inference_endpoints",
                                        api_params={"url": "<your-inference-endpoint-url>"},
                                        token=Secret.from_token("<your-api-key>"))

result = generator.run(messages)
print(result)
```

#### With self-hosted text generation inference

```python
from haystack_integrations.components.generators.huggingface_api import HuggingFaceAPIChatGenerator
from haystack.dataclasses import ChatMessage

messages = [ChatMessage.from_system("\nYou are a helpful, respectful and honest assistant"),
            ChatMessage.from_user("What's Natural Language Processing?")]

generator = HuggingFaceAPIChatGenerator(api_type="text_generation_inference",
                                        api_params={"url": "http://localhost:8080"})

result = generator.run(messages)
print(result)
```

#### __init__

```python
__init__(
    api_type: HFGenerationAPIType | str,
    api_params: dict[str, str],
    token: Secret | None = Secret.from_env_var(
        ["HF_API_TOKEN", "HF_TOKEN"], strict=False
    ),
    generation_kwargs: dict[str, Any] | None = None,
    stop_words: list[str] | None = None,
    streaming_callback: StreamingCallbackT | None = None,
    tools: ToolsType | None = None,
) -> None
```

Initialize the HuggingFaceAPIChatGenerator instance.

**Parameters:**

- **api_type** (<code>HFGenerationAPIType | str</code>) – The type of Hugging Face API to use. Available types:
- `text_generation_inference`: See [TGI](https://github.com/huggingface/text-generation-inference).
- `inference_endpoints`: See [Inference Endpoints](https://huggingface.co/inference-endpoints).
- `serverless_inference_api`: See
  [Serverless Inference API - Inference Providers](https://huggingface.co/docs/inference-providers).
- **api_params** (<code>dict\[str, str\]</code>) – A dictionary with the following keys:
- `model`: Hugging Face model ID. Required when `api_type` is `SERVERLESS_INFERENCE_API`.
- `provider`: Provider name. Recommended when `api_type` is `SERVERLESS_INFERENCE_API`.
- `url`: URL of the inference endpoint. Required when `api_type` is `INFERENCE_ENDPOINTS` or
  `TEXT_GENERATION_INFERENCE`.
- Other parameters specific to the chosen API type, such as `timeout`, `headers`, etc.
- **token** (<code>Secret | None</code>) – The Hugging Face token to use as HTTP bearer authorization.
  Check your HF token in your [account settings](https://huggingface.co/settings/tokens).
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – A dictionary with keyword arguments to customize text generation.
  Some examples: `max_tokens`, `temperature`, `top_p`.
  For details, see [Hugging Face chat_completion documentation](https://huggingface.co/docs/huggingface_hub/package_reference/inference_client#huggingface_hub.InferenceClient.chat_completion).
- **stop_words** (<code>list\[str\] | None</code>) – An optional list of strings representing the stop words.
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – An optional callable for handling streaming responses.
- **tools** (<code>ToolsType | None</code>) – A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
  The chosen model should support tool/function calling, according to the model card.
  Support for tools in the Hugging Face API and TGI is not yet fully refined and you may experience
  unexpected behavior.

**Raises:**

- <code>ValueError</code> – If the required `model` or `url` is missing from `api_params`, the `url` is invalid, the `api_type`
  is unknown, `tools` and `streaming_callback` are used together, or duplicate tool names are provided.

#### warm_up

```python
warm_up() -> None
```

Warm up the Hugging Face API chat generator.

This will warm up the tools registered in the chat generator.
This method is idempotent and will only warm up the tools once.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize this component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary containing the serialized component.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> HuggingFaceAPIChatGenerator
```

Deserialize this component from a dictionary.

#### run

```python
run(
    messages: list[ChatMessage] | str,
    generation_kwargs: dict[str, Any] | None = None,
    tools: ToolsType | None = None,
    streaming_callback: StreamingCallbackT | None = None,
) -> dict[str, list[ChatMessage]]
```

Invoke the text generation inference based on the provided messages and generation parameters.

**Parameters:**

- **messages** (<code>list\[ChatMessage\] | str</code>) – A list of ChatMessage objects representing the input messages. If a string is provided, it is converted
  to a list containing a ChatMessage with user role.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments for text generation.
- **tools** (<code>ToolsType | None</code>) – A list of tools or a Toolset for which the model can prepare calls. If set, it will override
  the `tools` parameter set during component initialization. This parameter can accept either a
  list of `Tool` objects or a `Toolset` instance.
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – An optional callable for handling streaming responses. If set, it will override the `streaming_callback`
  parameter set during component initialization.

**Returns:**

- <code>dict\[str, list\[ChatMessage\]\]</code> – A dictionary with the following keys:
- `replies`: A list containing the generated responses as ChatMessage objects.

**Raises:**

- <code>ValueError</code> – If `tools` and a streaming callback are used together, or if duplicate tool names are provided.

#### run_async

```python
run_async(
    messages: list[ChatMessage] | str,
    generation_kwargs: dict[str, Any] | None = None,
    tools: ToolsType | None = None,
    streaming_callback: StreamingCallbackT | None = None,
) -> dict[str, list[ChatMessage]]
```

Asynchronously invokes the text generation inference based on the provided messages and generation parameters.

This is the asynchronous version of the `run` method. It has the same parameters
and return values but can be used with `await` in an async code.

**Parameters:**

- **messages** (<code>list\[ChatMessage\] | str</code>) – A list of ChatMessage objects representing the input messages. If a string is provided, it is converted
  to a list containing a ChatMessage with user role.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments for text generation.
- **tools** (<code>ToolsType | None</code>) – A list of tools or a Toolset for which the model can prepare calls. If set, it will override the `tools`
  parameter set during component initialization. This parameter can accept either a list of `Tool` objects
  or a `Toolset` instance.
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – An optional callable for handling streaming responses. If set, it will override the `streaming_callback`
  parameter set during component initialization.

**Returns:**

- <code>dict\[str, list\[ChatMessage\]\]</code> – A dictionary with the following keys:
- `replies`: A list containing the generated responses as ChatMessage objects.

**Raises:**

- <code>ValueError</code> – If `tools` and a streaming callback are used together, or if duplicate tool names are provided.

## haystack_integrations.components.rankers.huggingface_api.ranker

### TruncationDirection

Bases: <code>str</code>, <code>Enum</code>

Defines the direction to truncate text when input length exceeds the model's limit.

Attributes:
LEFT: Truncate text from the left side (start of text).
RIGHT: Truncate text from the right side (end of text).

### HuggingFaceTEIRanker

Ranks documents based on their semantic similarity to the query.

It can be used with a Text Embeddings Inference (TEI) API endpoint:

- [Self-hosted Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference)
- [Hugging Face Inference Endpoints](https://huggingface.co/inference-endpoints)

Usage example:

```python
from haystack import Document
from haystack.utils import Secret

from haystack_integrations.components.rankers.huggingface_api import HuggingFaceTEIRanker

reranker = HuggingFaceTEIRanker(
    url="http://localhost:8080",
    top_k=5,
    timeout=30,
    token=Secret.from_token("my_api_token")
)

docs = [Document(content="The capital of France is Paris"), Document(content="The capital of Germany is Berlin")]

result = reranker.run(query="What is the capital of France?", documents=docs)

ranked_docs = result["documents"]
print(ranked_docs)
# >> {'documents': [Document(id=..., content: 'the capital of France is Paris', score: 0.9979767),
# >>                Document(id=..., content: 'the capital of Germany is Berlin', score: 0.13982213)]}
```

#### __init__

```python
__init__(
    *,
    url: str,
    top_k: int = 10,
    raw_scores: bool = False,
    timeout: int | None = 30,
    max_retries: int = 3,
    retry_status_codes: list[int] | None = None,
    token: Secret | None = Secret.from_env_var(
        ["HF_API_TOKEN", "HF_TOKEN"], strict=False
    )
) -> None
```

Initializes the TEI reranker component.

**Parameters:**

- **url** (<code>str</code>) – Base URL of the TEI reranking service (for example, "https://api.example.com").
- **top_k** (<code>int</code>) – Maximum number of top documents to return.
- **raw_scores** (<code>bool</code>) – If True, include raw relevance scores in the API payload.
- **timeout** (<code>int | None</code>) – Request timeout in seconds.
- **max_retries** (<code>int</code>) – Maximum number of retry attempts for failed requests.
- **retry_status_codes** (<code>list\[int\] | None</code>) – List of HTTP status codes that will trigger a retry.
  When None, HTTP 408, 418, 429 and 503 will be retried (default: None).
- **token** (<code>Secret | None</code>) – The Hugging Face token to use as HTTP bearer authorization. Not always required
  depending on your TEI server configuration.
  Check your HF token in your [account settings](https://huggingface.co/settings/tokens).

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> HuggingFaceTEIRanker
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>HuggingFaceTEIRanker</code> – Deserialized component.

#### run

```python
run(
    query: str,
    documents: list[Document],
    top_k: int | None = None,
    truncation_direction: TruncationDirection | None = None,
) -> dict[str, list[Document]]
```

Reranks the provided documents by relevance to the query using the TEI API.

Before ranking, documents are deduplicated by their id, retaining only the document with the highest score
if a score is present.

**Parameters:**

- **query** (<code>str</code>) – The user query string to guide reranking.
- **documents** (<code>list\[Document\]</code>) – List of `Document` objects to rerank.
- **top_k** (<code>int | None</code>) – Optional override for the maximum number of documents to return.
- **truncation_direction** (<code>TruncationDirection | None</code>) – If set, enables text truncation in the specified direction.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: A list of reranked documents.

**Raises:**

- <code>RuntimeError</code> – - If the API request fails.
- <code>RuntimeError</code> – - If the API returns an error response.
- <code>TypeError</code> – - If the API response is not in the expected list format.

#### run_async

```python
run_async(
    query: str,
    documents: list[Document],
    top_k: int | None = None,
    truncation_direction: TruncationDirection | None = None,
) -> dict[str, list[Document]]
```

Asynchronously reranks the provided documents by relevance to the query using the TEI API.

Before ranking, documents are deduplicated by their id, retaining only the document with the highest score
if a score is present.

**Parameters:**

- **query** (<code>str</code>) – The user query string to guide reranking.
- **documents** (<code>list\[Document\]</code>) – List of `Document` objects to rerank.
- **top_k** (<code>int | None</code>) – Optional override for the maximum number of documents to return.
- **truncation_direction** (<code>TruncationDirection | None</code>) – If set, enables text truncation in the specified direction.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: A list of reranked documents.

**Raises:**

- <code>httpx.RequestError</code> – - If the API request fails.
- <code>RuntimeError</code> – - If the API returns an error response.
- <code>TypeError</code> – - If the API response is not in the expected list format.
