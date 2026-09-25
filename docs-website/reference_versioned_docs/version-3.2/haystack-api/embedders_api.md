---
title: "Embedders"
id: embedders-api
description: "Transforms queries into vectors to look for similar or relevant Documents."
slug: "/embedders-api"
---


## azure_document_embedder

### AzureOpenAIDocumentEmbedder

Bases: <code>OpenAIDocumentEmbedder</code>

Calculates document embeddings using OpenAI models deployed on Azure.

### Usage example

<!-- test-ignore -->

```python
from haystack import Document
from haystack.components.embedders import AzureOpenAIDocumentEmbedder

doc = Document(content="I love pizza!")
document_embedder = AzureOpenAIDocumentEmbedder()

result = document_embedder.run([doc])
print(result['documents'][0].embedding)

# [0.017020374536514282, -0.023255806416273117, ...]
```

#### __init__

```python
__init__(
    azure_endpoint: str | None = None,
    api_version: str | None = "2023-05-15",
    azure_deployment: str = "text-embedding-ada-002",
    dimensions: int | None = None,
    api_key: Secret | None = Secret.from_env_var(
        "AZURE_OPENAI_API_KEY", strict=False
    ),
    azure_ad_token: Secret | None = Secret.from_env_var(
        "AZURE_OPENAI_AD_TOKEN", strict=False
    ),
    organization: str | None = None,
    prefix: str = "",
    suffix: str = "",
    batch_size: int = 32,
    progress_bar: bool = True,
    meta_fields_to_embed: list[str] | None = None,
    embedding_separator: str = "\n",
    timeout: float | None = None,
    max_retries: int | None = None,
    *,
    default_headers: dict[str, str] | None = None,
    azure_ad_token_provider: AzureADTokenProvider | None = None,
    http_client_kwargs: dict[str, Any] | None = None,
    raise_on_failure: bool = False
) -> None
```

Creates an AzureOpenAIDocumentEmbedder component.

**Parameters:**

- **azure_endpoint** (<code>str | None</code>) – The endpoint of the model deployed on Azure.
- **api_version** (<code>str | None</code>) – The version of the API to use.
- **azure_deployment** (<code>str</code>) – The name of the model deployed on Azure. The default model is text-embedding-ada-002.
- **dimensions** (<code>int | None</code>) – The number of dimensions of the resulting embeddings. Only supported in text-embedding-3
  and later models.
- **api_key** (<code>Secret | None</code>) – The Azure OpenAI API key.
  You can set it with an environment variable `AZURE_OPENAI_API_KEY`, or pass with this
  parameter during initialization.
- **azure_ad_token** (<code>Secret | None</code>) – Microsoft Entra ID token, see Microsoft's
  [Entra ID](https://www.microsoft.com/en-us/security/business/identity-access/microsoft-entra-id)
  documentation for more information. You can set it with an environment variable
  `AZURE_OPENAI_AD_TOKEN`, or pass with this parameter during initialization.
  Previously called Azure Active Directory.
- **organization** (<code>str | None</code>) – Your organization ID. See OpenAI's
  [Setting Up Your Organization](https://platform.openai.com/docs/guides/production-best-practices/setting-up-your-organization)
  for more information.
- **prefix** (<code>str</code>) – A string to add at the beginning of each text.
- **suffix** (<code>str</code>) – A string to add at the end of each text.
- **batch_size** (<code>int</code>) – Number of documents to embed at once.
- **progress_bar** (<code>bool</code>) – If `True`, shows a progress bar when running.
- **meta_fields_to_embed** (<code>list\[str\] | None</code>) – List of metadata fields to embed along with the document text.
- **embedding_separator** (<code>str</code>) – Separator used to concatenate the metadata fields to the document text.
- **timeout** (<code>float | None</code>) – The timeout for `AzureOpenAI` client calls, in seconds.
  If not set, defaults to either the
  `OPENAI_TIMEOUT` environment variable, or 30 seconds.
- **max_retries** (<code>int | None</code>) – Maximum number of retries to contact AzureOpenAI after an internal error.
  If not set, defaults to either the `OPENAI_MAX_RETRIES` environment variable or to 5 retries.
- **default_headers** (<code>dict\[str, str\] | None</code>) – Default headers to send to the AzureOpenAI client.
- **azure_ad_token_provider** (<code>AzureADTokenProvider | None</code>) – A function that returns an Azure Active Directory token, will be invoked on
  every request.
- **http_client_kwargs** (<code>dict\[str, Any\] | None</code>) – A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
  For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).
- **raise_on_failure** (<code>bool</code>) – Whether to raise an exception if the embedding request fails. If `False`, the component will log the error
  and continue processing the remaining documents. If `True`, it will raise an exception on failure.

#### warm_up

```python
warm_up() -> None
```

Initializes the synchronous AzureOpenAI client.

#### warm_up_async

```python
warm_up_async() -> None
```

Initializes the asynchronous AzureOpenAI client on the serving event loop.

#### close

```python
close() -> None
```

Releases the synchronous AzureOpenAI client.

#### close_async

```python
close_async() -> None
```

Releases the asynchronous AzureOpenAI client.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> AzureOpenAIDocumentEmbedder
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>AzureOpenAIDocumentEmbedder</code> – Deserialized component.

## azure_text_embedder

### AzureOpenAITextEmbedder

Bases: <code>OpenAITextEmbedder</code>

Embeds strings using OpenAI models deployed on Azure.

### Usage example

<!-- test-ignore -->

```python
from haystack.components.embedders import AzureOpenAITextEmbedder

text_to_embed = "I love pizza!"
text_embedder = AzureOpenAITextEmbedder()

print(text_embedder.run(text_to_embed))

# {'embedding': [0.017020374536514282, -0.023255806416273117, ...],
# 'meta': {'model': 'text-embedding-ada-002-v2',
#          'usage': {'prompt_tokens': 4, 'total_tokens': 4}}}
```

#### __init__

```python
__init__(
    azure_endpoint: str | None = None,
    api_version: str | None = "2023-05-15",
    azure_deployment: str = "text-embedding-ada-002",
    dimensions: int | None = None,
    api_key: Secret | None = Secret.from_env_var(
        "AZURE_OPENAI_API_KEY", strict=False
    ),
    azure_ad_token: Secret | None = Secret.from_env_var(
        "AZURE_OPENAI_AD_TOKEN", strict=False
    ),
    organization: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    prefix: str = "",
    suffix: str = "",
    *,
    default_headers: dict[str, str] | None = None,
    azure_ad_token_provider: AzureADTokenProvider | None = None,
    http_client_kwargs: dict[str, Any] | None = None
) -> None
```

Creates an AzureOpenAITextEmbedder component.

**Parameters:**

- **azure_endpoint** (<code>str | None</code>) – The endpoint of the model deployed on Azure.
- **api_version** (<code>str | None</code>) – The version of the API to use.
- **azure_deployment** (<code>str</code>) – The name of the model deployed on Azure. The default model is text-embedding-ada-002.
- **dimensions** (<code>int | None</code>) – The number of dimensions the resulting output embeddings should have. Only supported in text-embedding-3
  and later models.
- **api_key** (<code>Secret | None</code>) – The Azure OpenAI API key.
  You can set it with an environment variable `AZURE_OPENAI_API_KEY`, or pass with this
  parameter during initialization.
- **azure_ad_token** (<code>Secret | None</code>) – Microsoft Entra ID token, see Microsoft's
  [Entra ID](https://www.microsoft.com/en-us/security/business/identity-access/microsoft-entra-id)
  documentation for more information. You can set it with an environment variable
  `AZURE_OPENAI_AD_TOKEN`, or pass with this parameter during initialization.
  Previously called Azure Active Directory.
- **organization** (<code>str | None</code>) – Your organization ID. See OpenAI's
  [Setting Up Your Organization](https://platform.openai.com/docs/guides/production-best-practices/setting-up-your-organization)
  for more information.
- **timeout** (<code>float | None</code>) – The timeout for `AzureOpenAI` client calls, in seconds.
  If not set, defaults to either the
  `OPENAI_TIMEOUT` environment variable, or 30 seconds.
- **max_retries** (<code>int | None</code>) – Maximum number of retries to contact AzureOpenAI after an internal error.
  If not set, defaults to either the `OPENAI_MAX_RETRIES` environment variable, or to 5 retries.
- **prefix** (<code>str</code>) – A string to add at the beginning of each text.
- **suffix** (<code>str</code>) – A string to add at the end of each text.
- **default_headers** (<code>dict\[str, str\] | None</code>) – Default headers to send to the AzureOpenAI client.
- **azure_ad_token_provider** (<code>AzureADTokenProvider | None</code>) – A function that returns an Azure Active Directory token, will be invoked on
  every request.
- **http_client_kwargs** (<code>dict\[str, Any\] | None</code>) – A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
  For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).

#### warm_up

```python
warm_up() -> None
```

Initializes the synchronous Azure OpenAI client.

#### warm_up_async

```python
warm_up_async() -> None
```

Initializes the asynchronous Azure OpenAI client on the serving event loop.

#### close

```python
close() -> None
```

Releases the synchronous Azure OpenAI client.

#### close_async

```python
close_async() -> None
```

Releases the asynchronous Azure OpenAI client.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> AzureOpenAITextEmbedder
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>AzureOpenAITextEmbedder</code> – Deserialized component.

## mock_document_embedder

### MockDocumentEmbedder

A Document Embedder that returns deterministic embeddings without calling any API.

It is a drop-in replacement for real Document Embedders (such as `OpenAIDocumentEmbedder`) in tests, smoke tests,
and quick prototypes. It implements the same interface (`run`, `run_async`, serialization) but never contacts an
external service, so it is fully deterministic and free to run.

The embedding is selected based on how the component is configured:

- **Deterministic (default)**: with no configuration, each document's embedding is derived from a hash of its
  (prepared) text. The same text always yields the same embedding, and different texts yield different
  embeddings, so the mock works in retrieval pipelines and is reproducible across runs and processes.
- **Fixed embedding**: pass an `embedding` vector. The same vector is assigned to every document.
- **Dynamic embedding**: pass an `embedding_fn` callable that receives the (prepared) text of a document and
  returns the embedding. This is useful when the embedding should depend on the input in a custom way.

Like real Document Embedders, the metadata fields listed in `meta_fields_to_embed` are concatenated with the
document content before embedding, so the deterministic embedding reflects the embedded metadata.

### Usage example

```python
from haystack import Document
from haystack.components.embedders import MockDocumentEmbedder

embedder = MockDocumentEmbedder(dimension=8)
result = embedder.run([Document(content="I love pizza!")])
print(result["documents"][0].embedding)  # a deterministic list of 8 floats
```

#### __init__

```python
__init__(
    embedding: list[float] | None = None,
    *,
    embedding_fn: EmbeddingFn | None = None,
    dimension: int = 768,
    model: str = "mock-model",
    meta: dict[str, Any] | None = None,
    prefix: str = "",
    suffix: str = "",
    meta_fields_to_embed: list[str] | None = None,
    embedding_separator: str = "\n",
    progress_bar: bool = False
) -> None
```

Creates an instance of MockDocumentEmbedder.

**Parameters:**

- **embedding** (<code>list\[float\] | None</code>) – An optional fixed embedding assigned to every document. Mutually exclusive with
  `embedding_fn`. If neither is provided, a deterministic embedding is derived from each document's text.
- **embedding_fn** (<code>EmbeddingFn | None</code>) – An optional callable that receives the prepared text of a document and returns the
  embedding as a list of floats. Mutually exclusive with `embedding`. To support serialization, pass a
  named function (lambdas and nested functions cannot be serialized).
- **dimension** (<code>int</code>) – The number of dimensions of the deterministic embedding. Ignored when `embedding` or
  `embedding_fn` is provided, since their length is determined by the value or callable.
- **model** (<code>str</code>) – The model name reported in the metadata. Purely cosmetic; no model is loaded.
- **meta** (<code>dict\[str, Any\] | None</code>) – Additional metadata merged into the output `meta`.
- **prefix** (<code>str</code>) – A string to add at the beginning of each text before embedding.
- **suffix** (<code>str</code>) – A string to add at the end of each text before embedding.
- **meta_fields_to_embed** (<code>list\[str\] | None</code>) – List of metadata fields to embed along with the document text.
- **embedding_separator** (<code>str</code>) – Separator used to concatenate the metadata fields to the document text.
- **progress_bar** (<code>bool</code>) – Accepted for interface compatibility with real Document Embedders and ignored.

**Raises:**

- <code>ValueError</code> – If both `embedding` and `embedding_fn` are provided, if `dimension` is not positive, or
  if `embedding` is an empty list.
- <code>TypeError</code> – If `embedding` is not a sequence of numbers.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the component to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> MockDocumentEmbedder
```

Deserialize the component from a dictionary.

#### warm_up

```python
warm_up() -> None
```

No-op warm up, provided for interface compatibility with real Embedders.

#### run

```python
run(documents: list[Document]) -> dict[str, Any]
```

Return the input documents with deterministic embeddings added, without calling any API.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – A list of documents to embed.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the following keys:
- `documents`: A list of documents with embeddings.
- `meta`: Metadata about the (mock) model.

**Raises:**

- <code>TypeError</code> – If `documents` is not a list of `Document` objects.

#### run_async

```python
run_async(documents: list[Document]) -> dict[str, Any]
```

Asynchronously return the input documents with deterministic embeddings added, without calling any API.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – A list of documents to embed.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the following keys:
- `documents`: A list of documents with embeddings.
- `meta`: Metadata about the (mock) model.

**Raises:**

- <code>TypeError</code> – If `documents` is not a list of `Document` objects.

## mock_text_embedder

### MockTextEmbedder

A Text Embedder that returns deterministic embeddings without calling any API.

It is a drop-in replacement for real Text Embedders (such as `OpenAITextEmbedder`) in tests, smoke tests, and
quick prototypes. It implements the same interface (`run`, `run_async`, serialization) but never contacts an
external service, so it is fully deterministic and free to run.

The embedding is selected based on how the component is configured:

- **Deterministic (default)**: with no configuration, the embedding is derived from a hash of the input text.
  The same text always yields the same embedding, and different texts yield different embeddings, so the mock
  works in retrieval pipelines and is reproducible across runs and processes.
- **Fixed embedding**: pass an `embedding` vector. The same vector is returned for every input.
- **Dynamic embedding**: pass an `embedding_fn` callable that receives the (prepared) text and returns the
  embedding. This is useful when the embedding should depend on the input in a custom way.

### Usage example

```python
from haystack.components.embedders import MockTextEmbedder

embedder = MockTextEmbedder(dimension=8)
result = embedder.run("I love pizza!")
print(result["embedding"])  # a deterministic list of 8 floats
```

#### __init__

```python
__init__(
    embedding: list[float] | None = None,
    *,
    embedding_fn: EmbeddingFn | None = None,
    dimension: int = 768,
    model: str = "mock-model",
    meta: dict[str, Any] | None = None,
    prefix: str = "",
    suffix: str = ""
) -> None
```

Creates an instance of MockTextEmbedder.

**Parameters:**

- **embedding** (<code>list\[float\] | None</code>) – An optional fixed embedding returned for every input. Mutually exclusive with
  `embedding_fn`. If neither is provided, a deterministic embedding is derived from the input text.
- **embedding_fn** (<code>EmbeddingFn | None</code>) – An optional callable that receives the prepared text (after `prefix`/`suffix` are
  applied) and returns the embedding as a list of floats. Mutually exclusive with `embedding`. To support
  serialization, pass a named function (lambdas and nested functions cannot be serialized).
- **dimension** (<code>int</code>) – The number of dimensions of the deterministic embedding. Ignored when `embedding` or
  `embedding_fn` is provided, since their length is determined by the value or callable.
- **model** (<code>str</code>) – The model name reported in the metadata. Purely cosmetic; no model is loaded.
- **meta** (<code>dict\[str, Any\] | None</code>) – Additional metadata merged into the output `meta`.
- **prefix** (<code>str</code>) – A string to add at the beginning of the text before embedding.
- **suffix** (<code>str</code>) – A string to add at the end of the text before embedding.

**Raises:**

- <code>ValueError</code> – If both `embedding` and `embedding_fn` are provided, if `dimension` is not positive, or
  if `embedding` is an empty list.
- <code>TypeError</code> – If `embedding` is not a sequence of numbers.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the component to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> MockTextEmbedder
```

Deserialize the component from a dictionary.

#### warm_up

```python
warm_up() -> None
```

No-op warm up, provided for interface compatibility with real Embedders.

#### run

```python
run(text: str) -> dict[str, Any]
```

Return a deterministic embedding for the input text without calling any API.

**Parameters:**

- **text** (<code>str</code>) – The text to embed.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the following keys:
- `embedding`: The embedding of the input text.
- `meta`: Metadata about the (mock) model.

**Raises:**

- <code>TypeError</code> – If `text` is not a string.

#### run_async

```python
run_async(text: str) -> dict[str, Any]
```

Asynchronously return a deterministic embedding for the input text without calling any API.

**Parameters:**

- **text** (<code>str</code>) – The text to embed.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the following keys:
- `embedding`: The embedding of the input text.
- `meta`: Metadata about the (mock) model.

**Raises:**

- <code>TypeError</code> – If `text` is not a string.

## openai_document_embedder

### OpenAIDocumentEmbedder

Computes document embeddings using OpenAI models.

### Usage example

<!-- test-ignore -->

```python
from haystack import Document
from haystack.components.embedders import OpenAIDocumentEmbedder

doc = Document(content="I love pizza!")
document_embedder = OpenAIDocumentEmbedder()
result = document_embedder.run([doc])

print(result['documents'][0].embedding)

# [0.017020374536514282, -0.023255806416273117, ...]
```

#### __init__

```python
__init__(
    api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
    model: str = "text-embedding-ada-002",
    dimensions: int | None = None,
    api_base_url: str | None = None,
    organization: str | None = None,
    prefix: str = "",
    suffix: str = "",
    batch_size: int = 32,
    progress_bar: bool = True,
    meta_fields_to_embed: list[str] | None = None,
    embedding_separator: str = "\n",
    timeout: float | None = None,
    max_retries: int | None = None,
    http_client_kwargs: dict[str, Any] | None = None,
    *,
    raise_on_failure: bool = False
) -> None
```

Creates an OpenAIDocumentEmbedder component.

Before initializing the component, you can set the 'OPENAI_TIMEOUT' and 'OPENAI_MAX_RETRIES'
environment variables to override the `timeout` and `max_retries` parameters respectively
in the OpenAI client.

**Parameters:**

- **api_key** (<code>Secret</code>) – The OpenAI API key.
  You can set it with an environment variable `OPENAI_API_KEY`, or pass with this parameter
  during initialization.
- **model** (<code>str</code>) – The name of the model to use for calculating embeddings.
  The default model is `text-embedding-ada-002`.
- **dimensions** (<code>int | None</code>) – The number of dimensions of the resulting embeddings. Only `text-embedding-3` and
  later models support this parameter.
- **api_base_url** (<code>str | None</code>) – Overrides the default base URL for all HTTP requests.
- **organization** (<code>str | None</code>) – Your OpenAI organization ID. See OpenAI's
  [Setting Up Your Organization](https://platform.openai.com/docs/guides/production-best-practices/setting-up-your-organization)
  for more information.
- **prefix** (<code>str</code>) – A string to add at the beginning of each text.
- **suffix** (<code>str</code>) – A string to add at the end of each text.
- **batch_size** (<code>int</code>) – Number of documents to embed at once.
- **progress_bar** (<code>bool</code>) – If `True`, shows a progress bar when running.
- **meta_fields_to_embed** (<code>list\[str\] | None</code>) – List of metadata fields to embed along with the document text.
- **embedding_separator** (<code>str</code>) – Separator used to concatenate the metadata fields to the document text.
- **timeout** (<code>float | None</code>) – Timeout for OpenAI client calls. If not set, it defaults to either the
  `OPENAI_TIMEOUT` environment variable, or 30 seconds.
- **max_retries** (<code>int | None</code>) – Maximum number of retries to contact OpenAI after an internal error.
  If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or 5 retries.
- **http_client_kwargs** (<code>dict\[str, Any\] | None</code>) – A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
  For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).
- **raise_on_failure** (<code>bool</code>) – Whether to raise an exception if the embedding request fails. If `False`, the component will log the error
  and continue processing the remaining documents. If `True`, it will raise an exception on failure.

#### warm_up

```python
warm_up() -> None
```

Initializes the synchronous OpenAI client.

#### warm_up_async

```python
warm_up_async() -> None
```

Initializes the asynchronous OpenAI client on the serving event loop.

#### close

```python
close() -> None
```

Releases the synchronous OpenAI client.

#### close_async

```python
close_async() -> None
```

Releases the asynchronous OpenAI client.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> OpenAIDocumentEmbedder
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>OpenAIDocumentEmbedder</code> – Deserialized component.

#### run

```python
run(documents: list[Document]) -> dict[str, Any]
```

Embeds a list of documents.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – A list of documents to embed.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the following keys:
- `documents`: A list of documents with embeddings.
- `meta`: Information about the usage of the model.

#### run_async

```python
run_async(documents: list[Document]) -> dict[str, Any]
```

Embeds a list of documents asynchronously.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – A list of documents to embed.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the following keys:
- `documents`: A list of documents with embeddings.
- `meta`: Information about the usage of the model.

## openai_text_embedder

### OpenAITextEmbedder

Embeds strings using OpenAI models.

You can use it to embed user query and send it to an embedding Retriever.

### Usage example

<!-- test-ignore -->

```python
from haystack.components.embedders import OpenAITextEmbedder

text_to_embed = "I love pizza!"
text_embedder = OpenAITextEmbedder()

print(text_embedder.run(text_to_embed))

# {'embedding': [0.017020374536514282, -0.023255806416273117, ...],
# 'meta': {'model': 'text-embedding-ada-002-v2',
#          'usage': {'prompt_tokens': 4, 'total_tokens': 4}}}
```

#### __init__

```python
__init__(
    api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
    model: str = "text-embedding-ada-002",
    dimensions: int | None = None,
    api_base_url: str | None = None,
    organization: str | None = None,
    prefix: str = "",
    suffix: str = "",
    timeout: float | None = None,
    max_retries: int | None = None,
    http_client_kwargs: dict[str, Any] | None = None,
) -> None
```

Creates an OpenAITextEmbedder component.

Before initializing the component, you can set the 'OPENAI_TIMEOUT' and 'OPENAI_MAX_RETRIES'
environment variables to override the `timeout` and `max_retries` parameters respectively
in the OpenAI client.

**Parameters:**

- **api_key** (<code>Secret</code>) – The OpenAI API key.
  You can set it with an environment variable `OPENAI_API_KEY`, or pass with this parameter
  during initialization.
- **model** (<code>str</code>) – The name of the model to use for calculating embeddings.
  The default model is `text-embedding-ada-002`.
- **dimensions** (<code>int | None</code>) – The number of dimensions of the resulting embeddings. Only `text-embedding-3` and
  later models support this parameter.
- **api_base_url** (<code>str | None</code>) – Overrides default base URL for all HTTP requests.
- **organization** (<code>str | None</code>) – Your organization ID. See OpenAI's
  [production best practices](https://platform.openai.com/docs/guides/production-best-practices/setting-up-your-organization)
  for more information.
- **prefix** (<code>str</code>) – A string to add at the beginning of each text to embed.
- **suffix** (<code>str</code>) – A string to add at the end of each text to embed.
- **timeout** (<code>float | None</code>) – Timeout for OpenAI client calls. If not set, it defaults to either the
  `OPENAI_TIMEOUT` environment variable, or 30 seconds.
- **max_retries** (<code>int | None</code>) – Maximum number of retries to contact OpenAI after an internal error.
  If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
- **http_client_kwargs** (<code>dict\[str, Any\] | None</code>) – A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
  For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).

#### warm_up

```python
warm_up() -> None
```

Initializes the synchronous OpenAI client.

#### warm_up_async

```python
warm_up_async() -> None
```

Initializes the asynchronous OpenAI client on the serving event loop.

#### close

```python
close() -> None
```

Releases the synchronous OpenAI client.

#### close_async

```python
close_async() -> None
```

Releases the asynchronous OpenAI client.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> OpenAITextEmbedder
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>OpenAITextEmbedder</code> – Deserialized component.

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
- `meta`: Information about the usage of the model.

#### run_async

```python
run_async(text: str) -> dict[str, Any]
```

Asynchronously embed a single string.

This is the asynchronous version of the `run` method. It has the same parameters and return values
but can be used with `await` in async code.

**Parameters:**

- **text** (<code>str</code>) – Text to embed.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the following keys:
- `embedding`: The embedding of the input text.
- `meta`: Information about the usage of the model.
