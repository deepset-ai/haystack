---
title: Embedders
id: embedders-api
description: Transforms queries into vectors to look for similar or relevant Documents.
slug: "/embedders-api"
---

<a id="azure_document_embedder"></a>

# Module azure\_document\_embedder

<a id="azure_document_embedder.AzureOpenAIDocumentEmbedder"></a>

## AzureOpenAIDocumentEmbedder

Calculates document embeddings using OpenAI models deployed on Azure.

### Usage example

```python
from haystack import Document
from haystack.components.embedders import AzureOpenAIDocumentEmbedder

doc = Document(content="I love pizza!")

document_embedder = AzureOpenAIDocumentEmbedder()

result = document_embedder.run([doc])
print(result['documents'][0].embedding)

# [0.017020374536514282, -0.023255806416273117, ...]
```

<a id="azure_document_embedder.AzureOpenAIDocumentEmbedder.__init__"></a>

#### AzureOpenAIDocumentEmbedder.\_\_init\_\_

```python
def __init__(azure_endpoint: Optional[str] = None,
             api_version: Optional[str] = "2023-05-15",
             azure_deployment: str = "text-embedding-ada-002",
             dimensions: Optional[int] = None,
             api_key: Optional[Secret] = Secret.from_env_var(
                 "AZURE_OPENAI_API_KEY", strict=False),
             azure_ad_token: Optional[Secret] = Secret.from_env_var(
                 "AZURE_OPENAI_AD_TOKEN", strict=False),
             organization: Optional[str] = None,
             prefix: str = "",
             suffix: str = "",
             batch_size: int = 32,
             progress_bar: bool = True,
             meta_fields_to_embed: Optional[list[str]] = None,
             embedding_separator: str = "\n",
             timeout: Optional[float] = None,
             max_retries: Optional[int] = None,
             *,
             default_headers: Optional[dict[str, str]] = None,
             azure_ad_token_provider: Optional[AzureADTokenProvider] = None,
             http_client_kwargs: Optional[dict[str, Any]] = None,
             raise_on_failure: bool = False)
```

Creates an AzureOpenAIDocumentEmbedder component.

**Arguments**:

- `azure_endpoint`: The endpoint of the model deployed on Azure.
- `api_version`: The version of the API to use.
- `azure_deployment`: The name of the model deployed on Azure. The default model is text-embedding-ada-002.
- `dimensions`: The number of dimensions of the resulting embeddings. Only supported in text-embedding-3
and later models.
- `api_key`: The Azure OpenAI API key.
You can set it with an environment variable `AZURE_OPENAI_API_KEY`, or pass with this
parameter during initialization.
- `azure_ad_token`: Microsoft Entra ID token, see Microsoft's
[Entra ID](https://www.microsoft.com/en-us/security/business/identity-access/microsoft-entra-id)
documentation for more information. You can set it with an environment variable
`AZURE_OPENAI_AD_TOKEN`, or pass with this parameter during initialization.
Previously called Azure Active Directory.
- `organization`: Your organization ID. See OpenAI's
[Setting Up Your Organization](https://platform.openai.com/docs/guides/production-best-practices/setting-up-your-organization)
for more information.
- `prefix`: A string to add at the beginning of each text.
- `suffix`: A string to add at the end of each text.
- `batch_size`: Number of documents to embed at once.
- `progress_bar`: If `True`, shows a progress bar when running.
- `meta_fields_to_embed`: List of metadata fields to embed along with the document text.
- `embedding_separator`: Separator used to concatenate the metadata fields to the document text.
- `timeout`: The timeout for `AzureOpenAI` client calls, in seconds.
If not set, defaults to either the
`OPENAI_TIMEOUT` environment variable, or 30 seconds.
- `max_retries`: Maximum number of retries to contact AzureOpenAI after an internal error.
If not set, defaults to either the `OPENAI_MAX_RETRIES` environment variable or to 5 retries.
- `default_headers`: Default headers to send to the AzureOpenAI client.
- `azure_ad_token_provider`: A function that returns an Azure Active Directory token, will be invoked on
every request.
- `http_client_kwargs`: A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/`client`).
- `raise_on_failure`: Whether to raise an exception if the embedding request fails. If `False`, the component will log the error
and continue processing the remaining documents. If `True`, it will raise an exception on failure.

<a id="azure_document_embedder.AzureOpenAIDocumentEmbedder.to_dict"></a>

#### AzureOpenAIDocumentEmbedder.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="azure_document_embedder.AzureOpenAIDocumentEmbedder.from_dict"></a>

#### AzureOpenAIDocumentEmbedder.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "AzureOpenAIDocumentEmbedder"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="azure_document_embedder.AzureOpenAIDocumentEmbedder.run"></a>

#### AzureOpenAIDocumentEmbedder.run

```python
@component.output_types(documents=list[Document], meta=dict[str, Any])
def run(documents: list[Document])
```

Embeds a list of documents.

**Arguments**:

- `documents`: A list of documents to embed.

**Returns**:

A dictionary with the following keys:
- `documents`: A list of documents with embeddings.
- `meta`: Information about the usage of the model.

<a id="azure_document_embedder.AzureOpenAIDocumentEmbedder.run_async"></a>

#### AzureOpenAIDocumentEmbedder.run\_async

```python
@component.output_types(documents=list[Document], meta=dict[str, Any])
async def run_async(documents: list[Document])
```

Embeds a list of documents asynchronously.

**Arguments**:

- `documents`: A list of documents to embed.

**Returns**:

A dictionary with the following keys:
- `documents`: A list of documents with embeddings.
- `meta`: Information about the usage of the model.

<a id="azure_text_embedder"></a>

# Module azure\_text\_embedder

<a id="azure_text_embedder.AzureOpenAITextEmbedder"></a>

## AzureOpenAITextEmbedder

Embeds strings using OpenAI models deployed on Azure.

### Usage example

```python
from haystack.components.embedders import AzureOpenAITextEmbedder

text_to_embed = "I love pizza!"

text_embedder = AzureOpenAITextEmbedder()

print(text_embedder.run(text_to_embed))

# {'embedding': [0.017020374536514282, -0.023255806416273117, ...],
# 'meta': {'model': 'text-embedding-ada-002-v2',
#          'usage': {'prompt_tokens': 4, 'total_tokens': 4}}}
```

<a id="azure_text_embedder.AzureOpenAITextEmbedder.__init__"></a>

#### AzureOpenAITextEmbedder.\_\_init\_\_

```python
def __init__(azure_endpoint: Optional[str] = None,
             api_version: Optional[str] = "2023-05-15",
             azure_deployment: str = "text-embedding-ada-002",
             dimensions: Optional[int] = None,
             api_key: Optional[Secret] = Secret.from_env_var(
                 "AZURE_OPENAI_API_KEY", strict=False),
             azure_ad_token: Optional[Secret] = Secret.from_env_var(
                 "AZURE_OPENAI_AD_TOKEN", strict=False),
             organization: Optional[str] = None,
             timeout: Optional[float] = None,
             max_retries: Optional[int] = None,
             prefix: str = "",
             suffix: str = "",
             *,
             default_headers: Optional[dict[str, str]] = None,
             azure_ad_token_provider: Optional[AzureADTokenProvider] = None,
             http_client_kwargs: Optional[dict[str, Any]] = None)
```

Creates an AzureOpenAITextEmbedder component.

**Arguments**:

- `azure_endpoint`: The endpoint of the model deployed on Azure.
- `api_version`: The version of the API to use.
- `azure_deployment`: The name of the model deployed on Azure. The default model is text-embedding-ada-002.
- `dimensions`: The number of dimensions the resulting output embeddings should have. Only supported in text-embedding-3
and later models.
- `api_key`: The Azure OpenAI API key.
You can set it with an environment variable `AZURE_OPENAI_API_KEY`, or pass with this
parameter during initialization.
- `azure_ad_token`: Microsoft Entra ID token, see Microsoft's
[Entra ID](https://www.microsoft.com/en-us/security/business/identity-access/microsoft-entra-id)
documentation for more information. You can set it with an environment variable
`AZURE_OPENAI_AD_TOKEN`, or pass with this parameter during initialization.
Previously called Azure Active Directory.
- `organization`: Your organization ID. See OpenAI's
[Setting Up Your Organization](https://platform.openai.com/docs/guides/production-best-practices/setting-up-your-organization)
for more information.
- `timeout`: The timeout for `AzureOpenAI` client calls, in seconds.
If not set, defaults to either the
`OPENAI_TIMEOUT` environment variable, or 30 seconds.
- `max_retries`: Maximum number of retries to contact AzureOpenAI after an internal error.
If not set, defaults to either the `OPENAI_MAX_RETRIES` environment variable, or to 5 retries.
- `prefix`: A string to add at the beginning of each text.
- `suffix`: A string to add at the end of each text.
- `default_headers`: Default headers to send to the AzureOpenAI client.
- `azure_ad_token_provider`: A function that returns an Azure Active Directory token, will be invoked on
every request.
- `http_client_kwargs`: A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/`client`).

<a id="azure_text_embedder.AzureOpenAITextEmbedder.to_dict"></a>

#### AzureOpenAITextEmbedder.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="azure_text_embedder.AzureOpenAITextEmbedder.from_dict"></a>

#### AzureOpenAITextEmbedder.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "AzureOpenAITextEmbedder"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="azure_text_embedder.AzureOpenAITextEmbedder.run"></a>

#### AzureOpenAITextEmbedder.run

```python
@component.output_types(embedding=list[float], meta=dict[str, Any])
def run(text: str)
```

Embeds a single string.

**Arguments**:

- `text`: Text to embed.

**Returns**:

A dictionary with the following keys:
- `embedding`: The embedding of the input text.
- `meta`: Information about the usage of the model.

<a id="azure_text_embedder.AzureOpenAITextEmbedder.run_async"></a>

#### AzureOpenAITextEmbedder.run\_async

```python
@component.output_types(embedding=list[float], meta=dict[str, Any])
async def run_async(text: str)
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

<a id="hugging_face_api_document_embedder"></a>

# Module hugging\_face\_api\_document\_embedder

<a id="hugging_face_api_document_embedder.HuggingFaceAPIDocumentEmbedder"></a>

## HuggingFaceAPIDocumentEmbedder

Embeds documents using Hugging Face APIs.

Use it with the following Hugging Face APIs:
- [Free Serverless Inference API](https://huggingface.co/inference-api)
- [Paid Inference Endpoints](https://huggingface.co/inference-endpoints)
- [Self-hosted Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference)


### Usage examples

#### With free serverless inference API

```python
from haystack.components.embedders import HuggingFaceAPIDocumentEmbedder
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
from haystack.components.embedders import HuggingFaceAPIDocumentEmbedder
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
from haystack.components.embedders import HuggingFaceAPIDocumentEmbedder
from haystack.dataclasses import Document

doc = Document(content="I love pizza!")

doc_embedder = HuggingFaceAPIDocumentEmbedder(api_type="text_embeddings_inference",
                                              api_params={"url": "http://localhost:8080"})

result = document_embedder.run([doc])
print(result["documents"][0].embedding)

# [0.017020374536514282, -0.023255806416273117, ...]
```

<a id="hugging_face_api_document_embedder.HuggingFaceAPIDocumentEmbedder.__init__"></a>

#### HuggingFaceAPIDocumentEmbedder.\_\_init\_\_

```python
def __init__(api_type: Union[HFEmbeddingAPIType, str],
             api_params: dict[str, str],
             token: Optional[Secret] = Secret.from_env_var(
                 ["HF_API_TOKEN", "HF_TOKEN"], strict=False),
             prefix: str = "",
             suffix: str = "",
             truncate: Optional[bool] = True,
             normalize: Optional[bool] = False,
             batch_size: int = 32,
             progress_bar: bool = True,
             meta_fields_to_embed: Optional[list[str]] = None,
             embedding_separator: str = "\n")
```

Creates a HuggingFaceAPIDocumentEmbedder component.

**Arguments**:

- `api_type`: The type of Hugging Face API to use.
- `api_params`: A dictionary with the following keys:
- `model`: Hugging Face model ID. Required when `api_type` is `SERVERLESS_INFERENCE_API`.
- `url`: URL of the inference endpoint. Required when `api_type` is `INFERENCE_ENDPOINTS` or
`TEXT_EMBEDDINGS_INFERENCE`.
- `token`: The Hugging Face token to use as HTTP bearer authorization.
Check your HF token in your [account settings](https://huggingface.co/settings/tokens).
- `prefix`: A string to add at the beginning of each text.
- `suffix`: A string to add at the end of each text.
- `truncate`: Truncates the input text to the maximum length supported by the model.
Applicable when `api_type` is `TEXT_EMBEDDINGS_INFERENCE`, or `INFERENCE_ENDPOINTS`
if the backend uses Text Embeddings Inference.
If `api_type` is `SERVERLESS_INFERENCE_API`, this parameter is ignored.
- `normalize`: Normalizes the embeddings to unit length.
Applicable when `api_type` is `TEXT_EMBEDDINGS_INFERENCE`, or `INFERENCE_ENDPOINTS`
if the backend uses Text Embeddings Inference.
If `api_type` is `SERVERLESS_INFERENCE_API`, this parameter is ignored.
- `batch_size`: Number of documents to process at once.
- `progress_bar`: If `True`, shows a progress bar when running.
- `meta_fields_to_embed`: List of metadata fields to embed along with the document text.
- `embedding_separator`: Separator used to concatenate the metadata fields to the document text.

<a id="hugging_face_api_document_embedder.HuggingFaceAPIDocumentEmbedder.to_dict"></a>

#### HuggingFaceAPIDocumentEmbedder.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="hugging_face_api_document_embedder.HuggingFaceAPIDocumentEmbedder.from_dict"></a>

#### HuggingFaceAPIDocumentEmbedder.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "HuggingFaceAPIDocumentEmbedder"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="hugging_face_api_document_embedder.HuggingFaceAPIDocumentEmbedder.run"></a>

#### HuggingFaceAPIDocumentEmbedder.run

```python
@component.output_types(documents=list[Document])
def run(documents: list[Document])
```

Embeds a list of documents.

**Arguments**:

- `documents`: Documents to embed.

**Returns**:

A dictionary with the following keys:
- `documents`: A list of documents with embeddings.

<a id="hugging_face_api_document_embedder.HuggingFaceAPIDocumentEmbedder.run_async"></a>

#### HuggingFaceAPIDocumentEmbedder.run\_async

```python
@component.output_types(documents=list[Document])
async def run_async(documents: list[Document])
```

Embeds a list of documents asynchronously.

**Arguments**:

- `documents`: Documents to embed.

**Returns**:

A dictionary with the following keys:
- `documents`: A list of documents with embeddings.

<a id="hugging_face_api_text_embedder"></a>

# Module hugging\_face\_api\_text\_embedder

<a id="hugging_face_api_text_embedder.HuggingFaceAPITextEmbedder"></a>

## HuggingFaceAPITextEmbedder

Embeds strings using Hugging Face APIs.

Use it with the following Hugging Face APIs:
- [Free Serverless Inference API](https://huggingface.co/inference-api)
- [Paid Inference Endpoints](https://huggingface.co/inference-endpoints)
- [Self-hosted Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference)

### Usage examples

#### With free serverless inference API

```python
from haystack.components.embedders import HuggingFaceAPITextEmbedder
from haystack.utils import Secret

text_embedder = HuggingFaceAPITextEmbedder(api_type="serverless_inference_api",
                                           api_params={"model": "BAAI/bge-small-en-v1.5"},
                                           token=Secret.from_token("<your-api-key>"))

print(text_embedder.run("I love pizza!"))

# {'embedding': [0.017020374536514282, -0.023255806416273117, ...],
```

#### With paid inference endpoints

```python
from haystack.components.embedders import HuggingFaceAPITextEmbedder
from haystack.utils import Secret
text_embedder = HuggingFaceAPITextEmbedder(api_type="inference_endpoints",
                                           api_params={"model": "BAAI/bge-small-en-v1.5"},
                                           token=Secret.from_token("<your-api-key>"))

print(text_embedder.run("I love pizza!"))

# {'embedding': [0.017020374536514282, -0.023255806416273117, ...],
```

#### With self-hosted text embeddings inference

```python
from haystack.components.embedders import HuggingFaceAPITextEmbedder
from haystack.utils import Secret

text_embedder = HuggingFaceAPITextEmbedder(api_type="text_embeddings_inference",
                                           api_params={"url": "http://localhost:8080"})

print(text_embedder.run("I love pizza!"))

# {'embedding': [0.017020374536514282, -0.023255806416273117, ...],
```

<a id="hugging_face_api_text_embedder.HuggingFaceAPITextEmbedder.__init__"></a>

#### HuggingFaceAPITextEmbedder.\_\_init\_\_

```python
def __init__(api_type: Union[HFEmbeddingAPIType, str],
             api_params: dict[str, str],
             token: Optional[Secret] = Secret.from_env_var(
                 ["HF_API_TOKEN", "HF_TOKEN"], strict=False),
             prefix: str = "",
             suffix: str = "",
             truncate: Optional[bool] = True,
             normalize: Optional[bool] = False)
```

Creates a HuggingFaceAPITextEmbedder component.

**Arguments**:

- `api_type`: The type of Hugging Face API to use.
- `api_params`: A dictionary with the following keys:
- `model`: Hugging Face model ID. Required when `api_type` is `SERVERLESS_INFERENCE_API`.
- `url`: URL of the inference endpoint. Required when `api_type` is `INFERENCE_ENDPOINTS` or
`TEXT_EMBEDDINGS_INFERENCE`.
- `token`: The Hugging Face token to use as HTTP bearer authorization.
Check your HF token in your [account settings](https://huggingface.co/settings/tokens).
- `prefix`: A string to add at the beginning of each text.
- `suffix`: A string to add at the end of each text.
- `truncate`: Truncates the input text to the maximum length supported by the model.
Applicable when `api_type` is `TEXT_EMBEDDINGS_INFERENCE`, or `INFERENCE_ENDPOINTS`
if the backend uses Text Embeddings Inference.
If `api_type` is `SERVERLESS_INFERENCE_API`, this parameter is ignored.
- `normalize`: Normalizes the embeddings to unit length.
Applicable when `api_type` is `TEXT_EMBEDDINGS_INFERENCE`, or `INFERENCE_ENDPOINTS`
if the backend uses Text Embeddings Inference.
If `api_type` is `SERVERLESS_INFERENCE_API`, this parameter is ignored.

<a id="hugging_face_api_text_embedder.HuggingFaceAPITextEmbedder.to_dict"></a>

#### HuggingFaceAPITextEmbedder.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="hugging_face_api_text_embedder.HuggingFaceAPITextEmbedder.from_dict"></a>

#### HuggingFaceAPITextEmbedder.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "HuggingFaceAPITextEmbedder"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="hugging_face_api_text_embedder.HuggingFaceAPITextEmbedder.run"></a>

#### HuggingFaceAPITextEmbedder.run

```python
@component.output_types(embedding=list[float])
def run(text: str)
```

Embeds a single string.

**Arguments**:

- `text`: Text to embed.

**Returns**:

A dictionary with the following keys:
- `embedding`: The embedding of the input text.

<a id="hugging_face_api_text_embedder.HuggingFaceAPITextEmbedder.run_async"></a>

#### HuggingFaceAPITextEmbedder.run\_async

```python
@component.output_types(embedding=list[float])
async def run_async(text: str)
```

Embeds a single string asynchronously.

**Arguments**:

- `text`: Text to embed.

**Returns**:

A dictionary with the following keys:
- `embedding`: The embedding of the input text.

<a id="openai_document_embedder"></a>

# Module openai\_document\_embedder

<a id="openai_document_embedder.OpenAIDocumentEmbedder"></a>

## OpenAIDocumentEmbedder

Computes document embeddings using OpenAI models.

### Usage example

```python
from haystack import Document
from haystack.components.embedders import OpenAIDocumentEmbedder

doc = Document(content="I love pizza!")

document_embedder = OpenAIDocumentEmbedder()

result = document_embedder.run([doc])
print(result['documents'][0].embedding)

# [0.017020374536514282, -0.023255806416273117, ...]
```

<a id="openai_document_embedder.OpenAIDocumentEmbedder.__init__"></a>

#### OpenAIDocumentEmbedder.\_\_init\_\_

```python
def __init__(api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
             model: str = "text-embedding-ada-002",
             dimensions: Optional[int] = None,
             api_base_url: Optional[str] = None,
             organization: Optional[str] = None,
             prefix: str = "",
             suffix: str = "",
             batch_size: int = 32,
             progress_bar: bool = True,
             meta_fields_to_embed: Optional[list[str]] = None,
             embedding_separator: str = "\n",
             timeout: Optional[float] = None,
             max_retries: Optional[int] = None,
             http_client_kwargs: Optional[dict[str, Any]] = None,
             *,
             raise_on_failure: bool = False)
```

Creates an OpenAIDocumentEmbedder component.

Before initializing the component, you can set the 'OPENAI_TIMEOUT' and 'OPENAI_MAX_RETRIES'
environment variables to override the `timeout` and `max_retries` parameters respectively
in the OpenAI client.

**Arguments**:

- `api_key`: The OpenAI API key.
You can set it with an environment variable `OPENAI_API_KEY`, or pass with this parameter
during initialization.
- `model`: The name of the model to use for calculating embeddings.
The default model is `text-embedding-ada-002`.
- `dimensions`: The number of dimensions of the resulting embeddings. Only `text-embedding-3` and
later models support this parameter.
- `api_base_url`: Overrides the default base URL for all HTTP requests.
- `organization`: Your OpenAI organization ID. See OpenAI's
[Setting Up Your Organization](https://platform.openai.com/docs/guides/production-best-practices/setting-up-your-organization)
for more information.
- `prefix`: A string to add at the beginning of each text.
- `suffix`: A string to add at the end of each text.
- `batch_size`: Number of documents to embed at once.
- `progress_bar`: If `True`, shows a progress bar when running.
- `meta_fields_to_embed`: List of metadata fields to embed along with the document text.
- `embedding_separator`: Separator used to concatenate the metadata fields to the document text.
- `timeout`: Timeout for OpenAI client calls. If not set, it defaults to either the
`OPENAI_TIMEOUT` environment variable, or 30 seconds.
- `max_retries`: Maximum number of retries to contact OpenAI after an internal error.
If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or 5 retries.
- `http_client_kwargs`: A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/`client`).
- `raise_on_failure`: Whether to raise an exception if the embedding request fails. If `False`, the component will log the error
and continue processing the remaining documents. If `True`, it will raise an exception on failure.

<a id="openai_document_embedder.OpenAIDocumentEmbedder.to_dict"></a>

#### OpenAIDocumentEmbedder.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="openai_document_embedder.OpenAIDocumentEmbedder.from_dict"></a>

#### OpenAIDocumentEmbedder.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "OpenAIDocumentEmbedder"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="openai_document_embedder.OpenAIDocumentEmbedder.run"></a>

#### OpenAIDocumentEmbedder.run

```python
@component.output_types(documents=list[Document], meta=dict[str, Any])
def run(documents: list[Document])
```

Embeds a list of documents.

**Arguments**:

- `documents`: A list of documents to embed.

**Returns**:

A dictionary with the following keys:
- `documents`: A list of documents with embeddings.
- `meta`: Information about the usage of the model.

<a id="openai_document_embedder.OpenAIDocumentEmbedder.run_async"></a>

#### OpenAIDocumentEmbedder.run\_async

```python
@component.output_types(documents=list[Document], meta=dict[str, Any])
async def run_async(documents: list[Document])
```

Embeds a list of documents asynchronously.

**Arguments**:

- `documents`: A list of documents to embed.

**Returns**:

A dictionary with the following keys:
- `documents`: A list of documents with embeddings.
- `meta`: Information about the usage of the model.

<a id="openai_text_embedder"></a>

# Module openai\_text\_embedder

<a id="openai_text_embedder.OpenAITextEmbedder"></a>

## OpenAITextEmbedder

Embeds strings using OpenAI models.

You can use it to embed user query and send it to an embedding Retriever.

### Usage example

```python
from haystack.components.embedders import OpenAITextEmbedder

text_to_embed = "I love pizza!"

text_embedder = OpenAITextEmbedder()

print(text_embedder.run(text_to_embed))

# {'embedding': [0.017020374536514282, -0.023255806416273117, ...],
# 'meta': {'model': 'text-embedding-ada-002-v2',
#          'usage': {'prompt_tokens': 4, 'total_tokens': 4}}}
```

<a id="openai_text_embedder.OpenAITextEmbedder.__init__"></a>

#### OpenAITextEmbedder.\_\_init\_\_

```python
def __init__(api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
             model: str = "text-embedding-ada-002",
             dimensions: Optional[int] = None,
             api_base_url: Optional[str] = None,
             organization: Optional[str] = None,
             prefix: str = "",
             suffix: str = "",
             timeout: Optional[float] = None,
             max_retries: Optional[int] = None,
             http_client_kwargs: Optional[dict[str, Any]] = None)
```

Creates an OpenAITextEmbedder component.

Before initializing the component, you can set the 'OPENAI_TIMEOUT' and 'OPENAI_MAX_RETRIES'
environment variables to override the `timeout` and `max_retries` parameters respectively
in the OpenAI client.

**Arguments**:

- `api_key`: The OpenAI API key.
You can set it with an environment variable `OPENAI_API_KEY`, or pass with this parameter
during initialization.
- `model`: The name of the model to use for calculating embeddings.
The default model is `text-embedding-ada-002`.
- `dimensions`: The number of dimensions of the resulting embeddings. Only `text-embedding-3` and
later models support this parameter.
- `api_base_url`: Overrides default base URL for all HTTP requests.
- `organization`: Your organization ID. See OpenAI's
[production best practices](https://platform.openai.com/docs/guides/production-best-practices/setting-up-your-organization)
for more information.
- `prefix`: A string to add at the beginning of each text to embed.
- `suffix`: A string to add at the end of each text to embed.
- `timeout`: Timeout for OpenAI client calls. If not set, it defaults to either the
`OPENAI_TIMEOUT` environment variable, or 30 seconds.
- `max_retries`: Maximum number of retries to contact OpenAI after an internal error.
If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
- `http_client_kwargs`: A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/`client`).

<a id="openai_text_embedder.OpenAITextEmbedder.to_dict"></a>

#### OpenAITextEmbedder.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="openai_text_embedder.OpenAITextEmbedder.from_dict"></a>

#### OpenAITextEmbedder.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "OpenAITextEmbedder"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="openai_text_embedder.OpenAITextEmbedder.run"></a>

#### OpenAITextEmbedder.run

```python
@component.output_types(embedding=list[float], meta=dict[str, Any])
def run(text: str)
```

Embeds a single string.

**Arguments**:

- `text`: Text to embed.

**Returns**:

A dictionary with the following keys:
- `embedding`: The embedding of the input text.
- `meta`: Information about the usage of the model.

<a id="openai_text_embedder.OpenAITextEmbedder.run_async"></a>

#### OpenAITextEmbedder.run\_async

```python
@component.output_types(embedding=list[float], meta=dict[str, Any])
async def run_async(text: str)
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

<a id="sentence_transformers_document_embedder"></a>

# Module sentence\_transformers\_document\_embedder

<a id="sentence_transformers_document_embedder.SentenceTransformersDocumentEmbedder"></a>

## SentenceTransformersDocumentEmbedder

Calculates document embeddings using Sentence Transformers models.

It stores the embeddings in the `embedding` metadata field of each document.
You can also embed documents' metadata.
Use this component in indexing pipelines to embed input documents
and send them to DocumentWriter to write a into a Document Store.

### Usage example:

```python
from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
doc = Document(content="I love pizza!")
doc_embedder = SentenceTransformersDocumentEmbedder()
doc_embedder.warm_up()

result = doc_embedder.run([doc])
print(result['documents'][0].embedding)

# [-0.07804739475250244, 0.1498992145061493, ...]
```

<a id="sentence_transformers_document_embedder.SentenceTransformersDocumentEmbedder.__init__"></a>

#### SentenceTransformersDocumentEmbedder.\_\_init\_\_

```python
def __init__(model: str = "sentence-transformers/all-mpnet-base-v2",
             device: Optional[ComponentDevice] = None,
             token: Optional[Secret] = Secret.from_env_var(
                 ["HF_API_TOKEN", "HF_TOKEN"], strict=False),
             prefix: str = "",
             suffix: str = "",
             batch_size: int = 32,
             progress_bar: bool = True,
             normalize_embeddings: bool = False,
             meta_fields_to_embed: Optional[list[str]] = None,
             embedding_separator: str = "\n",
             trust_remote_code: bool = False,
             local_files_only: bool = False,
             truncate_dim: Optional[int] = None,
             model_kwargs: Optional[dict[str, Any]] = None,
             tokenizer_kwargs: Optional[dict[str, Any]] = None,
             config_kwargs: Optional[dict[str, Any]] = None,
             precision: Literal["float32", "int8", "uint8", "binary",
                                "ubinary"] = "float32",
             encode_kwargs: Optional[dict[str, Any]] = None,
             backend: Literal["torch", "onnx", "openvino"] = "torch")
```

Creates a SentenceTransformersDocumentEmbedder component.

**Arguments**:

- `model`: The model to use for calculating embeddings.
Pass a local path or ID of the model on Hugging Face.
- `device`: The device to use for loading the model.
Overrides the default device.
- `token`: The API token to download private models from Hugging Face.
- `prefix`: A string to add at the beginning of each document text.
Can be used to prepend the text with an instruction, as required by some embedding models,
such as E5 and bge.
- `suffix`: A string to add at the end of each document text.
- `batch_size`: Number of documents to embed at once.
- `progress_bar`: If `True`, shows a progress bar when embedding documents.
- `normalize_embeddings`: If `True`, the embeddings are normalized using L2 normalization, so that each embedding has a norm of 1.
- `meta_fields_to_embed`: List of metadata fields to embed along with the document text.
- `embedding_separator`: Separator used to concatenate the metadata fields to the document text.
- `trust_remote_code`: If `False`, allows only Hugging Face verified model architectures.
If `True`, allows custom models and scripts.
- `local_files_only`: If `True`, does not attempt to download the model from Hugging Face Hub and only looks at local files.
- `truncate_dim`: The dimension to truncate sentence embeddings to. `None` does no truncation.
If the model wasn't trained with Matryoshka Representation Learning,
truncating embeddings can significantly affect performance.
- `model_kwargs`: Additional keyword arguments for `AutoModelForSequenceClassification.from_pretrained`
when loading the model. Refer to specific model documentation for available kwargs.
- `tokenizer_kwargs`: Additional keyword arguments for `AutoTokenizer.from_pretrained` when loading the tokenizer.
Refer to specific model documentation for available kwargs.
- `config_kwargs`: Additional keyword arguments for `AutoConfig.from_pretrained` when loading the model configuration.
- `precision`: The precision to use for the embeddings.
All non-float32 precisions are quantized embeddings.
Quantized embeddings are smaller and faster to compute, but may have a lower accuracy.
They are useful for reducing the size of the embeddings of a corpus for semantic search, among other tasks.
- `encode_kwargs`: Additional keyword arguments for `SentenceTransformer.encode` when embedding documents.
This parameter is provided for fine customization. Be careful not to clash with already set parameters and
avoid passing parameters that change the output type.
- `backend`: The backend to use for the Sentence Transformers model. Choose from "torch", "onnx", or "openvino".
Refer to the [Sentence Transformers documentation](https://sbert.net/docs/sentence_transformer/usage/efficiency.html)
for more information on acceleration and quantization options.

<a id="sentence_transformers_document_embedder.SentenceTransformersDocumentEmbedder.to_dict"></a>

#### SentenceTransformersDocumentEmbedder.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="sentence_transformers_document_embedder.SentenceTransformersDocumentEmbedder.from_dict"></a>

#### SentenceTransformersDocumentEmbedder.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str,
                              Any]) -> "SentenceTransformersDocumentEmbedder"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="sentence_transformers_document_embedder.SentenceTransformersDocumentEmbedder.warm_up"></a>

#### SentenceTransformersDocumentEmbedder.warm\_up

```python
def warm_up()
```

Initializes the component.

<a id="sentence_transformers_document_embedder.SentenceTransformersDocumentEmbedder.run"></a>

#### SentenceTransformersDocumentEmbedder.run

```python
@component.output_types(documents=list[Document])
def run(documents: list[Document])
```

Embed a list of documents.

**Arguments**:

- `documents`: Documents to embed.

**Returns**:

A dictionary with the following keys:
- `documents`: Documents with embeddings.

<a id="sentence_transformers_text_embedder"></a>

# Module sentence\_transformers\_text\_embedder

<a id="sentence_transformers_text_embedder.SentenceTransformersTextEmbedder"></a>

## SentenceTransformersTextEmbedder

Embeds strings using Sentence Transformers models.

You can use it to embed user query and send it to an embedding retriever.

Usage example:
```python
from haystack.components.embedders import SentenceTransformersTextEmbedder

text_to_embed = "I love pizza!"

text_embedder = SentenceTransformersTextEmbedder()
text_embedder.warm_up()

print(text_embedder.run(text_to_embed))

# {'embedding': [-0.07804739475250244, 0.1498992145061493,, ...]}
```

<a id="sentence_transformers_text_embedder.SentenceTransformersTextEmbedder.__init__"></a>

#### SentenceTransformersTextEmbedder.\_\_init\_\_

```python
def __init__(model: str = "sentence-transformers/all-mpnet-base-v2",
             device: Optional[ComponentDevice] = None,
             token: Optional[Secret] = Secret.from_env_var(
                 ["HF_API_TOKEN", "HF_TOKEN"], strict=False),
             prefix: str = "",
             suffix: str = "",
             batch_size: int = 32,
             progress_bar: bool = True,
             normalize_embeddings: bool = False,
             trust_remote_code: bool = False,
             local_files_only: bool = False,
             truncate_dim: Optional[int] = None,
             model_kwargs: Optional[dict[str, Any]] = None,
             tokenizer_kwargs: Optional[dict[str, Any]] = None,
             config_kwargs: Optional[dict[str, Any]] = None,
             precision: Literal["float32", "int8", "uint8", "binary",
                                "ubinary"] = "float32",
             encode_kwargs: Optional[dict[str, Any]] = None,
             backend: Literal["torch", "onnx", "openvino"] = "torch")
```

Create a SentenceTransformersTextEmbedder component.

**Arguments**:

- `model`: The model to use for calculating embeddings.
Specify the path to a local model or the ID of the model on Hugging Face.
- `device`: Overrides the default device used to load the model.
- `token`: An API token to use private models from Hugging Face.
- `prefix`: A string to add at the beginning of each text to be embedded.
You can use it to prepend the text with an instruction, as required by some embedding models,
such as E5 and bge.
- `suffix`: A string to add at the end of each text to embed.
- `batch_size`: Number of texts to embed at once.
- `progress_bar`: If `True`, shows a progress bar for calculating embeddings.
If `False`, disables the progress bar.
- `normalize_embeddings`: If `True`, the embeddings are normalized using L2 normalization, so that the embeddings have a norm of 1.
- `trust_remote_code`: If `False`, permits only Hugging Face verified model architectures.
If `True`, permits custom models and scripts.
- `local_files_only`: If `True`, does not attempt to download the model from Hugging Face Hub and only looks at local files.
- `truncate_dim`: The dimension to truncate sentence embeddings to. `None` does no truncation.
If the model has not been trained with Matryoshka Representation Learning,
truncation of embeddings can significantly affect performance.
- `model_kwargs`: Additional keyword arguments for `AutoModelForSequenceClassification.from_pretrained`
when loading the model. Refer to specific model documentation for available kwargs.
- `tokenizer_kwargs`: Additional keyword arguments for `AutoTokenizer.from_pretrained` when loading the tokenizer.
Refer to specific model documentation for available kwargs.
- `config_kwargs`: Additional keyword arguments for `AutoConfig.from_pretrained` when loading the model configuration.
- `precision`: The precision to use for the embeddings.
All non-float32 precisions are quantized embeddings.
Quantized embeddings are smaller in size and faster to compute, but may have a lower accuracy.
They are useful for reducing the size of the embeddings of a corpus for semantic search, among other tasks.
- `encode_kwargs`: Additional keyword arguments for `SentenceTransformer.encode` when embedding texts.
This parameter is provided for fine customization. Be careful not to clash with already set parameters and
avoid passing parameters that change the output type.
- `backend`: The backend to use for the Sentence Transformers model. Choose from "torch", "onnx", or "openvino".
Refer to the [Sentence Transformers documentation](https://sbert.net/docs/sentence_transformer/usage/efficiency.html)
for more information on acceleration and quantization options.

<a id="sentence_transformers_text_embedder.SentenceTransformersTextEmbedder.to_dict"></a>

#### SentenceTransformersTextEmbedder.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="sentence_transformers_text_embedder.SentenceTransformersTextEmbedder.from_dict"></a>

#### SentenceTransformersTextEmbedder.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "SentenceTransformersTextEmbedder"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="sentence_transformers_text_embedder.SentenceTransformersTextEmbedder.warm_up"></a>

#### SentenceTransformersTextEmbedder.warm\_up

```python
def warm_up()
```

Initializes the component.

<a id="sentence_transformers_text_embedder.SentenceTransformersTextEmbedder.run"></a>

#### SentenceTransformersTextEmbedder.run

```python
@component.output_types(embedding=list[float])
def run(text: str)
```

Embed a single string.

**Arguments**:

- `text`: Text to embed.

**Returns**:

A dictionary with the following keys:
- `embedding`: The embedding of the input text.

<a id="sentence_transformers_sparse_document_embedder"></a>

# Module sentence\_transformers\_sparse\_document\_embedder

<a id="sentence_transformers_sparse_document_embedder.SentenceTransformersSparseDocumentEmbedder"></a>

## SentenceTransformersSparseDocumentEmbedder

Calculates document sparse embeddings using sparse embedding models from Sentence Transformers.

It stores the sparse embeddings in the `sparse_embedding` metadata field of each document.
You can also embed documents' metadata.
Use this component in indexing pipelines to embed input documents
and send them to DocumentWriter to write a into a Document Store.

### Usage example:

```python
from haystack import Document
from haystack.components.embedders import SentenceTransformersSparseDocumentEmbedder

doc = Document(content="I love pizza!")
doc_embedder = SentenceTransformersSparseDocumentEmbedder()
doc_embedder.warm_up()

result = doc_embedder.run([doc])
print(result['documents'][0].sparse_embedding)

# SparseEmbedding(indices=[999, 1045, ...], values=[0.918, 0.867, ...])
```

<a id="sentence_transformers_sparse_document_embedder.SentenceTransformersSparseDocumentEmbedder.__init__"></a>

#### SentenceTransformersSparseDocumentEmbedder.\_\_init\_\_

```python
def __init__(*,
             model: str = "prithivida/Splade_PP_en_v2",
             device: Optional[ComponentDevice] = None,
             token: Optional[Secret] = Secret.from_env_var(
                 ["HF_API_TOKEN", "HF_TOKEN"], strict=False),
             prefix: str = "",
             suffix: str = "",
             batch_size: int = 32,
             progress_bar: bool = True,
             meta_fields_to_embed: Optional[list[str]] = None,
             embedding_separator: str = "\n",
             trust_remote_code: bool = False,
             local_files_only: bool = False,
             model_kwargs: Optional[dict[str, Any]] = None,
             tokenizer_kwargs: Optional[dict[str, Any]] = None,
             config_kwargs: Optional[dict[str, Any]] = None,
             backend: Literal["torch", "onnx", "openvino"] = "torch")
```

Creates a SentenceTransformersSparseDocumentEmbedder component.

**Arguments**:

- `model`: The model to use for calculating sparse embeddings.
Pass a local path or ID of the model on Hugging Face.
- `device`: The device to use for loading the model.
Overrides the default device.
- `token`: The API token to download private models from Hugging Face.
- `prefix`: A string to add at the beginning of each document text.
- `suffix`: A string to add at the end of each document text.
- `batch_size`: Number of documents to embed at once.
- `progress_bar`: If `True`, shows a progress bar when embedding documents.
- `meta_fields_to_embed`: List of metadata fields to embed along with the document text.
- `embedding_separator`: Separator used to concatenate the metadata fields to the document text.
- `trust_remote_code`: If `False`, allows only Hugging Face verified model architectures.
If `True`, allows custom models and scripts.
- `local_files_only`: If `True`, does not attempt to download the model from Hugging Face Hub and only looks at local files.
- `model_kwargs`: Additional keyword arguments for `AutoModelForSequenceClassification.from_pretrained`
when loading the model. Refer to specific model documentation for available kwargs.
- `tokenizer_kwargs`: Additional keyword arguments for `AutoTokenizer.from_pretrained` when loading the tokenizer.
Refer to specific model documentation for available kwargs.
- `config_kwargs`: Additional keyword arguments for `AutoConfig.from_pretrained` when loading the model configuration.
- `backend`: The backend to use for the Sentence Transformers model. Choose from "torch", "onnx", or "openvino".
Refer to the [Sentence Transformers documentation](https://sbert.net/docs/sentence_transformer/usage/efficiency.html)
for more information on acceleration and quantization options.

<a id="sentence_transformers_sparse_document_embedder.SentenceTransformersSparseDocumentEmbedder.to_dict"></a>

#### SentenceTransformersSparseDocumentEmbedder.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="sentence_transformers_sparse_document_embedder.SentenceTransformersSparseDocumentEmbedder.from_dict"></a>

#### SentenceTransformersSparseDocumentEmbedder.from\_dict

```python
@classmethod
def from_dict(
        cls, data: dict[str,
                        Any]) -> "SentenceTransformersSparseDocumentEmbedder"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="sentence_transformers_sparse_document_embedder.SentenceTransformersSparseDocumentEmbedder.warm_up"></a>

#### SentenceTransformersSparseDocumentEmbedder.warm\_up

```python
def warm_up()
```

Initializes the component.

<a id="sentence_transformers_sparse_document_embedder.SentenceTransformersSparseDocumentEmbedder.run"></a>

#### SentenceTransformersSparseDocumentEmbedder.run

```python
@component.output_types(documents=list[Document])
def run(documents: list[Document])
```

Embed a list of documents.

**Arguments**:

- `documents`: Documents to embed.

**Returns**:

A dictionary with the following keys:
- `documents`: Documents with sparse embeddings under the `sparse_embedding` field.

<a id="sentence_transformers_sparse_text_embedder"></a>

# Module sentence\_transformers\_sparse\_text\_embedder

<a id="sentence_transformers_sparse_text_embedder.SentenceTransformersSparseTextEmbedder"></a>

## SentenceTransformersSparseTextEmbedder

Embeds strings using sparse embedding models from Sentence Transformers.

You can use it to embed user query and send it to a sparse embedding retriever.

Usage example:
```python
from haystack.components.embedders import SentenceTransformersSparseTextEmbedder

text_to_embed = "I love pizza!"

text_embedder = SentenceTransformersSparseTextEmbedder()
text_embedder.warm_up()

print(text_embedder.run(text_to_embed))

# {'sparse_embedding': SparseEmbedding(indices=[999, 1045, ...], values=[0.918, 0.867, ...])}
```

<a id="sentence_transformers_sparse_text_embedder.SentenceTransformersSparseTextEmbedder.__init__"></a>

#### SentenceTransformersSparseTextEmbedder.\_\_init\_\_

```python
def __init__(*,
             model: str = "prithivida/Splade_PP_en_v2",
             device: Optional[ComponentDevice] = None,
             token: Optional[Secret] = Secret.from_env_var(
                 ["HF_API_TOKEN", "HF_TOKEN"], strict=False),
             prefix: str = "",
             suffix: str = "",
             trust_remote_code: bool = False,
             local_files_only: bool = False,
             model_kwargs: Optional[dict[str, Any]] = None,
             tokenizer_kwargs: Optional[dict[str, Any]] = None,
             config_kwargs: Optional[dict[str, Any]] = None,
             encode_kwargs: Optional[dict[str, Any]] = None,
             backend: Literal["torch", "onnx", "openvino"] = "torch")
```

Create a SentenceTransformersSparseTextEmbedder component.

**Arguments**:

- `model`: The model to use for calculating sparse embeddings.
Specify the path to a local model or the ID of the model on Hugging Face.
- `device`: Overrides the default device used to load the model.
- `token`: An API token to use private models from Hugging Face.
- `prefix`: A string to add at the beginning of each text to be embedded.
- `suffix`: A string to add at the end of each text to embed.
- `trust_remote_code`: If `False`, permits only Hugging Face verified model architectures.
If `True`, permits custom models and scripts.
- `local_files_only`: If `True`, does not attempt to download the model from Hugging Face Hub and only looks at local files.
- `model_kwargs`: Additional keyword arguments for `AutoModelForSequenceClassification.from_pretrained`
when loading the model. Refer to specific model documentation for available kwargs.
- `tokenizer_kwargs`: Additional keyword arguments for `AutoTokenizer.from_pretrained` when loading the tokenizer.
Refer to specific model documentation for available kwargs.
- `config_kwargs`: Additional keyword arguments for `AutoConfig.from_pretrained` when loading the model configuration.
- `backend`: The backend to use for the Sentence Transformers model. Choose from "torch", "onnx", or "openvino".
Refer to the [Sentence Transformers documentation](https://sbert.net/docs/sentence_transformer/usage/efficiency.html)
for more information on acceleration and quantization options.

<a id="sentence_transformers_sparse_text_embedder.SentenceTransformersSparseTextEmbedder.to_dict"></a>

#### SentenceTransformersSparseTextEmbedder.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="sentence_transformers_sparse_text_embedder.SentenceTransformersSparseTextEmbedder.from_dict"></a>

#### SentenceTransformersSparseTextEmbedder.from\_dict

```python
@classmethod
def from_dict(
        cls, data: dict[str, Any]) -> "SentenceTransformersSparseTextEmbedder"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="sentence_transformers_sparse_text_embedder.SentenceTransformersSparseTextEmbedder.warm_up"></a>

#### SentenceTransformersSparseTextEmbedder.warm\_up

```python
def warm_up()
```

Initializes the component.

<a id="sentence_transformers_sparse_text_embedder.SentenceTransformersSparseTextEmbedder.run"></a>

#### SentenceTransformersSparseTextEmbedder.run

```python
@component.output_types(sparse_embedding=SparseEmbedding)
def run(text: str)
```

Embed a single string.

**Arguments**:

- `text`: Text to embed.

**Returns**:

A dictionary with the following keys:
- `sparse_embedding`: The sparse embedding of the input text.

<a id="image/sentence_transformers_doc_image_embedder"></a>

# Module image/sentence\_transformers\_doc\_image\_embedder

<a id="image/sentence_transformers_doc_image_embedder.SentenceTransformersDocumentImageEmbedder"></a>

## SentenceTransformersDocumentImageEmbedder

A component for computing Document embeddings based on images using Sentence Transformers models.

The embedding of each Document is stored in the `embedding` field of the Document.

### Usage example
```python
from haystack import Document
from haystack.components.embedders.image import SentenceTransformersDocumentImageEmbedder

embedder = SentenceTransformersDocumentImageEmbedder(model="sentence-transformers/clip-ViT-B-32")
embedder.warm_up()

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
#           embedding=vector of size 512),
#  ...]
```

<a id="image/sentence_transformers_doc_image_embedder.SentenceTransformersDocumentImageEmbedder.__init__"></a>

#### SentenceTransformersDocumentImageEmbedder.\_\_init\_\_

```python
def __init__(*,
             file_path_meta_field: str = "file_path",
             root_path: Optional[str] = None,
             model: str = "sentence-transformers/clip-ViT-B-32",
             device: Optional[ComponentDevice] = None,
             token: Optional[Secret] = Secret.from_env_var(
                 ["HF_API_TOKEN", "HF_TOKEN"], strict=False),
             batch_size: int = 32,
             progress_bar: bool = True,
             normalize_embeddings: bool = False,
             trust_remote_code: bool = False,
             local_files_only: bool = False,
             model_kwargs: Optional[dict[str, Any]] = None,
             tokenizer_kwargs: Optional[dict[str, Any]] = None,
             config_kwargs: Optional[dict[str, Any]] = None,
             precision: Literal["float32", "int8", "uint8", "binary",
                                "ubinary"] = "float32",
             encode_kwargs: Optional[dict[str, Any]] = None,
             backend: Literal["torch", "onnx", "openvino"] = "torch") -> None
```

Creates a SentenceTransformersDocumentEmbedder component.

**Arguments**:

- `file_path_meta_field`: The metadata field in the Document that contains the file path to the image or PDF.
- `root_path`: The root directory path where document files are located. If provided, file paths in
document metadata will be resolved relative to this path. If None, file paths are treated as absolute paths.
- `model`: The Sentence Transformers model to use for calculating embeddings. Pass a local path or ID of the model on
Hugging Face. To be used with this component, the model must be able to embed images and text into the same
vector space. Compatible models include:
- "sentence-transformers/clip-ViT-B-32"
- "sentence-transformers/clip-ViT-L-14"
- "sentence-transformers/clip-ViT-B-16"
- "sentence-transformers/clip-ViT-B-32-multilingual-v1"
- "jinaai/jina-embeddings-v4"
- "jinaai/jina-clip-v1"
- "jinaai/jina-clip-v2".
- `device`: The device to use for loading the model.
Overrides the default device.
- `token`: The API token to download private models from Hugging Face.
- `batch_size`: Number of documents to embed at once.
- `progress_bar`: If `True`, shows a progress bar when embedding documents.
- `normalize_embeddings`: If `True`, the embeddings are normalized using L2 normalization, so that each embedding has a norm of 1.
- `trust_remote_code`: If `False`, allows only Hugging Face verified model architectures.
If `True`, allows custom models and scripts.
- `local_files_only`: If `True`, does not attempt to download the model from Hugging Face Hub and only looks at local files.
- `model_kwargs`: Additional keyword arguments for `AutoModelForSequenceClassification.from_pretrained`
when loading the model. Refer to specific model documentation for available kwargs.
- `tokenizer_kwargs`: Additional keyword arguments for `AutoTokenizer.from_pretrained` when loading the tokenizer.
Refer to specific model documentation for available kwargs.
- `config_kwargs`: Additional keyword arguments for `AutoConfig.from_pretrained` when loading the model configuration.
- `precision`: The precision to use for the embeddings.
All non-float32 precisions are quantized embeddings.
Quantized embeddings are smaller and faster to compute, but may have a lower accuracy.
They are useful for reducing the size of the embeddings of a corpus for semantic search, among other tasks.
- `encode_kwargs`: Additional keyword arguments for `SentenceTransformer.encode` when embedding documents.
This parameter is provided for fine customization. Be careful not to clash with already set parameters and
avoid passing parameters that change the output type.
- `backend`: The backend to use for the Sentence Transformers model. Choose from "torch", "onnx", or "openvino".
Refer to the [Sentence Transformers documentation](https://sbert.net/docs/sentence_transformer/usage/efficiency.html)
for more information on acceleration and quantization options.

<a id="image/sentence_transformers_doc_image_embedder.SentenceTransformersDocumentImageEmbedder.to_dict"></a>

#### SentenceTransformersDocumentImageEmbedder.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="image/sentence_transformers_doc_image_embedder.SentenceTransformersDocumentImageEmbedder.from_dict"></a>

#### SentenceTransformersDocumentImageEmbedder.from\_dict

```python
@classmethod
def from_dict(
        cls, data: dict[str,
                        Any]) -> "SentenceTransformersDocumentImageEmbedder"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="image/sentence_transformers_doc_image_embedder.SentenceTransformersDocumentImageEmbedder.warm_up"></a>

#### SentenceTransformersDocumentImageEmbedder.warm\_up

```python
def warm_up() -> None
```

Initializes the component.

<a id="image/sentence_transformers_doc_image_embedder.SentenceTransformersDocumentImageEmbedder.run"></a>

#### SentenceTransformersDocumentImageEmbedder.run

```python
@component.output_types(documents=list[Document])
def run(documents: list[Document]) -> dict[str, list[Document]]
```

Embed a list of documents.

**Arguments**:

- `documents`: Documents to embed.

**Returns**:

A dictionary with the following keys:
- `documents`: Documents with embeddings.
