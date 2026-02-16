---
title: "Embedders"
id: embedders-api
description: "Transforms queries into vectors to look for similar or relevant Documents."
slug: "/embedders-api"
---


## `AzureOpenAIDocumentEmbedder`

Bases: <code>OpenAIDocumentEmbedder</code>

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

### `__init__`

```python
__init__(azure_endpoint: str | None = None, api_version: str | None = '2023-05-15', azure_deployment: str = 'text-embedding-ada-002', dimensions: int | None = None, api_key: Secret | None = Secret.from_env_var('AZURE_OPENAI_API_KEY', strict=False), azure_ad_token: Secret | None = Secret.from_env_var('AZURE_OPENAI_AD_TOKEN', strict=False), organization: str | None = None, prefix: str = '', suffix: str = '', batch_size: int = 32, progress_bar: bool = True, meta_fields_to_embed: list[str] | None = None, embedding_separator: str = '\n', timeout: float | None = None, max_retries: int | None = None, *, default_headers: dict[str, str] | None = None, azure_ad_token_provider: AzureADTokenProvider | None = None, http_client_kwargs: dict[str, Any] | None = None, raise_on_failure: bool = False)
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

### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

### `from_dict`

```python
from_dict(data: dict[str, Any]) -> AzureOpenAIDocumentEmbedder
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>AzureOpenAIDocumentEmbedder</code> – Deserialized component.

### `run`

```python
run(documents: list[Document])
```

Embeds a list of documents.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – A list of documents to embed.

**Returns:**

- – A dictionary with the following keys:
- `documents`: A list of documents with embeddings.
- `meta`: Information about the usage of the model.

### `run_async`

```python
run_async(documents: list[Document])
```

Embeds a list of documents asynchronously.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – A list of documents to embed.

**Returns:**

- – A dictionary with the following keys:
- `documents`: A list of documents with embeddings.
- `meta`: Information about the usage of the model.

## `AzureOpenAITextEmbedder`

Bases: <code>OpenAITextEmbedder</code>

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

### `__init__`

```python
__init__(azure_endpoint: str | None = None, api_version: str | None = '2023-05-15', azure_deployment: str = 'text-embedding-ada-002', dimensions: int | None = None, api_key: Secret | None = Secret.from_env_var('AZURE_OPENAI_API_KEY', strict=False), azure_ad_token: Secret | None = Secret.from_env_var('AZURE_OPENAI_AD_TOKEN', strict=False), organization: str | None = None, timeout: float | None = None, max_retries: int | None = None, prefix: str = '', suffix: str = '', *, default_headers: dict[str, str] | None = None, azure_ad_token_provider: AzureADTokenProvider | None = None, http_client_kwargs: dict[str, Any] | None = None)
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

### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

### `run`

```python
run(text: str)
```

Embeds a single string.

**Parameters:**

- **text** (<code>str</code>) – Text to embed.

**Returns:**

- – A dictionary with the following keys:
- `embedding`: The embedding of the input text.
- `meta`: Information about the usage of the model.

### `from_dict`

```python
from_dict(data: dict[str, Any]) -> AzureOpenAITextEmbedder
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>AzureOpenAITextEmbedder</code> – Deserialized component.

### `run_async`

```python
run_async(text: str)
```

Asynchronously embed a single string.

This is the asynchronous version of the `run` method. It has the same parameters and return values
but can be used with `await` in async code.

**Parameters:**

- **text** (<code>str</code>) – Text to embed.

**Returns:**

- – A dictionary with the following keys:
- `embedding`: The embedding of the input text.
- `meta`: Information about the usage of the model.

## `HuggingFaceAPIDocumentEmbedder`

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

### `__init__`

```python
__init__(api_type: HFEmbeddingAPIType | str, api_params: dict[str, str], token: Secret | None = Secret.from_env_var(['HF_API_TOKEN', 'HF_TOKEN'], strict=False), prefix: str = '', suffix: str = '', truncate: bool | None = True, normalize: bool | None = False, batch_size: int = 32, progress_bar: bool = True, meta_fields_to_embed: list[str] | None = None, embedding_separator: str = '\n')
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

### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

### `from_dict`

```python
from_dict(data: dict[str, Any]) -> HuggingFaceAPIDocumentEmbedder
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>HuggingFaceAPIDocumentEmbedder</code> – Deserialized component.

### `run`

```python
run(documents: list[Document])
```

Embeds a list of documents.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – Documents to embed.

**Returns:**

- – A dictionary with the following keys:
- `documents`: A list of documents with embeddings.

### `run_async`

```python
run_async(documents: list[Document])
```

Embeds a list of documents asynchronously.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – Documents to embed.

**Returns:**

- – A dictionary with the following keys:
- `documents`: A list of documents with embeddings.

## `HuggingFaceAPITextEmbedder`

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

### `__init__`

```python
__init__(api_type: HFEmbeddingAPIType | str, api_params: dict[str, str], token: Secret | None = Secret.from_env_var(['HF_API_TOKEN', 'HF_TOKEN'], strict=False), prefix: str = '', suffix: str = '', truncate: bool | None = True, normalize: bool | None = False)
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

### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

### `from_dict`

```python
from_dict(data: dict[str, Any]) -> HuggingFaceAPITextEmbedder
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>HuggingFaceAPITextEmbedder</code> – Deserialized component.

### `run`

```python
run(text: str)
```

Embeds a single string.

**Parameters:**

- **text** (<code>str</code>) – Text to embed.

**Returns:**

- – A dictionary with the following keys:
- `embedding`: The embedding of the input text.

### `run_async`

```python
run_async(text: str)
```

Embeds a single string asynchronously.

**Parameters:**

- **text** (<code>str</code>) – Text to embed.

**Returns:**

- – A dictionary with the following keys:
- `embedding`: The embedding of the input text.

## `SentenceTransformersDocumentImageEmbedder`

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

### `__init__`

```python
__init__(*, file_path_meta_field: str = 'file_path', root_path: str | None = None, model: str = 'sentence-transformers/clip-ViT-B-32', device: ComponentDevice | None = None, token: Secret | None = Secret.from_env_var(['HF_API_TOKEN', 'HF_TOKEN'], strict=False), batch_size: int = 32, progress_bar: bool = True, normalize_embeddings: bool = False, trust_remote_code: bool = False, local_files_only: bool = False, model_kwargs: dict[str, Any] | None = None, tokenizer_kwargs: dict[str, Any] | None = None, config_kwargs: dict[str, Any] | None = None, precision: Literal['float32', 'int8', 'uint8', 'binary', 'ubinary'] = 'float32', encode_kwargs: dict[str, Any] | None = None, backend: Literal['torch', 'onnx', 'openvino'] = 'torch') -> None
```

Creates a SentenceTransformersDocumentEmbedder component.

**Parameters:**

- **file_path_meta_field** (<code>str</code>) – The metadata field in the Document that contains the file path to the image or PDF.
- **root_path** (<code>str | None</code>) – The root directory path where document files are located. If provided, file paths in
  document metadata will be resolved relative to this path. If None, file paths are treated as absolute paths.
- **model** (<code>str</code>) – The Sentence Transformers model to use for calculating embeddings. Pass a local path or ID of the model on
  Hugging Face. To be used with this component, the model must be able to embed images and text into the same
  vector space. Compatible models include:
- "sentence-transformers/clip-ViT-B-32"
- "sentence-transformers/clip-ViT-L-14"
- "sentence-transformers/clip-ViT-B-16"
- "sentence-transformers/clip-ViT-B-32-multilingual-v1"
- "jinaai/jina-embeddings-v4"
- "jinaai/jina-clip-v1"
- "jinaai/jina-clip-v2".
- **device** (<code>ComponentDevice | None</code>) – The device to use for loading the model.
  Overrides the default device.
- **token** (<code>Secret | None</code>) – The API token to download private models from Hugging Face.
- **batch_size** (<code>int</code>) – Number of documents to embed at once.
- **progress_bar** (<code>bool</code>) – If `True`, shows a progress bar when embedding documents.
- **normalize_embeddings** (<code>bool</code>) – If `True`, the embeddings are normalized using L2 normalization, so that each embedding has a norm of 1.
- **trust_remote_code** (<code>bool</code>) – If `False`, allows only Hugging Face verified model architectures.
  If `True`, allows custom models and scripts.
- **local_files_only** (<code>bool</code>) – If `True`, does not attempt to download the model from Hugging Face Hub and only looks at local files.
- **model_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments for `AutoModelForSequenceClassification.from_pretrained`
  when loading the model. Refer to specific model documentation for available kwargs.
- **tokenizer_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments for `AutoTokenizer.from_pretrained` when loading the tokenizer.
  Refer to specific model documentation for available kwargs.
- **config_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments for `AutoConfig.from_pretrained` when loading the model configuration.
- **precision** (<code>Literal['float32', 'int8', 'uint8', 'binary', 'ubinary']</code>) – The precision to use for the embeddings.
  All non-float32 precisions are quantized embeddings.
  Quantized embeddings are smaller and faster to compute, but may have a lower accuracy.
  They are useful for reducing the size of the embeddings of a corpus for semantic search, among other tasks.
- **encode_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments for `SentenceTransformer.encode` when embedding documents.
  This parameter is provided for fine customization. Be careful not to clash with already set parameters and
  avoid passing parameters that change the output type.
- **backend** (<code>Literal['torch', 'onnx', 'openvino']</code>) – The backend to use for the Sentence Transformers model. Choose from "torch", "onnx", or "openvino".
  Refer to the [Sentence Transformers documentation](https://sbert.net/docs/sentence_transformer/usage/efficiency.html)
  for more information on acceleration and quantization options.

### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

### `from_dict`

```python
from_dict(data: dict[str, Any]) -> SentenceTransformersDocumentImageEmbedder
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>SentenceTransformersDocumentImageEmbedder</code> – Deserialized component.

### `warm_up`

```python
warm_up() -> None
```

Initializes the component.

### `run`

```python
run(documents: list[Document]) -> dict[str, list[Document]]
```

Embed a list of documents.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – Documents to embed.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: Documents with embeddings.

## `OpenAIDocumentEmbedder`

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

### `__init__`

```python
__init__(api_key: Secret = Secret.from_env_var('OPENAI_API_KEY'), model: str = 'text-embedding-ada-002', dimensions: int | None = None, api_base_url: str | None = None, organization: str | None = None, prefix: str = '', suffix: str = '', batch_size: int = 32, progress_bar: bool = True, meta_fields_to_embed: list[str] | None = None, embedding_separator: str = '\n', timeout: float | None = None, max_retries: int | None = None, http_client_kwargs: dict[str, Any] | None = None, *, raise_on_failure: bool = False)
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

### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

### `from_dict`

```python
from_dict(data: dict[str, Any]) -> OpenAIDocumentEmbedder
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>OpenAIDocumentEmbedder</code> – Deserialized component.

### `run`

```python
run(documents: list[Document])
```

Embeds a list of documents.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – A list of documents to embed.

**Returns:**

- – A dictionary with the following keys:
- `documents`: A list of documents with embeddings.
- `meta`: Information about the usage of the model.

### `run_async`

```python
run_async(documents: list[Document])
```

Embeds a list of documents asynchronously.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – A list of documents to embed.

**Returns:**

- – A dictionary with the following keys:
- `documents`: A list of documents with embeddings.
- `meta`: Information about the usage of the model.

## `OpenAITextEmbedder`

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

### `__init__`

```python
__init__(api_key: Secret = Secret.from_env_var('OPENAI_API_KEY'), model: str = 'text-embedding-ada-002', dimensions: int | None = None, api_base_url: str | None = None, organization: str | None = None, prefix: str = '', suffix: str = '', timeout: float | None = None, max_retries: int | None = None, http_client_kwargs: dict[str, Any] | None = None)
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

### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

### `from_dict`

```python
from_dict(data: dict[str, Any]) -> OpenAITextEmbedder
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>OpenAITextEmbedder</code> – Deserialized component.

### `run`

```python
run(text: str)
```

Embeds a single string.

**Parameters:**

- **text** (<code>str</code>) – Text to embed.

**Returns:**

- – A dictionary with the following keys:
- `embedding`: The embedding of the input text.
- `meta`: Information about the usage of the model.

### `run_async`

```python
run_async(text: str)
```

Asynchronously embed a single string.

This is the asynchronous version of the `run` method. It has the same parameters and return values
but can be used with `await` in async code.

**Parameters:**

- **text** (<code>str</code>) – Text to embed.

**Returns:**

- – A dictionary with the following keys:
- `embedding`: The embedding of the input text.
- `meta`: Information about the usage of the model.

## `SentenceTransformersDocumentEmbedder`

Calculates document embeddings using Sentence Transformers models.

It stores the embeddings in the `embedding` metadata field of each document.
You can also embed documents' metadata.
Use this component in indexing pipelines to embed input documents
and send them to DocumentWriter to write into a Document Store.

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

### `__init__`

```python
__init__(model: str = 'sentence-transformers/all-mpnet-base-v2', device: ComponentDevice | None = None, token: Secret | None = Secret.from_env_var(['HF_API_TOKEN', 'HF_TOKEN'], strict=False), prefix: str = '', suffix: str = '', batch_size: int = 32, progress_bar: bool = True, normalize_embeddings: bool = False, meta_fields_to_embed: list[str] | None = None, embedding_separator: str = '\n', trust_remote_code: bool = False, local_files_only: bool = False, truncate_dim: int | None = None, model_kwargs: dict[str, Any] | None = None, tokenizer_kwargs: dict[str, Any] | None = None, config_kwargs: dict[str, Any] | None = None, precision: Literal['float32', 'int8', 'uint8', 'binary', 'ubinary'] = 'float32', encode_kwargs: dict[str, Any] | None = None, backend: Literal['torch', 'onnx', 'openvino'] = 'torch', revision: str | None = None)
```

Creates a SentenceTransformersDocumentEmbedder component.

**Parameters:**

- **model** (<code>str</code>) – The model to use for calculating embeddings.
  Pass a local path or ID of the model on Hugging Face.
- **device** (<code>ComponentDevice | None</code>) – The device to use for loading the model.
  Overrides the default device.
- **token** (<code>Secret | None</code>) – The API token to download private models from Hugging Face.
- **prefix** (<code>str</code>) – A string to add at the beginning of each document text.
  Can be used to prepend the text with an instruction, as required by some embedding models,
  such as E5 and bge.
- **suffix** (<code>str</code>) – A string to add at the end of each document text.
- **batch_size** (<code>int</code>) – Number of documents to embed at once.
- **progress_bar** (<code>bool</code>) – If `True`, shows a progress bar when embedding documents.
- **normalize_embeddings** (<code>bool</code>) – If `True`, the embeddings are normalized using L2 normalization, so that each embedding has a norm of 1.
- **meta_fields_to_embed** (<code>list\[str\] | None</code>) – List of metadata fields to embed along with the document text.
- **embedding_separator** (<code>str</code>) – Separator used to concatenate the metadata fields to the document text.
- **trust_remote_code** (<code>bool</code>) – If `False`, allows only Hugging Face verified model architectures.
  If `True`, allows custom models and scripts.
- **local_files_only** (<code>bool</code>) – If `True`, does not attempt to download the model from Hugging Face Hub and only looks at local files.
- **truncate_dim** (<code>int | None</code>) – The dimension to truncate sentence embeddings to. `None` does no truncation.
  If the model wasn't trained with Matryoshka Representation Learning,
  truncating embeddings can significantly affect performance.
- **model_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments for `AutoModelForSequenceClassification.from_pretrained`
  when loading the model. Refer to specific model documentation for available kwargs.
- **tokenizer_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments for `AutoTokenizer.from_pretrained` when loading the tokenizer.
  Refer to specific model documentation for available kwargs.
- **config_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments for `AutoConfig.from_pretrained` when loading the model configuration.
- **precision** (<code>Literal['float32', 'int8', 'uint8', 'binary', 'ubinary']</code>) – The precision to use for the embeddings.
  All non-float32 precisions are quantized embeddings.
  Quantized embeddings are smaller and faster to compute, but may have a lower accuracy.
  They are useful for reducing the size of the embeddings of a corpus for semantic search, among other tasks.
- **encode_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments for `SentenceTransformer.encode` when embedding documents.
  This parameter is provided for fine customization. Be careful not to clash with already set parameters and
  avoid passing parameters that change the output type.
- **backend** (<code>Literal['torch', 'onnx', 'openvino']</code>) – The backend to use for the Sentence Transformers model. Choose from "torch", "onnx", or "openvino".
  Refer to the [Sentence Transformers documentation](https://sbert.net/docs/sentence_transformer/usage/efficiency.html)
  for more information on acceleration and quantization options.
- **revision** (<code>str | None</code>) – The specific model version to use. It can be a branch name, a tag name, or a commit id,
  for a stored model on Hugging Face.

### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

### `from_dict`

```python
from_dict(data: dict[str, Any]) -> SentenceTransformersDocumentEmbedder
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>SentenceTransformersDocumentEmbedder</code> – Deserialized component.

### `warm_up`

```python
warm_up()
```

Initializes the component.

### `run`

```python
run(documents: list[Document])
```

Embed a list of documents.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – Documents to embed.

**Returns:**

- – A dictionary with the following keys:
- `documents`: Documents with embeddings.

## `SentenceTransformersSparseDocumentEmbedder`

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

### `__init__`

```python
__init__(*, model: str = 'prithivida/Splade_PP_en_v2', device: ComponentDevice | None = None, token: Secret | None = Secret.from_env_var(['HF_API_TOKEN', 'HF_TOKEN'], strict=False), prefix: str = '', suffix: str = '', batch_size: int = 32, progress_bar: bool = True, meta_fields_to_embed: list[str] | None = None, embedding_separator: str = '\n', trust_remote_code: bool = False, local_files_only: bool = False, model_kwargs: dict[str, Any] | None = None, tokenizer_kwargs: dict[str, Any] | None = None, config_kwargs: dict[str, Any] | None = None, backend: Literal['torch', 'onnx', 'openvino'] = 'torch', revision: str | None = None)
```

Creates a SentenceTransformersSparseDocumentEmbedder component.

**Parameters:**

- **model** (<code>str</code>) – The model to use for calculating sparse embeddings.
  Pass a local path or ID of the model on Hugging Face.
- **device** (<code>ComponentDevice | None</code>) – The device to use for loading the model.
  Overrides the default device.
- **token** (<code>Secret | None</code>) – The API token to download private models from Hugging Face.
- **prefix** (<code>str</code>) – A string to add at the beginning of each document text.
- **suffix** (<code>str</code>) – A string to add at the end of each document text.
- **batch_size** (<code>int</code>) – Number of documents to embed at once.
- **progress_bar** (<code>bool</code>) – If `True`, shows a progress bar when embedding documents.
- **meta_fields_to_embed** (<code>list\[str\] | None</code>) – List of metadata fields to embed along with the document text.
- **embedding_separator** (<code>str</code>) – Separator used to concatenate the metadata fields to the document text.
- **trust_remote_code** (<code>bool</code>) – If `False`, allows only Hugging Face verified model architectures.
  If `True`, allows custom models and scripts.
- **local_files_only** (<code>bool</code>) – If `True`, does not attempt to download the model from Hugging Face Hub and only looks at local files.
- **model_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments for `AutoModelForSequenceClassification.from_pretrained`
  when loading the model. Refer to specific model documentation for available kwargs.
- **tokenizer_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments for `AutoTokenizer.from_pretrained` when loading the tokenizer.
  Refer to specific model documentation for available kwargs.
- **config_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments for `AutoConfig.from_pretrained` when loading the model configuration.
- **backend** (<code>Literal['torch', 'onnx', 'openvino']</code>) – The backend to use for the Sentence Transformers model. Choose from "torch", "onnx", or "openvino".
  Refer to the [Sentence Transformers documentation](https://sbert.net/docs/sentence_transformer/usage/efficiency.html)
  for more information on acceleration and quantization options.
- **revision** (<code>str | None</code>) – The specific model version to use. It can be a branch name, a tag name, or a commit id,
  for a stored model on Hugging Face.

### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

### `from_dict`

```python
from_dict(data: dict[str, Any]) -> SentenceTransformersSparseDocumentEmbedder
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>SentenceTransformersSparseDocumentEmbedder</code> – Deserialized component.

### `warm_up`

```python
warm_up()
```

Initializes the component.

### `run`

```python
run(documents: list[Document])
```

Embed a list of documents.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – Documents to embed.

**Returns:**

- – A dictionary with the following keys:
- `documents`: Documents with sparse embeddings under the `sparse_embedding` field.

## `SentenceTransformersSparseTextEmbedder`

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

### `__init__`

```python
__init__(*, model: str = 'prithivida/Splade_PP_en_v2', device: ComponentDevice | None = None, token: Secret | None = Secret.from_env_var(['HF_API_TOKEN', 'HF_TOKEN'], strict=False), prefix: str = '', suffix: str = '', trust_remote_code: bool = False, local_files_only: bool = False, model_kwargs: dict[str, Any] | None = None, tokenizer_kwargs: dict[str, Any] | None = None, config_kwargs: dict[str, Any] | None = None, encode_kwargs: dict[str, Any] | None = None, backend: Literal['torch', 'onnx', 'openvino'] = 'torch', revision: str | None = None)
```

Create a SentenceTransformersSparseTextEmbedder component.

**Parameters:**

- **model** (<code>str</code>) – The model to use for calculating sparse embeddings.
  Specify the path to a local model or the ID of the model on Hugging Face.
- **device** (<code>ComponentDevice | None</code>) – Overrides the default device used to load the model.
- **token** (<code>Secret | None</code>) – An API token to use private models from Hugging Face.
- **prefix** (<code>str</code>) – A string to add at the beginning of each text to be embedded.
- **suffix** (<code>str</code>) – A string to add at the end of each text to embed.
- **trust_remote_code** (<code>bool</code>) – If `False`, permits only Hugging Face verified model architectures.
  If `True`, permits custom models and scripts.
- **local_files_only** (<code>bool</code>) – If `True`, does not attempt to download the model from Hugging Face Hub and only looks at local files.
- **model_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments for `AutoModelForSequenceClassification.from_pretrained`
  when loading the model. Refer to specific model documentation for available kwargs.
- **tokenizer_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments for `AutoTokenizer.from_pretrained` when loading the tokenizer.
  Refer to specific model documentation for available kwargs.
- **config_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments for `AutoConfig.from_pretrained` when loading the model configuration.
- **backend** (<code>Literal['torch', 'onnx', 'openvino']</code>) – The backend to use for the Sentence Transformers model. Choose from "torch", "onnx", or "openvino".
  Refer to the [Sentence Transformers documentation](https://sbert.net/docs/sentence_transformer/usage/efficiency.html)
  for more information on acceleration and quantization options.
- **revision** (<code>str | None</code>) – The specific model version to use. It can be a branch name, a tag name, or a commit id,
  for a stored model on Hugging Face.

### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

### `from_dict`

```python
from_dict(data: dict[str, Any]) -> SentenceTransformersSparseTextEmbedder
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>SentenceTransformersSparseTextEmbedder</code> – Deserialized component.

### `warm_up`

```python
warm_up()
```

Initializes the component.

### `run`

```python
run(text: str)
```

Embed a single string.

**Parameters:**

- **text** (<code>str</code>) – Text to embed.

**Returns:**

- – A dictionary with the following keys:
- `sparse_embedding`: The sparse embedding of the input text.

## `SentenceTransformersTextEmbedder`

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

### `__init__`

```python
__init__(model: str = 'sentence-transformers/all-mpnet-base-v2', device: ComponentDevice | None = None, token: Secret | None = Secret.from_env_var(['HF_API_TOKEN', 'HF_TOKEN'], strict=False), prefix: str = '', suffix: str = '', batch_size: int = 32, progress_bar: bool = True, normalize_embeddings: bool = False, trust_remote_code: bool = False, local_files_only: bool = False, truncate_dim: int | None = None, model_kwargs: dict[str, Any] | None = None, tokenizer_kwargs: dict[str, Any] | None = None, config_kwargs: dict[str, Any] | None = None, precision: Literal['float32', 'int8', 'uint8', 'binary', 'ubinary'] = 'float32', encode_kwargs: dict[str, Any] | None = None, backend: Literal['torch', 'onnx', 'openvino'] = 'torch', revision: str | None = None)
```

Create a SentenceTransformersTextEmbedder component.

**Parameters:**

- **model** (<code>str</code>) – The model to use for calculating embeddings.
  Specify the path to a local model or the ID of the model on Hugging Face.
- **device** (<code>ComponentDevice | None</code>) – Overrides the default device used to load the model.
- **token** (<code>Secret | None</code>) – An API token to use private models from Hugging Face.
- **prefix** (<code>str</code>) – A string to add at the beginning of each text to be embedded.
  You can use it to prepend the text with an instruction, as required by some embedding models,
  such as E5 and bge.
- **suffix** (<code>str</code>) – A string to add at the end of each text to embed.
- **batch_size** (<code>int</code>) – Number of texts to embed at once.
- **progress_bar** (<code>bool</code>) – If `True`, shows a progress bar for calculating embeddings.
  If `False`, disables the progress bar.
- **normalize_embeddings** (<code>bool</code>) – If `True`, the embeddings are normalized using L2 normalization, so that the embeddings have a norm of 1.
- **trust_remote_code** (<code>bool</code>) – If `False`, permits only Hugging Face verified model architectures.
  If `True`, permits custom models and scripts.
- **local_files_only** (<code>bool</code>) – If `True`, does not attempt to download the model from Hugging Face Hub and only looks at local files.
- **truncate_dim** (<code>int | None</code>) – The dimension to truncate sentence embeddings to. `None` does no truncation.
  If the model has not been trained with Matryoshka Representation Learning,
  truncation of embeddings can significantly affect performance.
- **model_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments for `AutoModelForSequenceClassification.from_pretrained`
  when loading the model. Refer to specific model documentation for available kwargs.
- **tokenizer_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments for `AutoTokenizer.from_pretrained` when loading the tokenizer.
  Refer to specific model documentation for available kwargs.
- **config_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments for `AutoConfig.from_pretrained` when loading the model configuration.
- **precision** (<code>Literal['float32', 'int8', 'uint8', 'binary', 'ubinary']</code>) – The precision to use for the embeddings.
  All non-float32 precisions are quantized embeddings.
  Quantized embeddings are smaller in size and faster to compute, but may have a lower accuracy.
  They are useful for reducing the size of the embeddings of a corpus for semantic search, among other tasks.
- **encode_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments for `SentenceTransformer.encode` when embedding texts.
  This parameter is provided for fine customization. Be careful not to clash with already set parameters and
  avoid passing parameters that change the output type.
- **backend** (<code>Literal['torch', 'onnx', 'openvino']</code>) – The backend to use for the Sentence Transformers model. Choose from "torch", "onnx", or "openvino".
  Refer to the [Sentence Transformers documentation](https://sbert.net/docs/sentence_transformer/usage/efficiency.html)
  for more information on acceleration and quantization options.
- **revision** (<code>str | None</code>) – The specific model version to use. It can be a branch name, a tag name, or a commit id,
  for a stored model on Hugging Face.

### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

### `from_dict`

```python
from_dict(data: dict[str, Any]) -> SentenceTransformersTextEmbedder
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>SentenceTransformersTextEmbedder</code> – Deserialized component.

### `warm_up`

```python
warm_up()
```

Initializes the component.

### `run`

```python
run(text: str)
```

Embed a single string.

**Parameters:**

- **text** (<code>str</code>) – Text to embed.

**Returns:**

- – A dictionary with the following keys:
- `embedding`: The embedding of the input text.
