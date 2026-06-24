---
title: "Sentence Transformers"
id: integrations-sentence-transformers
description: "Sentence Transformers integration for Haystack"
slug: "/integrations-sentence-transformers"
---


## haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_doc_image_embedder

### SentenceTransformersDocumentImageEmbedder

A component for computing Document embeddings based on images using Sentence Transformers models.

The embedding of each Document is stored in the `embedding` field of the Document.

### Usage example

```python
from haystack import Document
from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersDocumentImageEmbedder,
)

embedder = SentenceTransformersDocumentImageEmbedder(model="sentence-transformers/clip-ViT-B-32")

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

#### __init__

```python
__init__(
    *,
    file_path_meta_field: str = "file_path",
    root_path: str | None = None,
    model: str = "sentence-transformers/clip-ViT-B-32",
    device: ComponentDevice | None = None,
    token: Secret | None = Secret.from_env_var(
        ["HF_API_TOKEN", "HF_TOKEN"], strict=False
    ),
    batch_size: int = 32,
    progress_bar: bool = True,
    normalize_embeddings: bool = False,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    model_kwargs: dict[str, Any] | None = None,
    tokenizer_kwargs: dict[str, Any] | None = None,
    config_kwargs: dict[str, Any] | None = None,
    precision: Literal[
        "float32", "int8", "uint8", "binary", "ubinary"
    ] = "float32",
    encode_kwargs: dict[str, Any] | None = None,
    backend: Literal["torch", "onnx", "openvino"] = "torch"
) -> None
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

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> SentenceTransformersDocumentImageEmbedder
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>SentenceTransformersDocumentImageEmbedder</code> – Deserialized component.

#### warm_up

```python
warm_up() -> None
```

Initializes the component.

#### run

```python
run(documents: list[Document]) -> dict[str, list[Document]]
```

Embed a list of documents.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – Documents to embed.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: Documents with embeddings.

## haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_document_embedder

### SentenceTransformersDocumentEmbedder

Calculates document embeddings using Sentence Transformers models.

It stores the embeddings in the `embedding` metadata field of each document.
You can also embed documents' metadata.
Use this component in indexing pipelines to embed input documents
and send them to DocumentWriter to write into a Document Store.

### Usage example:

```python
from haystack import Document
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersDocumentEmbedder
doc = Document(content="I love pizza!")
doc_embedder = SentenceTransformersDocumentEmbedder()

result = doc_embedder.run([doc])
print(result['documents'][0].embedding)

# [-0.07804739475250244, 0.1498992145061493, ...]
```

#### __init__

```python
__init__(
    *,
    model: str = "sentence-transformers/all-mpnet-base-v2",
    device: ComponentDevice | None = None,
    token: Secret | None = Secret.from_env_var(
        ["HF_API_TOKEN", "HF_TOKEN"], strict=False
    ),
    prefix: str = "",
    suffix: str = "",
    batch_size: int = 32,
    progress_bar: bool = True,
    normalize_embeddings: bool = False,
    meta_fields_to_embed: list[str] | None = None,
    embedding_separator: str = "\n",
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    truncate_dim: int | None = None,
    model_kwargs: dict[str, Any] | None = None,
    tokenizer_kwargs: dict[str, Any] | None = None,
    config_kwargs: dict[str, Any] | None = None,
    precision: Literal[
        "float32", "int8", "uint8", "binary", "ubinary"
    ] = "float32",
    encode_kwargs: dict[str, Any] | None = None,
    backend: Literal["torch", "onnx", "openvino"] = "torch",
    revision: str | None = None
) -> None
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

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> SentenceTransformersDocumentEmbedder
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>SentenceTransformersDocumentEmbedder</code> – Deserialized component.

#### warm_up

```python
warm_up() -> None
```

Initializes the component.

#### run

```python
run(documents: list[Document]) -> dict[str, list[Document]]
```

Embed a list of documents.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – Documents to embed.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: Documents with embeddings.

## haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_sparse_document_embedder

### SentenceTransformersSparseDocumentEmbedder

Calculates document sparse embeddings using sparse embedding models from Sentence Transformers.

It stores the sparse embeddings in the `sparse_embedding` metadata field of each document.
You can also embed documents' metadata.
Use this component in indexing pipelines to embed input documents
and send them to DocumentWriter to write a into a Document Store.

### Usage example:

```python
from haystack import Document
from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersSparseDocumentEmbedder,
)

doc = Document(content="I love pizza!")
doc_embedder = SentenceTransformersSparseDocumentEmbedder()

result = doc_embedder.run([doc])
print(result['documents'][0].sparse_embedding)

# SparseEmbedding(indices=[999, 1045, ...], values=[0.918, 0.867, ...])
```

#### __init__

```python
__init__(
    *,
    model: str = "prithivida/Splade_PP_en_v2",
    device: ComponentDevice | None = None,
    token: Secret | None = Secret.from_env_var(
        ["HF_API_TOKEN", "HF_TOKEN"], strict=False
    ),
    prefix: str = "",
    suffix: str = "",
    batch_size: int = 32,
    progress_bar: bool = True,
    meta_fields_to_embed: list[str] | None = None,
    embedding_separator: str = "\n",
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    model_kwargs: dict[str, Any] | None = None,
    tokenizer_kwargs: dict[str, Any] | None = None,
    config_kwargs: dict[str, Any] | None = None,
    backend: Literal["torch", "onnx", "openvino"] = "torch",
    revision: str | None = None
) -> None
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

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> SentenceTransformersSparseDocumentEmbedder
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>SentenceTransformersSparseDocumentEmbedder</code> – Deserialized component.

#### warm_up

```python
warm_up() -> None
```

Initializes the component.

#### run

```python
run(documents: list[Document]) -> dict[str, list[Document]]
```

Embed a list of documents.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – Documents to embed.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: Documents with sparse embeddings under the `sparse_embedding` field.

## haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_sparse_text_embedder

### SentenceTransformersSparseTextEmbedder

Embeds strings using sparse embedding models from Sentence Transformers.

You can use it to embed user query and send it to a sparse embedding retriever.

Usage example:

```python
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersSparseTextEmbedder

text_to_embed = "I love pizza!"

text_embedder = SentenceTransformersSparseTextEmbedder()

print(text_embedder.run(text_to_embed))

# {'sparse_embedding': SparseEmbedding(indices=[999, 1045, ...], values=[0.918, 0.867, ...])}
```

#### __init__

```python
__init__(
    *,
    model: str = "prithivida/Splade_PP_en_v2",
    device: ComponentDevice | None = None,
    token: Secret | None = Secret.from_env_var(
        ["HF_API_TOKEN", "HF_TOKEN"], strict=False
    ),
    prefix: str = "",
    suffix: str = "",
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    model_kwargs: dict[str, Any] | None = None,
    tokenizer_kwargs: dict[str, Any] | None = None,
    config_kwargs: dict[str, Any] | None = None,
    backend: Literal["torch", "onnx", "openvino"] = "torch",
    revision: str | None = None
) -> None
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

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> SentenceTransformersSparseTextEmbedder
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>SentenceTransformersSparseTextEmbedder</code> – Deserialized component.

#### warm_up

```python
warm_up() -> None
```

Initializes the component.

#### run

```python
run(text: str) -> dict[str, Any]
```

Embed a single string.

**Parameters:**

- **text** (<code>str</code>) – Text to embed.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the following keys:
- `sparse_embedding`: The sparse embedding of the input text.

## haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_text_embedder

### SentenceTransformersTextEmbedder

Embeds strings using Sentence Transformers models.

You can use it to embed user query and send it to an embedding retriever.

Usage example:

```python
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersTextEmbedder

text_to_embed = "I love pizza!"

text_embedder = SentenceTransformersTextEmbedder()

print(text_embedder.run(text_to_embed))

# {'embedding': [-0.07804739475250244, 0.1498992145061493,, ...]}
```

#### __init__

```python
__init__(
    *,
    model: str = "sentence-transformers/all-mpnet-base-v2",
    device: ComponentDevice | None = None,
    token: Secret | None = Secret.from_env_var(
        ["HF_API_TOKEN", "HF_TOKEN"], strict=False
    ),
    prefix: str = "",
    suffix: str = "",
    batch_size: int = 32,
    progress_bar: bool = True,
    normalize_embeddings: bool = False,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    truncate_dim: int | None = None,
    model_kwargs: dict[str, Any] | None = None,
    tokenizer_kwargs: dict[str, Any] | None = None,
    config_kwargs: dict[str, Any] | None = None,
    precision: Literal[
        "float32", "int8", "uint8", "binary", "ubinary"
    ] = "float32",
    encode_kwargs: dict[str, Any] | None = None,
    backend: Literal["torch", "onnx", "openvino"] = "torch",
    revision: str | None = None
) -> None
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

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> SentenceTransformersTextEmbedder
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>SentenceTransformersTextEmbedder</code> – Deserialized component.

#### warm_up

```python
warm_up() -> None
```

Initializes the component.

#### run

```python
run(text: str) -> dict[str, Any]
```

Embed a single string.

**Parameters:**

- **text** (<code>str</code>) – Text to embed.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the following keys:
- `embedding`: The embedding of the input text.

## haystack_integrations.components.rankers.sentence_transformers.sentence_transformers_diversity

### DiversityRankingStrategy

Bases: <code>Enum</code>

The strategy to use for diversity ranking.

#### from_str

```python
from_str(string: str) -> DiversityRankingStrategy
```

Convert a string to a Strategy enum.

### DiversityRankingSimilarity

Bases: <code>Enum</code>

The similarity metric to use for comparing embeddings.

#### from_str

```python
from_str(string: str) -> DiversityRankingSimilarity
```

Convert a string to a Similarity enum.

### SentenceTransformersDiversityRanker

A Diversity Ranker based on Sentence Transformers.

Applies a document ranking algorithm based on one of the two strategies:

1. Greedy Diversity Order:

   Implements a document ranking algorithm that orders documents in a way that maximizes the overall diversity
   of the documents based on their similarity to the query.

   It uses a pre-trained Sentence Transformers model to embed the query and
   the documents.

1. Maximum Margin Relevance:

   Implements a document ranking algorithm that orders documents based on their Maximum Margin Relevance (MMR)
   scores.

   MMR scores are calculated for each document based on their relevance to the query and diversity from already
   selected documents. The algorithm iteratively selects documents based on their MMR scores, balancing between
   relevance to the query and diversity from already selected documents. The 'lambda_threshold' controls the
   trade-off between relevance and diversity.

Before ranking, documents are deduplicated by their id, retaining only the document with the highest score
if a score is present.

### Usage example

```python
from haystack import Document
from haystack_integrations.components.rankers.sentence_transformers import SentenceTransformersDiversityRanker

ranker = SentenceTransformersDiversityRanker(
    model="sentence-transformers/all-MiniLM-L6-v2", similarity="cosine", strategy="greedy_diversity_order"
)

docs = [Document(content="Paris"), Document(content="Berlin")]
query = "What is the capital of germany?"
output = ranker.run(query=query, documents=docs)
docs = output["documents"]
```

#### __init__

```python
__init__(
    *,
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 10,
    device: ComponentDevice | None = None,
    token: Secret | None = Secret.from_env_var(
        ["HF_API_TOKEN", "HF_TOKEN"], strict=False
    ),
    similarity: str | DiversityRankingSimilarity = "cosine",
    query_prefix: str = "",
    query_suffix: str = "",
    document_prefix: str = "",
    document_suffix: str = "",
    meta_fields_to_embed: list[str] | None = None,
    embedding_separator: str = "\n",
    strategy: str | DiversityRankingStrategy = "greedy_diversity_order",
    lambda_threshold: float = 0.5,
    model_kwargs: dict[str, Any] | None = None,
    tokenizer_kwargs: dict[str, Any] | None = None,
    config_kwargs: dict[str, Any] | None = None,
    backend: Literal["torch", "onnx", "openvino"] = "torch"
) -> None
```

Initialize a SentenceTransformersDiversityRanker.

**Parameters:**

- **model** (<code>str</code>) – Local path or name of the model in Hugging Face's model hub,
  such as `'sentence-transformers/all-MiniLM-L6-v2'`.
- **top_k** (<code>int</code>) – The maximum number of Documents to return per query.
- **device** (<code>ComponentDevice | None</code>) – The device on which the model is loaded. If `None`, the default device is automatically
  selected.
- **token** (<code>Secret | None</code>) – The API token used to download private models from Hugging Face.
- **similarity** (<code>str | DiversityRankingSimilarity</code>) – Similarity metric for comparing embeddings. Can be set to "dot_product" (default) or
  "cosine".
- **query_prefix** (<code>str</code>) – A string to add to the beginning of the query text before ranking.
  Can be used to prepend the text with an instruction, as required by some embedding models,
  such as E5 and BGE.
- **query_suffix** (<code>str</code>) – A string to add to the end of the query text before ranking.
- **document_prefix** (<code>str</code>) – A string to add to the beginning of each Document text before ranking.
  Can be used to prepend the text with an instruction, as required by some embedding models,
  such as E5 and BGE.
- **document_suffix** (<code>str</code>) – A string to add to the end of each Document text before ranking.
- **meta_fields_to_embed** (<code>list\[str\] | None</code>) – List of meta fields that should be embedded along with the Document content.
- **embedding_separator** (<code>str</code>) – Separator used to concatenate the meta fields to the Document content.
- **strategy** (<code>str | DiversityRankingStrategy</code>) – The strategy to use for diversity ranking. Can be either "greedy_diversity_order" or
  "maximum_margin_relevance".
- **lambda_threshold** (<code>float</code>) – The trade-off parameter between relevance and diversity. Only used when strategy is
  "maximum_margin_relevance".
- **model_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments for `AutoModelForSequenceClassification.from_pretrained`
  when loading the model. Refer to specific model documentation for available kwargs.
- **tokenizer_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments for `AutoTokenizer.from_pretrained` when loading the tokenizer.
  Refer to specific model documentation for available kwargs.
- **config_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments for `AutoConfig.from_pretrained` when loading the model configuration.
- **backend** (<code>Literal['torch', 'onnx', 'openvino']</code>) – The backend to use for the Sentence Transformers model. Choose from "torch", "onnx", or "openvino".
  Refer to the [Sentence Transformers documentation](https://sbert.net/docs/sentence_transformer/usage/efficiency.html)
  for more information on acceleration and quantization options.

#### warm_up

```python
warm_up() -> None
```

Initializes the component.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> SentenceTransformersDiversityRanker
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to deserialize from.

**Returns:**

- <code>SentenceTransformersDiversityRanker</code> – The deserialized component.

#### run

```python
run(
    query: str,
    documents: list[Document],
    top_k: int | None = None,
    lambda_threshold: float | None = None,
) -> dict[str, list[Document]]
```

Rank the documents based on their diversity.

**Parameters:**

- **query** (<code>str</code>) – The search query.
- **documents** (<code>list\[Document\]</code>) – List of Document objects to be ranker.
- **top_k** (<code>int | None</code>) – Optional. An integer to override the top_k set during initialization.
- **lambda_threshold** (<code>float | None</code>) – Override the trade-off parameter between relevance and diversity. Only used when
  strategy is "maximum_margin_relevance".

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following key:
- `documents`: List of Document objects that have been selected based on the diversity ranking.

**Raises:**

- <code>ValueError</code> – If the top_k value is less than or equal to 0.

## haystack_integrations.components.rankers.sentence_transformers.sentence_transformers_similarity

### SentenceTransformersSimilarityRanker

Ranks documents based on their semantic similarity to the query.

It uses a pre-trained cross-encoder model from Hugging Face to embed the query and the documents.

### Usage example

```python
from haystack import Document
from haystack_integrations.components.rankers.sentence_transformers import SentenceTransformersSimilarityRanker

ranker = SentenceTransformersSimilarityRanker()
docs = [Document(content="Paris"), Document(content="Berlin")]
query = "City in Germany"
result = ranker.run(query=query, documents=docs)
docs = result["documents"]
print(docs[0].content)
```

#### __init__

```python
__init__(
    *,
    model: str | Path = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device: ComponentDevice | None = None,
    token: Secret | None = Secret.from_env_var(
        ["HF_API_TOKEN", "HF_TOKEN"], strict=False
    ),
    top_k: int = 10,
    query_prefix: str = "",
    query_suffix: str = "",
    document_prefix: str = "",
    document_suffix: str = "",
    meta_fields_to_embed: list[str] | None = None,
    embedding_separator: str = "\n",
    scale_score: bool = True,
    score_threshold: float | None = None,
    trust_remote_code: bool = False,
    model_kwargs: dict[str, Any] | None = None,
    tokenizer_kwargs: dict[str, Any] | None = None,
    config_kwargs: dict[str, Any] | None = None,
    backend: Literal["torch", "onnx", "openvino"] = "torch",
    batch_size: int = 16
) -> None
```

Creates an instance of SentenceTransformersSimilarityRanker.

**Parameters:**

- **model** (<code>str | Path</code>) – The ranking model. Pass a local path or the Hugging Face model name of a cross-encoder model.
- **device** (<code>ComponentDevice | None</code>) – The device on which the model is loaded. If `None`, the default device is automatically selected.
- **token** (<code>Secret | None</code>) – The API token to download private models from Hugging Face.
- **top_k** (<code>int</code>) – The maximum number of documents to return per query.
- **query_prefix** (<code>str</code>) – A string to add at the beginning of the query text before ranking.
  Use it to prepend the text with an instruction, as required by reranking models like `bge`.
- **query_suffix** (<code>str</code>) – A string to add at the end of the query text before ranking.
  Use it to append the text with an instruction, as required by reranking models like `qwen`.
- **document_prefix** (<code>str</code>) – A string to add at the beginning of each document before ranking. You can use it to prepend the document
  with an instruction, as required by embedding models like `bge`.
- **document_suffix** (<code>str</code>) – A string to add at the end of each document before ranking. You can use it to append the document
  with an instruction, as required by embedding models like `qwen`.
- **meta_fields_to_embed** (<code>list\[str\] | None</code>) – List of metadata fields to embed with the document.
- **embedding_separator** (<code>str</code>) – Separator to concatenate metadata fields to the document.
- **scale_score** (<code>bool</code>) – If `True`, scales the raw logit predictions using a Sigmoid activation function.
  If `False`, disables scaling of the raw logit predictions.
- **score_threshold** (<code>float | None</code>) – Use it to return documents with a score above this threshold only.
- **trust_remote_code** (<code>bool</code>) – If `False`, allows only Hugging Face verified model architectures.
  If `True`, allows custom models and scripts.
- **model_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments for `AutoModelForSequenceClassification.from_pretrained`
  when loading the model. Refer to specific model documentation for available kwargs.
- **tokenizer_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments for `AutoTokenizer.from_pretrained` when loading the tokenizer.
  Refer to specific model documentation for available kwargs.
- **config_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments for `AutoConfig.from_pretrained` when loading the model configuration.
- **backend** (<code>Literal['torch', 'onnx', 'openvino']</code>) – The backend to use for the Sentence Transformers model. Choose from "torch", "onnx", or "openvino".
  Refer to the [Sentence Transformers documentation](https://sbert.net/docs/sentence_transformer/usage/efficiency.html)
  for more information on acceleration and quantization options.
- **batch_size** (<code>int</code>) – The batch size to use for inference. The higher the batch size, the more memory is required.
  If you run into memory issues, reduce the batch size.

**Raises:**

- <code>ValueError</code> – If `top_k` is not > 0.

#### warm_up

```python
warm_up() -> None
```

Initializes the component.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> SentenceTransformersSimilarityRanker
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>SentenceTransformersSimilarityRanker</code> – Deserialized component.

#### run

```python
run(
    *,
    query: str,
    documents: list[Document],
    top_k: int | None = None,
    scale_score: bool | None = None,
    score_threshold: float | None = None
) -> dict[str, list[Document]]
```

Returns a list of documents ranked by their similarity to the given query.

Before ranking, documents are deduplicated by their id, retaining only the document with the highest score
if a score is present.

**Parameters:**

- **query** (<code>str</code>) – The input query to compare the documents to.
- **documents** (<code>list\[Document\]</code>) – A list of documents to be ranked.
- **top_k** (<code>int | None</code>) – The maximum number of documents to return.
- **scale_score** (<code>bool | None</code>) – If `True`, scales the raw logit predictions using a Sigmoid activation function.
  If `False`, disables scaling of the raw logit predictions.
  If set, overrides the value set at initialization.
- **score_threshold** (<code>float | None</code>) – Use it to return documents only with a score above this threshold.
  If set, overrides the value set at initialization.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: A list of documents closest to the query, sorted from most similar to least similar.

**Raises:**

- <code>ValueError</code> – If `top_k` is not > 0.
