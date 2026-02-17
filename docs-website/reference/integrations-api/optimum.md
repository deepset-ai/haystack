---
title: "Optimum"
id: integrations-optimum
description: "Optimum integration for Haystack"
slug: "/integrations-optimum"
---


## `haystack_integrations.components.embedders.optimum.optimization`

### `OptimumEmbedderOptimizationMode`

Bases: <code>Enum</code>

[ONXX Optimization modes](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/optimization)
support by the Optimum Embedders.

#### `from_str`

```python
from_str(string: str) -> OptimumEmbedderOptimizationMode
```

Create an optimization mode from a string.

**Parameters:**

- **string** (<code>str</code>) – String to convert.

**Returns:**

- <code>OptimumEmbedderOptimizationMode</code> – Optimization mode.

### `OptimumEmbedderOptimizationConfig`

Configuration for Optimum Embedder Optimization.

**Parameters:**

- **mode** (<code>OptimumEmbedderOptimizationMode</code>) – Optimization mode.
- **for_gpu** (<code>bool</code>) – Whether to optimize for GPUs.

#### `to_optimum_config`

```python
to_optimum_config() -> OptimizationConfig
```

Convert the configuration to a Optimum configuration.

**Returns:**

- <code>OptimizationConfig</code> – Optimum configuration.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Convert the configuration to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> OptimumEmbedderOptimizationConfig
```

Create an optimization configuration from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>OptimumEmbedderOptimizationConfig</code> – Optimization configuration.

## `haystack_integrations.components.embedders.optimum.optimum_document_embedder`

### `OptimumDocumentEmbedder`

A component for computing `Document` embeddings using models loaded with the
[HuggingFace Optimum](https://huggingface.co/docs/optimum/index) library,
leveraging the ONNX runtime for high-speed inference.

The embedding of each Document is stored in the `embedding` field of the Document.

Usage example:

```python
from haystack.dataclasses import Document
from haystack_integrations.components.embedders.optimum import OptimumDocumentEmbedder

doc = Document(content="I love pizza!")

document_embedder = OptimumDocumentEmbedder(model="sentence-transformers/all-mpnet-base-v2")
# Components warm up automatically on first run.

result = document_embedder.run([doc])
print(result["documents"][0].embedding)

# [0.017020374536514282, -0.023255806416273117, ...]
```

#### `__init__`

```python
__init__(
    model: str = "sentence-transformers/all-mpnet-base-v2",
    token: Secret | None = Secret.from_env_var("HF_API_TOKEN", strict=False),
    prefix: str = "",
    suffix: str = "",
    normalize_embeddings: bool = True,
    onnx_execution_provider: str = "CPUExecutionProvider",
    pooling_mode: str | OptimumEmbedderPooling | None = None,
    model_kwargs: dict[str, Any] | None = None,
    working_dir: str | None = None,
    optimizer_settings: OptimumEmbedderOptimizationConfig | None = None,
    quantizer_settings: OptimumEmbedderQuantizationConfig | None = None,
    batch_size: int = 32,
    progress_bar: bool = True,
    meta_fields_to_embed: list[str] | None = None,
    embedding_separator: str = "\n",
) -> None
```

Create a OptimumDocumentEmbedder component.

**Parameters:**

- **model** (<code>str</code>) – A string representing the model id on HF Hub.

- **token** (<code>Secret | None</code>) – The HuggingFace token to use as HTTP bearer authorization.

- **prefix** (<code>str</code>) – A string to add to the beginning of each text.

- **suffix** (<code>str</code>) – A string to add to the end of each text.

- **normalize_embeddings** (<code>bool</code>) – Whether to normalize the embeddings to unit length.

- **onnx_execution_provider** (<code>str</code>) – The [execution provider](https://onnxruntime.ai/docs/execution-providers/)
  to use for ONNX models.

  Note: Using the TensorRT execution provider
  TensorRT requires to build its inference engine ahead of inference,
  which takes some time due to the model optimization and nodes fusion.
  To avoid rebuilding the engine every time the model is loaded, ONNX
  Runtime provides a pair of options to save the engine: `trt_engine_cache_enable`
  and `trt_engine_cache_path`. We recommend setting these two provider
  options using the `model_kwargs` parameter, when using the TensorRT execution provider.
  The usage is as follows:

  ```python
  embedder = OptimumDocumentEmbedder(
      model="sentence-transformers/all-mpnet-base-v2",
      onnx_execution_provider="TensorrtExecutionProvider",
      model_kwargs={
          "provider_options": {
              "trt_engine_cache_enable": True,
              "trt_engine_cache_path": "tmp/trt_cache",
          }
      },
  )
  ```

- **pooling_mode** (<code>str | OptimumEmbedderPooling | None</code>) – The pooling mode to use. When `None`, pooling mode will be inferred from the model config.

- **model_kwargs** (<code>dict\[str, Any\] | None</code>) – Dictionary containing additional keyword arguments to pass to the model.
  In case of duplication, these kwargs override `model`, `onnx_execution_provider`
  and `token` initialization parameters.

- **working_dir** (<code>str | None</code>) – The directory to use for storing intermediate files
  generated during model optimization/quantization. Required
  for optimization and quantization.

- **optimizer_settings** (<code>OptimumEmbedderOptimizationConfig | None</code>) – Configuration for Optimum Embedder Optimization.
  If `None`, no additional optimization is be applied.

- **quantizer_settings** (<code>OptimumEmbedderQuantizationConfig | None</code>) – Configuration for Optimum Embedder Quantization.
  If `None`, no quantization is be applied.

- **batch_size** (<code>int</code>) – Number of Documents to encode at once.

- **progress_bar** (<code>bool</code>) – Whether to show a progress bar or not.

- **meta_fields_to_embed** (<code>list\[str\] | None</code>) – List of meta fields that should be embedded along with the Document text.

- **embedding_separator** (<code>str</code>) – Separator used to concatenate the meta fields to the Document text.

#### `warm_up`

```python
warm_up() -> None
```

Initializes the component.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> OptimumDocumentEmbedder
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to deserialize from.

**Returns:**

- <code>OptimumDocumentEmbedder</code> – The deserialized component.

#### `run`

```python
run(documents: list[Document]) -> dict[str, list[Document]]
```

Embed a list of Documents.
The embedding of each Document is stored in the `embedding` field of the Document.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – A list of Documents to embed.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – The updated Documents with their embeddings.

**Raises:**

- <code>TypeError</code> – If the input is not a list of Documents.

## `haystack_integrations.components.embedders.optimum.optimum_text_embedder`

### `OptimumTextEmbedder`

A component to embed text using models loaded with the
[HuggingFace Optimum](https://huggingface.co/docs/optimum/index) library,
leveraging the ONNX runtime for high-speed inference.

Usage example:

```python
from haystack_integrations.components.embedders.optimum import OptimumTextEmbedder

text_to_embed = "I love pizza!"

text_embedder = OptimumTextEmbedder(model="sentence-transformers/all-mpnet-base-v2")
# Components warm up automatically on first run.

print(text_embedder.run(text_to_embed))

# {'embedding': [-0.07804739475250244, 0.1498992145061493,, ...]}
```

#### `__init__`

```python
__init__(
    model: str = "sentence-transformers/all-mpnet-base-v2",
    token: Secret | None = Secret.from_env_var("HF_API_TOKEN", strict=False),
    prefix: str = "",
    suffix: str = "",
    normalize_embeddings: bool = True,
    onnx_execution_provider: str = "CPUExecutionProvider",
    pooling_mode: str | OptimumEmbedderPooling | None = None,
    model_kwargs: dict[str, Any] | None = None,
    working_dir: str | None = None,
    optimizer_settings: OptimumEmbedderOptimizationConfig | None = None,
    quantizer_settings: OptimumEmbedderQuantizationConfig | None = None,
)
```

Create a OptimumTextEmbedder component.

**Parameters:**

- **model** (<code>str</code>) – A string representing the model id on HF Hub.

- **token** (<code>Secret | None</code>) – The HuggingFace token to use as HTTP bearer authorization.

- **prefix** (<code>str</code>) – A string to add to the beginning of each text.

- **suffix** (<code>str</code>) – A string to add to the end of each text.

- **normalize_embeddings** (<code>bool</code>) – Whether to normalize the embeddings to unit length.

- **onnx_execution_provider** (<code>str</code>) – The [execution provider](https://onnxruntime.ai/docs/execution-providers/)
  to use for ONNX models.

  Note: Using the TensorRT execution provider
  TensorRT requires to build its inference engine ahead of inference,
  which takes some time due to the model optimization and nodes fusion.
  To avoid rebuilding the engine every time the model is loaded, ONNX
  Runtime provides a pair of options to save the engine: `trt_engine_cache_enable`
  and `trt_engine_cache_path`. We recommend setting these two provider
  options using the `model_kwargs` parameter, when using the TensorRT execution provider.
  The usage is as follows:

  ```python
  embedder = OptimumDocumentEmbedder(
      model="sentence-transformers/all-mpnet-base-v2",
      onnx_execution_provider="TensorrtExecutionProvider",
      model_kwargs={
          "provider_options": {
              "trt_engine_cache_enable": True,
              "trt_engine_cache_path": "tmp/trt_cache",
          }
      },
  )
  ```

- **pooling_mode** (<code>str | OptimumEmbedderPooling | None</code>) – The pooling mode to use. When `None`, pooling mode will be inferred from the model config.

- **model_kwargs** (<code>dict\[str, Any\] | None</code>) – Dictionary containing additional keyword arguments to pass to the model.
  In case of duplication, these kwargs override `model`, `onnx_execution_provider`
  and `token` initialization parameters.

- **working_dir** (<code>str | None</code>) – The directory to use for storing intermediate files
  generated during model optimization/quantization. Required
  for optimization and quantization.

- **optimizer_settings** (<code>OptimumEmbedderOptimizationConfig | None</code>) – Configuration for Optimum Embedder Optimization.
  If `None`, no additional optimization is be applied.

- **quantizer_settings** (<code>OptimumEmbedderQuantizationConfig | None</code>) – Configuration for Optimum Embedder Quantization.
  If `None`, no quantization is be applied.

#### `warm_up`

```python
warm_up()
```

Initializes the component.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> OptimumTextEmbedder
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to deserialize from.

**Returns:**

- <code>OptimumTextEmbedder</code> – The deserialized component.

#### `run`

```python
run(text: str) -> dict[str, list[float]]
```

Embed a string.

**Parameters:**

- **text** (<code>str</code>) – The text to embed.

**Returns:**

- <code>dict\[str, list\[float\]\]</code> – The embeddings of the text.

**Raises:**

- <code>TypeError</code> – If the input is not a string.

## `haystack_integrations.components.embedders.optimum.pooling`

### `OptimumEmbedderPooling`

Bases: <code>Enum</code>

Pooling modes support by the Optimum Embedders.

#### `from_str`

```python
from_str(string: str) -> OptimumEmbedderPooling
```

Create a pooling mode from a string.

**Parameters:**

- **string** (<code>str</code>) – String to convert.

**Returns:**

- <code>OptimumEmbedderPooling</code> – Pooling mode.

## `haystack_integrations.components.embedders.optimum.quantization`

### `OptimumEmbedderQuantizationMode`

Bases: <code>Enum</code>

[Dynamic Quantization modes](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/quantization)
support by the Optimum Embedders.

#### `from_str`

```python
from_str(string: str) -> OptimumEmbedderQuantizationMode
```

Create an quantization mode from a string.

**Parameters:**

- **string** (<code>str</code>) – String to convert.

**Returns:**

- <code>OptimumEmbedderQuantizationMode</code> – Quantization mode.

### `OptimumEmbedderQuantizationConfig`

Configuration for Optimum Embedder Quantization.

**Parameters:**

- **mode** (<code>OptimumEmbedderQuantizationMode</code>) – Quantization mode.
- **per_channel** (<code>bool</code>) – Whether to apply per-channel quantization.

#### `to_optimum_config`

```python
to_optimum_config() -> QuantizationConfig
```

Convert the configuration to a Optimum configuration.

**Returns:**

- <code>QuantizationConfig</code> – Optimum configuration.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Convert the configuration to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> OptimumEmbedderQuantizationConfig
```

Create a configuration from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>OptimumEmbedderQuantizationConfig</code> – Quantization configuration.
