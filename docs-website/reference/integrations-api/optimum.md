---
title: "Optimum"
id: integrations-optimum
description: "Optimum integration for Haystack"
slug: "/integrations-optimum"
---

<a id="haystack_integrations.components.embedders.optimum.optimum_document_embedder"></a>

# Module haystack\_integrations.components.embedders.optimum.optimum\_document\_embedder

<a id="haystack_integrations.components.embedders.optimum.optimum_document_embedder.OptimumDocumentEmbedder"></a>

## OptimumDocumentEmbedder

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
document_embedder.warm_up()

result = document_embedder.run([doc])
print(result["documents"][0].embedding)

# [0.017020374536514282, -0.023255806416273117, ...]
```

<a id="haystack_integrations.components.embedders.optimum.optimum_document_embedder.OptimumDocumentEmbedder.__init__"></a>

#### OptimumDocumentEmbedder.\_\_init\_\_

```python
def __init__(
        model: str = "sentence-transformers/all-mpnet-base-v2",
        token: Optional[Secret] = Secret.from_env_var("HF_API_TOKEN",
                                                      strict=False),
        prefix: str = "",
        suffix: str = "",
        normalize_embeddings: bool = True,
        onnx_execution_provider: str = "CPUExecutionProvider",
        pooling_mode: Optional[Union[str, OptimumEmbedderPooling]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        working_dir: Optional[str] = None,
        optimizer_settings: Optional[OptimumEmbedderOptimizationConfig] = None,
        quantizer_settings: Optional[OptimumEmbedderQuantizationConfig] = None,
        batch_size: int = 32,
        progress_bar: bool = True,
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n")
```

Create a OptimumDocumentEmbedder component.

**Arguments**:

- `model`: A string representing the model id on HF Hub.
- `token`: The HuggingFace token to use as HTTP bearer authorization.
- `prefix`: A string to add to the beginning of each text.
- `suffix`: A string to add to the end of each text.
- `normalize_embeddings`: Whether to normalize the embeddings to unit length.
- `onnx_execution_provider`: The [execution provider](https://onnxruntime.ai/docs/execution-providers/)
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
- `pooling_mode`: The pooling mode to use. When `None`, pooling mode will be inferred from the model config.
- `model_kwargs`: Dictionary containing additional keyword arguments to pass to the model.
In case of duplication, these kwargs override `model`, `onnx_execution_provider`
and `token` initialization parameters.
- `working_dir`: The directory to use for storing intermediate files
generated during model optimization/quantization. Required
for optimization and quantization.
- `optimizer_settings`: Configuration for Optimum Embedder Optimization.
If `None`, no additional optimization is be applied.
- `quantizer_settings`: Configuration for Optimum Embedder Quantization.
If `None`, no quantization is be applied.
- `batch_size`: Number of Documents to encode at once.
- `progress_bar`: Whether to show a progress bar or not.
- `meta_fields_to_embed`: List of meta fields that should be embedded along with the Document text.
- `embedding_separator`: Separator used to concatenate the meta fields to the Document text.

<a id="haystack_integrations.components.embedders.optimum.optimum_document_embedder.OptimumDocumentEmbedder.warm_up"></a>

#### OptimumDocumentEmbedder.warm\_up

```python
def warm_up()
```

Initializes the component.

<a id="haystack_integrations.components.embedders.optimum.optimum_document_embedder.OptimumDocumentEmbedder.to_dict"></a>

#### OptimumDocumentEmbedder.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.embedders.optimum.optimum_document_embedder.OptimumDocumentEmbedder.from_dict"></a>

#### OptimumDocumentEmbedder.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "OptimumDocumentEmbedder"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: The dictionary to deserialize from.

**Returns**:

The deserialized component.

<a id="haystack_integrations.components.embedders.optimum.optimum_document_embedder.OptimumDocumentEmbedder.run"></a>

#### OptimumDocumentEmbedder.run

```python
@component.output_types(documents=List[Document])
def run(documents: List[Document]) -> Dict[str, List[Document]]
```

Embed a list of Documents.

The embedding of each Document is stored in the `embedding` field of the Document.

**Arguments**:

- `documents`: A list of Documents to embed.

**Raises**:

- `RuntimeError`: If the component was not initialized.
- `TypeError`: If the input is not a list of Documents.

**Returns**:

The updated Documents with their embeddings.

<a id="haystack_integrations.components.embedders.optimum.optimum_text_embedder"></a>

# Module haystack\_integrations.components.embedders.optimum.optimum\_text\_embedder

<a id="haystack_integrations.components.embedders.optimum.optimum_text_embedder.OptimumTextEmbedder"></a>

## OptimumTextEmbedder

A component to embed text using models loaded with the
[HuggingFace Optimum](https://huggingface.co/docs/optimum/index) library,
leveraging the ONNX runtime for high-speed inference.

Usage example:
```python
from haystack_integrations.components.embedders.optimum import OptimumTextEmbedder

text_to_embed = "I love pizza!"

text_embedder = OptimumTextEmbedder(model="sentence-transformers/all-mpnet-base-v2")
text_embedder.warm_up()

print(text_embedder.run(text_to_embed))

# {'embedding': [-0.07804739475250244, 0.1498992145061493,, ...]}
```

<a id="haystack_integrations.components.embedders.optimum.optimum_text_embedder.OptimumTextEmbedder.__init__"></a>

#### OptimumTextEmbedder.\_\_init\_\_

```python
def __init__(
        model: str = "sentence-transformers/all-mpnet-base-v2",
        token: Optional[Secret] = Secret.from_env_var("HF_API_TOKEN",
                                                      strict=False),
        prefix: str = "",
        suffix: str = "",
        normalize_embeddings: bool = True,
        onnx_execution_provider: str = "CPUExecutionProvider",
        pooling_mode: Optional[Union[str, OptimumEmbedderPooling]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        working_dir: Optional[str] = None,
        optimizer_settings: Optional[OptimumEmbedderOptimizationConfig] = None,
        quantizer_settings: Optional[OptimumEmbedderQuantizationConfig] = None
)
```

Create a OptimumTextEmbedder component.

**Arguments**:

- `model`: A string representing the model id on HF Hub.
- `token`: The HuggingFace token to use as HTTP bearer authorization.
- `prefix`: A string to add to the beginning of each text.
- `suffix`: A string to add to the end of each text.
- `normalize_embeddings`: Whether to normalize the embeddings to unit length.
- `onnx_execution_provider`: The [execution provider](https://onnxruntime.ai/docs/execution-providers/)
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
- `pooling_mode`: The pooling mode to use. When `None`, pooling mode will be inferred from the model config.
- `model_kwargs`: Dictionary containing additional keyword arguments to pass to the model.
In case of duplication, these kwargs override `model`, `onnx_execution_provider`
and `token` initialization parameters.
- `working_dir`: The directory to use for storing intermediate files
generated during model optimization/quantization. Required
for optimization and quantization.
- `optimizer_settings`: Configuration for Optimum Embedder Optimization.
If `None`, no additional optimization is be applied.
- `quantizer_settings`: Configuration for Optimum Embedder Quantization.
If `None`, no quantization is be applied.

<a id="haystack_integrations.components.embedders.optimum.optimum_text_embedder.OptimumTextEmbedder.warm_up"></a>

#### OptimumTextEmbedder.warm\_up

```python
def warm_up()
```

Initializes the component.

<a id="haystack_integrations.components.embedders.optimum.optimum_text_embedder.OptimumTextEmbedder.to_dict"></a>

#### OptimumTextEmbedder.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.embedders.optimum.optimum_text_embedder.OptimumTextEmbedder.from_dict"></a>

#### OptimumTextEmbedder.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "OptimumTextEmbedder"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: The dictionary to deserialize from.

**Returns**:

The deserialized component.

<a id="haystack_integrations.components.embedders.optimum.optimum_text_embedder.OptimumTextEmbedder.run"></a>

#### OptimumTextEmbedder.run

```python
@component.output_types(embedding=List[float])
def run(text: str) -> Dict[str, List[float]]
```

Embed a string.

**Arguments**:

- `text`: The text to embed.

**Raises**:

- `RuntimeError`: If the component was not initialized.
- `TypeError`: If the input is not a string.

**Returns**:

The embeddings of the text.

<a id="haystack_integrations.components.embedders.optimum.pooling"></a>

# Module haystack\_integrations.components.embedders.optimum.pooling

<a id="haystack_integrations.components.embedders.optimum.pooling.OptimumEmbedderPooling"></a>

## OptimumEmbedderPooling

Pooling modes support by the Optimum Embedders.

<a id="haystack_integrations.components.embedders.optimum.pooling.OptimumEmbedderPooling.CLS"></a>

#### CLS

Perform CLS Pooling on the output of the embedding model
using the first token (CLS token).

<a id="haystack_integrations.components.embedders.optimum.pooling.OptimumEmbedderPooling.MEAN"></a>

#### MEAN

Perform Mean Pooling on the output of the embedding model.

<a id="haystack_integrations.components.embedders.optimum.pooling.OptimumEmbedderPooling.MAX"></a>

#### MAX

Perform Max Pooling on the output of the embedding model
using the maximum value in each dimension over all the tokens.

<a id="haystack_integrations.components.embedders.optimum.pooling.OptimumEmbedderPooling.MEAN_SQRT_LEN"></a>

#### MEAN\_SQRT\_LEN

Perform mean-pooling on the output of the embedding model but
divide by the square root of the sequence length.

<a id="haystack_integrations.components.embedders.optimum.pooling.OptimumEmbedderPooling.WEIGHTED_MEAN"></a>

#### WEIGHTED\_MEAN

Perform weighted (position) mean pooling on the output of the
embedding model.

<a id="haystack_integrations.components.embedders.optimum.pooling.OptimumEmbedderPooling.LAST_TOKEN"></a>

#### LAST\_TOKEN

Perform Last Token Pooling on the output of the embedding model.

<a id="haystack_integrations.components.embedders.optimum.pooling.OptimumEmbedderPooling.from_str"></a>

#### OptimumEmbedderPooling.from\_str

```python
@classmethod
def from_str(cls, string: str) -> "OptimumEmbedderPooling"
```

Create a pooling mode from a string.

**Arguments**:

- `string`: String to convert.

**Returns**:

Pooling mode.

<a id="haystack_integrations.components.embedders.optimum.optimization"></a>

# Module haystack\_integrations.components.embedders.optimum.optimization

<a id="haystack_integrations.components.embedders.optimum.optimization.OptimumEmbedderOptimizationMode"></a>

## OptimumEmbedderOptimizationMode

[ONXX Optimization modes](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/optimization)
support by the Optimum Embedders.

<a id="haystack_integrations.components.embedders.optimum.optimization.OptimumEmbedderOptimizationMode.O1"></a>

#### O1

Basic general optimizations.

<a id="haystack_integrations.components.embedders.optimum.optimization.OptimumEmbedderOptimizationMode.O2"></a>

#### O2

Basic and extended general optimizations, transformers-specific fusions.

<a id="haystack_integrations.components.embedders.optimum.optimization.OptimumEmbedderOptimizationMode.O3"></a>

#### O3

Same as O2 with Gelu approximation.

<a id="haystack_integrations.components.embedders.optimum.optimization.OptimumEmbedderOptimizationMode.O4"></a>

#### O4

Same as O3 with mixed precision.

<a id="haystack_integrations.components.embedders.optimum.optimization.OptimumEmbedderOptimizationMode.from_str"></a>

#### OptimumEmbedderOptimizationMode.from\_str

```python
@classmethod
def from_str(cls, string: str) -> "OptimumEmbedderOptimizationMode"
```

Create an optimization mode from a string.

**Arguments**:

- `string`: String to convert.

**Returns**:

Optimization mode.

<a id="haystack_integrations.components.embedders.optimum.optimization.OptimumEmbedderOptimizationConfig"></a>

## OptimumEmbedderOptimizationConfig

Configuration for Optimum Embedder Optimization.

**Arguments**:

- `mode`: Optimization mode.
- `for_gpu`: Whether to optimize for GPUs.

<a id="haystack_integrations.components.embedders.optimum.optimization.OptimumEmbedderOptimizationConfig.to_optimum_config"></a>

#### OptimumEmbedderOptimizationConfig.to\_optimum\_config

```python
def to_optimum_config() -> OptimizationConfig
```

Convert the configuration to a Optimum configuration.

**Returns**:

Optimum configuration.

<a id="haystack_integrations.components.embedders.optimum.optimization.OptimumEmbedderOptimizationConfig.to_dict"></a>

#### OptimumEmbedderOptimizationConfig.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Convert the configuration to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.embedders.optimum.optimization.OptimumEmbedderOptimizationConfig.from_dict"></a>

#### OptimumEmbedderOptimizationConfig.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str,
                              Any]) -> "OptimumEmbedderOptimizationConfig"
```

Create an optimization configuration from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Optimization configuration.

<a id="haystack_integrations.components.embedders.optimum.quantization"></a>

# Module haystack\_integrations.components.embedders.optimum.quantization

<a id="haystack_integrations.components.embedders.optimum.quantization.OptimumEmbedderQuantizationMode"></a>

## OptimumEmbedderQuantizationMode

[Dynamic Quantization modes](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/quantization)
support by the Optimum Embedders.

<a id="haystack_integrations.components.embedders.optimum.quantization.OptimumEmbedderQuantizationMode.ARM64"></a>

#### ARM64

Quantization for the ARM64 architecture.

<a id="haystack_integrations.components.embedders.optimum.quantization.OptimumEmbedderQuantizationMode.AVX2"></a>

#### AVX2

Quantization with AVX-2 instructions.

<a id="haystack_integrations.components.embedders.optimum.quantization.OptimumEmbedderQuantizationMode.AVX512"></a>

#### AVX512

Quantization with AVX-512 instructions.

<a id="haystack_integrations.components.embedders.optimum.quantization.OptimumEmbedderQuantizationMode.AVX512_VNNI"></a>

#### AVX512\_VNNI

Quantization with AVX-512 and VNNI instructions.

<a id="haystack_integrations.components.embedders.optimum.quantization.OptimumEmbedderQuantizationMode.from_str"></a>

#### OptimumEmbedderQuantizationMode.from\_str

```python
@classmethod
def from_str(cls, string: str) -> "OptimumEmbedderQuantizationMode"
```

Create an quantization mode from a string.

**Arguments**:

- `string`: String to convert.

**Returns**:

Quantization mode.

<a id="haystack_integrations.components.embedders.optimum.quantization.OptimumEmbedderQuantizationConfig"></a>

## OptimumEmbedderQuantizationConfig

Configuration for Optimum Embedder Quantization.

**Arguments**:

- `mode`: Quantization mode.
- `per_channel`: Whether to apply per-channel quantization.

<a id="haystack_integrations.components.embedders.optimum.quantization.OptimumEmbedderQuantizationConfig.to_optimum_config"></a>

#### OptimumEmbedderQuantizationConfig.to\_optimum\_config

```python
def to_optimum_config() -> QuantizationConfig
```

Convert the configuration to a Optimum configuration.

**Returns**:

Optimum configuration.

<a id="haystack_integrations.components.embedders.optimum.quantization.OptimumEmbedderQuantizationConfig.to_dict"></a>

#### OptimumEmbedderQuantizationConfig.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Convert the configuration to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.embedders.optimum.quantization.OptimumEmbedderQuantizationConfig.from_dict"></a>

#### OptimumEmbedderQuantizationConfig.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str,
                              Any]) -> "OptimumEmbedderQuantizationConfig"
```

Create a configuration from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Quantization configuration.
