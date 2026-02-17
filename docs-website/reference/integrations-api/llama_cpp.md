---
title: "Llama.cpp"
id: integrations-llama-cpp
description: "Llama.cpp integration for Haystack"
slug: "/integrations-llama-cpp"
---


## `haystack_integrations.components.generators.llama_cpp.chat.chat_generator`

### `LlamaCppChatGenerator`

Provides an interface to generate text using LLM via llama.cpp.

[llama.cpp](https://github.com/ggml-org/llama.cpp) is a project written in C/C++ for efficient inference of LLMs.
It employs the quantized GGUF format, suitable for running these models on standard machines (even without GPUs).
Supports both text-only and multimodal (text + image) models like LLaVA.

Usage example:

```python
from haystack_integrations.components.generators.llama_cpp import LlamaCppChatGenerator
user_message = [ChatMessage.from_user("Who is the best American actor?")]
generator = LlamaCppGenerator(model="zephyr-7b-beta.Q4_0.gguf", n_ctx=2048, n_batch=512)

print(generator.run(user_message, generation_kwargs={"max_tokens": 128}))
# {"replies": [ChatMessage(content="John Cusack", role=<ChatRole.ASSISTANT: "assistant">, name=None, meta={...})}
```

Usage example with multimodal (image + text):

```python
from haystack.dataclasses import ChatMessage, ImageContent

# Create an image from file path or base64
image_content = ImageContent.from_file_path("path/to/your/image.jpg")

# Create a multimodal message with both text and image
messages = [ChatMessage.from_user(content_parts=["What's in this image?", image_content])]

# Initialize with multimodal support
generator = LlamaCppChatGenerator(
    model="llava-v1.5-7b-q4_0.gguf",
    chat_handler_name="Llava15ChatHandler",  # Use llava-1-5 handler
    model_clip_path="mmproj-model-f16.gguf",  # CLIP model
    n_ctx=4096  # Larger context for image processing
)
generator.warm_up()

result = generator.run(messages)
print(result)
```

#### `__init__`

```python
__init__(
    model: str,
    n_ctx: int | None = 0,
    n_batch: int | None = 512,
    model_kwargs: dict[str, Any] | None = None,
    generation_kwargs: dict[str, Any] | None = None,
    *,
    tools: ToolsType | None = None,
    streaming_callback: StreamingCallbackT | None = None,
    chat_handler_name: str | None = None,
    model_clip_path: str | None = None
) -> None
```

**Parameters:**

- **model** (<code>str</code>) – The path of a quantized model for text generation, for example, "zephyr-7b-beta.Q4_0.gguf".
  If the model path is also specified in the `model_kwargs`, this parameter will be ignored.
- **n_ctx** (<code>int | None</code>) – The number of tokens in the context. When set to 0, the context will be taken from the model.
- **n_batch** (<code>int | None</code>) – Prompt processing maximum batch size.
- **model_kwargs** (<code>dict\[str, Any\] | None</code>) – Dictionary containing keyword arguments used to initialize the LLM for text generation.
  These keyword arguments provide fine-grained control over the model loading.
  In case of duplication, these kwargs override `model`, `n_ctx`, and `n_batch` init parameters.
  For more information on the available kwargs, see
  [llama.cpp documentation](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__init__).
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – A dictionary containing keyword arguments to customize text generation.
  For more information on the available kwargs, see
  [llama.cpp documentation](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_chat_completion).
- **tools** (<code>ToolsType | None</code>) – A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
  Each tool should have a unique name.
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – A callback function that is called when a new token is received from the stream.
- **chat_handler_name** (<code>str | None</code>) – Name of the chat handler for multimodal models.
  Common options include: "Llava16ChatHandler", "MoondreamChatHandler", "Qwen25VLChatHandler".
  For other handlers, check
  [llama-cpp-python documentation](https://llama-cpp-python.readthedocs.io/en/latest/#multi-modal-models).
- **model_clip_path** (<code>str | None</code>) – Path to the CLIP model for vision processing (e.g., "mmproj.bin").
  Required when chat_handler_name is provided for multimodal models.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> LlamaCppChatGenerator
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>LlamaCppChatGenerator</code> – Deserialized component.

#### `run`

```python
run(
    messages: list[ChatMessage],
    generation_kwargs: dict[str, Any] | None = None,
    *,
    tools: ToolsType | None = None,
    streaming_callback: StreamingCallbackT | None = None
) -> dict[str, list[ChatMessage]]
```

Run the text generation model on the given list of ChatMessages.

**Parameters:**

- **messages** (<code>list\[ChatMessage\]</code>) – A list of ChatMessage instances representing the input messages.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – A dictionary containing keyword arguments to customize text generation.
  For more information on the available kwargs, see
  [llama.cpp documentation](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_chat_completion).
- **tools** (<code>ToolsType | None</code>) – A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
  Each tool should have a unique name. If set, it will override the `tools` parameter set during
  component initialization.
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – A callback function that is called when a new token is received from the stream.
  If set, it will override the `streaming_callback` parameter set during component initialization.

**Returns:**

- <code>dict\[str, list\[ChatMessage\]\]</code> – A dictionary with the following keys:
- `replies`: The responses from the model

#### `run_async`

```python
run_async(
    messages: list[ChatMessage],
    generation_kwargs: dict[str, Any] | None = None,
    *,
    tools: ToolsType | None = None,
    streaming_callback: StreamingCallbackT | None = None
) -> dict[str, list[ChatMessage]]
```

Async version of run. Runs the text generation model on the given list of ChatMessages.

Uses a thread pool to avoid blocking the event loop, since llama-cpp-python provides
only synchronous inference.

**Parameters:**

- **messages** (<code>list\[ChatMessage\]</code>) – A list of ChatMessage instances representing the input messages.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – A dictionary containing keyword arguments to customize text generation.
  For more information on the available kwargs, see
  [llama.cpp documentation](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_chat_completion).
- **tools** (<code>ToolsType | None</code>) – A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
  Each tool should have a unique name. If set, it will override the `tools` parameter set during
  component initialization.
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – A callback function that is called when a new token is received from the stream.
  If set, it will override the `streaming_callback` parameter set during component initialization.

**Returns:**

- <code>dict\[str, list\[ChatMessage\]\]</code> – A dictionary with the following keys:
- `replies`: The responses from the model

## `haystack_integrations.components.generators.llama_cpp.generator`

### `LlamaCppGenerator`

Provides an interface to generate text using LLM via llama.cpp.

[llama.cpp](https://github.com/ggml-org/llama.cpp) is a project written in C/C++ for efficient inference of LLMs.
It employs the quantized GGUF format, suitable for running these models on standard machines (even without GPUs).

Usage example:

```python
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator
generator = LlamaCppGenerator(model="zephyr-7b-beta.Q4_0.gguf", n_ctx=2048, n_batch=512)

print(generator.run("Who is the best American actor?", generation_kwargs={"max_tokens": 128}))
# {'replies': ['John Cusack'], 'meta': [{"object": "text_completion", ...}]}
```

#### `__init__`

```python
__init__(
    model: str,
    n_ctx: int | None = 0,
    n_batch: int | None = 512,
    model_kwargs: dict[str, Any] | None = None,
    generation_kwargs: dict[str, Any] | None = None,
) -> None
```

**Parameters:**

- **model** (<code>str</code>) – The path of a quantized model for text generation, for example, "zephyr-7b-beta.Q4_0.gguf".
  If the model path is also specified in the `model_kwargs`, this parameter will be ignored.
- **n_ctx** (<code>int | None</code>) – The number of tokens in the context. When set to 0, the context will be taken from the model.
- **n_batch** (<code>int | None</code>) – Prompt processing maximum batch size.
- **model_kwargs** (<code>dict\[str, Any\] | None</code>) – Dictionary containing keyword arguments used to initialize the LLM for text generation.
  These keyword arguments provide fine-grained control over the model loading.
  In case of duplication, these kwargs override `model`, `n_ctx`, and `n_batch` init parameters.
  For more information on the available kwargs, see
  [llama.cpp documentation](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__init__).
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – A dictionary containing keyword arguments to customize text generation.
  For more information on the available kwargs, see
  [llama.cpp documentation](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_completion).

#### `run`

```python
run(
    prompt: str, generation_kwargs: dict[str, Any] | None = None
) -> dict[str, list[str] | list[dict[str, Any]]]
```

Run the text generation model on the given prompt.

**Parameters:**

- **prompt** (<code>str</code>) – the prompt to be sent to the generative model.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – A dictionary containing keyword arguments to customize text generation.
  For more information on the available kwargs, see
  [llama.cpp documentation](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_completion).

**Returns:**

- <code>dict\[str, list\[str\] | list\[dict\[str, Any\]\]\]</code> – A dictionary with the following keys:
- `replies`: the list of replies generated by the model.
- `meta`: metadata about the request.
