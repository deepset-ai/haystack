---
title: "Ollama"
id: integrations-ollama
description: "Ollama integration for Haystack"
slug: "/integrations-ollama"
---


## haystack_integrations.components.embedders.ollama.document_embedder

### OllamaDocumentEmbedder

Computes the embeddings of a list of Documents and stores the obtained vectors in the embedding field of each
Document. It uses embedding models compatible with the Ollama Library.

Usage example:

```python
from haystack import Document
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder

doc = Document(content="What do llamas say once you have thanked them? No probllama!")
document_embedder = OllamaDocumentEmbedder()

result = document_embedder.run([doc])
print(result['documents'][0].embedding)
```

#### __init__

```python
__init__(
    model: str = "nomic-embed-text",
    url: str = "http://localhost:11434",
    generation_kwargs: dict[str, Any] | None = None,
    timeout: int = 120,
    keep_alive: float | str | None = None,
    prefix: str = "",
    suffix: str = "",
    progress_bar: bool = True,
    meta_fields_to_embed: list[str] | None = None,
    embedding_separator: str = "\n",
    batch_size: int = 32,
)
```

**Parameters:**

- **model** (<code>str</code>) – The name of the model to use. The model should be available in the running Ollama instance.
- **url** (<code>str</code>) – The URL of a running Ollama instance.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Optional arguments to pass to the Ollama generation endpoint, such as temperature, top_p, and others.
  See the available arguments in
  [Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).
- **timeout** (<code>int</code>) – The number of seconds before throwing a timeout error from the Ollama API.
- **keep_alive** (<code>float | str | None</code>) – The option that controls how long the model will stay loaded into memory following the request.
  If not set, it will use the default value from the Ollama (5 minutes).
  The value can be set to:
- a duration string (such as "10m" or "24h")
- a number in seconds (such as 3600)
- any negative number which will keep the model loaded in memory (e.g. -1 or "-1m")
- '0' which will unload the model immediately after generating a response.
- **prefix** (<code>str</code>) – A string to add at the beginning of each text.
- **suffix** (<code>str</code>) – A string to add at the end of each text.
- **progress_bar** (<code>bool</code>) – If `True`, shows a progress bar when running.
- **meta_fields_to_embed** (<code>list\[str\] | None</code>) – List of metadata fields to embed along with the document text.
- **embedding_separator** (<code>str</code>) – Separator used to concatenate the metadata fields to the document text.
- **batch_size** (<code>int</code>) – Number of documents to process at once.

#### run

```python
run(
    documents: list[Document], generation_kwargs: dict[str, Any] | None = None
) -> dict[str, list[Document] | dict[str, Any]]
```

Runs an Ollama Model to compute embeddings of the provided documents.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – Documents to be converted to an embedding.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Optional arguments to pass to the Ollama generation endpoint, such as temperature,
  top_p, etc. See the
  [Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).

**Returns:**

- <code>dict\[str, list\[Document\] | dict\[str, Any\]\]</code> – A dictionary with the following keys:
- `documents`: Documents with embedding information attached
- `meta`: The metadata collected during the embedding process

#### run_async

```python
run_async(
    documents: list[Document], generation_kwargs: dict[str, Any] | None = None
) -> dict[str, list[Document] | dict[str, Any]]
```

Asynchronously run an Ollama Model to compute embeddings of the provided documents.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – Documents to be converted to an embedding.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Optional arguments to pass to the Ollama generation endpoint, such as temperature,
  top_p, etc. See the
  [Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).

**Returns:**

- <code>dict\[str, list\[Document\] | dict\[str, Any\]\]</code> – A dictionary with the following keys:
- `documents`: Documents with embedding information attached
- `meta`: The metadata collected during the embedding process

## haystack_integrations.components.embedders.ollama.text_embedder

### OllamaTextEmbedder

Computes the embeddings of a list of Documents and stores the obtained vectors in the embedding field of
each Document. It uses embedding models compatible with the Ollama Library.

Usage example:

```python
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder

embedder = OllamaTextEmbedder()
result = embedder.run(text="What do llamas say once you have thanked them? No probllama!")
print(result['embedding'])
```

#### __init__

```python
__init__(
    model: str = "nomic-embed-text",
    url: str = "http://localhost:11434",
    generation_kwargs: dict[str, Any] | None = None,
    timeout: int = 120,
    keep_alive: float | str | None = None,
)
```

**Parameters:**

- **model** (<code>str</code>) – The name of the model to use. The model should be available in the running Ollama instance.
- **url** (<code>str</code>) – The URL of a running Ollama instance.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Optional arguments to pass to the Ollama generation endpoint, such as temperature,
  top_p, and others. See the available arguments in
  [Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).
- **timeout** (<code>int</code>) – The number of seconds before throwing a timeout error from the Ollama API.
- **keep_alive** (<code>float | str | None</code>) – The option that controls how long the model will stay loaded into memory following the request.
  If not set, it will use the default value from the Ollama (5 minutes).
  The value can be set to:
- a duration string (such as "10m" or "24h")
- a number in seconds (such as 3600)
- any negative number which will keep the model loaded in memory (e.g. -1 or "-1m")
- '0' which will unload the model immediately after generating a response.

#### run

```python
run(
    text: str, generation_kwargs: dict[str, Any] | None = None
) -> dict[str, list[float] | dict[str, Any]]
```

Runs an Ollama Model to compute embeddings of the provided text.

**Parameters:**

- **text** (<code>str</code>) – Text to be converted to an embedding.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Optional arguments to pass to the Ollama generation endpoint, such as temperature,
  top_p, etc. See the
  [Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).

**Returns:**

- <code>dict\[str, list\[float\] | dict\[str, Any\]\]</code> – A dictionary with the following keys:
- `embedding`: The computed embeddings
- `meta`: The metadata collected during the embedding process

#### run_async

```python
run_async(
    text: str, generation_kwargs: dict[str, Any] | None = None
) -> dict[str, list[float] | dict[str, Any]]
```

Asynchronously run an Ollama Model to compute embeddings of the provided text.

**Parameters:**

- **text** (<code>str</code>) – Text to be converted to an embedding.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Optional arguments to pass to the Ollama generation endpoint, such as temperature,
  top_p, etc. See the
  [Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).

**Returns:**

- <code>dict\[str, list\[float\] | dict\[str, Any\]\]</code> – A dictionary with the following keys:
- `embedding`: The computed embeddings
- `meta`: The metadata collected during the embedding process

## haystack_integrations.components.generators.ollama.chat.chat_generator

### OllamaChatGenerator

Haystack Chat Generator for models served with Ollama (https://ollama.ai).

Supports streaming, tool calls, reasoning, and structured outputs.

Usage example:

```python
from haystack_integrations.components.generators.ollama.chat import OllamaChatGenerator
from haystack.dataclasses import ChatMessage

llm = OllamaChatGenerator(model="qwen3:0.6b")
result = llm.run(messages=[ChatMessage.from_user("What is the capital of France?")])
print(result)
```

#### __init__

```python
__init__(
    model: str = "qwen3:0.6b",
    url: str = "http://localhost:11434",
    generation_kwargs: dict[str, Any] | None = None,
    timeout: int = 120,
    keep_alive: float | str | None = None,
    streaming_callback: Callable[[StreamingChunk], None] | None = None,
    tools: ToolsType | None = None,
    response_format: None | Literal["json"] | JsonSchemaValue | None = None,
    think: bool | Literal["low", "medium", "high"] = False,
)
```

**Parameters:**

- **model** (<code>str</code>) – The name of the model to use. The model must already be present (pulled) in the running Ollama instance.
- **url** (<code>str</code>) – The base URL of the Ollama server (default "http://localhost:11434").
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Optional arguments to pass to the Ollama generation endpoint, such as temperature,
  top_p, and others. See the available arguments in
  [Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).
- **timeout** (<code>int</code>) – The number of seconds before throwing a timeout error from the Ollama API.
- **think** (<code>bool | Literal['low', 'medium', 'high']</code>) – If True, the model will "think" before producing a response.
  Only [thinking models](https://ollama.com/search?c=thinking) support this feature.
  Some models like gpt-oss support different levels of thinking: "low", "medium", "high".
  The intermediate "thinking" output can be found by inspecting the `reasoning` property of the returned
  `ChatMessage`.
- **keep_alive** (<code>float | str | None</code>) – The option that controls how long the model will stay loaded into memory following the request.
  If not set, it will use the default value from the Ollama (5 minutes).
  The value can be set to:
- a duration string (such as "10m" or "24h")
- a number in seconds (such as 3600)
- any negative number which will keep the model loaded in memory (e.g. -1 or "-1m")
- '0' which will unload the model immediately after generating a response.
- **streaming_callback** (<code>Callable\\[[StreamingChunk\], None\] | None</code>) – A callback function that is called when a new token is received from the stream.
  The callback function accepts StreamingChunk as an argument.
- **tools** (<code>ToolsType | None</code>) – A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
  Each tool should have a unique name. Not all models support tools. For a list of models compatible
  with tools, see the [models page](https://ollama.com/search?c=tools).
- **response_format** (<code>None | Literal['json'] | JsonSchemaValue | None</code>) – The format for structured model outputs. The value can be:
- None: No specific structure or format is applied to the response. The response is returned as-is.
- "json": The response is formatted as a JSON object.
- JSON Schema: The response is formatted as a JSON object
  that adheres to the specified JSON Schema. (needs Ollama ≥ 0.1.34)

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> OllamaChatGenerator
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>OllamaChatGenerator</code> – Deserialized component.

#### run

```python
run(
    messages: list[ChatMessage],
    generation_kwargs: dict[str, Any] | None = None,
    tools: ToolsType | None = None,
    *,
    streaming_callback: StreamingCallbackT | None = None
) -> dict[str, list[ChatMessage]]
```

Runs an Ollama Model on a given chat history.

**Parameters:**

- **messages** (<code>list\[ChatMessage\]</code>) – A list of ChatMessage instances representing the input messages.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Per-call overrides for Ollama inference options.
  These are merged on top of the instance-level `generation_kwargs`.
  Optional arguments to pass to the Ollama generation endpoint, such as temperature, top_p, etc. See the
  [Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).
- **tools** (<code>ToolsType | None</code>) – A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
  If set, it will override the `tools` parameter set during component initialization.
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – A callable to receive `StreamingChunk` objects as they
  arrive. Supplying a callback (here or in the constructor) switches
  the component into streaming mode.

**Returns:**

- <code>dict\[str, list\[ChatMessage\]\]</code> – A dictionary with the following keys:
- `replies`: A list of ChatMessages containing the model's response

#### run_async

```python
run_async(
    messages: list[ChatMessage],
    generation_kwargs: dict[str, Any] | None = None,
    tools: ToolsType | None = None,
    *,
    streaming_callback: StreamingCallbackT | None = None
) -> dict[str, list[ChatMessage]]
```

Async version of run. Runs an Ollama Model on a given chat history.

**Parameters:**

- **messages** (<code>list\[ChatMessage\]</code>) – A list of ChatMessage instances representing the input messages.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Per-call overrides for Ollama inference options.
  These are merged on top of the instance-level `generation_kwargs`.
- **tools** (<code>ToolsType | None</code>) – A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
  If set, it will override the `tools` parameter set during component initialization.
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – A callable to receive `StreamingChunk` objects as they arrive.
  Supplying a callback switches the component into streaming mode.

**Returns:**

- <code>dict\[str, list\[ChatMessage\]\]</code> – A dictionary with the following keys:
- `replies`: A list of ChatMessages containing the model's response

## haystack_integrations.components.generators.ollama.generator

### OllamaGenerator

Provides an interface to generate text using an LLM running on Ollama.

Usage example:

```python
from haystack_integrations.components.generators.ollama import OllamaGenerator

generator = OllamaGenerator(model="zephyr",
                            url = "http://localhost:11434",
                            generation_kwargs={
                            "num_predict": 100,
                            "temperature": 0.9,
                            })

print(generator.run("Who is the best American actor?"))
```

#### __init__

```python
__init__(
    model: str = "orca-mini",
    url: str = "http://localhost:11434",
    generation_kwargs: dict[str, Any] | None = None,
    system_prompt: str | None = None,
    template: str | None = None,
    raw: bool = False,
    timeout: int = 120,
    keep_alive: float | str | None = None,
    streaming_callback: Callable[[StreamingChunk], None] | None = None,
)
```

**Parameters:**

- **model** (<code>str</code>) – The name of the model to use. The model should be available in the running Ollama instance.
- **url** (<code>str</code>) – The URL of a running Ollama instance.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Optional arguments to pass to the Ollama generation endpoint, such as temperature,
  top_p, and others. See the available arguments in
  [Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).
- **system_prompt** (<code>str | None</code>) – Optional system message (overrides what is defined in the Ollama Modelfile).
- **template** (<code>str | None</code>) – The full prompt template (overrides what is defined in the Ollama Modelfile).
- **raw** (<code>bool</code>) – If True, no formatting will be applied to the prompt. You may choose to use the raw parameter
  if you are specifying a full templated prompt in your API request.
- **timeout** (<code>int</code>) – The number of seconds before throwing a timeout error from the Ollama API.
- **streaming_callback** (<code>Callable\\[[StreamingChunk\], None\] | None</code>) – A callback function that is called when a new token is received from the stream.
  The callback function accepts StreamingChunk as an argument.
- **keep_alive** (<code>float | str | None</code>) – The option that controls how long the model will stay loaded into memory following the request.
  If not set, it will use the default value from the Ollama (5 minutes).
  The value can be set to:
- a duration string (such as "10m" or "24h")
- a number in seconds (such as 3600)
- any negative number which will keep the model loaded in memory (e.g. -1 or "-1m")
- '0' which will unload the model immediately after generating a response.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> OllamaGenerator
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>OllamaGenerator</code> – Deserialized component.

#### run

```python
run(
    prompt: str,
    generation_kwargs: dict[str, Any] | None = None,
    *,
    streaming_callback: Callable[[StreamingChunk], None] | None = None
) -> dict[str, list[Any]]
```

Runs an Ollama Model on the given prompt.

**Parameters:**

- **prompt** (<code>str</code>) – The prompt to generate a response for.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Optional arguments to pass to the Ollama generation endpoint, such as temperature,
  top_p, and others. See the available arguments in
  [Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).
- **streaming_callback** (<code>Callable\\[[StreamingChunk\], None\] | None</code>) – A callback function that is called when a new token is received from the stream.

**Returns:**

- <code>dict\[str, list\[Any\]\]</code> – A dictionary with the following keys:
- `replies`: The responses from the model
- `meta`: The metadata collected during the run
