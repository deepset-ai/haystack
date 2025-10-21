---
title: "Ollama"
id: integrations-ollama
description: "Ollama integration for Haystack"
slug: "/integrations-ollama"
---

<a id="haystack_integrations.components.generators.ollama.generator"></a>

# Module haystack\_integrations.components.generators.ollama.generator

<a id="haystack_integrations.components.generators.ollama.generator.OllamaGenerator"></a>

## OllamaGenerator

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

<a id="haystack_integrations.components.generators.ollama.generator.OllamaGenerator.__init__"></a>

#### OllamaGenerator.\_\_init\_\_

```python
def __init__(model: str = "orca-mini",
             url: str = "http://localhost:11434",
             generation_kwargs: Optional[Dict[str, Any]] = None,
             system_prompt: Optional[str] = None,
             template: Optional[str] = None,
             raw: bool = False,
             timeout: int = 120,
             keep_alive: Optional[Union[float, str]] = None,
             streaming_callback: Optional[Callable[[StreamingChunk],
                                                   None]] = None)
```

**Arguments**:

- `model`: The name of the model to use. The model should be available in the running Ollama instance.
- `url`: The URL of a running Ollama instance.
- `generation_kwargs`: Optional arguments to pass to the Ollama generation endpoint, such as temperature,
top_p, and others. See the available arguments in
[Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).
- `system_prompt`: Optional system message (overrides what is defined in the Ollama Modelfile).
- `template`: The full prompt template (overrides what is defined in the Ollama Modelfile).
- `raw`: If True, no formatting will be applied to the prompt. You may choose to use the raw parameter
if you are specifying a full templated prompt in your API request.
- `timeout`: The number of seconds before throwing a timeout error from the Ollama API.
- `streaming_callback`: A callback function that is called when a new token is received from the stream.
The callback function accepts StreamingChunk as an argument.
- `keep_alive`: The option that controls how long the model will stay loaded into memory following the request.
If not set, it will use the default value from the Ollama (5 minutes).
The value can be set to:
- a duration string (such as "10m" or "24h")
- a number in seconds (such as 3600)
- any negative number which will keep the model loaded in memory (e.g. -1 or "-1m")
- '0' which will unload the model immediately after generating a response.

<a id="haystack_integrations.components.generators.ollama.generator.OllamaGenerator.to_dict"></a>

#### OllamaGenerator.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.generators.ollama.generator.OllamaGenerator.from_dict"></a>

#### OllamaGenerator.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "OllamaGenerator"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.generators.ollama.generator.OllamaGenerator.run"></a>

#### OllamaGenerator.run

```python
@component.output_types(replies=List[str], meta=List[Dict[str, Any]])
def run(
    prompt: str,
    generation_kwargs: Optional[Dict[str, Any]] = None,
    *,
    streaming_callback: Optional[Callable[[StreamingChunk], None]] = None
) -> Dict[str, List[Any]]
```

Runs an Ollama Model on the given prompt.

**Arguments**:

- `prompt`: The prompt to generate a response for.
- `generation_kwargs`: Optional arguments to pass to the Ollama generation endpoint, such as temperature,
top_p, and others. See the available arguments in
[Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).
- `streaming_callback`: A callback function that is called when a new token is received from the stream.

**Returns**:

A dictionary with the following keys:
- `replies`: The responses from the model
- `meta`: The metadata collected during the run

<a id="haystack_integrations.components.generators.ollama.chat.chat_generator"></a>

# Module haystack\_integrations.components.generators.ollama.chat.chat\_generator

<a id="haystack_integrations.components.generators.ollama.chat.chat_generator.OllamaChatGenerator"></a>

## OllamaChatGenerator

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

<a id="haystack_integrations.components.generators.ollama.chat.chat_generator.OllamaChatGenerator.__init__"></a>

#### OllamaChatGenerator.\_\_init\_\_

```python
def __init__(model: str = "qwen3:0.6b",
             url: str = "http://localhost:11434",
             generation_kwargs: Optional[Dict[str, Any]] = None,
             timeout: int = 120,
             keep_alive: Optional[Union[float, str]] = None,
             streaming_callback: Optional[Callable[[StreamingChunk],
                                                   None]] = None,
             tools: Optional[Union[List[Tool], Toolset]] = None,
             response_format: Optional[Union[None, Literal["json"],
                                             JsonSchemaValue]] = None,
             think: Union[bool, Literal["low", "medium", "high"]] = False)
```

:param model:

The name of the model to use. The model must already be present (pulled) in the running Ollama instance.
:param url:
    The base URL of the Ollama server (default "http://localhost:11434").
:param generation_kwargs:
    Optional arguments to pass to the Ollama generation endpoint, such as temperature,
    top_p, and others. See the available arguments in
    [Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).
:param timeout:
    The number of seconds before throwing a timeout error from the Ollama API.
:param think
    If True, the model will "think" before producing a response.
    Only [thinking models](https://ollama.com/search?c=thinking) support this feature.
    Some models like gpt-oss support different levels of thinking: "low", "medium", "high".
    The intermediate "thinking" output can be found by inspecting the `reasoning` property of the returned
    `ChatMessage`.
:param keep_alive:
    The option that controls how long the model will stay loaded into memory following the request.
    If not set, it will use the default value from the Ollama (5 minutes).
    The value can be set to:
    - a duration string (such as "10m" or "24h")
    - a number in seconds (such as 3600)
    - any negative number which will keep the model loaded in memory (e.g. -1 or "-1m")
    - '0' which will unload the model immediately after generating a response.
:param streaming_callback:
    A callback function that is called when a new token is received from the stream.
    The callback function accepts StreamingChunk as an argument.
:param tools:
    A list of `haystack.tools.Tool` or a `haystack.tools.Toolset`. Duplicate tool names raise a `ValueError`.
    Not all models support tools. For a list of models compatible with tools, see the
    [models page](https://ollama.com/search?c=tools).
:param response_format:
    The format for structured model outputs. The value can be:
    - None: No specific structure or format is applied to the response. The response is returned as-is.
    - "json": The response is formatted as a JSON object.
    - JSON Schema: The response is formatted as a JSON object
        that adheres to the specified JSON Schema. (needs Ollama ≥ 0.1.34)


<a id="haystack_integrations.components.generators.ollama.chat.chat_generator.OllamaChatGenerator.to_dict"></a>

#### OllamaChatGenerator.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.generators.ollama.chat.chat_generator.OllamaChatGenerator.from_dict"></a>

#### OllamaChatGenerator.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "OllamaChatGenerator"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.generators.ollama.chat.chat_generator.OllamaChatGenerator.run"></a>

#### OllamaChatGenerator.run

```python
@component.output_types(replies=List[ChatMessage])
def run(
    messages: List[ChatMessage],
    generation_kwargs: Optional[Dict[str, Any]] = None,
    tools: Optional[Union[List[Tool], Toolset]] = None,
    *,
    streaming_callback: Optional[StreamingCallbackT] = None
) -> Dict[str, List[ChatMessage]]
```

Runs an Ollama Model on a given chat history.

**Arguments**:

- `messages`: A list of ChatMessage instances representing the input messages.
- `generation_kwargs`: Per-call overrides for Ollama inference options.
These are merged on top of the instance-level `generation_kwargs`.
Optional arguments to pass to the Ollama generation endpoint, such as temperature, top_p, etc. See the
[Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).
- `tools`: A list of tools or a Toolset for which the model can prepare calls. This parameter can accept either a
list of `Tool` objects or a `Toolset` instance. If set, it will override the `tools` parameter set
during component initialization.
- `streaming_callback`: A callable to receive `StreamingChunk` objects as they
arrive.  Supplying a callback (here or in the constructor) switches
the component into streaming mode.

**Returns**:

A dictionary with the following keys:
- `replies`: A list of ChatMessages containing the model's response

<a id="haystack_integrations.components.generators.ollama.chat.chat_generator.OllamaChatGenerator.run_async"></a>

#### OllamaChatGenerator.run\_async

```python
@component.output_types(replies=List[ChatMessage])
async def run_async(
    messages: List[ChatMessage],
    generation_kwargs: Optional[Dict[str, Any]] = None,
    tools: Optional[Union[List[Tool], Toolset]] = None,
    *,
    streaming_callback: Optional[StreamingCallbackT] = None
) -> Dict[str, List[ChatMessage]]
```

Async version of run. Runs an Ollama Model on a given chat history.

**Arguments**:

- `messages`: A list of ChatMessage instances representing the input messages.
- `generation_kwargs`: Per-call overrides for Ollama inference options.
These are merged on top of the instance-level `generation_kwargs`.
- `tools`: A list of tools or a Toolset for which the model can prepare calls.
If set, it will override the `tools` parameter set during component initialization.
- `streaming_callback`: A callable to receive `StreamingChunk` objects as they arrive.
Supplying a callback switches the component into streaming mode.

**Returns**:

A dictionary with the following keys:
- `replies`: A list of ChatMessages containing the model's response

<a id="haystack_integrations.components.embedders.ollama.document_embedder"></a>

# Module haystack\_integrations.components.embedders.ollama.document\_embedder

<a id="haystack_integrations.components.embedders.ollama.document_embedder.OllamaDocumentEmbedder"></a>

## OllamaDocumentEmbedder

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

<a id="haystack_integrations.components.embedders.ollama.document_embedder.OllamaDocumentEmbedder.__init__"></a>

#### OllamaDocumentEmbedder.\_\_init\_\_

```python
def __init__(model: str = "nomic-embed-text",
             url: str = "http://localhost:11434",
             generation_kwargs: Optional[Dict[str, Any]] = None,
             timeout: int = 120,
             keep_alive: Optional[Union[float, str]] = None,
             prefix: str = "",
             suffix: str = "",
             progress_bar: bool = True,
             meta_fields_to_embed: Optional[List[str]] = None,
             embedding_separator: str = "\n",
             batch_size: int = 32)
```

**Arguments**:

- `model`: The name of the model to use. The model should be available in the running Ollama instance.
- `url`: The URL of a running Ollama instance.
- `generation_kwargs`: Optional arguments to pass to the Ollama generation endpoint, such as temperature, top_p, and others.
See the available arguments in
[Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).
- `timeout`: The number of seconds before throwing a timeout error from the Ollama API.
- `keep_alive`: The option that controls how long the model will stay loaded into memory following the request.
If not set, it will use the default value from the Ollama (5 minutes).
The value can be set to:
- a duration string (such as "10m" or "24h")
- a number in seconds (such as 3600)
- any negative number which will keep the model loaded in memory (e.g. -1 or "-1m")
- '0' which will unload the model immediately after generating a response.
- `prefix`: A string to add at the beginning of each text.
- `suffix`: A string to add at the end of each text.
- `progress_bar`: If `True`, shows a progress bar when running.
- `meta_fields_to_embed`: List of metadata fields to embed along with the document text.
- `embedding_separator`: Separator used to concatenate the metadata fields to the document text.
- `batch_size`: Number of documents to process at once.

<a id="haystack_integrations.components.embedders.ollama.document_embedder.OllamaDocumentEmbedder.run"></a>

#### OllamaDocumentEmbedder.run

```python
@component.output_types(documents=List[Document], meta=Dict[str, Any])
def run(
    documents: List[Document],
    generation_kwargs: Optional[Dict[str, Any]] = None
) -> Dict[str, Union[List[Document], Dict[str, Any]]]
```

Runs an Ollama Model to compute embeddings of the provided documents.

**Arguments**:

- `documents`: Documents to be converted to an embedding.
- `generation_kwargs`: Optional arguments to pass to the Ollama generation endpoint, such as temperature,
top_p, etc. See the
[Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).

**Returns**:

A dictionary with the following keys:
- `documents`: Documents with embedding information attached
- `meta`: The metadata collected during the embedding process

<a id="haystack_integrations.components.embedders.ollama.document_embedder.OllamaDocumentEmbedder.run_async"></a>

#### OllamaDocumentEmbedder.run\_async

```python
@component.output_types(documents=List[Document], meta=Dict[str, Any])
async def run_async(
    documents: List[Document],
    generation_kwargs: Optional[Dict[str, Any]] = None
) -> Dict[str, Union[List[Document], Dict[str, Any]]]
```

Asynchronously run an Ollama Model to compute embeddings of the provided documents.

**Arguments**:

- `documents`: Documents to be converted to an embedding.
- `generation_kwargs`: Optional arguments to pass to the Ollama generation endpoint, such as temperature,
top_p, etc. See the
[Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).

**Returns**:

A dictionary with the following keys:
- `documents`: Documents with embedding information attached
- `meta`: The metadata collected during the embedding process

<a id="haystack_integrations.components.embedders.ollama.text_embedder"></a>

# Module haystack\_integrations.components.embedders.ollama.text\_embedder

<a id="haystack_integrations.components.embedders.ollama.text_embedder.OllamaTextEmbedder"></a>

## OllamaTextEmbedder

Computes the embeddings of a list of Documents and stores the obtained vectors in the embedding field of
each Document. It uses embedding models compatible with the Ollama Library.

Usage example:
```python
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder

embedder = OllamaTextEmbedder()
result = embedder.run(text="What do llamas say once you have thanked them? No probllama!")
print(result['embedding'])
```

<a id="haystack_integrations.components.embedders.ollama.text_embedder.OllamaTextEmbedder.__init__"></a>

#### OllamaTextEmbedder.\_\_init\_\_

```python
def __init__(model: str = "nomic-embed-text",
             url: str = "http://localhost:11434",
             generation_kwargs: Optional[Dict[str, Any]] = None,
             timeout: int = 120,
             keep_alive: Optional[Union[float, str]] = None)
```

**Arguments**:

- `model`: The name of the model to use. The model should be available in the running Ollama instance.
- `url`: The URL of a running Ollama instance.
- `generation_kwargs`: Optional arguments to pass to the Ollama generation endpoint, such as temperature,
top_p, and others. See the available arguments in
[Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).
- `timeout`: The number of seconds before throwing a timeout error from the Ollama API.
- `keep_alive`: The option that controls how long the model will stay loaded into memory following the request.
If not set, it will use the default value from the Ollama (5 minutes).
The value can be set to:
- a duration string (such as "10m" or "24h")
- a number in seconds (such as 3600)
- any negative number which will keep the model loaded in memory (e.g. -1 or "-1m")
- '0' which will unload the model immediately after generating a response.

<a id="haystack_integrations.components.embedders.ollama.text_embedder.OllamaTextEmbedder.run"></a>

#### OllamaTextEmbedder.run

```python
@component.output_types(embedding=List[float], meta=Dict[str, Any])
def run(
    text: str,
    generation_kwargs: Optional[Dict[str, Any]] = None
) -> Dict[str, Union[List[float], Dict[str, Any]]]
```

Runs an Ollama Model to compute embeddings of the provided text.

**Arguments**:

- `text`: Text to be converted to an embedding.
- `generation_kwargs`: Optional arguments to pass to the Ollama generation endpoint, such as temperature,
top_p, etc. See the
[Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).

**Returns**:

A dictionary with the following keys:
- `embedding`: The computed embeddings
- `meta`: The metadata collected during the embedding process

<a id="haystack_integrations.components.embedders.ollama.text_embedder.OllamaTextEmbedder.run_async"></a>

#### OllamaTextEmbedder.run\_async

```python
@component.output_types(embedding=List[float], meta=Dict[str, Any])
async def run_async(
    text: str,
    generation_kwargs: Optional[Dict[str, Any]] = None
) -> Dict[str, Union[List[float], Dict[str, Any]]]
```

Asynchronously run an Ollama Model to compute embeddings of the provided text.

**Arguments**:

- `text`: Text to be converted to an embedding.
- `generation_kwargs`: Optional arguments to pass to the Ollama generation endpoint, such as temperature,
top_p, etc. See the
[Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).

**Returns**:

A dictionary with the following keys:
- `embedding`: The computed embeddings
- `meta`: The metadata collected during the embedding process
