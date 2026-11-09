---
title: "TwelveLabs"
id: integrations-twelvelabs
description: "TwelveLabs integration for Haystack"
slug: "/integrations-twelvelabs"
---


## haystack_integrations.components.converters.twelvelabs.video_converter

### TwelveLabsVideoConverter

Converts videos to Haystack Documents using TwelveLabs Pegasus.

Pegasus is a video-language model that analyzes a video on the fly (its
visuals **and** its own audio ASR) and returns text. Each source video
becomes one Document whose content is Pegasus's analysis (e.g. a description
plus a transcript) — no frame extraction or separate transcription step.

Sources may be publicly accessible direct video URLs or local file paths
(uploaded to TwelveLabs, up to 200 MB).

### Usage example

```python
from haystack_integrations.components.converters.twelvelabs import TwelveLabsVideoConverter

# Set the TWELVELABS_API_KEY environment variable
converter = TwelveLabsVideoConverter()
result = converter.run(sources=["https://example.com/clip.mp4"])
print(result["documents"][0].content)
```

#### __init__

```python
__init__(
    *,
    api_key: Secret = Secret.from_env_var("TWELVELABS_API_KEY"),
    model: str = DEFAULT_MODEL,
    prompt: str = DEFAULT_PROMPT,
    temperature: float = 0.2,
    max_tokens: int = 16384
) -> None
```

Create a TwelveLabsVideoConverter.

**Parameters:**

- **api_key** (<code>Secret</code>) – The TwelveLabs API key. Read from the `TWELVELABS_API_KEY`
  environment variable by default.
- **model** (<code>str</code>) – The Pegasus model name (`pegasus1.5` or `pegasus1.2`).
- **prompt** (<code>str</code>) – The analysis prompt sent to Pegasus for each video.
- **temperature** (<code>float</code>) – Sampling temperature (0-1).
- **max_tokens** (<code>int</code>) – Maximum output tokens per analysis.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> TwelveLabsVideoConverter
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>TwelveLabsVideoConverter</code> – Deserialized component.

#### run

```python
run(
    sources: list[str],
    meta: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> dict[str, list[Document]]
```

Convert videos to Documents with Pegasus.

**Parameters:**

- **sources** (<code>list\[str\]</code>) – Video sources — publicly accessible direct video URLs or
  local file paths.
- **meta** (<code>dict\[str, Any\] | list\[dict\[str, Any\]\] | None</code>) – Optional metadata to attach to the produced Documents. Either
  a single dict applied to all, or a list aligned with `sources`.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with key `documents`: the produced Documents.

## haystack_integrations.components.embedders.twelvelabs.document_embedder

### TwelveLabsDocumentEmbedder

Embeds the text content of Documents using TwelveLabs Marengo.

Computes a Marengo embedding for each Document's `content` and stores it on
`Document.embedding`. Because Marengo embeds text, images, audio, and video
into one shared space, these embeddings support cross-modal retrieval.

### Usage example

```python
from haystack import Document
from haystack_integrations.components.embedders.twelvelabs import TwelveLabsDocumentEmbedder

# Set the TWELVELABS_API_KEY environment variable
doc_embedder = TwelveLabsDocumentEmbedder()
docs = [Document(content="a cat playing piano")]
docs = doc_embedder.run(documents=docs)["documents"]
print(docs[0].embedding)
```

#### __init__

```python
__init__(
    *,
    api_key: Secret = Secret.from_env_var("TWELVELABS_API_KEY"),
    model: str = DEFAULT_MODEL,
    prefix: str = "",
    suffix: str = "",
    batch_size: int = 32,
    progress_bar: bool = True,
    meta_fields_to_embed: list[str] | None = None,
    embedding_separator: str = "\n"
) -> None
```

Create a TwelveLabsDocumentEmbedder.

**Parameters:**

- **api_key** (<code>Secret</code>) – The TwelveLabs API key. Read from the `TWELVELABS_API_KEY`
  environment variable by default.
- **model** (<code>str</code>) – The Marengo model name.
- **prefix** (<code>str</code>) – A string to add to the beginning of each text before embedding.
- **suffix** (<code>str</code>) – A string to add to the end of each text before embedding.
- **batch_size** (<code>int</code>) – Number of Documents per batch; within a batch `run_async` embeds concurrently.
- **progress_bar** (<code>bool</code>) – Whether to show a progress bar while embedding. Can be helpful
  to disable in production deployments to keep the logs clean.
- **meta_fields_to_embed** (<code>list\[str\] | None</code>) – List of meta fields that should be embedded along with the Document text.
- **embedding_separator** (<code>str</code>) – Separator used to concatenate the meta fields to the Document text.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> TwelveLabsDocumentEmbedder
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>TwelveLabsDocumentEmbedder</code> – Deserialized component.

#### run

```python
run(documents: list[Document]) -> dict[str, Any]
```

Embed a list of Documents.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – The Documents to embed (their `content` is embedded).

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with keys:
- `documents`: New Documents that are copies of the inputs with `embedding` populated.
- `meta`: Metadata about the request (the model used).

**Raises:**

- <code>TypeError</code> – If the input is not a list of Documents.

#### run_async

```python
run_async(documents: list[Document]) -> dict[str, Any]
```

Asynchronously embed a list of Documents.

Documents within each batch of `batch_size` are embedded concurrently.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – The Documents to embed.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with keys `documents` (copies with `embedding` populated) and `meta`.

**Raises:**

- <code>TypeError</code> – If the input is not a list of Documents.

## haystack_integrations.components.embedders.twelvelabs.text_embedder

### TwelveLabsTextEmbedder

Embeds strings using TwelveLabs Marengo.

Marengo embeds text, images, audio, and video into a single shared vector
space, so embeddings from this component are directly comparable (cosine
similarity) with image/video embeddings from the same model — enabling
cross-modal retrieval. Use it to embed a query before searching a document
store populated with Marengo embeddings.

### Usage example

```python
from haystack_integrations.components.embedders.twelvelabs import TwelveLabsTextEmbedder

# Set the TWELVELABS_API_KEY environment variable
text_embedder = TwelveLabsTextEmbedder()
result = text_embedder.run(text="a cat playing piano")
print(result["embedding"])
```

#### __init__

```python
__init__(
    *,
    api_key: Secret = Secret.from_env_var("TWELVELABS_API_KEY"),
    model: str = DEFAULT_MODEL,
    prefix: str = "",
    suffix: str = ""
) -> None
```

Create a TwelveLabsTextEmbedder.

**Parameters:**

- **api_key** (<code>Secret</code>) – The TwelveLabs API key. Read from the `TWELVELABS_API_KEY`
  environment variable by default.
- **model** (<code>str</code>) – The Marengo model name.
- **prefix** (<code>str</code>) – A string to add to the beginning of the text before embedding.
- **suffix** (<code>str</code>) – A string to add to the end of the text before embedding.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> TwelveLabsTextEmbedder
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>TwelveLabsTextEmbedder</code> – Deserialized component.

#### run

```python
run(text: str) -> dict[str, Any]
```

Embed a single string.

**Parameters:**

- **text** (<code>str</code>) – The string to embed.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with keys:
- `embedding`: The embedding vector for the input string.
- `meta`: Metadata about the request (the model used).

**Raises:**

- <code>TypeError</code> – If the input is not a string.

#### run_async

```python
run_async(text: str) -> dict[str, Any]
```

Asynchronously embed a single string.

**Parameters:**

- **text** (<code>str</code>) – The string to embed.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with keys `embedding` and `meta`.

**Raises:**

- <code>TypeError</code> – If the input is not a string.
