---
title: "Summarizers"
id: experimental-summarizers-api
description: "Components that summarize texts into concise versions."
slug: "/experimental-summarizers-api"
---


## `haystack_experimental.components.summarizers.llm_summarizer`

### `LLMSummarizer`

Summarizes text using a language model.

It's inspired by code from the OpenAI blog post: https://cookbook.openai.com/examples/summarizing_long_documents

Example

```python
from haystack_experimental.components.summarizers.summarizer import Summarizer
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack import Document

text = ("Machine learning is a subset of artificial intelligence that provides systems "
        "the ability to automatically learn and improve from experience without being "
        "explicitly programmed. The process of learning begins with observations or data. "
        "Supervised learning algorithms build a mathematical model of sample data, known as "
        "training data, in order to make predictions or decisions. Unsupervised learning "
        "algorithms take a set of data that contains only inputs and find structure in the data. "
        "Reinforcement learning is an area of machine learning where an agent learns to behave "
        "in an environment by performing actions and seeing the results. Deep learning uses "
        "artificial neural networks to model complex patterns in data. Neural networks consist "
        "of layers of connected nodes, each performing a simple computation.")

doc = Document(content=text)
chat_generator = OpenAIChatGenerator(model="gpt-4")
summarizer = Summarizer(chat_generator=chat_generator)
summarizer.run(documents=[doc])
```

#### `__init__`

```python
__init__(
    chat_generator: ChatGenerator,
    system_prompt: str | None = "Rewrite this text in summarized form.",
    summary_detail: float = 0,
    minimum_chunk_size: int | None = 500,
    chunk_delimiter: str = ".",
    summarize_recursively: bool = False,
    split_overlap: int = 0,
)
```

```
    Initialize the Summarizer component.

    :param chat_generator: A ChatGenerator instance to use for summarization.
    :param system_prompt: The prompt to instruct the LLM to summarise text, if not given defaults to:
        "Rewrite this text in summarized form."
    :param summary_detail: The level of detail for the summary (0-1), defaults to 0.
        This parameter controls the trade-off between conciseness and completeness by adjusting how many
        chunks the text is divided into. At detail=0, the text is processed as a single chunk (or very few
        chunks), producing the most concise summary. At detail=1, the text is split into the maximum number
        of chunks allowed by minimum_chunk_size, enabling more granular analysis and detailed summaries.
        The formula uses linear interpolation: num_chunks = 1 + detail * (max_chunks - 1), where max_chunks
        is determined by dividing the document length by minimum_chunk_size.
    :param minimum_chunk_size: The minimum token count per chunk, defaults to 500
    :param chunk_delimiter: The character used to determine separator priority.
        "." uses sentence-based splitting, "
```

" uses paragraph-based splitting, defaults to "."
:param summarize_recursively: Whether to use previous summaries as context, defaults to False.
:param split_overlap: Number of tokens to overlap between consecutive chunks, defaults to 0.

#### `warm_up`

```python
warm_up()
```

Warm up the chat generator and document splitter components.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> LLMSummarizer
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary with serialized data.

**Returns:**

- <code>LLMSummarizer</code> – An instance of the component.

#### `num_tokens`

```python
num_tokens(text: str) -> int
```

Estimates the token count for a given text.

Uses the RecursiveDocumentSplitter's tokenization logic for consistency.

**Parameters:**

- **text** (<code>str</code>) – The text to tokenize

**Returns:**

- <code>int</code> – The estimated token count

#### `summarize`

```python
summarize(
    text: str,
    detail: float,
    minimum_chunk_size: int,
    summarize_recursively: bool = False,
) -> str
```

Summarizes text by splitting it into optimally-sized chunks and processing each with an LLM.

**Parameters:**

- **text** (<code>str</code>) – Text to summarize
- **detail** (<code>float</code>) – Detail level (0-1) where 0 is most concise and 1 is most detailed
- **minimum_chunk_size** (<code>int</code>) – Minimum token count per chunk
- **summarize_recursively** (<code>bool</code>) – Whether to use previous summaries as context

**Returns:**

- <code>str</code> – The textual content summarized by the LLM.

**Raises:**

- <code>ValueError</code> – If detail is not between 0 and 1

#### `run`

```python
run(
    *,
    documents: list[Document],
    detail: float | None = None,
    minimum_chunk_size: int | None = None,
    summarize_recursively: bool | None = None,
    system_prompt: str | None = None
) -> dict[str, list[Document]]
```

Run the summarizer on a list of documents.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – List of documents to summarize
- **detail** (<code>float | None</code>) – The level of detail for the summary (0-1), defaults to 0 overwriting the component's default.
- **minimum_chunk_size** (<code>int | None</code>) – The minimum token count per chunk, defaults to 500 overwriting the
  component's default.
- **system_prompt** (<code>str | None</code>) – If given it will overwrite prompt given at init time or the default one.
- **summarize_recursively** (<code>bool | None</code>) – Whether to use previous summaries as context, defaults to False overwriting the
  component's default.

**Raises:**

- <code>RuntimeError</code> – If the component wasn't warmed up.
