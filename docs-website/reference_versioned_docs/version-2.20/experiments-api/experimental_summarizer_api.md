---
title: "Summarizers"
id: experimental-summarizers-api
description: "Components that summarize texts into concise versions."
slug: "/experimental-summarizers-api"
---

<a id="haystack_experimental.components.summarizers.llm_summarizer"></a>

## Module haystack\_experimental.components.summarizers.llm\_summarizer

<a id="haystack_experimental.components.summarizers.llm_summarizer.LLMSummarizer"></a>

### LLMSummarizer

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

<a id="haystack_experimental.components.summarizers.llm_summarizer.LLMSummarizer.__init__"></a>

#### LLMSummarizer.\_\_init\_\_

```python
def __init__(
        chat_generator: ChatGenerator,
        system_prompt: Optional[str] = "Rewrite this text in summarized form.",
        summary_detail: float = 0,
        minimum_chunk_size: Optional[int] = 500,
        chunk_delimiter: str = ".",
        summarize_recursively: bool = False,
        split_overlap: int = 0)
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
" uses paragraph-based splitting, defaults to "."
        :param summarize_recursively: Whether to use previous summaries as context, defaults to False.
        :param split_overlap: Number of tokens to overlap between consecutive chunks, defaults to 0.


<a id="haystack_experimental.components.summarizers.llm_summarizer.LLMSummarizer.warm_up"></a>

#### LLMSummarizer.warm\_up

```python
def warm_up()
```

Warm up the chat generator and document splitter components.

<a id="haystack_experimental.components.summarizers.llm_summarizer.LLMSummarizer.to_dict"></a>

#### LLMSummarizer.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_experimental.components.summarizers.llm_summarizer.LLMSummarizer.from_dict"></a>

#### LLMSummarizer.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "LLMSummarizer"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary with serialized data.

**Returns**:

An instance of the component.

<a id="haystack_experimental.components.summarizers.llm_summarizer.LLMSummarizer.num_tokens"></a>

#### LLMSummarizer.num\_tokens

```python
def num_tokens(text: str) -> int
```

Estimates the token count for a given text.

Uses the RecursiveDocumentSplitter's tokenization logic for consistency.

**Arguments**:

- `text`: The text to tokenize

**Returns**:

The estimated token count

<a id="haystack_experimental.components.summarizers.llm_summarizer.LLMSummarizer.summarize"></a>

#### LLMSummarizer.summarize

```python
def summarize(text: str,
              detail: float,
              minimum_chunk_size: int,
              summarize_recursively: bool = False) -> str
```

Summarizes text by splitting it into optimally-sized chunks and processing each with an LLM.

**Arguments**:

- `text`: Text to summarize
- `detail`: Detail level (0-1) where 0 is most concise and 1 is most detailed
- `minimum_chunk_size`: Minimum token count per chunk
- `summarize_recursively`: Whether to use previous summaries as context

**Raises**:

- `ValueError`: If detail is not between 0 and 1

**Returns**:

The textual content summarized by the LLM.

<a id="haystack_experimental.components.summarizers.llm_summarizer.LLMSummarizer.run"></a>

#### LLMSummarizer.run

```python
@component.output_types(summary=list[Document])
def run(*,
        documents: list[Document],
        detail: Optional[float] = None,
        minimum_chunk_size: Optional[int] = None,
        summarize_recursively: Optional[bool] = None,
        system_prompt: Optional[str] = None) -> dict[str, list[Document]]
```

Run the summarizer on a list of documents.

**Arguments**:

- `documents`: List of documents to summarize
- `detail`: The level of detail for the summary (0-1), defaults to 0 overwriting the component's default.
- `minimum_chunk_size`: The minimum token count per chunk, defaults to 500 overwriting the
component's default.
- `system_prompt`: If given it will overwrite prompt given at init time or the default one.
- `summarize_recursively`: Whether to use previous summaries as context, defaults to False overwriting the
component's default.

**Raises**:

- `RuntimeError`: If the component wasn't warmed up.

