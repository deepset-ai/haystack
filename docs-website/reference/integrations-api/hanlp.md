---
title: "HanLP"
id: integrations-hanlp
description: "HanLP integration for Haystack"
slug: "/integrations-hanlp"
---


## `haystack_integrations.components.preprocessors.hanlp.chinese_document_splitter`

### `ChineseDocumentSplitter`

A DocumentSplitter for Chinese text.

'coarse' represents coarse granularity Chinese word segmentation, 'fine' represents fine granularity word
segmentation, default is coarse granularity word segmentation.

Unlike English where words are usually separated by spaces,
Chinese text is written continuously without spaces between words.
Chinese words can consist of multiple characters.
For example, the English word "America" is translated to "美国" in Chinese,
which consists of two characters but is treated as a single word.
Similarly, "Portugal" is "葡萄牙" in Chinese, three characters but one word.
Therefore, splitting by word means splitting by these multi-character tokens,
not simply by single characters or spaces.

### Usage example

```python
doc = Document(content=
    "这是第一句话，这是第二句话，这是第三句话。"
    "这是第四句话，这是第五句话，这是第六句话！"
    "这是第七句话，这是第八句话，这是第九句话？"
)

splitter = ChineseDocumentSplitter(
    split_by="word", split_length=10, split_overlap=3, respect_sentence_boundary=True
)
result = splitter.run(documents=[doc])
print(result["documents"])
```

#### `__init__`

```python
__init__(
    split_by: Literal[
        "word", "sentence", "passage", "page", "line", "period", "function"
    ] = "word",
    split_length: int = 1000,
    split_overlap: int = 200,
    split_threshold: int = 0,
    respect_sentence_boundary: bool = False,
    splitting_function: Callable | None = None,
    granularity: Literal["coarse", "fine"] = "coarse",
) -> None
```

Initialize the ChineseDocumentSplitter component.

**Parameters:**

- **split_by** (<code>Literal['word', 'sentence', 'passage', 'page', 'line', 'period', 'function']</code>) – The unit for splitting your documents. Choose from:
- `word` for splitting by spaces (" ")
- `period` for splitting by periods (".")
- `page` for splitting by form feed ("\\f")
- `passage` for splitting by double line breaks ("\\n\\n")
- `line` for splitting each line ("\\n")
- `sentence` for splitting by HanLP sentence tokenizer
- **split_length** (<code>int</code>) – The maximum number of units in each split.
- **split_overlap** (<code>int</code>) – The number of overlapping units for each split.
- **split_threshold** (<code>int</code>) – The minimum number of units per split. If a split has fewer units
  than the threshold, it's attached to the previous split.
- **respect_sentence_boundary** (<code>bool</code>) – Choose whether to respect sentence boundaries when splitting by "word".
  If True, uses HanLP to detect sentence boundaries, ensuring splits occur only between sentences.
- **splitting_function** (<code>Callable | None</code>) – Necessary when `split_by` is set to "function".
  This is a function which must accept a single `str` as input and return a `list` of `str` as output,
  representing the chunks after splitting.
- **granularity** (<code>Literal['coarse', 'fine']</code>) – The granularity of Chinese word segmentation, either 'coarse' or 'fine'.

**Raises:**

- <code>ValueError</code> – If the granularity is not 'coarse' or 'fine'.

#### `run`

```python
run(documents: list[Document]) -> dict[str, list[Document]]
```

Split documents into smaller chunks.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – The documents to split.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary containing the split documents.

**Raises:**

- <code>RuntimeError</code> – If the Chinese word segmentation model is not loaded.

#### `warm_up`

```python
warm_up() -> None
```

Warm up the component by loading the necessary models.

#### `chinese_sentence_split`

```python
chinese_sentence_split(text: str) -> list[dict[str, Any]]
```

Split Chinese text into sentences.

**Parameters:**

- **text** (<code>str</code>) – The text to split.

**Returns:**

- <code>list\[dict\[str, Any\]\]</code> – A list of split sentences.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> ChineseDocumentSplitter
```

Deserializes the component from a dictionary.
