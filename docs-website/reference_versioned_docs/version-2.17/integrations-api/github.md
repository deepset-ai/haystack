---
title: "HanLP"
id: integrations-hanlp
description: "HanLP integration for Haystack"
slug: "/integrations-hanlp"
---

<a id="haystack_integrations.components.preprocessors.hanlp.chinese_document_splitter"></a>

## Module haystack\_integrations.components.preprocessors.hanlp.chinese\_document\_splitter

<a id="haystack_integrations.components.preprocessors.hanlp.chinese_document_splitter.ChineseDocumentSplitter"></a>

### ChineseDocumentSplitter

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
splitter.warm_up()
result = splitter.run(documents=[doc])
print(result["documents"])
```

<a id="haystack_integrations.components.preprocessors.hanlp.chinese_document_splitter.ChineseDocumentSplitter.__init__"></a>

#### ChineseDocumentSplitter.\_\_init\_\_

```python
def __init__(split_by: Literal["word", "sentence", "passage", "page", "line",
                               "period", "function"] = "word",
             split_length: int = 1000,
             split_overlap: int = 200,
             split_threshold: int = 0,
             respect_sentence_boundary: bool = False,
             splitting_function: Optional[Callable] = None,
             granularity: Literal["coarse", "fine"] = "coarse")
```

Initialize the ChineseDocumentSplitter component.

**Arguments**:

- `split_by`: The unit for splitting your documents. Choose from:
- `word` for splitting by spaces (" ")
- `period` for splitting by periods (".")
- `page` for splitting by form feed ("\f")
- `passage` for splitting by double line breaks ("\n\n")
- `line` for splitting each line ("\n")
- `sentence` for splitting by HanLP sentence tokenizer
- `split_length`: The maximum number of units in each split.
- `split_overlap`: The number of overlapping units for each split.
- `split_threshold`: The minimum number of units per split. If a split has fewer units
than the threshold, it's attached to the previous split.
- `respect_sentence_boundary`: Choose whether to respect sentence boundaries when splitting by "word".
If True, uses HanLP to detect sentence boundaries, ensuring splits occur only between sentences.
- `splitting_function`: Necessary when `split_by` is set to "function".
This is a function which must accept a single `str` as input and return a `list` of `str` as output,
representing the chunks after splitting.
- `granularity`: The granularity of Chinese word segmentation, either 'coarse' or 'fine'.

**Raises**:

- `ValueError`: If the granularity is not 'coarse' or 'fine'.

<a id="haystack_integrations.components.preprocessors.hanlp.chinese_document_splitter.ChineseDocumentSplitter.run"></a>

#### ChineseDocumentSplitter.run

```python
@component.output_types(documents=list[Document])
def run(documents: List[Document]) -> Dict[str, List[Document]]
```

Split documents into smaller chunks.

**Arguments**:

- `documents`: The documents to split.

**Raises**:

- `RuntimeError`: If the Chinese word segmentation model is not loaded.

**Returns**:

A dictionary containing the split documents.

<a id="haystack_integrations.components.preprocessors.hanlp.chinese_document_splitter.ChineseDocumentSplitter.warm_up"></a>

#### ChineseDocumentSplitter.warm\_up

```python
def warm_up() -> None
```

Warm up the component by loading the necessary models.

<a id="haystack_integrations.components.preprocessors.hanlp.chinese_document_splitter.ChineseDocumentSplitter.chinese_sentence_split"></a>

#### ChineseDocumentSplitter.chinese\_sentence\_split

```python
def chinese_sentence_split(text: str) -> List[Dict[str, Any]]
```

Split Chinese text into sentences.

**Arguments**:

- `text`: The text to split.

**Returns**:

A list of split sentences.

<a id="haystack_integrations.components.preprocessors.hanlp.chinese_document_splitter.ChineseDocumentSplitter.to_dict"></a>

#### ChineseDocumentSplitter.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

<a id="haystack_integrations.components.preprocessors.hanlp.chinese_document_splitter.ChineseDocumentSplitter.from_dict"></a>

#### ChineseDocumentSplitter.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "ChineseDocumentSplitter"
```

Deserializes the component from a dictionary.
