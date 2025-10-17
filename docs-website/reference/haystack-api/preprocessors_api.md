---
title: "PreProcessors"
id: preprocessors-api
description: "Preprocess your Documents and texts. Clean, split, and more."
slug: "/preprocessors-api"
---

<a id="csv_document_cleaner"></a>

# Module csv\_document\_cleaner

<a id="csv_document_cleaner.CSVDocumentCleaner"></a>

## CSVDocumentCleaner

A component for cleaning CSV documents by removing empty rows and columns.

This component processes CSV content stored in Documents, allowing
for the optional ignoring of a specified number of rows and columns before performing
the cleaning operation. Additionally, it provides options to keep document IDs and
control whether empty rows and columns should be removed.

<a id="csv_document_cleaner.CSVDocumentCleaner.__init__"></a>

#### CSVDocumentCleaner.\_\_init\_\_

```python
def __init__(*,
             ignore_rows: int = 0,
             ignore_columns: int = 0,
             remove_empty_rows: bool = True,
             remove_empty_columns: bool = True,
             keep_id: bool = False) -> None
```

Initializes the CSVDocumentCleaner component.

**Arguments**:

- `ignore_rows`: Number of rows to ignore from the top of the CSV table before processing.
- `ignore_columns`: Number of columns to ignore from the left of the CSV table before processing.
- `remove_empty_rows`: Whether to remove rows that are entirely empty.
- `remove_empty_columns`: Whether to remove columns that are entirely empty.
- `keep_id`: Whether to retain the original document ID in the output document.
Rows and columns ignored using these parameters are preserved in the final output, meaning
they are not considered when removing empty rows and columns.

<a id="csv_document_cleaner.CSVDocumentCleaner.run"></a>

#### CSVDocumentCleaner.run

```python
@component.output_types(documents=list[Document])
def run(documents: list[Document]) -> dict[str, list[Document]]
```

Cleans CSV documents by removing empty rows and columns while preserving specified ignored rows and columns.

**Arguments**:

- `documents`: List of Documents containing CSV-formatted content.

**Returns**:

A dictionary with a list of cleaned Documents under the key "documents".
Processing steps:
1. Reads each document's content as a CSV table.
2. Retains the specified number of `ignore_rows` from the top and `ignore_columns` from the left.
3. Drops any rows and columns that are entirely empty (if enabled by `remove_empty_rows` and
    `remove_empty_columns`).
4. Reattaches the ignored rows and columns to maintain their original positions.
5. Returns the cleaned CSV content as a new `Document` object, with an option to retain the original
    document ID.

<a id="csv_document_splitter"></a>

# Module csv\_document\_splitter

<a id="csv_document_splitter.CSVDocumentSplitter"></a>

## CSVDocumentSplitter

A component for splitting CSV documents into sub-tables based on split arguments.

The splitter supports two modes of operation:
- identify consecutive empty rows or columns that exceed a given threshold
and uses them as delimiters to segment the document into smaller tables.
- split each row into a separate sub-table, represented as a Document.

<a id="csv_document_splitter.CSVDocumentSplitter.__init__"></a>

#### CSVDocumentSplitter.\_\_init\_\_

```python
def __init__(row_split_threshold: Optional[int] = 2,
             column_split_threshold: Optional[int] = 2,
             read_csv_kwargs: Optional[dict[str, Any]] = None,
             split_mode: SplitMode = "threshold") -> None
```

Initializes the CSVDocumentSplitter component.

**Arguments**:

- `row_split_threshold`: The minimum number of consecutive empty rows required to trigger a split.
- `column_split_threshold`: The minimum number of consecutive empty columns required to trigger a split.
- `read_csv_kwargs`: Additional keyword arguments to pass to `pandas.read_csv`.
By default, the component with options:
- `header=None`
- `skip_blank_lines=False` to preserve blank lines
- `dtype=object` to prevent type inference (e.g., converting numbers to floats).
See https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html for more information.
- `split_mode`: If `threshold`, the component will split the document based on the number of
consecutive empty rows or columns that exceed the `row_split_threshold` or `column_split_threshold`.
If `row-wise`, the component will split each row into a separate sub-table.

<a id="csv_document_splitter.CSVDocumentSplitter.run"></a>

#### CSVDocumentSplitter.run

```python
@component.output_types(documents=list[Document])
def run(documents: list[Document]) -> dict[str, list[Document]]
```

Processes and splits a list of CSV documents into multiple sub-tables.

**Splitting Process:**
1. Applies a row-based split if `row_split_threshold` is provided.
2. Applies a column-based split if `column_split_threshold` is provided.
3. If both thresholds are specified, performs a recursive split by rows first, then columns, ensuring
   further fragmentation of any sub-tables that still contain empty sections.
4. Sorts the resulting sub-tables based on their original positions within the document.

**Arguments**:

- `documents`: A list of Documents containing CSV-formatted content.
Each document is assumed to contain one or more tables separated by empty rows or columns.

**Returns**:

A dictionary with a key `"documents"`, mapping to a list of new `Document` objects,
each representing an extracted sub-table from the original CSV.
    The metadata of each document includes:
        - A field `source_id` to track the original document.
        - A field `row_idx_start` to indicate the starting row index of the sub-table in the original table.
        - A field `col_idx_start` to indicate the starting column index of the sub-table in the original table.
        - A field `split_id` to indicate the order of the split in the original document.
        - All other metadata copied from the original document.

- If a document cannot be processed, it is returned unchanged.
- The `meta` field from the original document is preserved in the split documents.

<a id="document_cleaner"></a>

# Module document\_cleaner

<a id="document_cleaner.DocumentCleaner"></a>

## DocumentCleaner

Cleans the text in the documents.

It removes extra whitespaces,
empty lines, specified substrings, regexes,
page headers and footers (in this order).

### Usage example:

```python
from haystack import Document
from haystack.components.preprocessors import DocumentCleaner

doc = Document(content="This   is  a  document  to  clean\n\n\nsubstring to remove")

cleaner = DocumentCleaner(remove_substrings = ["substring to remove"])
result = cleaner.run(documents=[doc])

assert result["documents"][0].content == "This is a document to clean "
```

<a id="document_cleaner.DocumentCleaner.__init__"></a>

#### DocumentCleaner.\_\_init\_\_

```python
def __init__(remove_empty_lines: bool = True,
             remove_extra_whitespaces: bool = True,
             remove_repeated_substrings: bool = False,
             keep_id: bool = False,
             remove_substrings: Optional[list[str]] = None,
             remove_regex: Optional[str] = None,
             unicode_normalization: Optional[Literal["NFC", "NFKC", "NFD",
                                                     "NFKD"]] = None,
             ascii_only: bool = False)
```

Initialize DocumentCleaner.

**Arguments**:

- `remove_empty_lines`: If `True`, removes empty lines.
- `remove_extra_whitespaces`: If `True`, removes extra whitespaces.
- `remove_repeated_substrings`: If `True`, removes repeated substrings (headers and footers) from pages.
Pages must be separated by a form feed character "\f",
which is supported by `TextFileToDocument` and `AzureOCRDocumentConverter`.
- `remove_substrings`: List of substrings to remove from the text.
- `remove_regex`: Regex to match and replace substrings by "".
- `keep_id`: If `True`, keeps the IDs of the original documents.
- `unicode_normalization`: Unicode normalization form to apply to the text.
Note: This will run before any other steps.
- `ascii_only`: Whether to convert the text to ASCII only.
Will remove accents from characters and replace them with ASCII characters.
Other non-ASCII characters will be removed.
Note: This will run before any pattern matching or removal.

<a id="document_cleaner.DocumentCleaner.run"></a>

#### DocumentCleaner.run

```python
@component.output_types(documents=list[Document])
def run(documents: list[Document])
```

Cleans up the documents.

**Arguments**:

- `documents`: List of Documents to clean.

**Raises**:

- `TypeError`: if documents is not a list of Documents.

**Returns**:

A dictionary with the following key:
- `documents`: List of cleaned Documents.

<a id="document_preprocessor"></a>

# Module document\_preprocessor

<a id="document_preprocessor.DocumentPreprocessor"></a>

## DocumentPreprocessor

A SuperComponent that first splits and then cleans documents.

This component consists of a DocumentSplitter followed by a DocumentCleaner in a single pipeline.
It takes a list of documents as input and returns a processed list of documents.

Usage example:
```python
from haystack import Document
from haystack.components.preprocessors import DocumentPreprocessor

doc = Document(content="I love pizza!")
preprocessor = DocumentPreprocessor()
result = preprocessor.run(documents=[doc])
print(result["documents"])
```

<a id="document_preprocessor.DocumentPreprocessor.__init__"></a>

#### DocumentPreprocessor.\_\_init\_\_

```python
def __init__(*,
             split_by: Literal["function", "page", "passage", "period", "word",
                               "line", "sentence"] = "word",
             split_length: int = 250,
             split_overlap: int = 0,
             split_threshold: int = 0,
             splitting_function: Optional[Callable[[str], list[str]]] = None,
             respect_sentence_boundary: bool = False,
             language: Language = "en",
             use_split_rules: bool = True,
             extend_abbreviations: bool = True,
             remove_empty_lines: bool = True,
             remove_extra_whitespaces: bool = True,
             remove_repeated_substrings: bool = False,
             keep_id: bool = False,
             remove_substrings: Optional[list[str]] = None,
             remove_regex: Optional[str] = None,
             unicode_normalization: Optional[Literal["NFC", "NFKC", "NFD",
                                                     "NFKD"]] = None,
             ascii_only: bool = False) -> None
```

Initialize a DocumentPreProcessor that first splits and then cleans documents.

**Splitter Parameters**:

**Arguments**:

- `split_by`: The unit of splitting: "function", "page", "passage", "period", "word", "line", or "sentence".
- `split_length`: The maximum number of units (words, lines, pages, and so on) in each split.
- `split_overlap`: The number of overlapping units between consecutive splits.
- `split_threshold`: The minimum number of units per split. If a split is smaller than this, it's merged
with the previous split.
- `splitting_function`: A custom function for splitting if `split_by="function"`.
- `respect_sentence_boundary`: If `True`, splits by words but tries not to break inside a sentence.
- `language`: Language used by the sentence tokenizer if `split_by="sentence"` or
`respect_sentence_boundary=True`.
- `use_split_rules`: Whether to apply additional splitting heuristics for the sentence splitter.
- `extend_abbreviations`: Whether to extend the sentence splitter with curated abbreviations for certain
languages.

**Cleaner Parameters**:
- `remove_empty_lines`: If `True`, removes empty lines.
- `remove_extra_whitespaces`: If `True`, removes extra whitespaces.
- `remove_repeated_substrings`: If `True`, removes repeated substrings like headers/footers across pages.
- `keep_id`: If `True`, keeps the original document IDs.
- `remove_substrings`: A list of strings to remove from the document content.
- `remove_regex`: A regex pattern whose matches will be removed from the document content.
- `unicode_normalization`: Unicode normalization form to apply to the text, for example `"NFC"`.
- `ascii_only`: If `True`, converts text to ASCII only.

<a id="document_preprocessor.DocumentPreprocessor.to_dict"></a>

#### DocumentPreprocessor.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serialize SuperComponent to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="document_preprocessor.DocumentPreprocessor.from_dict"></a>

#### DocumentPreprocessor.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "DocumentPreprocessor"
```

Deserializes the SuperComponent from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized SuperComponent.

<a id="document_splitter"></a>

# Module document\_splitter

<a id="document_splitter.DocumentSplitter"></a>

## DocumentSplitter

Splits long documents into smaller chunks.

This is a common preprocessing step during indexing. It helps Embedders create meaningful semantic representations
and prevents exceeding language model context limits.

The DocumentSplitter is compatible with the following DocumentStores:
- [Astra](https://docs.haystack.deepset.ai/docs/astradocumentstore)
- [Chroma](https://docs.haystack.deepset.ai/docs/chromadocumentstore) limited support, overlapping information is
  not stored
- [Elasticsearch](https://docs.haystack.deepset.ai/docs/elasticsearch-document-store)
- [OpenSearch](https://docs.haystack.deepset.ai/docs/opensearch-document-store)
- [Pgvector](https://docs.haystack.deepset.ai/docs/pgvectordocumentstore)
- [Pinecone](https://docs.haystack.deepset.ai/docs/pinecone-document-store) limited support, overlapping
   information is not stored
- [Qdrant](https://docs.haystack.deepset.ai/docs/qdrant-document-store)
- [Weaviate](https://docs.haystack.deepset.ai/docs/weaviatedocumentstore)

### Usage example

```python
from haystack import Document
from haystack.components.preprocessors import DocumentSplitter

doc = Document(content="Moonlight shimmered softly, wolves howled nearby, night enveloped everything.")

splitter = DocumentSplitter(split_by="word", split_length=3, split_overlap=0)
result = splitter.run(documents=[doc])
```

<a id="document_splitter.DocumentSplitter.__init__"></a>

#### DocumentSplitter.\_\_init\_\_

```python
def __init__(split_by: Literal["function", "page", "passage", "period", "word",
                               "line", "sentence"] = "word",
             split_length: int = 200,
             split_overlap: int = 0,
             split_threshold: int = 0,
             splitting_function: Optional[Callable[[str], list[str]]] = None,
             respect_sentence_boundary: bool = False,
             language: Language = "en",
             use_split_rules: bool = True,
             extend_abbreviations: bool = True,
             *,
             skip_empty_documents: bool = True)
```

Initialize DocumentSplitter.

**Arguments**:

- `split_by`: The unit for splitting your documents. Choose from:
- `word` for splitting by spaces (" ")
- `period` for splitting by periods (".")
- `page` for splitting by form feed ("\f")
- `passage` for splitting by double line breaks ("\n\n")
- `line` for splitting each line ("\n")
- `sentence` for splitting by NLTK sentence tokenizer
- `split_length`: The maximum number of units in each split.
- `split_overlap`: The number of overlapping units for each split.
- `split_threshold`: The minimum number of units per split. If a split has fewer units
than the threshold, it's attached to the previous split.
- `splitting_function`: Necessary when `split_by` is set to "function".
This is a function which must accept a single `str` as input and return a `list` of `str` as output,
representing the chunks after splitting.
- `respect_sentence_boundary`: Choose whether to respect sentence boundaries when splitting by "word".
If True, uses NLTK to detect sentence boundaries, ensuring splits occur only between sentences.
- `language`: Choose the language for the NLTK tokenizer. The default is English ("en").
- `use_split_rules`: Choose whether to use additional split rules when splitting by `sentence`.
- `extend_abbreviations`: Choose whether to extend NLTK's PunktTokenizer abbreviations with a list
of curated abbreviations, if available. This is currently supported for English ("en") and German ("de").
- `skip_empty_documents`: Choose whether to skip documents with empty content. Default is True.
Set to False when downstream components in the Pipeline (like LLMDocumentContentExtractor) can extract text
from non-textual documents.

<a id="document_splitter.DocumentSplitter.warm_up"></a>

#### DocumentSplitter.warm\_up

```python
def warm_up()
```

Warm up the DocumentSplitter by loading the sentence tokenizer.

<a id="document_splitter.DocumentSplitter.run"></a>

#### DocumentSplitter.run

```python
@component.output_types(documents=list[Document])
def run(documents: list[Document])
```

Split documents into smaller parts.

Splits documents by the unit expressed in `split_by`, with a length of `split_length`
and an overlap of `split_overlap`.

**Arguments**:

- `documents`: The documents to split.

**Raises**:

- `TypeError`: if the input is not a list of Documents.
- `ValueError`: if the content of a document is None.

**Returns**:

A dictionary with the following key:
- `documents`: List of documents with the split texts. Each document includes:
- A metadata field `source_id` to track the original document.
- A metadata field `page_number` to track the original page number.
- All other metadata copied from the original document.

<a id="document_splitter.DocumentSplitter.to_dict"></a>

#### DocumentSplitter.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

<a id="document_splitter.DocumentSplitter.from_dict"></a>

#### DocumentSplitter.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "DocumentSplitter"
```

Deserializes the component from a dictionary.

<a id="hierarchical_document_splitter"></a>

# Module hierarchical\_document\_splitter

<a id="hierarchical_document_splitter.HierarchicalDocumentSplitter"></a>

## HierarchicalDocumentSplitter

Splits a documents into different block sizes building a hierarchical tree structure of blocks of different sizes.

The root node of the tree is the original document, the leaf nodes are the smallest blocks. The blocks in between
are connected such that the smaller blocks are children of the parent-larger blocks.

## Usage example
```python
from haystack import Document
from haystack.components.preprocessors import HierarchicalDocumentSplitter

doc = Document(content="This is a simple test document")
splitter = HierarchicalDocumentSplitter(block_sizes={3, 2}, split_overlap=0, split_by="word")
splitter.run([doc])
>> {'documents': [Document(id=3f7..., content: 'This is a simple test document', meta: {'block_size': 0, 'parent_id': None, 'children_ids': ['5ff..', '8dc..'], 'level': 0}),
>> Document(id=5ff.., content: 'This is a ', meta: {'block_size': 3, 'parent_id': '3f7..', 'children_ids': ['f19..', '52c..'], 'level': 1, 'source_id': '3f7..', 'page_number': 1, 'split_id': 0, 'split_idx_start': 0}),
>> Document(id=8dc.., content: 'simple test document', meta: {'block_size': 3, 'parent_id': '3f7..', 'children_ids': ['39d..', 'e23..'], 'level': 1, 'source_id': '3f7..', 'page_number': 1, 'split_id': 1, 'split_idx_start': 10}),
>> Document(id=f19.., content: 'This is ', meta: {'block_size': 2, 'parent_id': '5ff..', 'children_ids': [], 'level': 2, 'source_id': '5ff..', 'page_number': 1, 'split_id': 0, 'split_idx_start': 0}),
>> Document(id=52c.., content: 'a ', meta: {'block_size': 2, 'parent_id': '5ff..', 'children_ids': [], 'level': 2, 'source_id': '5ff..', 'page_number': 1, 'split_id': 1, 'split_idx_start': 8}),
>> Document(id=39d.., content: 'simple test ', meta: {'block_size': 2, 'parent_id': '8dc..', 'children_ids': [], 'level': 2, 'source_id': '8dc..', 'page_number': 1, 'split_id': 0, 'split_idx_start': 0}),
>> Document(id=e23.., content: 'document', meta: {'block_size': 2, 'parent_id': '8dc..', 'children_ids': [], 'level': 2, 'source_id': '8dc..', 'page_number': 1, 'split_id': 1, 'split_idx_start': 12})]}
```

<a id="hierarchical_document_splitter.HierarchicalDocumentSplitter.__init__"></a>

#### HierarchicalDocumentSplitter.\_\_init\_\_

```python
def __init__(block_sizes: set[int],
             split_overlap: int = 0,
             split_by: Literal["word", "sentence", "page",
                               "passage"] = "word")
```

Initialize HierarchicalDocumentSplitter.

**Arguments**:

- `block_sizes`: Set of block sizes to split the document into. The blocks are split in descending order.
- `split_overlap`: The number of overlapping units for each split.
- `split_by`: The unit for splitting your documents.

<a id="hierarchical_document_splitter.HierarchicalDocumentSplitter.run"></a>

#### HierarchicalDocumentSplitter.run

```python
@component.output_types(documents=list[Document])
def run(documents: list[Document])
```

Builds a hierarchical document structure for each document in a list of documents.

**Arguments**:

- `documents`: List of Documents to split into hierarchical blocks.

**Returns**:

List of HierarchicalDocument

<a id="hierarchical_document_splitter.HierarchicalDocumentSplitter.build_hierarchy_from_doc"></a>

#### HierarchicalDocumentSplitter.build\_hierarchy\_from\_doc

```python
def build_hierarchy_from_doc(document: Document) -> list[Document]
```

Build a hierarchical tree document structure from a single document.

Given a document, this function splits the document into hierarchical blocks of different sizes represented
as HierarchicalDocument objects.

**Arguments**:

- `document`: Document to split into hierarchical blocks.

**Returns**:

List of HierarchicalDocument

<a id="hierarchical_document_splitter.HierarchicalDocumentSplitter.to_dict"></a>

#### HierarchicalDocumentSplitter.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Returns a dictionary representation of the component.

**Returns**:

Serialized dictionary representation of the component.

<a id="hierarchical_document_splitter.HierarchicalDocumentSplitter.from_dict"></a>

#### HierarchicalDocumentSplitter.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "HierarchicalDocumentSplitter"
```

Deserialize this component from a dictionary.

**Arguments**:

- `data`: The dictionary to deserialize and create the component.

**Returns**:

The deserialized component.

<a id="recursive_splitter"></a>

# Module recursive\_splitter

<a id="recursive_splitter.RecursiveDocumentSplitter"></a>

## RecursiveDocumentSplitter

Recursively chunk text into smaller chunks.

This component is used to split text into smaller chunks, it does so by recursively applying a list of separators
to the text.

The separators are applied in the order they are provided, typically this is a list of separators that are
applied in a specific order, being the last separator the most specific one.

Each separator is applied to the text, it then checks each of the resulting chunks, it keeps the chunks that
are within the split_length, for the ones that are larger than the split_length, it applies the next separator in the
list to the remaining text.

This is done until all chunks are smaller than the split_length parameter.

**Example**:

  
```python
from haystack import Document
from haystack.components.preprocessors import RecursiveDocumentSplitter

chunker = RecursiveDocumentSplitter(split_length=260, split_overlap=0, separators=["\n\n", "\n", ".", " "])
text = ('''Artificial intelligence (AI) - Introduction

AI, in its broadest sense, is intelligence exhibited by machines, particularly computer systems.
AI technology is widely used throughout industry, government, and science. Some high-profile applications include advanced web search engines; recommendation systems; interacting via human speech; autonomous vehicles; generative and creative tools; and superhuman play and analysis in strategy games.''')
chunker.warm_up()
doc = Document(content=text)
doc_chunks = chunker.run([doc])
print(doc_chunks["documents"])
>[
>Document(id=..., content: 'Artificial intelligence (AI) - Introduction\n\n', meta: {'original_id': '...', 'split_id': 0, 'split_idx_start': 0, '_split_overlap': []})
>Document(id=..., content: 'AI, in its broadest sense, is intelligence exhibited by machines, particularly computer systems.\n', meta: {'original_id': '...', 'split_id': 1, 'split_idx_start': 45, '_split_overlap': []})
>Document(id=..., content: 'AI technology is widely used throughout industry, government, and science.', meta: {'original_id': '...', 'split_id': 2, 'split_idx_start': 142, '_split_overlap': []})
>Document(id=..., content: ' Some high-profile applications include advanced web search engines; recommendation systems; interac...', meta: {'original_id': '...', 'split_id': 3, 'split_idx_start': 216, '_split_overlap': []})
>]
```

<a id="recursive_splitter.RecursiveDocumentSplitter.__init__"></a>

#### RecursiveDocumentSplitter.\_\_init\_\_

```python
def __init__(*,
             split_length: int = 200,
             split_overlap: int = 0,
             split_unit: Literal["word", "char", "token"] = "word",
             separators: Optional[list[str]] = None,
             sentence_splitter_params: Optional[dict[str, Any]] = None)
```

Initializes a RecursiveDocumentSplitter.

**Arguments**:

- `split_length`: The maximum length of each chunk by default in words, but can be in characters or tokens.
See the `split_units` parameter.
- `split_overlap`: The number of characters to overlap between consecutive chunks.
- `split_unit`: The unit of the split_length parameter. It can be either "word", "char", or "token".
If "token" is selected, the text will be split into tokens using the tiktoken tokenizer (o200k_base).
- `separators`: An optional list of separator strings to use for splitting the text. The string
separators will be treated as regular expressions unless the separator is "sentence", in that case the
text will be split into sentences using a custom sentence tokenizer based on NLTK.
See: haystack.components.preprocessors.sentence_tokenizer.SentenceSplitter.
If no separators are provided, the default separators ["\n\n", "sentence", "\n", " "] are used.
- `sentence_splitter_params`: Optional parameters to pass to the sentence tokenizer.
See: haystack.components.preprocessors.sentence_tokenizer.SentenceSplitter for more information.

**Raises**:

- `ValueError`: If the overlap is greater than or equal to the chunk size or if the overlap is negative, or
if any separator is not a string.

<a id="recursive_splitter.RecursiveDocumentSplitter.warm_up"></a>

#### RecursiveDocumentSplitter.warm\_up

```python
def warm_up() -> None
```

Warm up the sentence tokenizer and tiktoken tokenizer if needed.

<a id="recursive_splitter.RecursiveDocumentSplitter.run"></a>

#### RecursiveDocumentSplitter.run

```python
@component.output_types(documents=list[Document])
def run(documents: list[Document]) -> dict[str, list[Document]]
```

Split a list of documents into documents with smaller chunks of text.

**Arguments**:

- `documents`: List of Documents to split.

**Raises**:

- `RuntimeError`: If the component wasn't warmed up but requires it for sentence splitting or tokenization.

**Returns**:

A dictionary containing a key "documents" with a List of Documents with smaller chunks of text corresponding
to the input documents.

<a id="text_cleaner"></a>

# Module text\_cleaner

<a id="text_cleaner.TextCleaner"></a>

## TextCleaner

Cleans text strings.

It can remove substrings matching a list of regular expressions, convert text to lowercase,
remove punctuation, and remove numbers.
Use it to clean up text data before evaluation.

### Usage example

```python
from haystack.components.preprocessors import TextCleaner

text_to_clean = "1Moonlight shimmered softly, 300 Wolves howled nearby, Night enveloped everything."

cleaner = TextCleaner(convert_to_lowercase=True, remove_punctuation=False, remove_numbers=True)
result = cleaner.run(texts=[text_to_clean])
```

<a id="text_cleaner.TextCleaner.__init__"></a>

#### TextCleaner.\_\_init\_\_

```python
def __init__(remove_regexps: Optional[list[str]] = None,
             convert_to_lowercase: bool = False,
             remove_punctuation: bool = False,
             remove_numbers: bool = False)
```

Initializes the TextCleaner component.

**Arguments**:

- `remove_regexps`: A list of regex patterns to remove matching substrings from the text.
- `convert_to_lowercase`: If `True`, converts all characters to lowercase.
- `remove_punctuation`: If `True`, removes punctuation from the text.
- `remove_numbers`: If `True`, removes numerical digits from the text.

<a id="text_cleaner.TextCleaner.run"></a>

#### TextCleaner.run

```python
@component.output_types(texts=list[str])
def run(texts: list[str]) -> dict[str, Any]
```

Cleans up the given list of strings.

**Arguments**:

- `texts`: List of strings to clean.

**Returns**:

A dictionary with the following key:
- `texts`:  the cleaned list of strings.

