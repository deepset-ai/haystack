<a name="base"></a>
# Module base

<a name="base.BasePreProcessor"></a>
## BasePreProcessor Objects

```python
class BasePreProcessor(BaseComponent)
```

<a name="base.BasePreProcessor.process"></a>
#### process

```python
 | process(document: dict, clean_whitespace: Optional[bool] = True, clean_header_footer: Optional[bool] = False, clean_empty_lines: Optional[bool] = True, split_by: Optional[str] = "word", split_length: Optional[int] = 1000, split_overlap: Optional[int] = None, split_respect_sentence_boundary: Optional[bool] = True) -> List[dict]
```

Perform document cleaning and splitting. Takes a single document as input and returns a list of documents.

<a name="preprocessor"></a>
# Module preprocessor

<a name="preprocessor.PreProcessor"></a>
## PreProcessor Objects

```python
class PreProcessor(BasePreProcessor)
```

<a name="preprocessor.PreProcessor.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(clean_whitespace: bool = True, clean_header_footer: bool = False, clean_empty_lines: bool = True, split_by: str = "word", split_length: int = 1000, split_overlap: int = 0, split_respect_sentence_boundary: bool = True)
```

**Arguments**:

- `clean_header_footer`: Use heuristic to remove footers and headers across different pages by searching
                             for the longest common string. This heuristic uses exact matches and therefore
                             works well for footers like "Copyright 2019 by XXX", but won't detect "Page 3 of 4"
                             or similar.
- `clean_whitespace`: Strip whitespaces before or after each line in the text.
- `clean_empty_lines`: Remove more than two empty lines in the text.
- `split_by`: Unit for splitting the document. Can be "word", "sentence", or "passage". Set to None to disable splitting.
- `split_length`: Max. number of the above split unit (e.g. words) that are allowed in one document. For instance, if n -> 10 & split_by ->
                   "sentence", then each output document will have 10 sentences.
- `split_overlap`: Word overlap between two adjacent documents after a split.
                      Setting this to a positive number essentially enables the sliding window approach.
                      For example, if split_by -> `word`,
                      split_length -> 5 & split_overlap -> 2, then the splits would be like:
                      [w1 w2 w3 w4 w5, w4 w5 w6 w7 w8, w7 w8 w10 w11 w12].
                      Set the value to 0 to ensure there is no overlap among the documents after splitting.
- `split_respect_sentence_boundary`: Whether to split in partial sentences if split_by -> `word`. If set
                                        to True, the individual split will always have complete sentences &
                                        the number of words will be <= split_length.

<a name="preprocessor.PreProcessor.process"></a>
#### process

```python
 | process(document: dict, clean_whitespace: Optional[bool] = None, clean_header_footer: Optional[bool] = None, clean_empty_lines: Optional[bool] = None, split_by: Optional[str] = None, split_length: Optional[int] = None, split_overlap: Optional[int] = None, split_respect_sentence_boundary: Optional[bool] = None) -> List[dict]
```

Perform document cleaning and splitting. Takes a single document as input and returns a list of documents.

<a name="preprocessor.PreProcessor.clean"></a>
#### clean

```python
 | clean(document: dict, clean_whitespace: bool, clean_header_footer: bool, clean_empty_lines: bool) -> dict
```

Perform document cleaning on a single document and return a single document. This method will deal with whitespaces, headers, footers
and empty lines. Its exact functionality is defined by the parameters passed into PreProcessor.__init__().

<a name="preprocessor.PreProcessor.split"></a>
#### split

```python
 | split(document: dict, split_by: str, split_length: int, split_overlap: int, split_respect_sentence_boundary: bool) -> List[dict]
```

Perform document splitting on a single document. This method can split on different units, at different lengths,
with different strides. It can also respect sentence boundaries. Its exact functionality is defined by
the parameters passed into PreProcessor.__init__(). Takes a single document as input and returns a list of documents.

<a name="utils"></a>
# Module utils

<a name="utils.eval_data_from_json"></a>
#### eval\_data\_from\_json

```python
eval_data_from_json(filename: str, max_docs: Union[int, bool] = None, preprocessor: PreProcessor = None, open_domain: bool = False) -> Tuple[List[Document], List[Label]]
```

Read Documents + Labels from a SQuAD-style file.
Document and Labels can then be indexed to the DocumentStore and be used for evaluation.

**Arguments**:

- `filename`: Path to file in SQuAD format
- `max_docs`: This sets the number of documents that will be loaded. By default, this is set to None, thus reading in all available eval documents.
- `open_domain`: Set this to True if your file is an open domain dataset where two different answers to the same question might be found in different contexts.

**Returns**:

(List of Documents, List of Labels)

<a name="utils.eval_data_from_jsonl"></a>
#### eval\_data\_from\_jsonl

```python
eval_data_from_jsonl(filename: str, batch_size: Optional[int] = None, max_docs: Union[int, bool] = None, preprocessor: PreProcessor = None, open_domain: bool = False) -> Generator[Tuple[List[Document], List[Label]], None, None]
```

Read Documents + Labels from a SQuAD-style file in jsonl format, i.e. one document per line.
Document and Labels can then be indexed to the DocumentStore and be used for evaluation.

This is a generator which will yield one tuple per iteration containing a list
of batch_size documents and a list with the documents' labels.
If batch_size is set to None, this method will yield all documents and labels.

**Arguments**:

- `filename`: Path to file in SQuAD format
- `max_docs`: This sets the number of documents that will be loaded. By default, this is set to None, thus reading in all available eval documents.
- `open_domain`: Set this to True if your file is an open domain dataset where two different answers to the same question might be found in different contexts.

**Returns**:

(List of Documents, List of Labels)

<a name="utils.convert_files_to_dicts"></a>
#### convert\_files\_to\_dicts

```python
convert_files_to_dicts(dir_path: str, clean_func: Optional[Callable] = None, split_paragraphs: bool = False) -> List[dict]
```

Convert all files(.txt, .pdf, .docx) in the sub-directories of the given path to Python dicts that can be written to a
Document Store.

**Arguments**:

- `dir_path`: path for the documents to be written to the DocumentStore
- `clean_func`: a custom cleaning function that gets applied to each doc (input: str, output:str)
- `split_paragraphs`: split text in paragraphs.

**Returns**:

None

<a name="utils.tika_convert_files_to_dicts"></a>
#### tika\_convert\_files\_to\_dicts

```python
tika_convert_files_to_dicts(dir_path: str, clean_func: Optional[Callable] = None, split_paragraphs: bool = False, merge_short: bool = True, merge_lowercase: bool = True) -> List[dict]
```

Convert all files(.txt, .pdf) in the sub-directories of the given path to Python dicts that can be written to a
Document Store.

**Arguments**:

- `merge_lowercase`: allow conversion of merged paragraph to lowercase
- `merge_short`: allow merging of short paragraphs
- `dir_path`: path for the documents to be written to the DocumentStore
- `clean_func`: a custom cleaning function that gets applied to each doc (input: str, output:str)
- `split_paragraphs`: split text in paragraphs.

**Returns**:

None

<a name="utils.fetch_archive_from_http"></a>
#### fetch\_archive\_from\_http

```python
fetch_archive_from_http(url: str, output_dir: str, proxies: Optional[dict] = None)
```

Fetch an archive (zip or tar.gz) from a url via http and extract content to an output directory.

**Arguments**:

- `url`: http address
:type url: str
- `output_dir`: local path
:type output_dir: str
- `proxies`: proxies details as required by requests library
:type proxies: dict

**Returns**:

bool if anything got fetched

<a name="utils.squad_json_to_jsonl"></a>
#### squad\_json\_to\_jsonl

```python
squad_json_to_jsonl(squad_file: str, output_file: str)
```

Converts a SQuAD-json-file into jsonl format with one document per line.

**Arguments**:

- `squad_file`: SQuAD-file in json format.
:type squad_file: str
- `output_file`: Name of output file (SQuAD in jsonl format)
:type output_file: str

<a name="cleaning"></a>
# Module cleaning

<a name="cleaning.clean_wiki_text"></a>
#### clean\_wiki\_text

```python
clean_wiki_text(text: str) -> str
```

Clean wikipedia text by removing multiple new lines, removing extremely short lines,
adding paragraph breaks and removing empty paragraphs

