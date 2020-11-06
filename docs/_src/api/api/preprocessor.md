<a name="utils"></a>
# utils

<a name="utils.eval_data_from_file"></a>
#### eval\_data\_from\_file

```python
eval_data_from_file(filename: str, max_docs: Union[int, bool] = None) -> Tuple[List[Document], List[Label]]
```

Read Documents + Labels from a SQuAD-style file.
Document and Labels can then be indexed to the DocumentStore and be used for evaluation.

**Arguments**:

- `filename`: Path to file in SQuAD format
- `max_docs`: This sets the number of documents that will be loaded. By default, this is set to None, thus reading in all available eval documents.

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

<a name="preprocessor"></a>
# preprocessor

<a name="preprocessor.PreProcessor"></a>
## PreProcessor

```python
class PreProcessor(BasePreProcessor)
```

<a name="preprocessor.PreProcessor.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(clean_whitespace: Optional[bool] = True, clean_header_footer: Optional[bool] = False, clean_empty_lines: Optional[bool] = True, split_by: Optional[str] = "word", split_length: Optional[int] = 1000, split_stride: Optional[int] = None, split_respect_sentence_boundary: Optional[bool] = True)
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
- `split_stride`: Length of striding window over the splits. For example, if split_by -> `word`,
split_length -> 5 & split_stride -> 2, then the splits would be like:
[w1 w2 w3 w4 w5, w4 w5 w6 w7 w8, w7 w8 w10 w11 w12].
Set the value to None to disable striding behaviour.
- `split_respect_sentence_boundary`: Whether to split in partial sentences if split_by -> `word`. If set
to True, the individual split will always have complete sentences &
the number of words will be <= split_length.

<a name="base"></a>
# base

<a name="cleaning"></a>
# cleaning

