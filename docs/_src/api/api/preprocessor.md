<a name="cleaning"></a>
# cleaning

<a name="utils"></a>
# utils

<a name="utils.eval_data_from_file"></a>
#### eval\_data\_from\_file

```python
eval_data_from_file(filename: str) -> Tuple[List[Document], List[Label]]
```

Read Documents + Labels from a SQuAD-style file.
Document and Labels can then be indexed to the DocumentStore and be used for evaluation.

**Arguments**:

- `filename`: Path to file in SQuAD format

**Returns**:

(List of Documents, List of Labels)

<a name="utils.convert_files_to_dicts"></a>
#### convert\_files\_to\_dicts

```python
convert_files_to_dicts(dir_path: str, clean_func: Optional[Callable] = None, split_paragraphs: bool = False) -> List[dict]
```

Convert all files(.txt, .pdf) in the sub-directories of the given path to Python dicts that can be written to a
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

