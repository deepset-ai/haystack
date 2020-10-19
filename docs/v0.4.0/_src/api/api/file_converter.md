<a name="txt"></a>
# txt

<a name="txt.TextConverter"></a>
## TextConverter

```python
class TextConverter(BaseConverter)
```

<a name="txt.TextConverter.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(remove_numeric_tables: Optional[bool] = False, remove_whitespace: Optional[bool] = None, remove_empty_lines: Optional[bool] = None, remove_header_footer: Optional[bool] = None, valid_languages: Optional[List[str]] = None)
```

**Arguments**:

- `remove_numeric_tables`: This option uses heuristics to remove numeric rows from the tables.
The tabular structures in documents might be noise for the reader model if it
does not have table parsing capability for finding answers. However, tables
may also have long strings that could possible candidate for searching answers.
The rows containing strings are thus retained in this option.
- `remove_whitespace`: strip whitespaces before or after each line in the text.
- `remove_empty_lines`: remove more than two empty lines in the text.
- `remove_header_footer`: use heuristic to remove footers and headers across different pages by searching
for the longest common string. This heuristic uses exact matches and therefore
works well for footers like "Copyright 2019 by XXX", but won't detect "Page 3 of 4"
or similar.
- `valid_languages`: validate languages from a list of languages specified in the ISO 639-1
(https://en.wikipedia.org/wiki/ISO_639-1) format.
This option can be used to add test for encoding errors. If the extracted text is
not one of the valid languages, then it might likely be encoding error resulting
in garbled text.

<a name="docx"></a>
# docx

<a name="docx.DocxToTextConverter"></a>
## DocxToTextConverter

```python
class DocxToTextConverter(BaseConverter)
```

<a name="docx.DocxToTextConverter.convert"></a>
#### convert

```python
 | convert(file_path: Path, meta: Optional[Dict[str, str]] = None) -> Dict[str, Any]
```

Extract text from a .docx file.
Note: As docx doesn't contain "page" information, we actually extract and return a list of paragraphs here.
For compliance with other converters we nevertheless opted for keeping the methods name.

**Arguments**:

- `file_path`: Path to the .docx file you want to convert

<a name="tika"></a>
# tika

<a name="tika.TikaConverter"></a>
## TikaConverter

```python
class TikaConverter(BaseConverter)
```

<a name="tika.TikaConverter.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(tika_url: str = "http://localhost:9998/tika", remove_numeric_tables: Optional[bool] = False, remove_whitespace: Optional[bool] = None, remove_empty_lines: Optional[bool] = None, remove_header_footer: Optional[bool] = None, valid_languages: Optional[List[str]] = None)
```

**Arguments**:

- `tika_url`: URL of the Tika server
- `remove_numeric_tables`: This option uses heuristics to remove numeric rows from the tables.
The tabular structures in documents might be noise for the reader model if it
does not have table parsing capability for finding answers. However, tables
may also have long strings that could possible candidate for searching answers.
The rows containing strings are thus retained in this option.
- `remove_whitespace`: strip whitespaces before or after each line in the text.
- `remove_empty_lines`: remove more than two empty lines in the text.
- `remove_header_footer`: use heuristic to remove footers and headers across different pages by searching
for the longest common string. This heuristic uses exact matches and therefore
works well for footers like "Copyright 2019 by XXX", but won't detect "Page 3 of 4"
or similar.
- `valid_languages`: validate languages from a list of languages specified in the ISO 639-1
(https://en.wikipedia.org/wiki/ISO_639-1) format.
This option can be used to add test for encoding errors. If the extracted text is
not one of the valid languages, then it might likely be encoding error resulting
in garbled text.

<a name="tika.TikaConverter.convert"></a>
#### convert

```python
 | convert(file_path: Path, meta: Optional[Dict[str, str]] = None) -> Dict[str, Any]
```

**Arguments**:

- `file_path`: Path of file to be converted.

**Returns**:

a list of pages and the extracted meta data of the file.

<a name="base"></a>
# base

<a name="base.BaseConverter"></a>
## BaseConverter

```python
class BaseConverter()
```

Base class for implementing file converts to transform input documents to text format for ingestion in DocumentStore.

<a name="base.BaseConverter.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(remove_numeric_tables: Optional[bool] = None, remove_header_footer: Optional[bool] = None, remove_whitespace: Optional[bool] = None, remove_empty_lines: Optional[bool] = None, valid_languages: Optional[List[str]] = None)
```

**Arguments**:

- `remove_numeric_tables`: This option uses heuristics to remove numeric rows from the tables.
The tabular structures in documents might be noise for the reader model if it
does not have table parsing capability for finding answers. However, tables
may also have long strings that could possible candidate for searching answers.
The rows containing strings are thus retained in this option.
- `remove_header_footer`: use heuristic to remove footers and headers across different pages by searching
for the longest common string. This heuristic uses exact matches and therefore
works well for footers like "Copyright 2019 by XXX", but won't detect "Page 3 of 4"
or similar.
- `remove_whitespace`: strip whitespaces before or after each line in the text.
- `remove_empty_lines`: remove more than two empty lines in the text.
- `valid_languages`: validate languages from a list of languages specified in the ISO 639-1
(https://en.wikipedia.org/wiki/ISO_639-1) format.
This option can be used to add test for encoding errors. If the extracted text is
not one of the valid languages, then it might likely be encoding error resulting
in garbled text.

<a name="base.BaseConverter.convert"></a>
#### convert

```python
 | @abstractmethod
 | convert(file_path: Path, meta: Optional[Dict[str, str]]) -> Dict[str, Any]
```

Convert a file to a dictionary containing the text and any associated meta data.

File converters may extract file meta like name or size. In addition to it, user
supplied meta data like author, url, external IDs can be supplied as a dictionary.

**Arguments**:

- `file_path`: path of the file to convert
- `meta`: dictionary of meta data key-value pairs to append in the returned document.

<a name="base.BaseConverter.validate_language"></a>
#### validate\_language

```python
 | validate_language(text: str) -> bool
```

Validate if the language of the text is one of valid languages.

<a name="base.BaseConverter.find_and_remove_header_footer"></a>
#### find\_and\_remove\_header\_footer

```python
 | find_and_remove_header_footer(pages: List[str], n_chars: int, n_first_pages_to_ignore: int, n_last_pages_to_ignore: int) -> Tuple[List[str], Optional[str], Optional[str]]
```

Heuristic to find footers and headers across different pages by searching for the longest common string.
For headers we only search in the first n_chars characters (for footer: last n_chars).
Note: This heuristic uses exact matches and therefore works well for footers like "Copyright 2019 by XXX",
but won't detect "Page 3 of 4" or similar.

**Arguments**:

- `pages`: list of strings, one string per page
- `n_chars`: number of first/last characters where the header/footer shall be searched in
- `n_first_pages_to_ignore`: number of first pages to ignore (e.g. TOCs often don't contain footer/header)
- `n_last_pages_to_ignore`: number of last pages to ignore

**Returns**:

(cleaned pages, found_header_str, found_footer_str)

<a name="pdf"></a>
# pdf

<a name="pdf.PDFToTextConverter"></a>
## PDFToTextConverter

```python
class PDFToTextConverter(BaseConverter)
```

<a name="pdf.PDFToTextConverter.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(remove_numeric_tables: Optional[bool] = False, remove_whitespace: Optional[bool] = None, remove_empty_lines: Optional[bool] = None, remove_header_footer: Optional[bool] = None, valid_languages: Optional[List[str]] = None)
```

**Arguments**:

- `remove_numeric_tables`: This option uses heuristics to remove numeric rows from the tables.
The tabular structures in documents might be noise for the reader model if it
does not have table parsing capability for finding answers. However, tables
may also have long strings that could possible candidate for searching answers.
The rows containing strings are thus retained in this option.
- `remove_whitespace`: strip whitespaces before or after each line in the text.
- `remove_empty_lines`: remove more than two empty lines in the text.
- `remove_header_footer`: use heuristic to remove footers and headers across different pages by searching
for the longest common string. This heuristic uses exact matches and therefore
works well for footers like "Copyright 2019 by XXX", but won't detect "Page 3 of 4"
or similar.
- `valid_languages`: validate languages from a list of languages specified in the ISO 639-1
(https://en.wikipedia.org/wiki/ISO_639-1) format.
This option can be used to add test for encoding errors. If the extracted text is
not one of the valid languages, then it might likely be encoding error resulting
in garbled text.

