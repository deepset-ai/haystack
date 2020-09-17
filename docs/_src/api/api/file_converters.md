<!---
title: "File Converters"
metaTitle: "File Converters"
metaDescription: ""
slug: "/docs/file_converters"
date: "2020-09-03"
id: "file_convertersmd"
--->

<a name="indexing.file_converters.txt"></a>
# indexing.file\_converters.txt

<a name="indexing.file_converters.txt.TextConverter"></a>
## TextConverter Objects

```python
class TextConverter(BaseConverter)
```

<a name="indexing.file_converters.txt.TextConverter.__init__"></a>
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

<a name="indexing.file_converters.docx"></a>
# indexing.file\_converters.docx

<a name="indexing.file_converters.docx.DocxToTextConverter"></a>
## DocxToTextConverter Objects

```python
class DocxToTextConverter(BaseConverter)
```

<a name="indexing.file_converters.docx.DocxToTextConverter.extract_pages"></a>
#### extract\_pages

```python
 | extract_pages(file_path: Path) -> Tuple[List[str], Optional[Dict[str, Any]]]
```

Extract text from a .docx file.
Note: As docx doesn't contain "page" information, we actually extract and return a list of paragraphs here.
For compliance with other converters we nevertheless opted for keeping the methods name.

**Arguments**:

- `file_path`: Path to the .docx file you want to convert

<a name="indexing.file_converters.tika"></a>
# indexing.file\_converters.tika

<a name="indexing.file_converters.tika.TikaConverter"></a>
## TikaConverter Objects

```python
class TikaConverter(BaseConverter)
```

<a name="indexing.file_converters.tika.TikaConverter.__init__"></a>
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

<a name="indexing.file_converters.tika.TikaConverter.extract_pages"></a>
#### extract\_pages

```python
 | extract_pages(file_path: Path) -> Tuple[List[str], Optional[Dict[str, Any]]]
```

**Arguments**:

- `file_path`: Path of file to be converted.

**Returns**:

a list of pages and the extracted meta data of the file.

<a name="indexing.file_converters.base"></a>
# indexing.file\_converters.base

<a name="indexing.file_converters.base.BaseConverter"></a>
## BaseConverter Objects

```python
class BaseConverter()
```

Base class for implementing file converts to transform input documents to text format for indexing in database.

<a name="indexing.file_converters.base.BaseConverter.__init__"></a>
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

<a name="indexing.file_converters.base.BaseConverter.validate_language"></a>
#### validate\_language

```python
 | validate_language(text: str) -> bool
```

Validate if the language of the text is one of valid languages.

<a name="indexing.file_converters.base.BaseConverter.find_and_remove_header_footer"></a>
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

<a name="indexing.file_converters.pdf"></a>
# indexing.file\_converters.pdf

<a name="indexing.file_converters.pdf.PDFToTextConverter"></a>
## PDFToTextConverter Objects

```python
class PDFToTextConverter(BaseConverter)
```

<a name="indexing.file_converters.pdf.PDFToTextConverter.__init__"></a>
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

