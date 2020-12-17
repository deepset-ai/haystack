<a name="txt"></a>
# Module txt

<a name="txt.TextConverter"></a>
## TextConverter Objects

```python
class TextConverter(BaseConverter)
```

<a name="txt.TextConverter.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(remove_numeric_tables: Optional[bool] = False, valid_languages: Optional[List[str]] = None)
```

**Arguments**:

- `remove_numeric_tables`: This option uses heuristics to remove numeric rows from the tables.
The tabular structures in documents might be noise for the reader model if it
does not have table parsing capability for finding answers. However, tables
may also have long strings that could possible candidate for searching answers.
The rows containing strings are thus retained in this option.
- `valid_languages`: validate languages from a list of languages specified in the ISO 639-1
(https://en.wikipedia.org/wiki/ISO_639-1) format.
This option can be used to add test for encoding errors. If the extracted text is
not one of the valid languages, then it might likely be encoding error resulting
in garbled text.

<a name="txt.TextConverter.convert"></a>
#### convert

```python
 | convert(file_path: Path, meta: Optional[Dict[str, str]] = None, encoding: str = "utf-8") -> Dict[str, Any]
```

Reads text from a txt file and executes optional preprocessing steps.

**Arguments**:

- `file_path`: Path of the file to convert
- `meta`: Optional meta data that should be associated with the the document (e.g. name)
- `encoding`: Encoding of the file

**Returns**:

Dict of format {"text": "The text from file", "meta": meta}}

<a name="docx"></a>
# Module docx

<a name="docx.DocxToTextConverter"></a>
## DocxToTextConverter Objects

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
# Module tika

<a name="tika.TikaConverter"></a>
## TikaConverter Objects

```python
class TikaConverter(BaseConverter)
```

<a name="tika.TikaConverter.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(tika_url: str = "http://localhost:9998/tika", remove_numeric_tables: Optional[bool] = False, valid_languages: Optional[List[str]] = None)
```

**Arguments**:

- `tika_url`: URL of the Tika server
- `remove_numeric_tables`: This option uses heuristics to remove numeric rows from the tables.
The tabular structures in documents might be noise for the reader model if it
does not have table parsing capability for finding answers. However, tables
may also have long strings that could possible candidate for searching answers.
The rows containing strings are thus retained in this option.
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
# Module base

<a name="base.BaseConverter"></a>
## BaseConverter Objects

```python
class BaseConverter()
```

Base class for implementing file converts to transform input documents to text format for ingestion in DocumentStore.

<a name="base.BaseConverter.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(remove_numeric_tables: Optional[bool] = None, valid_languages: Optional[List[str]] = None)
```

**Arguments**:

- `remove_numeric_tables`: This option uses heuristics to remove numeric rows from the tables.
The tabular structures in documents might be noise for the reader model if it
does not have table parsing capability for finding answers. However, tables
may also have long strings that could possible candidate for searching answers.
The rows containing strings are thus retained in this option.
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

<a name="pdf"></a>
# Module pdf

<a name="pdf.PDFToTextConverter"></a>
## PDFToTextConverter Objects

```python
class PDFToTextConverter(BaseConverter)
```

<a name="pdf.PDFToTextConverter.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(remove_numeric_tables: Optional[bool] = False, valid_languages: Optional[List[str]] = None)
```

**Arguments**:

- `remove_numeric_tables`: This option uses heuristics to remove numeric rows from the tables.
The tabular structures in documents might be noise for the reader model if it
does not have table parsing capability for finding answers. However, tables
may also have long strings that could possible candidate for searching answers.
The rows containing strings are thus retained in this option.
- `valid_languages`: validate languages from a list of languages specified in the ISO 639-1
(https://en.wikipedia.org/wiki/ISO_639-1) format.
This option can be used to add test for encoding errors. If the extracted text is
not one of the valid languages, then it might likely be encoding error resulting
in garbled text.

<a name="pdf.PDFToTextConverter.convert"></a>
#### convert

```python
 | convert(file_path: Path, meta: Optional[Dict[str, str]] = None) -> Dict[str, Any]
```

Extract text from a .pdf file.

**Arguments**:

- `file_path`: Path to the .pdf file you want to convert

