<a id="base"></a>

# Module base

<a id="base.BaseConverter"></a>

## BaseConverter

```python
class BaseConverter(BaseComponent)
```

Base class for implementing file converts to transform input documents to text format for ingestion in DocumentStore.

<a id="base.BaseConverter.__init__"></a>

#### \_\_init\_\_

```python
def __init__(remove_numeric_tables: bool = False, valid_languages: Optional[List[str]] = None)
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

<a id="base.BaseConverter.convert"></a>

#### convert

```python
@abstractmethod
def convert(file_path: Path, meta: Optional[Dict[str, str]], remove_numeric_tables: Optional[bool] = None, valid_languages: Optional[List[str]] = None, encoding: Optional[str] = "utf-8") -> List[Dict[str, Any]]
```

Convert a file to a dictionary containing the text and any associated meta data.

File converters may extract file meta like name or size. In addition to it, user
supplied meta data like author, url, external IDs can be supplied as a dictionary.

**Arguments**:

- `file_path`: path of the file to convert
- `meta`: dictionary of meta data key-value pairs to append in the returned document.
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
- `encoding`: Select the file encoding (default is `utf-8`)

<a id="base.BaseConverter.validate_language"></a>

#### validate\_language

```python
def validate_language(text: str, valid_languages: Optional[List[str]] = None) -> bool
```

Validate if the language of the text is one of valid languages.

<a id="docx"></a>

# Module docx

<a id="docx.DocxToTextConverter"></a>

## DocxToTextConverter

```python
class DocxToTextConverter(BaseConverter)
```

<a id="docx.DocxToTextConverter.convert"></a>

#### convert

```python
def convert(file_path: Path, meta: Optional[Dict[str, str]] = None, remove_numeric_tables: Optional[bool] = None, valid_languages: Optional[List[str]] = None, encoding: Optional[str] = None) -> List[Dict[str, Any]]
```

Extract text from a .docx file.

Note: As docx doesn't contain "page" information, we actually extract and return a list of paragraphs here.
For compliance with other converters we nevertheless opted for keeping the methods name.

**Arguments**:

- `file_path`: Path to the .docx file you want to convert
- `meta`: dictionary of meta data key-value pairs to append in the returned document.
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
- `encoding`: Not applicable

<a id="image"></a>

# Module image

<a id="image.ImageToTextConverter"></a>

## ImageToTextConverter

```python
class ImageToTextConverter(BaseConverter)
```

<a id="image.ImageToTextConverter.__init__"></a>

#### \_\_init\_\_

```python
def __init__(remove_numeric_tables: bool = False, valid_languages: Optional[List[str]] = ["eng"])
```

**Arguments**:

- `remove_numeric_tables`: This option uses heuristics to remove numeric rows from the tables.
The tabular structures in documents might be noise for the reader model if it
does not have table parsing capability for finding answers. However, tables
may also have long strings that could possible candidate for searching answers.
The rows containing strings are thus retained in this option.
- `valid_languages`: validate languages from a list of languages specified here
(https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html)
This option can be used to add test for encoding errors. If the extracted text is
not one of the valid languages, then it might likely be encoding error resulting
in garbled text. Run the following line of code to check available language packs:
# List of available languages
print(pytesseract.get_languages(config=''))

<a id="image.ImageToTextConverter.convert"></a>

#### convert

```python
def convert(file_path: Union[Path, str], meta: Optional[Dict[str, str]] = None, remove_numeric_tables: Optional[bool] = None, valid_languages: Optional[List[str]] = None, encoding: Optional[str] = "utf-8") -> List[Dict[str, Any]]
```

Extract text from image file using the pytesseract library (https://github.com/madmaze/pytesseract)

**Arguments**:

- `file_path`: path to image file
- `meta`: Optional dictionary with metadata that shall be attached to all resulting documents.
Can be any custom keys and values.
- `remove_numeric_tables`: This option uses heuristics to remove numeric rows from the tables.
The tabular structures in documents might be noise for the reader model if it
does not have table parsing capability for finding answers. However, tables
may also have long strings that could possible candidate for searching answers.
The rows containing strings are thus retained in this option.
- `valid_languages`: validate languages from a list of languages supported by tessarect
(https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html).
This option can be used to add test for encoding errors. If the extracted text is
not one of the valid languages, then it might likely be encoding error resulting
in garbled text.

<a id="markdown"></a>

# Module markdown

<a id="markdown.MarkdownConverter"></a>

## MarkdownConverter

```python
class MarkdownConverter(BaseConverter)
```

<a id="markdown.MarkdownConverter.convert"></a>

#### convert

```python
def convert(file_path: Path, meta: Optional[Dict[str, str]] = None, remove_numeric_tables: Optional[bool] = None, valid_languages: Optional[List[str]] = None, encoding: Optional[str] = "utf-8") -> List[Dict[str, Any]]
```

Reads text from a txt file and executes optional preprocessing steps.

**Arguments**:

- `file_path`: path of the file to convert
- `meta`: dictionary of meta data key-value pairs to append in the returned document.
- `encoding`: Select the file encoding (default is `utf-8`)
- `remove_numeric_tables`: Not applicable
- `valid_languages`: Not applicable

**Returns**:

Dict of format {"text": "The text from file", "meta": meta}}

<a id="markdown.MarkdownConverter.markdown_to_text"></a>

#### markdown\_to\_text

```python
@staticmethod
def markdown_to_text(markdown_string: str) -> str
```

Converts a markdown string to plaintext

**Arguments**:

- `markdown_string`: String in markdown format

<a id="pdf"></a>

# Module pdf

<a id="pdf.PDFToTextConverter"></a>

## PDFToTextConverter

```python
class PDFToTextConverter(BaseConverter)
```

<a id="pdf.PDFToTextConverter.__init__"></a>

#### \_\_init\_\_

```python
def __init__(remove_numeric_tables: bool = False, valid_languages: Optional[List[str]] = None)
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

<a id="pdf.PDFToTextConverter.convert"></a>

#### convert

```python
def convert(file_path: Path, meta: Optional[Dict[str, str]] = None, remove_numeric_tables: Optional[bool] = None, valid_languages: Optional[List[str]] = None, encoding: Optional[str] = "Latin1") -> List[Dict[str, Any]]
```

Extract text from a .pdf file using the pdftotext library (https://www.xpdfreader.com/pdftotext-man.html)

**Arguments**:

- `file_path`: Path to the .pdf file you want to convert
- `meta`: Optional dictionary with metadata that shall be attached to all resulting documents.
Can be any custom keys and values.
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
- `encoding`: Encoding that will be passed as -enc parameter to pdftotext. "Latin 1" is the default encoding
of pdftotext. While this works well on many PDFs, it might be needed to switch to "UTF-8" or
others if your doc contains special characters (e.g. German Umlauts, Cyrillic characters ...).
Note: With "UTF-8" we experienced cases, where a simple "fi" gets wrongly parsed as
"xef\xac\x81c" (see test cases). That's why we keep "Latin 1" as default here.
(See list of available encodings by running `pdftotext -listenc` in the terminal)

<a id="pdf.PDFToTextOCRConverter"></a>

## PDFToTextOCRConverter

```python
class PDFToTextOCRConverter(BaseConverter)
```

<a id="pdf.PDFToTextOCRConverter.__init__"></a>

#### \_\_init\_\_

```python
def __init__(remove_numeric_tables: bool = False, valid_languages: Optional[List[str]] = ["eng"])
```

Extract text from image file using the pytesseract library (https://github.com/madmaze/pytesseract)

**Arguments**:

- `remove_numeric_tables`: This option uses heuristics to remove numeric rows from the tables.
The tabular structures in documents might be noise for the reader model if it
does not have table parsing capability for finding answers. However, tables
may also have long strings that could possible candidate for searching answers.
The rows containing strings are thus retained in this option.
- `valid_languages`: validate languages from a list of languages supported by tessarect
(https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html).
This option can be used to add test for encoding errors. If the extracted text is
not one of the valid languages, then it might likely be encoding error resulting
in garbled text.

<a id="pdf.PDFToTextOCRConverter.convert"></a>

#### convert

```python
def convert(file_path: Path, meta: Optional[Dict[str, str]] = None, remove_numeric_tables: Optional[bool] = None, valid_languages: Optional[List[str]] = None, encoding: Optional[str] = "utf-8") -> List[Dict[str, Any]]
```

Convert a file to a dictionary containing the text and any associated meta data.

File converters may extract file meta like name or size. In addition to it, user
supplied meta data like author, url, external IDs can be supplied as a dictionary.

**Arguments**:

- `file_path`: path of the file to convert
- `meta`: dictionary of meta data key-value pairs to append in the returned document.
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
- `encoding`: Select the file encoding (default is `utf-8`)

<a id="tika"></a>

# Module tika

<a id="tika.TikaConverter"></a>

## TikaConverter

```python
class TikaConverter(BaseConverter)
```

<a id="tika.TikaConverter.__init__"></a>

#### \_\_init\_\_

```python
def __init__(tika_url: str = "http://localhost:9998/tika", remove_numeric_tables: bool = False, valid_languages: Optional[List[str]] = None)
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

<a id="tika.TikaConverter.convert"></a>

#### convert

```python
def convert(file_path: Path, meta: Optional[Dict[str, str]] = None, remove_numeric_tables: Optional[bool] = None, valid_languages: Optional[List[str]] = None, encoding: Optional[str] = None) -> List[Dict[str, Any]]
```

**Arguments**:

- `file_path`: path of the file to convert
- `meta`: dictionary of meta data key-value pairs to append in the returned document.
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
- `encoding`: Not applicable

**Returns**:

a list of pages and the extracted meta data of the file.

<a id="txt"></a>

# Module txt

<a id="txt.TextConverter"></a>

## TextConverter

```python
class TextConverter(BaseConverter)
```

<a id="txt.TextConverter.convert"></a>

#### convert

```python
def convert(file_path: Path, meta: Optional[Dict[str, str]] = None, remove_numeric_tables: Optional[bool] = None, valid_languages: Optional[List[str]] = None, encoding: Optional[str] = "utf-8") -> List[Dict[str, Any]]
```

Reads text from a txt file and executes optional preprocessing steps.

**Arguments**:

- `file_path`: path of the file to convert
- `meta`: dictionary of meta data key-value pairs to append in the returned document.
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
- `encoding`: Select the file encoding (default is `utf-8`)

**Returns**:

Dict of format {"text": "The text from file", "meta": meta}}
