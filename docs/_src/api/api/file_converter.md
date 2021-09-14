<a id="base"></a>

# Module base

<a id="base.BaseConverter"></a>

## BaseConverter Objects

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

                              The tabular structures in documents might be noise for the reader model if it
                              does not have table parsing capability for finding answers. However, tables
                              may also have long strings that could possible candidate for searching answers.
                              The rows containing strings are thus retained in this option.
                        (https://en.wikipedia.org/wiki/ISO_639-1) format.
                        This option can be used to add test for encoding errors. If the extracted text is
                        not one of the valid languages, then it might likely be encoding error resulting
                        in garbled text.
- `remove_numeric_tables`: This option uses heuristics to remove numeric rows from the tables.
- `valid_languages`: validate languages from a list of languages specified in the ISO 639-1

<a id="base.BaseConverter.convert"></a>

#### convert

```python
@abstractmethod
def convert(file_path: Path, meta: Optional[Dict[str, str]], remove_numeric_tables: Optional[bool] = None, valid_languages: Optional[List[str]] = None, encoding: Optional[str] = "utf-8") -> Dict[str, Any]
```

Convert a file to a dictionary containing the text and any associated meta data.

File converters may extract file meta like name or size. In addition to it, user
supplied meta data like author, url, external IDs can be supplied as a dictionary.

**Arguments**:

                              The tabular structures in documents might be noise for the reader model if it
                              does not have table parsing capability for finding answers. However, tables
                              may also have long strings that could possible candidate for searching answers.
                              The rows containing strings are thus retained in this option.
                        (https://en.wikipedia.org/wiki/ISO_639-1) format.
                        This option can be used to add test for encoding errors. If the extracted text is
                        not one of the valid languages, then it might likely be encoding error resulting
                        in garbled text.
- `file_path`: path of the file to convert
- `meta`: dictionary of meta data key-value pairs to append in the returned document.
- `remove_numeric_tables`: This option uses heuristics to remove numeric rows from the tables.
- `valid_languages`: validate languages from a list of languages specified in the ISO 639-1
- `encoding`: Select the file encoding (default is `utf-8`)

<a id="base.BaseConverter.validate_language"></a>

#### validate\_language

```python
def validate_language(text: str) -> bool
```

Validate if the language of the text is one of valid languages.

<a id="base.FileTypeClassifier"></a>

## FileTypeClassifier Objects

```python
class FileTypeClassifier(BaseComponent)
```

Route files in an Indexing Pipeline to corresponding file converters.

<a id="base.FileTypeClassifier.run"></a>

#### run

```python
def run(file_paths: Union[Path, List[Path]])
```

Return the output based on file extension

<a id="txt"></a>

# Module txt

<a id="txt.TextConverter"></a>

## TextConverter Objects

```python
class TextConverter(BaseConverter)
```

<a id="txt.TextConverter.convert"></a>

#### convert

```python
def convert(file_path: Path, meta: Optional[Dict[str, str]] = None, remove_numeric_tables: Optional[bool] = None, valid_languages: Optional[List[str]] = None, encoding: Optional[str] = "utf-8") -> Dict[str, Any]
```

Reads text from a txt file and executes optional preprocessing steps.

**Arguments**:

                              The tabular structures in documents might be noise for the reader model if it
                              does not have table parsing capability for finding answers. However, tables
                              may also have long strings that could possible candidate for searching answers.
                              The rows containing strings are thus retained in this option.
                        (https://en.wikipedia.org/wiki/ISO_639-1) format.
                        This option can be used to add test for encoding errors. If the extracted text is
                        not one of the valid languages, then it might likely be encoding error resulting
                        in garbled text.

- `file_path`: path of the file to convert
- `meta`: dictionary of meta data key-value pairs to append in the returned document.
- `remove_numeric_tables`: This option uses heuristics to remove numeric rows from the tables.
- `valid_languages`: validate languages from a list of languages specified in the ISO 639-1
- `encoding`: Select the file encoding (default is `utf-8`)

**Returns**:

Dict of format {"text": "The text from file", "meta": meta}}

<a id="docx"></a>

# Module docx

<a id="docx.DocxToTextConverter"></a>

## DocxToTextConverter Objects

```python
class DocxToTextConverter(BaseConverter)
```

<a id="docx.DocxToTextConverter.convert"></a>

#### convert

```python
def convert(file_path: Path, meta: Optional[Dict[str, str]] = None, remove_numeric_tables: Optional[bool] = None, valid_languages: Optional[List[str]] = None, encoding: Optional[str] = None) -> Dict[str, Any]
```

Extract text from a .docx file.
Note: As docx doesn't contain "page" information, we actually extract and return a list of paragraphs here.
For compliance with other converters we nevertheless opted for keeping the methods name.

**Arguments**:

                              The tabular structures in documents might be noise for the reader model if it
                              does not have table parsing capability for finding answers. However, tables
                              may also have long strings that could possible candidate for searching answers.
                              The rows containing strings are thus retained in this option.
                        (https://en.wikipedia.org/wiki/ISO_639-1) format.
                        This option can be used to add test for encoding errors. If the extracted text is
                        not one of the valid languages, then it might likely be encoding error resulting
                        in garbled text.
- `file_path`: Path to the .docx file you want to convert
- `meta`: dictionary of meta data key-value pairs to append in the returned document.
- `remove_numeric_tables`: This option uses heuristics to remove numeric rows from the tables.
- `valid_languages`: validate languages from a list of languages specified in the ISO 639-1
- `encoding`: Not applicable

<a id="tika"></a>

# Module tika

<a id="tika.TikaConverter"></a>

## TikaConverter Objects

```python
class TikaConverter(BaseConverter)
```

<a id="tika.TikaConverter.__init__"></a>

#### \_\_init\_\_

```python
def __init__(tika_url: str = "http://localhost:9998/tika", remove_numeric_tables: bool = False, valid_languages: Optional[List[str]] = None)
```

**Arguments**:

                              The tabular structures in documents might be noise for the reader model if it
                              does not have table parsing capability for finding answers. However, tables
                              may also have long strings that could possible candidate for searching answers.
                              The rows containing strings are thus retained in this option.
                        (https://en.wikipedia.org/wiki/ISO_639-1) format.
                        This option can be used to add test for encoding errors. If the extracted text is
                        not one of the valid languages, then it might likely be encoding error resulting
                        in garbled text.
- `tika_url`: URL of the Tika server
- `remove_numeric_tables`: This option uses heuristics to remove numeric rows from the tables.
- `valid_languages`: validate languages from a list of languages specified in the ISO 639-1

<a id="tika.TikaConverter.convert"></a>

#### convert

```python
def convert(file_path: Path, meta: Optional[Dict[str, str]] = None, remove_numeric_tables: Optional[bool] = None, valid_languages: Optional[List[str]] = None, encoding: Optional[str] = None) -> Dict[str, Any]
```

**Arguments**:

                              The tabular structures in documents might be noise for the reader model if it
                              does not have table parsing capability for finding answers. However, tables
                              may also have long strings that could possible candidate for searching answers.
                              The rows containing strings are thus retained in this option.
                        (https://en.wikipedia.org/wiki/ISO_639-1) format.
                        This option can be used to add test for encoding errors. If the extracted text is
                        not one of the valid languages, then it might likely be encoding error resulting
                        in garbled text.

- `file_path`: path of the file to convert
- `meta`: dictionary of meta data key-value pairs to append in the returned document.
- `remove_numeric_tables`: This option uses heuristics to remove numeric rows from the tables.
- `valid_languages`: validate languages from a list of languages specified in the ISO 639-1
- `encoding`: Not applicable

**Returns**:

a list of pages and the extracted meta data of the file.

<a id="pdf"></a>

# Module pdf

<a id="pdf.PDFToTextConverter"></a>

## PDFToTextConverter Objects

```python
class PDFToTextConverter(BaseConverter)
```

<a id="pdf.PDFToTextConverter.__init__"></a>

#### \_\_init\_\_

```python
def __init__(remove_numeric_tables: bool = False, valid_languages: Optional[List[str]] = None)
```

**Arguments**:

                              The tabular structures in documents might be noise for the reader model if it
                              does not have table parsing capability for finding answers. However, tables
                              may also have long strings that could possible candidate for searching answers.
                              The rows containing strings are thus retained in this option.
                        (https://en.wikipedia.org/wiki/ISO_639-1) format.
                        This option can be used to add test for encoding errors. If the extracted text is
                        not one of the valid languages, then it might likely be encoding error resulting
                        in garbled text.
- `remove_numeric_tables`: This option uses heuristics to remove numeric rows from the tables.
- `valid_languages`: validate languages from a list of languages specified in the ISO 639-1

<a id="pdf.PDFToTextConverter.convert"></a>

#### convert

```python
def convert(file_path: Path, meta: Optional[Dict[str, str]] = None, remove_numeric_tables: Optional[bool] = None, valid_languages: Optional[List[str]] = None, encoding: Optional[str] = "Latin1") -> Dict[str, Any]
```

Extract text from a .pdf file using the pdftotext library (https://www.xpdfreader.com/pdftotext-man.html)

**Arguments**:

             Can be any custom keys and values.
                              The tabular structures in documents might be noise for the reader model if it
                              does not have table parsing capability for finding answers. However, tables
                              may also have long strings that could possible candidate for searching answers.
                              The rows containing strings are thus retained in this option.
                        (https://en.wikipedia.org/wiki/ISO_639-1) format.
                        This option can be used to add test for encoding errors. If the extracted text is
                        not one of the valid languages, then it might likely be encoding error resulting
                        in garbled text.
                 of pdftotext. While this works well on many PDFs, it might be needed to switch to "UTF-8" or
                 others if your doc contains special characters (e.g. German Umlauts, Cyrillic characters ...).
                 Note: With "UTF-8" we experienced cases, where a simple "fi" gets wrongly parsed as
                 "xef\xac\x81c" (see test cases). That's why we keep "Latin 1" as default here.
                 (See list of available encodings by running `pdftotext -listencodings` in the terminal)
- `file_path`: Path to the .pdf file you want to convert
- `meta`: Optional dictionary with metadata that shall be attached to all resulting documents.
- `remove_numeric_tables`: This option uses heuristics to remove numeric rows from the tables.
- `valid_languages`: validate languages from a list of languages specified in the ISO 639-1
- `encoding`: Encoding that will be passed as -enc parameter to pdftotext. "Latin 1" is the default encoding

<a id="pdf.PDFToTextOCRConverter"></a>

## PDFToTextOCRConverter Objects

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

                              The tabular structures in documents might be noise for the reader model if it
                              does not have table parsing capability for finding answers. However, tables
                              may also have long strings that could possible candidate for searching answers.
                              The rows containing strings are thus retained in this option.
                        (https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html).
                        This option can be used to add test for encoding errors. If the extracted text is
                        not one of the valid languages, then it might likely be encoding error resulting
                        in garbled text.
- `remove_numeric_tables`: This option uses heuristics to remove numeric rows from the tables.
- `valid_languages`: validate languages from a list of languages supported by tessarect

<a id="pdf.PDFToTextOCRConverter.convert"></a>

#### convert

```python
def convert(file_path: Path, meta: Optional[Dict[str, str]] = None, remove_numeric_tables: Optional[bool] = None, valid_languages: Optional[List[str]] = None, encoding: Optional[str] = "utf-8") -> Dict[str, Any]
```

Convert a file to a dictionary containing the text and any associated meta data.

File converters may extract file meta like name or size. In addition to it, user
supplied meta data like author, url, external IDs can be supplied as a dictionary.

**Arguments**:

                              The tabular structures in documents might be noise for the reader model if it
                              does not have table parsing capability for finding answers. However, tables
                              may also have long strings that could possible candidate for searching answers.
                              The rows containing strings are thus retained in this option.
                        (https://en.wikipedia.org/wiki/ISO_639-1) format.
                        This option can be used to add test for encoding errors. If the extracted text is
                        not one of the valid languages, then it might likely be encoding error resulting
                        in garbled text.
- `file_path`: path of the file to convert
- `meta`: dictionary of meta data key-value pairs to append in the returned document.
- `remove_numeric_tables`: This option uses heuristics to remove numeric rows from the tables.
- `valid_languages`: validate languages from a list of languages specified in the ISO 639-1
- `encoding`: Select the file encoding (default is `utf-8`)

