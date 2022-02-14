<a id="base"></a>

# Module base

<a id="base.BasePreProcessor"></a>

## BasePreProcessor

```python
class BasePreProcessor(BaseComponent)
```

<a id="base.BasePreProcessor.process"></a>

#### process

```python
def process(documents: Union[dict, List[dict]], clean_whitespace: Optional[bool] = True, clean_header_footer: Optional[bool] = False, clean_empty_lines: Optional[bool] = True, split_by: Optional[str] = "word", split_length: Optional[int] = 1000, split_overlap: Optional[int] = None, split_respect_sentence_boundary: Optional[bool] = True) -> List[dict]
```

Perform document cleaning and splitting. Takes a single document as input and returns a list of documents.

<a id="preprocessor"></a>

# Module preprocessor

<a id="preprocessor.PreProcessor"></a>

## PreProcessor

```python
class PreProcessor(BasePreProcessor)
```

<a id="preprocessor.PreProcessor.process"></a>

#### process

```python
def process(documents: Union[dict, List[dict]], clean_whitespace: Optional[bool] = None, clean_header_footer: Optional[bool] = None, clean_empty_lines: Optional[bool] = None, split_by: Optional[str] = None, split_length: Optional[int] = None, split_overlap: Optional[int] = None, split_respect_sentence_boundary: Optional[bool] = None) -> List[dict]
```

Perform document cleaning and splitting. Can take a single document or a list of documents as input and returns a list of documents.

<a id="preprocessor.PreProcessor.clean"></a>

#### clean

```python
def clean(document: dict, clean_whitespace: bool, clean_header_footer: bool, clean_empty_lines: bool) -> dict
```

Perform document cleaning on a single document and return a single document. This method will deal with whitespaces, headers, footers
and empty lines. Its exact functionality is defined by the parameters passed into PreProcessor.__init__().

<a id="preprocessor.PreProcessor.split"></a>

#### split

```python
def split(document: dict, split_by: str, split_length: int, split_overlap: int, split_respect_sentence_boundary: bool) -> List[dict]
```

Perform document splitting on a single document. This method can split on different units, at different lengths,
with different strides. It can also respect sentence boundaries. Its exact functionality is defined by
the parameters passed into PreProcessor.__init__(). Takes a single document as input and returns a list of documents.

