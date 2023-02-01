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

<a id="preprocessor.PreProcessor.__init__"></a>

#### \_\_init\_\_

```python
def __init__(clean_whitespace: bool = True, clean_header_footer: bool = False, clean_empty_lines: bool = True, split_by: str = "word", split_length: int = 200, split_overlap: int = 0, split_respect_sentence_boundary: bool = True, language: str = "en")
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
- `language`: The language used by "nltk.tokenize.sent_tokenize" in iso639 format. Available options: "en", "es", "de", "fr" & many more.

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
