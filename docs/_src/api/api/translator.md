<a id="base"></a>

# Module base

<a id="base.BaseTranslator"></a>

## BaseTranslator

```python
class BaseTranslator(BaseComponent)
```

Abstract class for a Translator component that translates either a query or a doc from language A to language B.

<a id="base.BaseTranslator.translate"></a>

#### translate

```python
@abstractmethod
def translate(results: List[Dict[str, Any]] = None, query: Optional[str] = None, documents: Optional[Union[List[Document], List[Answer], List[str], List[Dict[str, Any]]]] = None, dict_key: Optional[str] = None) -> Union[str, List[Document], List[Answer], List[str], List[Dict[str, Any]]]
```

Translate the passed query or a list of documents from language A to B.

<a id="base.BaseTranslator.run"></a>

#### run

```python
def run(results: List[Dict[str, Any]] = None, query: Optional[str] = None, documents: Optional[Union[List[Document], List[Answer], List[str], List[Dict[str, Any]]]] = None, answers: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None, dict_key: Optional[str] = None)
```

Method that gets executed when this class is used as a Node in a Haystack Pipeline

<a id="transformers"></a>

# Module transformers

<a id="transformers.TransformersTranslator"></a>

## TransformersTranslator

```python
class TransformersTranslator(BaseTranslator)
```

Translator component based on Seq2Seq models from Huggingface's transformers library.
Exemplary use cases:
- Translate a query from Language A to B (e.g. if you only have good models + documents in language B)
- Translate a document from Language A to B (e.g. if you want to return results in the native language of the user)

We currently recommend using OPUS models (see __init__() for details)

**Example:**

```python
|    DOCS = [
|        Document(text="Heinz von Foerster was an Austrian American scientist combining physics and philosophy,
|                       and widely attributed as the originator of Second-order cybernetics.")
|    ]
|    translator = TransformersTranslator(model_name_or_path="Helsinki-NLP/opus-mt-en-de")
|    res = translator.translate(documents=DOCS, query=None)
```

<a id="transformers.TransformersTranslator.translate"></a>

#### translate

```python
def translate(results: List[Dict[str, Any]] = None, query: Optional[str] = None, documents: Optional[Union[List[Document], List[Answer], List[str], List[Dict[str, Any]]]] = None, dict_key: Optional[str] = None) -> Union[str, List[Document], List[Answer], List[str], List[Dict[str, Any]]]
```

Run the actual translation. You can supply a query or a list of documents. Whatever is supplied will be translated.

**Arguments**:

- `results`: Generated QA pairs to translate
- `query`: The query string to translate
- `documents`: The documents to translate
- `dict_key`: If you pass a dictionary in `documents`, you can specify here the field which shall be translated.

