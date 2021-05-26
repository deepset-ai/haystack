<a name="base"></a>
# Module base

<a name="base.BaseTranslator"></a>
## BaseTranslator Objects

```python
class BaseTranslator(BaseComponent)
```

Abstract class for a Translator component that translates either a query or a doc from language A to language B.

<a name="base.BaseTranslator.translate"></a>
#### translate

```python
 | @abstractmethod
 | translate(query: Optional[str] = None, documents: Optional[Union[List[Document], List[str], List[Dict[str, Any]]]] = None, dict_key: Optional[str] = None, **kwargs) -> Union[str, List[Document], List[str], List[Dict[str, Any]]]
```

Translate the passed query or a list of documents from language A to B.

<a name="base.BaseTranslator.run"></a>
#### run

```python
 | run(query: Optional[str] = None, documents: Optional[Union[List[Document], List[str], List[Dict[str, Any]]]] = None, answers: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None, dict_key: Optional[str] = None, **kwargs)
```

Method that gets executed when this class is used as a Node in a Haystack Pipeline

<a name="transformers"></a>
# Module transformers

<a name="transformers.TransformersTranslator"></a>
## TransformersTranslator Objects

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

<a name="transformers.TransformersTranslator.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(model_name_or_path: str, tokenizer_name: Optional[str] = None, max_seq_len: Optional[int] = None, clean_up_tokenization_spaces: Optional[bool] = True)
```

Initialize the translator with a model that fits your targeted languages. While we support all seq2seq
models from Hugging Face's model hub, we recommend using the OPUS models from Helsiniki NLP. They provide plenty
of different models, usually one model per language pair and translation direction.
They have a pretty standardized naming that should help you find the right model:
- "Helsinki-NLP/opus-mt-en-de" => translating from English to German
- "Helsinki-NLP/opus-mt-de-en" => translating from German to English
- "Helsinki-NLP/opus-mt-fr-en" => translating from French to English
- "Helsinki-NLP/opus-mt-hi-en"=> translating from Hindi to English
...

They also have a few multilingual models that support multiple languages at once.

**Arguments**:

- `model_name_or_path`: Name of the seq2seq model that shall be used for translation.
                           Can be a remote name from Huggingface's modelhub or a local path.
- `tokenizer_name`: Optional tokenizer name. If not supplied, `model_name_or_path` will also be used for the
                       tokenizer.
- `max_seq_len`: The maximum sentence length the model accepts. (Optional)
- `clean_up_tokenization_spaces`: Whether or not to clean up the tokenization spaces. (default True)

<a name="transformers.TransformersTranslator.translate"></a>
#### translate

```python
 | translate(query: Optional[str] = None, documents: Optional[Union[List[Document], List[str], List[Dict[str, Any]]]] = None, dict_key: Optional[str] = None, **kwargs) -> Union[str, List[Document], List[str], List[Dict[str, Any]]]
```

Run the actual translation. You can supply a query or a list of documents. Whatever is supplied will be translated.

**Arguments**:

- `query`: The query string to translate
- `documents`: The documents to translate
- `dict_key`: 

