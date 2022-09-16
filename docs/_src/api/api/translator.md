<a id="base"></a>

# Module base

<a id="base.BaseTranslator"></a>

## BaseTranslator

```python
class BaseTranslator(BaseComponent)
```

Abstract class for a Translator component that translates either a query or a doc from language A to language B.

<a id="base.BaseTranslator.translate"></a>

#### BaseTranslator.translate

```python
@abstractmethod
def translate(
    results: List[Dict[str, Any]] = None,
    query: Optional[str] = None,
    documents: Optional[Union[List[Document], List[Answer], List[str],
                              List[Dict[str, Any]]]] = None,
    dict_key: Optional[str] = None
) -> Union[str, List[Document], List[Answer], List[str], List[Dict[str, Any]]]
```

Translate the passed query or a list of documents from language A to B.

<a id="base.BaseTranslator.run"></a>

#### BaseTranslator.run

```python
def run(results: List[Dict[str, Any]] = None,
        query: Optional[str] = None,
        documents: Optional[Union[List[Document], List[Answer], List[str],
                                  List[Dict[str, Any]]]] = None,
        answers: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        dict_key: Optional[str] = None)
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
|        Document(content="Heinz von Foerster was an Austrian American scientist combining physics and philosophy,
|                       and widely attributed as the originator of Second-order cybernetics.")
|    ]
|    translator = TransformersTranslator(model_name_or_path="Helsinki-NLP/opus-mt-en-de")
|    res = translator.translate(documents=DOCS, query=None)
```

<a id="transformers.TransformersTranslator.__init__"></a>

#### TransformersTranslator.\_\_init\_\_

```python
def __init__(model_name_or_path: str,
             tokenizer_name: Optional[str] = None,
             max_seq_len: Optional[int] = None,
             clean_up_tokenization_spaces: Optional[bool] = True,
             use_gpu: bool = True,
             progress_bar: bool = True,
             use_auth_token: Optional[Union[str, bool]] = None,
             devices: Optional[List[Union[str, torch.device]]] = None)
```

Initialize the translator with a model that fits your targeted languages. While we support all seq2seq

models from Hugging Face's model hub, we recommend using the OPUS models from Helsinki NLP. They provide plenty
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
- `use_gpu`: Whether to use GPU or the CPU. Falls back on CPU if no GPU is available.
- `progress_bar`: Whether to show a progress bar.
- `use_auth_token`: The API token used to download private models from Huggingface.
If this parameter is set to `True`, then the token generated when running
`transformers-cli login` (stored in ~/.huggingface) will be used.
Additional information can be found here
https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
- `devices`: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.
A list containing torch device objects and/or strings is supported (For example
[torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices
parameter is not used and a single cpu device is used for inference.

<a id="transformers.TransformersTranslator.translate"></a>

#### TransformersTranslator.translate

```python
def translate(
    results: Optional[List[Dict[str, Any]]] = None,
    query: Optional[str] = None,
    documents: Optional[Union[List[Document], List[Answer], List[str],
                              List[Dict[str, Any]]]] = None,
    dict_key: Optional[str] = None
) -> Union[str, List[Document], List[Answer], List[str], List[Dict[str, Any]]]
```

Run the actual translation. You can supply a query or a list of documents. Whatever is supplied will be translated.

**Arguments**:

- `results`: Generated QA pairs to translate
- `query`: The query string to translate
- `documents`: The documents to translate
- `dict_key`: If you pass a dictionary in `documents`, you can specify here the field which shall be translated.

<a id="transformers.TransformersTranslator.translate_batch"></a>

#### TransformersTranslator.translate\_batch

```python
def translate_batch(
    queries: Optional[List[str]] = None,
    documents: Optional[Union[List[Document], List[Answer],
                              List[List[Document]],
                              List[List[Answer]]]] = None,
    batch_size: Optional[int] = None
) -> List[Union[str, List[Document], List[Answer], List[str], List[Dict[
        str, Any]]]]
```

Run the actual translation. You can supply a single query, a list of queries or a list (of lists) of documents.

**Arguments**:

- `queries`: Single query or list of queries.
- `documents`: List of documents or list of lists of documets.
- `batch_size`: Not applicable.

