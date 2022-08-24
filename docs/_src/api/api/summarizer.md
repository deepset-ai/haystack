<a id="base"></a>

# Module base

<a id="base.BaseSummarizer"></a>

## BaseSummarizer

```python
class BaseSummarizer(BaseComponent)
```

Abstract class for Summarizer

<a id="base.BaseSummarizer.predict"></a>

#### BaseSummarizer.predict

```python
@abstractmethod
def predict(documents: List[Document], generate_single_summary: Optional[bool] = None) -> List[Document]
```

Abstract method for creating a summary.

**Arguments**:

- `documents`: Related documents (e.g. coming from a retriever) that the answer shall be conditioned on.
- `generate_single_summary`: Whether to generate a single summary for all documents or one summary per document.
If set to "True", all docs will be joined to a single string that will then
be summarized.
Important: The summary will depend on the order of the supplied documents!

**Returns**:

List of Documents, where Document.content contains the summarization and Document.meta["context"]
the original, not summarized text

<a id="transformers"></a>

# Module transformers

<a id="transformers.TransformersSummarizer"></a>

## TransformersSummarizer

```python
class TransformersSummarizer(BaseSummarizer)
```

Transformer based model to summarize the documents using the HuggingFace's transformers framework

You can use any model that has been fine-tuned on a summarization task. For example:
'`bart-large-cnn`', '`t5-small`', '`t5-base`', '`t5-large`', '`t5-3b`', '`t5-11b`'.
See the up-to-date list of available models on
`huggingface.co/models <https://huggingface.co/models?filter=summarization>`__

**Example**

```python
|     docs = [Document(text="PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions."
|            "The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be affected by"
|            "the shutoffs which were expected to last through at least midday tomorrow.")]
|
|     # Summarize
|     summary = summarizer.predict(
|        documents=docs,
|        generate_single_summary=True
|     )
|
|     # Show results (List of Documents, containing summary and original text)
|     print(summary)
|
|    [
|      {
|        "text": "California's largest electricity provider has turned off power to hundreds of thousands of customers.",
|        ...
|        "meta": {
|          "context": "PGE stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions. ..."
|              },
|        ...
|      },
```

<a id="transformers.TransformersSummarizer.__init__"></a>

#### TransformersSummarizer.\_\_init\_\_

```python
def __init__(model_name_or_path: str = "google/pegasus-xsum", model_version: Optional[str] = None, tokenizer: Optional[str] = None, max_length: int = 200, min_length: int = 5, use_gpu: bool = True, clean_up_tokenization_spaces: bool = True, separator_for_single_summary: str = " ", generate_single_summary: bool = False, batch_size: int = 16, progress_bar: bool = True, use_auth_token: Optional[Union[str, bool]] = None)
```

Load a Summarization model from Transformers.

See the up-to-date list of available models at
https://huggingface.co/models?filter=summarization

**Arguments**:

- `model_name_or_path`: Directory of a saved model or the name of a public model e.g.
'facebook/rag-token-nq', 'facebook/rag-sequence-nq'.
See https://huggingface.co/models?filter=summarization for full list of available models.
- `model_version`: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
- `tokenizer`: Name of the tokenizer (usually the same as model)
- `max_length`: Maximum length of summarized text
- `min_length`: Minimum length of summarized text
- `use_gpu`: Whether to use GPU (if available).
- `clean_up_tokenization_spaces`: Whether or not to clean up the potential extra spaces in the text output
- `separator_for_single_summary`: If `generate_single_summary=True` in `predict()`, we need to join all docs
into a single text. This separator appears between those subsequent docs.
- `generate_single_summary`: Whether to generate a single summary for all documents or one summary per document.
If set to "True", all docs will be joined to a single string that will then
be summarized.
Important: The summary will depend on the order of the supplied documents!
- `batch_size`: Number of documents to process at a time.
- `progress_bar`: Whether to show a progress bar.
- `use_auth_token`: The API token used to download private models from Huggingface.
If this parameter is set to `True`, then the token generated when running
`transformers-cli login` (stored in ~/.huggingface) will be used.
Additional information can be found here
https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained

<a id="transformers.TransformersSummarizer.predict"></a>

#### TransformersSummarizer.predict

```python
def predict(documents: List[Document], generate_single_summary: Optional[bool] = None) -> List[Document]
```

Produce the summarization from the supplied documents.

These document can for example be retrieved via the Retriever.

**Arguments**:

- `documents`: Related documents (e.g. coming from a retriever) that the answer shall be conditioned on.
- `generate_single_summary`: Whether to generate a single summary for all documents or one summary per document.
If set to "True", all docs will be joined to a single string that will then
be summarized.
Important: The summary will depend on the order of the supplied documents!

**Returns**:

List of Documents, where Document.text contains the summarization and Document.meta["context"]
the original, not summarized text

<a id="transformers.TransformersSummarizer.predict_batch"></a>

#### TransformersSummarizer.predict\_batch

```python
def predict_batch(documents: Union[List[Document], List[List[Document]]], generate_single_summary: Optional[bool] = None, batch_size: Optional[int] = None) -> Union[List[Document], List[List[Document]]]
```

Produce the summarization from the supplied documents.

These documents can for example be retrieved via the Retriever.

**Arguments**:

- `documents`: Single list of related documents or list of lists of related documents
(e.g. coming from a retriever) that the answer shall be conditioned on.
- `generate_single_summary`: Whether to generate a single summary for each provided document list or
one summary per document.
If set to "True", all docs of a document list will be joined to a single string
that will then be summarized.
Important: The summary will depend on the order of the supplied documents!
- `batch_size`: Number of Documents to process at a time.

