<a name="base"></a>
# Module base

<a name="base.BaseSummarizer"></a>
## BaseSummarizer Objects

```python
class BaseSummarizer(BaseComponent)
```

Abstract class for Summarizer

<a name="base.BaseSummarizer.predict"></a>
#### predict

```python
 | @abstractmethod
 | predict(documents: List[Document], generate_single_summary: Optional[bool] = None) -> List[Document]
```

Abstract method for creating a summary.

**Arguments**:

- `documents`: Related documents (e.g. coming from a retriever) that the answer shall be conditioned on.
- `generate_single_summary`: Whether to generate a single summary for all documents or one summary per document.
                                If set to "True", all docs will be joined to a single string that will then
                                be summarized.
                                Important: The summary will depend on the order of the supplied documents!

**Returns**:

List of Documents, where Document.text contains the summarization and Document.meta["context"]
         the original, not summarized text

<a name="transformers"></a>
# Module transformers

<a name="transformers.TransformersSummarizer"></a>
## TransformersSummarizer Objects

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

<a name="transformers.TransformersSummarizer.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(model_name_or_path: str = "google/pegasus-xsum", model_version: Optional[str] = None, tokenizer: Optional[str] = None, max_length: int = 200, min_length: int = 5, use_gpu: int = 0, clean_up_tokenization_spaces: bool = True, separator_for_single_summary: str = " ", generate_single_summary: bool = False)
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
- `use_gpu`: If < 0, then use cpu. If >= 0, this is the ordinal of the gpu to use
- `clean_up_tokenization_spaces`: Whether or not to clean up the potential extra spaces in the text output
- `separator_for_single_summary`: If `generate_single_summary=True` in `predict()`, we need to join all docs
                                     into a single text. This separator appears between those subsequent docs.
- `generate_single_summary`: Whether to generate a single summary for all documents or one summary per document.
                                If set to "True", all docs will be joined to a single string that will then
                                be summarized.
                                Important: The summary will depend on the order of the supplied documents!

<a name="transformers.TransformersSummarizer.predict"></a>
#### predict

```python
 | predict(documents: List[Document], generate_single_summary: Optional[bool] = None) -> List[Document]
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

