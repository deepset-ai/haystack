<a id="base"></a>

# Module base

<a id="base.BaseSummarizer"></a>

## BaseSummarizer

```python
class BaseSummarizer(BaseComponent)
```

Abstract class for Summarizer

<a id="base.BaseSummarizer.predict"></a>

#### predict

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

List of Documents, where Document.text contains the summarization and Document.meta["context"]
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

<a id="transformers.TransformersSummarizer.predict"></a>

#### predict

```python
def predict(documents: List[Document], generate_single_summary: Optional[bool] = None, truncation: bool = True) -> List[Document]
```

Produce the summarization from the supplied documents.

These document can for example be retrieved via the Retriever.

**Arguments**:

- `documents`: Related documents (e.g. coming from a retriever) that the answer shall be conditioned on.
- `generate_single_summary`: Whether to generate a single summary for all documents or one summary per document.
If set to "True", all docs will be joined to a single string that will then
be summarized.
Important: The summary will depend on the order of the supplied documents!
- `truncation`: Truncate to a maximum length accepted by the model

**Returns**:

List of Documents, where Document.text contains the summarization and Document.meta["context"]
the original, not summarized text

