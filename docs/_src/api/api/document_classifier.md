<a name="base"></a>
# Module base

<a name="base.BaseDocumentClassifier"></a>
## BaseDocumentClassifier Objects

```python
class BaseDocumentClassifier(BaseComponent)
```

<a name="base.BaseDocumentClassifier.timing"></a>
#### timing

```python
 | timing(fn, attr_name)
```

Wrapper method used to time functions.

<a name="transformers"></a>
# Module transformers

<a name="transformers.TransformersDocumentClassifier"></a>
## TransformersDocumentClassifier Objects

```python
class TransformersDocumentClassifier(BaseDocumentClassifier)
```

Transformer based model for document classification using the HuggingFace's transformers framework
(https://github.com/huggingface/transformers).
While the underlying model can vary (BERT, Roberta, DistilBERT ...), the interface remains the same.
This node classifies documents and adds the output from the classification step to the document's meta data.
The meta field of the document is a dictionary with the following format:
``'meta': {'name': '450_Baelor.txt', 'classification': {'label': 'neutral', 'probability' = 0.9997646, ...} }``

With this document_classifier, you can directly get predictions via predict()

 **Usage example:**
 ```python
|    ...
|    retriever = ElasticsearchRetriever(document_store=document_store)
|    document_classifier = TransformersDocumentClassifier(model_name_or_path="bhadresh-savani/distilbert-base-uncased-emotion")
|    p = Pipeline()
|    p.add_node(component=retriever, name="Retriever", inputs=["Query"])
|    p.add_node(component=document_classifier, name="Classifier", inputs=["Retriever"])
|    res = p.run(
|        query="Who is the father of Arya Stark?",
|        params={"Retriever": {"top_k": 10}}
|    )
|    
|    # print the classification results
|    print_documents(res, max_text_len=100, print_meta=True)
|    # or access the predicted class label directly
|    res["documents"][0].to_dict()["meta"]["classification"]["label"]
 ```

<a name="transformers.TransformersDocumentClassifier.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(model_name_or_path: str = "bhadresh-savani/distilbert-base-uncased-emotion", model_version: Optional[str] = None, tokenizer: Optional[str] = None, use_gpu: bool = True, return_all_scores: bool = False, task: str = 'text-classification', labels: Optional[List[str]] = None)
```

Load a text classification model from Transformers.
Available models for the task of text-classification include:
- ``'bhadresh-savani/distilbert-base-uncased-emotion'``
- ``'Hate-speech-CNERG/dehatebert-mono-english'``

Available models for the task of zero-shot-classification include:
- ``'valhalla/distilbart-mnli-12-3'``
- ``'cross-encoder/nli-distilroberta-base'``

See https://huggingface.co/models for full list of available models.
Filter for text classification models: https://huggingface.co/models?pipeline_tag=text-classification&sort=downloads
Filter for zero-shot classification models (NLI): https://huggingface.co/models?pipeline_tag=zero-shot-classification&sort=downloads&search=nli

**Arguments**:

- `model_name_or_path`: Directory of a saved model or the name of a public model e.g. 'bhadresh-savani/distilbert-base-uncased-emotion'.
See https://huggingface.co/models for full list of available models.
- `model_version`: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
- `tokenizer`: Name of the tokenizer (usually the same as model)
- `use_gpu`: Whether to use GPU (if available).
- `return_all_scores`: Whether to return all prediction scores or just the one of the predicted class. Only used for task 'text-classification'.
- `task`: 'text-classification' or 'zero-shot-classification'
- `labels`: Only used for task 'zero-shot-classification'. List of string defining class labels, e.g.,
["positive", "negative"] otherwise None. Given a LABEL, the sequence fed to the model is "<cls> sequence to
classify <sep> This example is LABEL . <sep>" and the model predicts whether that sequence is a contradiction
or an entailment.

<a name="transformers.TransformersDocumentClassifier.predict"></a>
#### predict

```python
 | predict(documents: List[Document]) -> List[Document]
```

Returns documents containing classification result in meta field

**Arguments**:

- `documents`: List of Document to classify

**Returns**:

List of Document enriched with meta information

