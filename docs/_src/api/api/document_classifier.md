<a id="base"></a>

# Module base

<a id="base.BaseDocumentClassifier"></a>

## BaseDocumentClassifier

```python
class BaseDocumentClassifier(BaseComponent)
```

<a id="base.BaseDocumentClassifier.timing"></a>

#### timing

```python
def timing(fn, attr_name)
```

Wrapper method used to time functions.

<a id="transformers"></a>

# Module transformers

<a id="transformers.TransformersDocumentClassifier"></a>

## TransformersDocumentClassifier

```python
class TransformersDocumentClassifier(BaseDocumentClassifier)
```

Transformer based model for document classification using the HuggingFace's transformers framework
(https://github.com/huggingface/transformers).
While the underlying model can vary (BERT, Roberta, DistilBERT ...), the interface remains the same.
This node classifies documents and adds the output from the classification step to the document's meta data.
The meta field of the document is a dictionary with the following format:
``'meta': {'name': '450_Baelor.txt', 'classification': {'label': 'neutral', 'probability' = 0.9997646, ...} }``

Classification is run on document's content field by default. If you want it to run on another field,
set the `classification_field` to one of document's meta fields.

With this document_classifier, you can directly get predictions via predict()

 **Usage example at query time:**
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

**Usage example at index time:**
 ```python
|    ...
|    converter = TextConverter()
|    preprocessor = Preprocessor()
|    document_store = ElasticsearchDocumentStore()
|    document_classifier = TransformersDocumentClassifier(model_name_or_path="bhadresh-savani/distilbert-base-uncased-emotion",
|                                                         batch_size=16)
|    p = Pipeline()
|    p.add_node(component=converter, name="TextConverter", inputs=["File"])
|    p.add_node(component=preprocessor, name="Preprocessor", inputs=["TextConverter"])
|    p.add_node(component=document_classifier, name="DocumentClassifier", inputs=["Preprocessor"])
|    p.add_node(component=document_store, name="DocumentStore", inputs=["DocumentClassifier"])
|    p.run(file_paths=file_paths)
 ```

<a id="transformers.TransformersDocumentClassifier.predict"></a>

#### predict

```python
def predict(documents: List[Document]) -> List[Document]
```

Returns documents containing classification result in meta field.

Documents are updated in place.

**Arguments**:

- `documents`: List of Document to classify

**Returns**:

List of Document enriched with meta information

