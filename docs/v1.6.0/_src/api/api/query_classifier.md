<a id="base"></a>

# Module base

<a id="base.BaseQueryClassifier"></a>

## BaseQueryClassifier

```python
class BaseQueryClassifier(BaseComponent)
```

Abstract class for Query Classifiers

<a id="sklearn"></a>

# Module sklearn

<a id="sklearn.SklearnQueryClassifier"></a>

## SklearnQueryClassifier

```python
class SklearnQueryClassifier(BaseQueryClassifier)
```

A node to classify an incoming query into one of two categories using a lightweight sklearn model. Depending on the result, the query flows to a different branch in your pipeline
and the further processing can be customized. You can define this by connecting the further pipeline to either `output_1` or `output_2` from this node.

**Example**:

  ```python
  |{
  |pipe = Pipeline()
  |pipe.add_node(component=SklearnQueryClassifier(), name="QueryClassifier", inputs=["Query"])
  |pipe.add_node(component=elastic_retriever, name="ElasticRetriever", inputs=["QueryClassifier.output_2"])
  |pipe.add_node(component=dpr_retriever, name="DPRRetriever", inputs=["QueryClassifier.output_1"])

  |# Keyword queries will use the ElasticRetriever
  |pipe.run("kubernetes aws")

  |# Semantic queries (questions, statements, sentences ...) will leverage the DPR retriever
  |pipe.run("How to manage kubernetes on aws")

  ```

  Models:

  Pass your own `Sklearn` binary classification model or use one of the following pretrained ones:
  1) Keywords vs. Questions/Statements (Default)
  query_classifier can be found [here](https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier/model.pickle)
  query_vectorizer can be found [here](https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier/vectorizer.pickle)
  output_1 => question/statement
  output_2 => keyword query
  [Readme](https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier/readme.txt)


  2) Questions vs. Statements
  query_classifier can be found [here](https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier_statements/model.pickle)
  query_vectorizer can be found [here](https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier_statements/vectorizer.pickle)
  output_1 => question
  output_2 => statement
  [Readme](https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier_statements/readme.txt)

  See also the [tutorial](https://haystack.deepset.ai/tutorials/pipelines) on pipelines.

<a id="sklearn.SklearnQueryClassifier.__init__"></a>

#### SklearnQueryClassifier.\_\_init\_\_

```python
def __init__(model_name_or_path: Union[
            str, Any
        ] = "https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier/model.pickle", vectorizer_name_or_path: Union[
            str, Any
        ] = "https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier/vectorizer.pickle", batch_size: Optional[int] = None)
```

**Arguments**:

- `model_name_or_path`: Gradient boosting based binary classifier to classify between keyword vs statement/question
queries or statement vs question queries.
- `vectorizer_name_or_path`: A ngram based Tfidf vectorizer for extracting features from query.
- `batch_size`: Number of queries to process at a time.

<a id="transformers"></a>

# Module transformers

<a id="transformers.TransformersQueryClassifier"></a>

## TransformersQueryClassifier

```python
class TransformersQueryClassifier(BaseQueryClassifier)
```

A node to classify an incoming query into one of two categories using a (small) BERT transformer model.
Depending on the result, the query flows to a different branch in your pipeline and the further processing
can be customized. You can define this by connecting the further pipeline to either `output_1` or `output_2`
from this node.

**Example**:

  ```python
  |{
  |pipe = Pipeline()
  |pipe.add_node(component=TransformersQueryClassifier(), name="QueryClassifier", inputs=["Query"])
  |pipe.add_node(component=elastic_retriever, name="ElasticRetriever", inputs=["QueryClassifier.output_2"])
  |pipe.add_node(component=dpr_retriever, name="DPRRetriever", inputs=["QueryClassifier.output_1"])

  |# Keyword queries will use the ElasticRetriever
  |pipe.run("kubernetes aws")

  |# Semantic queries (questions, statements, sentences ...) will leverage the DPR retriever
  |pipe.run("How to manage kubernetes on aws")

  ```

  Models:

  Pass your own `Transformer` binary classification model from file/huggingface or use one of the following
  pretrained ones hosted on Huggingface:
  1) Keywords vs. Questions/Statements (Default)
  model_name_or_path="shahrukhx01/bert-mini-finetune-question-detection"
  output_1 => question/statement
  output_2 => keyword query
  [Readme](https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier/readme.txt)


  2) Questions vs. Statements
  `model_name_or_path`="shahrukhx01/question-vs-statement-classifier"
  output_1 => question
  output_2 => statement
  [Readme](https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier_statements/readme.txt)


  See also the [tutorial](https://haystack.deepset.ai/tutorials/pipelines) on pipelines.

<a id="transformers.TransformersQueryClassifier.__init__"></a>

#### TransformersQueryClassifier.\_\_init\_\_

```python
def __init__(model_name_or_path: Union[Path, str] = "shahrukhx01/bert-mini-finetune-question-detection", use_gpu: bool = True, batch_size: Optional[int] = None)
```

**Arguments**:

- `model_name_or_path`: Transformer based fine tuned mini bert model for query classification
- `use_gpu`: Whether to use GPU (if available).
