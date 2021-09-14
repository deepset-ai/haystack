<a id="pipeline"></a>

# Module pipeline

<a id="pipeline.BasePipeline"></a>

## BasePipeline Objects

```python
class BasePipeline()
```

<a id="pipeline.BasePipeline.load_from_yaml"></a>

#### load\_from\_yaml

```python
@classmethod
def load_from_yaml(cls, path: Path, pipeline_name: Optional[str] = None, overwrite_with_env_variables: bool = True)
```

Load Pipeline from a YAML file defining the individual components and how they're tied together to form
a Pipeline. A single YAML can declare multiple Pipelines, in which case an explicit `pipeline_name` must
be passed.

Here's a sample configuration:

    ```yaml
    |   version: '0.8'
    |
    |    components:    # define all the building-blocks for Pipeline
    |    - name: MyReader       # custom-name for the component; helpful for visualization & debugging
    |      type: FARMReader    # Haystack Class name for the component
    |      params:
    |        no_ans_boost: -10
    |        model_name_or_path: deepset/roberta-base-squad2
    |    - name: MyESRetriever
    |      type: ElasticsearchRetriever
    |      params:
    |        document_store: MyDocumentStore    # params can reference other components defined in the YAML
    |        custom_query: null
    |    - name: MyDocumentStore
    |      type: ElasticsearchDocumentStore
    |      params:
    |        index: haystack_test
    |
    |    pipelines:    # multiple Pipelines can be defined using the components from above
    |    - name: my_query_pipeline    # a simple extractive-qa Pipeline
    |      nodes:
    |      - name: MyESRetriever
    |        inputs: [Query]
    |      - name: MyReader
    |        inputs: [MyESRetriever]
    ```

**Arguments**:

                                     to change index name param for an ElasticsearchDocumentStore, an env
                                     variable 'MYDOCSTORE_PARAMS_INDEX=documents-2021' can be set. Note that an
                                     `_` sign must be used to specify nested hierarchical properties.
- `path`: path of the YAML file.
- `pipeline_name`: if the YAML contains multiple pipelines, the pipeline_name to load must be set.
- `overwrite_with_env_variables`: Overwrite the YAML configuration with environment variables. For example,

<a id="pipeline.Pipeline"></a>

## Pipeline Objects

```python
class Pipeline(BasePipeline)
```

Pipeline brings together building blocks to build a complex search pipeline with Haystack & user-defined components.

Under-the-hood, a pipeline is represented as a directed acyclic graph of component nodes. It enables custom query
flows with options to branch queries(eg, extractive qa vs keyword match query), merge candidate documents for a
Reader from multiple Retrievers, or re-ranking of candidate documents.

<a id="pipeline.Pipeline.add_node"></a>

#### add\_node

```python
def add_node(component, name: str, inputs: List[str])
```

Add a new node to the pipeline.

**Arguments**:

                  (like Retriever, Reader, or Generator) or a user-defined object that implements a run()
                  method to process incoming data from predecessor node.
               of node is sufficient. For instance, a 'ElasticsearchRetriever' node would always output a single
               edge with a list of documents. It can be represented as ["ElasticsearchRetriever"].

               In cases when the predecessor node has multiple outputs, e.g., a "QueryClassifier", the output
               must be specified explicitly as "QueryClassifier.output_2".
- `component`: The object to be called when the data is passed to the node. It can be a Haystack component
- `name`: The name for the node. It must not contain any dots.
- `inputs`: A list of inputs to the node. If the predecessor node has a single outgoing edge, just the name

<a id="pipeline.Pipeline.get_node"></a>

#### get\_node

```python
def get_node(name: str) -> Optional[BaseComponent]
```

Get a node from the Pipeline.

**Arguments**:

- `name`: The name of the node.

<a id="pipeline.Pipeline.set_node"></a>

#### set\_node

```python
def set_node(name: str, component)
```

Set the component for a node in the Pipeline.

**Arguments**:

- `name`: The name of the node.
- `component`: The component object to be set at the node.

<a id="pipeline.Pipeline.get_nodes_by_class"></a>

#### get\_nodes\_by\_class

```python
def get_nodes_by_class(class_type) -> List[Any]
```

Gets all nodes in the pipeline that are an instance of a certain class (incl. subclasses).
This is for example helpful if you loaded a pipeline and then want to interact directly with the document store.
Example:
| from haystack.document_store.base import BaseDocumentStore
| INDEXING_PIPELINE = Pipeline.load_from_yaml(Path(PIPELINE_YAML_PATH), pipeline_name=INDEXING_PIPELINE_NAME)
| res = INDEXING_PIPELINE.get_nodes_by_class(class_type=BaseDocumentStore)

**Returns**:

List of components that are an instance the requested class

<a id="pipeline.Pipeline.get_document_store"></a>

#### get\_document\_store

```python
def get_document_store() -> Optional[BaseDocumentStore]
```

Return the document store object used in the current pipeline.

**Returns**:

Instance of DocumentStore or None

<a id="pipeline.Pipeline.draw"></a>

#### draw

```python
def draw(path: Path = Path("pipeline.png"))
```

Create a Graphviz visualization of the pipeline.

**Arguments**:

- `path`: the path to save the image.

<a id="pipeline.Pipeline.load_from_yaml"></a>

#### load\_from\_yaml

```python
@classmethod
def load_from_yaml(cls, path: Path, pipeline_name: Optional[str] = None, overwrite_with_env_variables: bool = True)
```

Load Pipeline from a YAML file defining the individual components and how they're tied together to form
a Pipeline. A single YAML can declare multiple Pipelines, in which case an explicit `pipeline_name` must
be passed.

Here's a sample configuration:

    ```yaml
    |   version: '0.8'
    |
    |    components:    # define all the building-blocks for Pipeline
    |    - name: MyReader       # custom-name for the component; helpful for visualization & debugging
    |      type: FARMReader    # Haystack Class name for the component
    |      params:
    |        no_ans_boost: -10
    |        model_name_or_path: deepset/roberta-base-squad2
    |    - name: MyESRetriever
    |      type: ElasticsearchRetriever
    |      params:
    |        document_store: MyDocumentStore    # params can reference other components defined in the YAML
    |        custom_query: null
    |    - name: MyDocumentStore
    |      type: ElasticsearchDocumentStore
    |      params:
    |        index: haystack_test
    |
    |    pipelines:    # multiple Pipelines can be defined using the components from above
    |    - name: my_query_pipeline    # a simple extractive-qa Pipeline
    |      nodes:
    |      - name: MyESRetriever
    |        inputs: [Query]
    |      - name: MyReader
    |        inputs: [MyESRetriever]
    ```

**Arguments**:

                                     to change index name param for an ElasticsearchDocumentStore, an env
                                     variable 'MYDOCSTORE_PARAMS_INDEX=documents-2021' can be set. Note that an
                                     `_` sign must be used to specify nested hierarchical properties.
- `path`: path of the YAML file.
- `pipeline_name`: if the YAML contains multiple pipelines, the pipeline_name to load must be set.
- `overwrite_with_env_variables`: Overwrite the YAML configuration with environment variables. For example,

<a id="pipeline.Pipeline.save_to_yaml"></a>

#### save\_to\_yaml

```python
def save_to_yaml(path: Path, return_defaults: bool = False)
```

Save a YAML configuration for the Pipeline that can be used with `Pipeline.load_from_yaml()`.

**Arguments**:

- `path`: path of the output YAML file.
- `return_defaults`: whether to output parameters that have the default values.

<a id="pipeline.BaseStandardPipeline"></a>

## BaseStandardPipeline Objects

```python
class BaseStandardPipeline(ABC)
```

<a id="pipeline.BaseStandardPipeline.add_node"></a>

#### add\_node

```python
def add_node(component, name: str, inputs: List[str])
```

Add a new node to the pipeline.

**Arguments**:

                  (like Retriever, Reader, or Generator) or a user-defined object that implements a run()
                  method to process incoming data from predecessor node.
               of node is sufficient. For instance, a 'ElasticsearchRetriever' node would always output a single
               edge with a list of documents. It can be represented as ["ElasticsearchRetriever"].

               In cases when the predecessor node has multiple outputs, e.g., a "QueryClassifier", the output
               must be specified explicitly as "QueryClassifier.output_2".
- `component`: The object to be called when the data is passed to the node. It can be a Haystack component
- `name`: The name for the node. It must not contain any dots.
- `inputs`: A list of inputs to the node. If the predecessor node has a single outgoing edge, just the name

<a id="pipeline.BaseStandardPipeline.get_node"></a>

#### get\_node

```python
def get_node(name: str)
```

Get a node from the Pipeline.

**Arguments**:

- `name`: The name of the node.

<a id="pipeline.BaseStandardPipeline.set_node"></a>

#### set\_node

```python
def set_node(name: str, component)
```

Set the component for a node in the Pipeline.

**Arguments**:

- `name`: The name of the node.
- `component`: The component object to be set at the node.

<a id="pipeline.BaseStandardPipeline.draw"></a>

#### draw

```python
def draw(path: Path = Path("pipeline.png"))
```

Create a Graphviz visualization of the pipeline.

**Arguments**:

- `path`: the path to save the image.

<a id="pipeline.ExtractiveQAPipeline"></a>

## ExtractiveQAPipeline Objects

```python
class ExtractiveQAPipeline(BaseStandardPipeline)
```

<a id="pipeline.ExtractiveQAPipeline.__init__"></a>

#### \_\_init\_\_

```python
def __init__(reader: BaseReader, retriever: BaseRetriever)
```

Initialize a Pipeline for Extractive Question Answering.

**Arguments**:

- `reader`: Reader instance
- `retriever`: Retriever instance

<a id="pipeline.ExtractiveQAPipeline.run"></a>

#### run

```python
def run(query: str, params: Optional[dict] = None)
```

**Arguments**:

               params={"retriever": {"top_k": 10}, "reader": {"top_k": 5}}
- `query`: the query string.
- `params`: params for the `retriever` and `reader`. For instance,

<a id="pipeline.DocumentSearchPipeline"></a>

## DocumentSearchPipeline Objects

```python
class DocumentSearchPipeline(BaseStandardPipeline)
```

<a id="pipeline.DocumentSearchPipeline.__init__"></a>

#### \_\_init\_\_

```python
def __init__(retriever: BaseRetriever)
```

Initialize a Pipeline for semantic document search.

**Arguments**:

- `retriever`: Retriever instance

<a id="pipeline.DocumentSearchPipeline.run"></a>

#### run

```python
def run(query: str, params: Optional[dict] = None)
```

**Arguments**:

- `query`: the query string.
- `params`: params for the `retriever` and `reader`. For instance, params={"retriever": {"top_k": 10}}

<a id="pipeline.GenerativeQAPipeline"></a>

## GenerativeQAPipeline Objects

```python
class GenerativeQAPipeline(BaseStandardPipeline)
```

<a id="pipeline.GenerativeQAPipeline.__init__"></a>

#### \_\_init\_\_

```python
def __init__(generator: BaseGenerator, retriever: BaseRetriever)
```

Initialize a Pipeline for Generative Question Answering.

**Arguments**:

- `generator`: Generator instance
- `retriever`: Retriever instance

<a id="pipeline.GenerativeQAPipeline.run"></a>

#### run

```python
def run(query: str, params: Optional[dict] = None)
```

**Arguments**:

               params={"retriever": {"top_k": 10}, "generator": {"top_k": 5}}
- `query`: the query string.
- `params`: params for the `retriever` and `generator`. For instance,

<a id="pipeline.SearchSummarizationPipeline"></a>

## SearchSummarizationPipeline Objects

```python
class SearchSummarizationPipeline(BaseStandardPipeline)
```

<a id="pipeline.SearchSummarizationPipeline.__init__"></a>

#### \_\_init\_\_

```python
def __init__(summarizer: BaseSummarizer, retriever: BaseRetriever, return_in_answer_format: bool = False)
```

Initialize a Pipeline that retrieves documents for a query and then summarizes those documents.

**Arguments**:

                                format used in other QA pipelines (True). With the latter, you can use this
                                pipeline as a "drop-in replacement" for other QA pipelines.
- `summarizer`: Summarizer instance
- `retriever`: Retriever instance
- `return_in_answer_format`: Whether the results should be returned as documents (False) or in the answer

<a id="pipeline.SearchSummarizationPipeline.run"></a>

#### run

```python
def run(query: str, params: Optional[dict] = None)
```

**Arguments**:

               params={"retriever": {"top_k": 10}, "summarizer": {"generate_single_summary": True}}
- `query`: the query string.
- `params`: params for the `retriever` and `summarizer`. For instance,

<a id="pipeline.FAQPipeline"></a>

## FAQPipeline Objects

```python
class FAQPipeline(BaseStandardPipeline)
```

<a id="pipeline.FAQPipeline.__init__"></a>

#### \_\_init\_\_

```python
def __init__(retriever: BaseRetriever)
```

Initialize a Pipeline for finding similar FAQs using semantic document search.

**Arguments**:

- `retriever`: Retriever instance

<a id="pipeline.FAQPipeline.run"></a>

#### run

```python
def run(query: str, params: Optional[dict] = None)
```

**Arguments**:

- `query`: the query string.
- `params`: params for the `retriever`. For instance, params={"retriever": {"top_k": 10}}

<a id="pipeline.TranslationWrapperPipeline"></a>

## TranslationWrapperPipeline Objects

```python
class TranslationWrapperPipeline(BaseStandardPipeline)
```

Takes an existing search pipeline and adds one "input translation node" after the Query and one
"output translation" node just before returning the results

<a id="pipeline.TranslationWrapperPipeline.__init__"></a>

#### \_\_init\_\_

```python
def __init__(input_translator: BaseTranslator, output_translator: BaseTranslator, pipeline: BaseStandardPipeline)
```

Wrap a given `pipeline` with the `input_translator` and `output_translator`.

**Arguments**:

                 Note that pipelines with split or merge nodes are currently not supported.
- `input_translator`: A Translator node that shall translate the input query from language A to B
- `output_translator`: A Translator node that shall translate the pipeline results from language B to A
- `pipeline`: The pipeline object (e.g. ExtractiveQAPipeline) you want to "wrap".

<a id="pipeline.QuestionGenerationPipeline"></a>

## QuestionGenerationPipeline Objects

```python
class QuestionGenerationPipeline(BaseStandardPipeline)
```

A simple pipeline that takes documents as input and generates
questions that it thinks can be answered by the documents.

<a id="pipeline.RetrieverQuestionGenerationPipeline"></a>

## RetrieverQuestionGenerationPipeline Objects

```python
class RetrieverQuestionGenerationPipeline(BaseStandardPipeline)
```

A simple pipeline that takes a query as input, performs retrieval, and then generates
questions that it thinks can be answered by the retrieved documents.

<a id="pipeline.QuestionAnswerGenerationPipeline"></a>

## QuestionAnswerGenerationPipeline Objects

```python
class QuestionAnswerGenerationPipeline(BaseStandardPipeline)
```

This is a pipeline which takes a document as input, generates questions that the model thinks can be answered by
this document, and then performs question answering of this questions using that single document.

<a id="pipeline.RootNode"></a>

## RootNode Objects

```python
class RootNode(BaseComponent)
```

RootNode feeds inputs together with corresponding params to a Pipeline.

<a id="pipeline.SklearnQueryClassifier"></a>

## SklearnQueryClassifier Objects

```python
class SklearnQueryClassifier(BaseComponent)
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

<a id="pipeline.SklearnQueryClassifier.__init__"></a>

#### \_\_init\_\_

```python
def __init__(model_name_or_path: Union[
            str, Any
        ] = "https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier/model.pickle", vectorizer_name_or_path: Union[
            str, Any
        ] = "https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier/vectorizer.pickle")
```

**Arguments**:

queries or statement vs question queries.
- `model_name_or_path`: Gradient boosting based binary classifier to classify between keyword vs statement/question
- `vectorizer_name_or_path`: A ngram based Tfidf vectorizer for extracting features from query.

<a id="pipeline.TransformersQueryClassifier"></a>

## TransformersQueryClassifier Objects

```python
class TransformersQueryClassifier(BaseComponent)
```

A node to classify an incoming query into one of two categories using a (small) BERT transformer model. Depending on the result, the query flows to a different branch in your pipeline
and the further processing can be customized. You can define this by connecting the further pipeline to either `output_1` or `output_2` from this node.

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
  
  Pass your own `Transformer` binary classification model from file/huggingface or use one of the following pretrained ones hosted on Huggingface:
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

<a id="pipeline.TransformersQueryClassifier.__init__"></a>

#### \_\_init\_\_

```python
def __init__(model_name_or_path: Union[
            Path, str
        ] = "shahrukhx01/bert-mini-finetune-question-detection")
```

**Arguments**:

- `model_name_or_path`: Transformer based fine tuned mini bert model for query classification

<a id="pipeline.JoinDocuments"></a>

## JoinDocuments Objects

```python
class JoinDocuments(BaseComponent)
```

A node to join documents outputted by multiple retriever nodes.

The node allows multiple join modes:
* concatenate: combine the documents from multiple nodes. Any duplicate documents are discarded.
* merge: merge scores of documents from multiple nodes. Optionally, each input score can be given a different
         `weight` & a `top_k` limit can be set. This mode can also be used for "reranking" retrieved documents.

<a id="pipeline.JoinDocuments.__init__"></a>

#### \_\_init\_\_

```python
def __init__(join_mode: str = "concatenate", weights: Optional[List[float]] = None, top_k_join: Optional[int] = None)
```

**Arguments**:

                  individual documents.
                adjusting document scores when using the `merge` join_mode. By default, equal weight is given
                to each retriever score. This param is not compatible with the `concatenate` join_mode.
- `join_mode`: `concatenate` to combine documents from multiple retrievers or `merge` to aggregate scores of
- `weights`: A node-wise list(length of list must be equal to the number of input nodes) of weights for
- `top_k_join`: Limit documents to top_k based on the resulting scores of the join.

<a id="pipeline.RayPipeline"></a>

## RayPipeline Objects

```python
class RayPipeline(Pipeline)
```

Ray (https://ray.io) is a framework for distributed computing.

Ray allows distributing a Pipeline's components across a cluster of machines. The individual components of a
Pipeline can be independently scaled. For instance, an extractive QA Pipeline deployment can have three replicas
of the Reader and a single replica for the Retriever. It enables efficient resource utilization by horizontally
scaling Components.

To set the number of replicas, add  `replicas` in the YAML config for the node in a pipeline:

        ```yaml
        |    components:
        |        ...
        |
        |    pipelines:
        |        - name: ray_query_pipeline
        |          type: RayPipeline
        |          nodes:
        |            - name: ESRetriever
        |              replicas: 2  # number of replicas to create on the Ray cluster
        |              inputs: [ Query ]
        ```

A RayPipeline can only be created with a YAML Pipeline config.
>>> from haystack.pipeline import RayPipeline
>>> pipeline = RayPipeline.load_from_yaml(path="my_pipelines.yaml", pipeline_name="my_query_pipeline")
>>> pipeline.run(query="What is the capital of Germany?")

By default, RayPipelines creates an instance of RayServe locally. To connect to an existing Ray instance,
set the `address` parameter when creating the RayPipeline instance.

<a id="pipeline.RayPipeline.__init__"></a>

#### \_\_init\_\_

```python
def __init__(address: str = None, **kwargs)
```

**Arguments**:

- `address`: The IP address for the Ray cluster. If set to None, a local Ray instance is started.
- `kwargs`: Optional parameters for initializing Ray.

<a id="pipeline.RayPipeline.load_from_yaml"></a>

#### load\_from\_yaml

```python
@classmethod
def load_from_yaml(cls, path: Path, pipeline_name: Optional[str] = None, overwrite_with_env_variables: bool = True, address: Optional[str] = None, **kwargs, ,)
```

Load Pipeline from a YAML file defining the individual components and how they're tied together to form
a Pipeline. A single YAML can declare multiple Pipelines, in which case an explicit `pipeline_name` must
be passed.

Here's a sample configuration:

    ```yaml
    |   version: '0.8'
    |
    |    components:    # define all the building-blocks for Pipeline
    |    - name: MyReader       # custom-name for the component; helpful for visualization & debugging
    |      type: FARMReader    # Haystack Class name for the component
    |      params:
    |        no_ans_boost: -10
    |        model_name_or_path: deepset/roberta-base-squad2
    |    - name: MyESRetriever
    |      type: ElasticsearchRetriever
    |      params:
    |        document_store: MyDocumentStore    # params can reference other components defined in the YAML
    |        custom_query: null
    |    - name: MyDocumentStore
    |      type: ElasticsearchDocumentStore
    |      params:
    |        index: haystack_test
    |
    |    pipelines:    # multiple Pipelines can be defined using the components from above
    |    - name: my_query_pipeline    # a simple extractive-qa Pipeline
    |      nodes:
    |      - name: MyESRetriever
    |        inputs: [Query]
    |      - name: MyReader
    |        inputs: [MyESRetriever]
    ```

**Arguments**:

                                     to change index name param for an ElasticsearchDocumentStore, an env
                                     variable 'MYDOCSTORE_PARAMS_INDEX=documents-2021' can be set. Note that an
                                     `_` sign must be used to specify nested hierarchical properties.
- `path`: path of the YAML file.
- `pipeline_name`: if the YAML contains multiple pipelines, the pipeline_name to load must be set.
- `overwrite_with_env_variables`: Overwrite the YAML configuration with environment variables. For example,
- `address`: The IP address for the Ray cluster. If set to None, a local Ray instance is started.

<a id="pipeline._RayDeploymentWrapper"></a>

## \_RayDeploymentWrapper Objects

```python
class _RayDeploymentWrapper()
```

Ray Serve supports calling of __init__ methods on the Classes to create "deployment" instances.

In case of Haystack, some Components like Retrievers have complex init methods that needs objects
like Document Stores.

This wrapper class encapsulates the initialization of Components. Given a Component Class
name, it creates an instance using the YAML Pipeline config.

<a id="pipeline._RayDeploymentWrapper.__init__"></a>

#### \_\_init\_\_

```python
def __init__(pipeline_config: dict, component_name: str)
```

Create an instance of Component.

**Arguments**:

- `pipeline_config`: Pipeline YAML parsed as a dict.
- `component_name`: Component Class name.

<a id="pipeline._RayDeploymentWrapper.__call__"></a>

#### \_\_call\_\_

```python
def __call__(*args, **kwargs)
```

Ray calls this method which is then re-directed to the corresponding component's run().

<a id="pipeline.MostSimilarDocumentsPipeline"></a>

## MostSimilarDocumentsPipeline Objects

```python
class MostSimilarDocumentsPipeline(BaseStandardPipeline)
```

<a id="pipeline.MostSimilarDocumentsPipeline.__init__"></a>

#### \_\_init\_\_

```python
def __init__(document_store: BaseDocumentStore)
```

Initialize a Pipeline for finding the most similar documents to a given document.
This pipeline can be helpful if you already show a relevant document to your end users and they want to search for just similar ones.

**Arguments**:

- `document_store`: Document Store instance with already stored embeddings.

<a id="pipeline.MostSimilarDocumentsPipeline.run"></a>

#### run

```python
def run(document_ids: List[str], top_k: int = 5)
```

**Arguments**:

- `document_ids`: document ids
- `top_k`: How many documents id to return against single document

