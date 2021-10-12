<a name="pipeline"></a>
# Module pipeline

<a name="pipeline.BasePipeline"></a>
## BasePipeline Objects

```python
class BasePipeline()
```

<a name="pipeline.BasePipeline.load_from_yaml"></a>
#### load\_from\_yaml

```python
 | @classmethod
 | load_from_yaml(cls, path: Path, pipeline_name: Optional[str] = None, overwrite_with_env_variables: bool = True)
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

- `path`: path of the YAML file.
- `pipeline_name`: if the YAML contains multiple pipelines, the pipeline_name to load must be set.
- `overwrite_with_env_variables`: Overwrite the YAML configuration with environment variables. For example,
                                     to change index name param for an ElasticsearchDocumentStore, an env
                                     variable 'MYDOCSTORE_PARAMS_INDEX=documents-2021' can be set. Note that an
                                     `_` sign must be used to specify nested hierarchical properties.

<a name="pipeline.Pipeline"></a>
## Pipeline Objects

```python
class Pipeline(BasePipeline)
```

Pipeline brings together building blocks to build a complex search pipeline with Haystack & user-defined components.

Under-the-hood, a pipeline is represented as a directed acyclic graph of component nodes. It enables custom query
flows with options to branch queries(eg, extractive qa vs keyword match query), merge candidate documents for a
Reader from multiple Retrievers, or re-ranking of candidate documents.

<a name="pipeline.Pipeline.add_node"></a>
#### add\_node

```python
 | add_node(component, name: str, inputs: List[str])
```

Add a new node to the pipeline.

**Arguments**:

- `component`: The object to be called when the data is passed to the node. It can be a Haystack component
                  (like Retriever, Reader, or Generator) or a user-defined object that implements a run()
                  method to process incoming data from predecessor node.
- `name`: The name for the node. It must not contain any dots.
- `inputs`: A list of inputs to the node. If the predecessor node has a single outgoing edge, just the name
               of node is sufficient. For instance, a 'ElasticsearchRetriever' node would always output a single
               edge with a list of documents. It can be represented as ["ElasticsearchRetriever"].

               In cases when the predecessor node has multiple outputs, e.g., a "QueryClassifier", the output
               must be specified explicitly as "QueryClassifier.output_2".

<a name="pipeline.Pipeline.get_node"></a>
#### get\_node

```python
 | get_node(name: str) -> Optional[BaseComponent]
```

Get a node from the Pipeline.

**Arguments**:

- `name`: The name of the node.

<a name="pipeline.Pipeline.set_node"></a>
#### set\_node

```python
 | set_node(name: str, component)
```

Set the component for a node in the Pipeline.

**Arguments**:

- `name`: The name of the node.
- `component`: The component object to be set at the node.

<a name="pipeline.Pipeline.run"></a>
#### run

```python
 | run(query: Optional[str] = None, file_paths: Optional[List[str]] = None, labels: Optional[MultiLabel] = None, documents: Optional[List[Document]] = None, meta: Optional[dict] = None, params: Optional[dict] = None, debug: Optional[bool] = None, debug_logs: Optional[bool] = None)
```

Runs the pipeline, one node at a time.

**Arguments**:

- `query`: The search query (for query pipelines only)
- `file_paths`: The files to index (for indexing pipelines only)
- `labels`: 
- `documents`: 
- `meta`: 
- `params`: Dictionary of parameters to be dispatched to the nodes.
               If you want to pass a param to all nodes, you can just use: {"top_k":10}
               If you want to pass it to targeted nodes, you can do:
               {"Retriever": {"top_k": 10}, "Reader": {"top_k": 3, "debug": True}}
- `debug`: Whether the pipeline should instruct nodes to collect debug information
              about their execution. By default these include the input parameters
              they received, the output they generated, and eventual logs (of any severity)
              emitted. All debug information can then be found in the dict returned
              by this method under the key "_debug"
- `debug_logs`: Whether all the logs of the node should be printed in the console,
                   regardless of their severity and of the existing logger's settings.

<a name="pipeline.Pipeline.get_nodes_by_class"></a>
#### get\_nodes\_by\_class

```python
 | get_nodes_by_class(class_type) -> List[Any]
```

Gets all nodes in the pipeline that are an instance of a certain class (incl. subclasses).
This is for example helpful if you loaded a pipeline and then want to interact directly with the document store.
Example:
| from haystack.document_store.base import BaseDocumentStore
| INDEXING_PIPELINE = Pipeline.load_from_yaml(Path(PIPELINE_YAML_PATH), pipeline_name=INDEXING_PIPELINE_NAME)
| res = INDEXING_PIPELINE.get_nodes_by_class(class_type=BaseDocumentStore)

**Returns**:

List of components that are an instance the requested class

<a name="pipeline.Pipeline.get_document_store"></a>
#### get\_document\_store

```python
 | get_document_store() -> Optional[BaseDocumentStore]
```

Return the document store object used in the current pipeline.

**Returns**:

Instance of DocumentStore or None

<a name="pipeline.Pipeline.draw"></a>
#### draw

```python
 | draw(path: Path = Path("pipeline.png"))
```

Create a Graphviz visualization of the pipeline.

**Arguments**:

- `path`: the path to save the image.

<a name="pipeline.Pipeline.load_from_yaml"></a>
#### load\_from\_yaml

```python
 | @classmethod
 | load_from_yaml(cls, path: Path, pipeline_name: Optional[str] = None, overwrite_with_env_variables: bool = True)
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

- `path`: path of the YAML file.
- `pipeline_name`: if the YAML contains multiple pipelines, the pipeline_name to load must be set.
- `overwrite_with_env_variables`: Overwrite the YAML configuration with environment variables. For example,
                                     to change index name param for an ElasticsearchDocumentStore, an env
                                     variable 'MYDOCSTORE_PARAMS_INDEX=documents-2021' can be set. Note that an
                                     `_` sign must be used to specify nested hierarchical properties.

<a name="pipeline.Pipeline.save_to_yaml"></a>
#### save\_to\_yaml

```python
 | save_to_yaml(path: Path, return_defaults: bool = False)
```

Save a YAML configuration for the Pipeline that can be used with `Pipeline.load_from_yaml()`.

**Arguments**:

- `path`: path of the output YAML file.
- `return_defaults`: whether to output parameters that have the default values.

<a name="pipeline.BaseStandardPipeline"></a>
## BaseStandardPipeline Objects

```python
class BaseStandardPipeline(ABC)
```

<a name="pipeline.BaseStandardPipeline.add_node"></a>
#### add\_node

```python
 | add_node(component, name: str, inputs: List[str])
```

Add a new node to the pipeline.

**Arguments**:

- `component`: The object to be called when the data is passed to the node. It can be a Haystack component
                  (like Retriever, Reader, or Generator) or a user-defined object that implements a run()
                  method to process incoming data from predecessor node.
- `name`: The name for the node. It must not contain any dots.
- `inputs`: A list of inputs to the node. If the predecessor node has a single outgoing edge, just the name
               of node is sufficient. For instance, a 'ElasticsearchRetriever' node would always output a single
               edge with a list of documents. It can be represented as ["ElasticsearchRetriever"].

               In cases when the predecessor node has multiple outputs, e.g., a "QueryClassifier", the output
               must be specified explicitly as "QueryClassifier.output_2".

<a name="pipeline.BaseStandardPipeline.get_node"></a>
#### get\_node

```python
 | get_node(name: str)
```

Get a node from the Pipeline.

**Arguments**:

- `name`: The name of the node.

<a name="pipeline.BaseStandardPipeline.set_node"></a>
#### set\_node

```python
 | set_node(name: str, component)
```

Set the component for a node in the Pipeline.

**Arguments**:

- `name`: The name of the node.
- `component`: The component object to be set at the node.

<a name="pipeline.BaseStandardPipeline.draw"></a>
#### draw

```python
 | draw(path: Path = Path("pipeline.png"))
```

Create a Graphviz visualization of the pipeline.

**Arguments**:

- `path`: the path to save the image.

<a name="pipeline.ExtractiveQAPipeline"></a>
## ExtractiveQAPipeline Objects

```python
class ExtractiveQAPipeline(BaseStandardPipeline)
```

<a name="pipeline.ExtractiveQAPipeline.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(reader: BaseReader, retriever: BaseRetriever)
```

Initialize a Pipeline for Extractive Question Answering.

**Arguments**:

- `reader`: Reader instance
- `retriever`: Retriever instance

<a name="pipeline.ExtractiveQAPipeline.run"></a>
#### run

```python
 | run(query: str, params: Optional[dict] = None)
```

**Arguments**:

- `query`: the query string.
- `params`: params for the `retriever` and `reader`. For instance,
               params={"retriever": {"top_k": 10}, "reader": {"top_k": 5}}

<a name="pipeline.DocumentSearchPipeline"></a>
## DocumentSearchPipeline Objects

```python
class DocumentSearchPipeline(BaseStandardPipeline)
```

<a name="pipeline.DocumentSearchPipeline.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(retriever: BaseRetriever)
```

Initialize a Pipeline for semantic document search.

**Arguments**:

- `retriever`: Retriever instance

<a name="pipeline.DocumentSearchPipeline.run"></a>
#### run

```python
 | run(query: str, params: Optional[dict] = None)
```

**Arguments**:

- `query`: the query string.
- `params`: params for the `retriever` and `reader`. For instance, params={"retriever": {"top_k": 10}}

<a name="pipeline.GenerativeQAPipeline"></a>
## GenerativeQAPipeline Objects

```python
class GenerativeQAPipeline(BaseStandardPipeline)
```

<a name="pipeline.GenerativeQAPipeline.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(generator: BaseGenerator, retriever: BaseRetriever)
```

Initialize a Pipeline for Generative Question Answering.

**Arguments**:

- `generator`: Generator instance
- `retriever`: Retriever instance

<a name="pipeline.GenerativeQAPipeline.run"></a>
#### run

```python
 | run(query: str, params: Optional[dict] = None)
```

**Arguments**:

- `query`: the query string.
- `params`: params for the `retriever` and `generator`. For instance,
               params={"retriever": {"top_k": 10}, "generator": {"top_k": 5}}

<a name="pipeline.SearchSummarizationPipeline"></a>
## SearchSummarizationPipeline Objects

```python
class SearchSummarizationPipeline(BaseStandardPipeline)
```

<a name="pipeline.SearchSummarizationPipeline.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(summarizer: BaseSummarizer, retriever: BaseRetriever, return_in_answer_format: bool = False)
```

Initialize a Pipeline that retrieves documents for a query and then summarizes those documents.

**Arguments**:

- `summarizer`: Summarizer instance
- `retriever`: Retriever instance
- `return_in_answer_format`: Whether the results should be returned as documents (False) or in the answer
                                format used in other QA pipelines (True). With the latter, you can use this
                                pipeline as a "drop-in replacement" for other QA pipelines.

<a name="pipeline.SearchSummarizationPipeline.run"></a>
#### run

```python
 | run(query: str, params: Optional[dict] = None)
```

**Arguments**:

- `query`: the query string.
- `params`: params for the `retriever` and `summarizer`. For instance,
               params={"retriever": {"top_k": 10}, "summarizer": {"generate_single_summary": True}}

<a name="pipeline.FAQPipeline"></a>
## FAQPipeline Objects

```python
class FAQPipeline(BaseStandardPipeline)
```

<a name="pipeline.FAQPipeline.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(retriever: BaseRetriever)
```

Initialize a Pipeline for finding similar FAQs using semantic document search.

**Arguments**:

- `retriever`: Retriever instance

<a name="pipeline.FAQPipeline.run"></a>
#### run

```python
 | run(query: str, params: Optional[dict] = None)
```

**Arguments**:

- `query`: the query string.
- `params`: params for the `retriever`. For instance, params={"retriever": {"top_k": 10}}

<a name="pipeline.TranslationWrapperPipeline"></a>
## TranslationWrapperPipeline Objects

```python
class TranslationWrapperPipeline(BaseStandardPipeline)
```

Takes an existing search pipeline and adds one "input translation node" after the Query and one
"output translation" node just before returning the results

<a name="pipeline.TranslationWrapperPipeline.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(input_translator: BaseTranslator, output_translator: BaseTranslator, pipeline: BaseStandardPipeline)
```

Wrap a given `pipeline` with the `input_translator` and `output_translator`.

**Arguments**:

- `input_translator`: A Translator node that shall translate the input query from language A to B
- `output_translator`: A Translator node that shall translate the pipeline results from language B to A
- `pipeline`: The pipeline object (e.g. ExtractiveQAPipeline) you want to "wrap".
                 Note that pipelines with split or merge nodes are currently not supported.

<a name="pipeline.QuestionGenerationPipeline"></a>
## QuestionGenerationPipeline Objects

```python
class QuestionGenerationPipeline(BaseStandardPipeline)
```

A simple pipeline that takes documents as input and generates
questions that it thinks can be answered by the documents.

<a name="pipeline.RetrieverQuestionGenerationPipeline"></a>
## RetrieverQuestionGenerationPipeline Objects

```python
class RetrieverQuestionGenerationPipeline(BaseStandardPipeline)
```

A simple pipeline that takes a query as input, performs retrieval, and then generates
questions that it thinks can be answered by the retrieved documents.

<a name="pipeline.QuestionAnswerGenerationPipeline"></a>
## QuestionAnswerGenerationPipeline Objects

```python
class QuestionAnswerGenerationPipeline(BaseStandardPipeline)
```

This is a pipeline which takes a document as input, generates questions that the model thinks can be answered by
this document, and then performs question answering of this questions using that single document.

<a name="pipeline.RootNode"></a>
## RootNode Objects

```python
class RootNode(BaseComponent)
```

RootNode feeds inputs together with corresponding params to a Pipeline.

<a name="pipeline.SklearnQueryClassifier"></a>
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

<a name="pipeline.SklearnQueryClassifier.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(model_name_or_path: Union[
 |             str, Any
 |         ] = "https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier/model.pickle", vectorizer_name_or_path: Union[
 |             str, Any
 |         ] = "https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier/vectorizer.pickle")
```

**Arguments**:

- `model_name_or_path`: Gradient boosting based binary classifier to classify between keyword vs statement/question
queries or statement vs question queries.
- `vectorizer_name_or_path`: A ngram based Tfidf vectorizer for extracting features from query.

<a name="pipeline.TransformersQueryClassifier"></a>
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

<a name="pipeline.TransformersQueryClassifier.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(model_name_or_path: Union[
 |             Path, str
 |         ] = "shahrukhx01/bert-mini-finetune-question-detection")
```

**Arguments**:

- `model_name_or_path`: Transformer based fine tuned mini bert model for query classification

<a name="pipeline.JoinDocuments"></a>
## JoinDocuments Objects

```python
class JoinDocuments(BaseComponent)
```

A node to join documents outputted by multiple retriever nodes.

The node allows multiple join modes:
* concatenate: combine the documents from multiple nodes. Any duplicate documents are discarded.
* merge: merge scores of documents from multiple nodes. Optionally, each input score can be given a different
         `weight` & a `top_k` limit can be set. This mode can also be used for "reranking" retrieved documents.

<a name="pipeline.JoinDocuments.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(join_mode: str = "concatenate", weights: Optional[List[float]] = None, top_k_join: Optional[int] = None)
```

**Arguments**:

- `join_mode`: `concatenate` to combine documents from multiple retrievers or `merge` to aggregate scores of
                  individual documents.
- `weights`: A node-wise list(length of list must be equal to the number of input nodes) of weights for
                adjusting document scores when using the `merge` join_mode. By default, equal weight is given
                to each retriever score. This param is not compatible with the `concatenate` join_mode.
- `top_k_join`: Limit documents to top_k based on the resulting scores of the join.

<a name="pipeline.RayPipeline"></a>
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

<a name="pipeline.RayPipeline.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(address: str = None, **kwargs)
```

**Arguments**:

- `address`: The IP address for the Ray cluster. If set to None, a local Ray instance is started.
- `kwargs`: Optional parameters for initializing Ray.

<a name="pipeline.RayPipeline.load_from_yaml"></a>
#### load\_from\_yaml

```python
 | @classmethod
 | load_from_yaml(cls, path: Path, pipeline_name: Optional[str] = None, overwrite_with_env_variables: bool = True, address: Optional[str] = None, **kwargs, ,)
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

- `path`: path of the YAML file.
- `pipeline_name`: if the YAML contains multiple pipelines, the pipeline_name to load must be set.
- `overwrite_with_env_variables`: Overwrite the YAML configuration with environment variables. For example,
                                     to change index name param for an ElasticsearchDocumentStore, an env
                                     variable 'MYDOCSTORE_PARAMS_INDEX=documents-2021' can be set. Note that an
                                     `_` sign must be used to specify nested hierarchical properties.
- `address`: The IP address for the Ray cluster. If set to None, a local Ray instance is started.

<a name="pipeline._RayDeploymentWrapper"></a>
## \_RayDeploymentWrapper Objects

```python
class _RayDeploymentWrapper()
```

Ray Serve supports calling of __init__ methods on the Classes to create "deployment" instances.

In case of Haystack, some Components like Retrievers have complex init methods that needs objects
like Document Stores.

This wrapper class encapsulates the initialization of Components. Given a Component Class
name, it creates an instance using the YAML Pipeline config.

<a name="pipeline._RayDeploymentWrapper.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(pipeline_config: dict, component_name: str)
```

Create an instance of Component.

**Arguments**:

- `pipeline_config`: Pipeline YAML parsed as a dict.
- `component_name`: Component Class name.

<a name="pipeline._RayDeploymentWrapper.__call__"></a>
#### \_\_call\_\_

```python
 | __call__(*args, **kwargs)
```

Ray calls this method which is then re-directed to the corresponding component's run().

<a name="pipeline.Docs2Answers"></a>
## Docs2Answers Objects

```python
class Docs2Answers(BaseComponent)
```

This Node is used to convert retrieved documents into predicted answers format.
It is useful for situations where you are calling a Retriever only pipeline via REST API.
This ensures that your output is in a compatible format.

<a name="pipeline.MostSimilarDocumentsPipeline"></a>
## MostSimilarDocumentsPipeline Objects

```python
class MostSimilarDocumentsPipeline(BaseStandardPipeline)
```

<a name="pipeline.MostSimilarDocumentsPipeline.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(document_store: BaseDocumentStore)
```

Initialize a Pipeline for finding the most similar documents to a given document.
This pipeline can be helpful if you already show a relevant document to your end users and they want to search for just similar ones.

**Arguments**:

- `document_store`: Document Store instance with already stored embeddings.

<a name="pipeline.MostSimilarDocumentsPipeline.run"></a>
#### run

```python
 | run(document_ids: List[str], top_k: int = 5)
```

**Arguments**:

- `document_ids`: document ids
- `top_k`: How many documents id to return against single document

