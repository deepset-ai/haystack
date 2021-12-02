<a name="base"></a>
# Module base

<a name="base.RootNode"></a>
## RootNode

```python
class RootNode(BaseComponent)
```

RootNode feeds inputs together with corresponding params to a Pipeline.

<a name="base.BasePipeline"></a>
## BasePipeline

```python
class BasePipeline()
```

Base class for pipelines, providing the most basic methods to load and save them in different ways. 
See also the `Pipeline` class for the actual pipeline logic.

<a name="base.BasePipeline.load_from_yaml"></a>
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

<a name="base.Pipeline"></a>
## Pipeline

```python
class Pipeline(BasePipeline)
```

Pipeline brings together building blocks to build a complex search pipeline with Haystack & user-defined components.

Under-the-hood, a pipeline is represented as a directed acyclic graph of component nodes. It enables custom query
flows with options to branch queries(eg, extractive qa vs keyword match query), merge candidate documents for a
Reader from multiple Retrievers, or re-ranking of candidate documents.

<a name="base.Pipeline.add_node"></a>
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

<a name="base.Pipeline.get_node"></a>
#### get\_node

```python
 | get_node(name: str) -> Optional[BaseComponent]
```

Get a node from the Pipeline.

**Arguments**:

- `name`: The name of the node.

<a name="base.Pipeline.set_node"></a>
#### set\_node

```python
 | set_node(name: str, component)
```

Set the component for a node in the Pipeline.

**Arguments**:

- `name`: The name of the node.
- `component`: The component object to be set at the node.

<a name="base.Pipeline.run"></a>
#### run

```python
 | run(query: Optional[str] = None, file_paths: Optional[List[str]] = None, labels: Optional[MultiLabel] = None, documents: Optional[List[Document]] = None, meta: Optional[dict] = None, params: Optional[dict] = None, debug: Optional[bool] = None)
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
              they received and the output they generated. All debug information can 
              then be found in the dict returned by this method under the key "_debug"

<a name="base.Pipeline.eval"></a>
#### eval

```python
 | eval(labels: List[MultiLabel], params: Optional[dict] = None, sas_model_name_or_path: str = None) -> EvaluationResult
```

Evaluates the pipeline by running the pipeline once per query in debug mode
and putting together all data that is needed for evaluation, e.g. calculating metrics.

**Arguments**:

- `labels`: The labels to evaluate on
- `params`: Dictionary of parameters to be dispatched to the nodes.
            If you want to pass a param to all nodes, you can just use: {"top_k":10}
            If you want to pass it to targeted nodes, you can do:
            {"Retriever": {"top_k": 10}, "Reader": {"top_k": 3, "debug": True}}
- `sas_model_name_or_path`: Name or path of "Semantic Answer Similarity (SAS) model". When set, the model will be used to calculate similarity between predictions and labels and generate the SAS metric.
            The SAS metric correlates better with human judgement of correct answers as it does not rely on string overlaps.
            Example: Prediction = "30%", Label = "thirty percent", EM and F1 would be overly pessimistic with both being 0, while SAS paints a more realistic picture.
            More info in the paper: https://arxiv.org/abs/2108.06130
            Models:
            - You can use Bi Encoders (sentence transformers) or cross encoders trained on Semantic Textual Similarity (STS) data.
            Not all cross encoders can be used because of different return types.
            If you use custom cross encoders please make sure they work with sentence_transformers.CrossEncoder class
            - Good default for multiple languages: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
            - Large, powerful, but slow model for English only: "cross-encoder/stsb-roberta-large"
            - Large model for German only: "deepset/gbert-large-sts"

<a name="base.Pipeline.get_nodes_by_class"></a>
#### get\_nodes\_by\_class

```python
 | get_nodes_by_class(class_type) -> List[Any]
```

Gets all nodes in the pipeline that are an instance of a certain class (incl. subclasses).
This is for example helpful if you loaded a pipeline and then want to interact directly with the document store.
Example:
| from haystack.document_stores.base import BaseDocumentStore
| INDEXING_PIPELINE = Pipeline.load_from_yaml(Path(PIPELINE_YAML_PATH), pipeline_name=INDEXING_PIPELINE_NAME)
| res = INDEXING_PIPELINE.get_nodes_by_class(class_type=BaseDocumentStore)

**Returns**:

List of components that are an instance the requested class

<a name="base.Pipeline.get_document_store"></a>
#### get\_document\_store

```python
 | get_document_store() -> Optional[BaseDocumentStore]
```

Return the document store object used in the current pipeline.

**Returns**:

Instance of DocumentStore or None

<a name="base.Pipeline.draw"></a>
#### draw

```python
 | draw(path: Path = Path("pipeline.png"))
```

Create a Graphviz visualization of the pipeline.

**Arguments**:

- `path`: the path to save the image.

<a name="base.Pipeline.load_from_yaml"></a>
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

<a name="base.Pipeline.save_to_yaml"></a>
#### save\_to\_yaml

```python
 | save_to_yaml(path: Path, return_defaults: bool = False)
```

Save a YAML configuration for the Pipeline that can be used with `Pipeline.load_from_yaml()`.

**Arguments**:

- `path`: path of the output YAML file.
- `return_defaults`: whether to output parameters that have the default values.

<a name="base.Pipeline.print_eval_report"></a>
#### print\_eval\_report

```python
 | print_eval_report(eval_result: EvaluationResult, n_wrong_examples: int = 3, metrics_filter: Optional[Dict[str, List[str]]] = None)
```

Prints evaluation report containing a metrics funnel and worst queries for further analysis.

**Arguments**:

- `eval_result`: The evaluation result, can be obtained by running eval().
- `n_wrong_examples`: The number of worst queries to show.
- `metrics_filter`: The metrics to show per node. If None all metrics will be shown.

<a name="base.RayPipeline"></a>
## RayPipeline

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

<a name="base.RayPipeline.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(address: str = None, **kwargs)
```

**Arguments**:

- `address`: The IP address for the Ray cluster. If set to None, a local Ray instance is started.
- `kwargs`: Optional parameters for initializing Ray.

<a name="base.RayPipeline.load_from_yaml"></a>
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

<a name="base._RayDeploymentWrapper"></a>
## \_RayDeploymentWrapper

```python
class _RayDeploymentWrapper()
```

Ray Serve supports calling of __init__ methods on the Classes to create "deployment" instances.

In case of Haystack, some Components like Retrievers have complex init methods that needs objects
like Document Stores.

This wrapper class encapsulates the initialization of Components. Given a Component Class
name, it creates an instance using the YAML Pipeline config.

<a name="base._RayDeploymentWrapper.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(pipeline_config: dict, component_name: str)
```

Create an instance of Component.

**Arguments**:

- `pipeline_config`: Pipeline YAML parsed as a dict.
- `component_name`: Component Class name.

<a name="base._RayDeploymentWrapper.__call__"></a>
#### \_\_call\_\_

```python
 | __call__(*args, **kwargs)
```

Ray calls this method which is then re-directed to the corresponding component's run().

<a name="standard_pipelines"></a>
# Module standard\_pipelines

<a name="standard_pipelines.BaseStandardPipeline"></a>
## BaseStandardPipeline

```python
class BaseStandardPipeline(ABC)
```

Base class for pre-made standard Haystack pipelines.
This class does not inherit from Pipeline.

<a name="standard_pipelines.BaseStandardPipeline.add_node"></a>
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

<a name="standard_pipelines.BaseStandardPipeline.get_node"></a>
#### get\_node

```python
 | get_node(name: str)
```

Get a node from the Pipeline.

**Arguments**:

- `name`: The name of the node.

<a name="standard_pipelines.BaseStandardPipeline.set_node"></a>
#### set\_node

```python
 | set_node(name: str, component)
```

Set the component for a node in the Pipeline.

**Arguments**:

- `name`: The name of the node.
- `component`: The component object to be set at the node.

<a name="standard_pipelines.BaseStandardPipeline.draw"></a>
#### draw

```python
 | draw(path: Path = Path("pipeline.png"))
```

Create a Graphviz visualization of the pipeline.

**Arguments**:

- `path`: the path to save the image.

<a name="standard_pipelines.BaseStandardPipeline.save_to_yaml"></a>
#### save\_to\_yaml

```python
 | save_to_yaml(path: Path, return_defaults: bool = False)
```

Save a YAML configuration for the Pipeline that can be used with `Pipeline.load_from_yaml()`.

**Arguments**:

- `path`: path of the output YAML file.
- `return_defaults`: whether to output parameters that have the default values.

<a name="standard_pipelines.BaseStandardPipeline.load_from_yaml"></a>
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

<a name="standard_pipelines.BaseStandardPipeline.get_nodes_by_class"></a>
#### get\_nodes\_by\_class

```python
 | get_nodes_by_class(class_type) -> List[Any]
```

Gets all nodes in the pipeline that are an instance of a certain class (incl. subclasses).
This is for example helpful if you loaded a pipeline and then want to interact directly with the document store.
Example:
```python
| from haystack.document_stores.base import BaseDocumentStore
| INDEXING_PIPELINE = Pipeline.load_from_yaml(Path(PIPELINE_YAML_PATH), pipeline_name=INDEXING_PIPELINE_NAME)
| res = INDEXING_PIPELINE.get_nodes_by_class(class_type=BaseDocumentStore)
```

**Returns**:

List of components that are an instance of the requested class

<a name="standard_pipelines.BaseStandardPipeline.get_document_store"></a>
#### get\_document\_store

```python
 | get_document_store() -> Optional[BaseDocumentStore]
```

Return the document store object used in the current pipeline.

**Returns**:

Instance of DocumentStore or None

<a name="standard_pipelines.BaseStandardPipeline.eval"></a>
#### eval

```python
 | eval(labels: List[MultiLabel], params: Optional[dict], sas_model_name_or_path: str = None) -> EvaluationResult
```

Evaluates the pipeline by running the pipeline once per query in debug mode
and putting together all data that is needed for evaluation, e.g. calculating metrics.

**Arguments**:

- `labels`: The labels to evaluate on
- `params`: Params for the `retriever` and `reader`. For instance,
               params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
- `sas_model_name_or_path`: SentenceTransformers semantic textual similarity model to be used for sas value calculation,
                            should be path or string pointing to downloadable models.

<a name="standard_pipelines.ExtractiveQAPipeline"></a>
## ExtractiveQAPipeline

```python
class ExtractiveQAPipeline(BaseStandardPipeline)
```

Pipeline for Extractive Question Answering.

<a name="standard_pipelines.ExtractiveQAPipeline.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(reader: BaseReader, retriever: BaseRetriever)
```

**Arguments**:

- `reader`: Reader instance
- `retriever`: Retriever instance

<a name="standard_pipelines.ExtractiveQAPipeline.run"></a>
#### run

```python
 | run(query: str, params: Optional[dict] = None, debug: Optional[bool] = None)
```

**Arguments**:

- `query`: The search query string.
- `params`: Params for the `retriever` and `reader`. For instance,
               params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
- `debug`: Whether the pipeline should instruct nodes to collect debug information
              about their execution. By default these include the input parameters
              they received and the output they generated. 
              All debug information can then be found in the dict returned
              by this method under the key "_debug"

<a name="standard_pipelines.DocumentSearchPipeline"></a>
## DocumentSearchPipeline

```python
class DocumentSearchPipeline(BaseStandardPipeline)
```

Pipeline for semantic document search.

<a name="standard_pipelines.DocumentSearchPipeline.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(retriever: BaseRetriever)
```

**Arguments**:

- `retriever`: Retriever instance

<a name="standard_pipelines.DocumentSearchPipeline.run"></a>
#### run

```python
 | run(query: str, params: Optional[dict] = None, debug: Optional[bool] = None)
```

**Arguments**:

- `query`: the query string.
- `params`: params for the `retriever` and `reader`. For instance, params={"Retriever": {"top_k": 10}}
- `debug`: Whether the pipeline should instruct nodes to collect debug information
      about their execution. By default these include the input parameters
      they received and the output they generated.
      All debug information can then be found in the dict returned
      by this method under the key "_debug"

<a name="standard_pipelines.GenerativeQAPipeline"></a>
## GenerativeQAPipeline

```python
class GenerativeQAPipeline(BaseStandardPipeline)
```

Pipeline for Generative Question Answering.

<a name="standard_pipelines.GenerativeQAPipeline.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(generator: BaseGenerator, retriever: BaseRetriever)
```

**Arguments**:

- `generator`: Generator instance
- `retriever`: Retriever instance

<a name="standard_pipelines.GenerativeQAPipeline.run"></a>
#### run

```python
 | run(query: str, params: Optional[dict] = None, debug: Optional[bool] = None)
```

**Arguments**:

- `query`: the query string.
- `params`: params for the `retriever` and `generator`. For instance,
               params={"Retriever": {"top_k": 10}, "Generator": {"top_k": 5}}
- `debug`: Whether the pipeline should instruct nodes to collect debug information
      about their execution. By default these include the input parameters
      they received and the output they generated.
      All debug information can then be found in the dict returned
      by this method under the key "_debug"

<a name="standard_pipelines.SearchSummarizationPipeline"></a>
## SearchSummarizationPipeline

```python
class SearchSummarizationPipeline(BaseStandardPipeline)
```

Pipeline that retrieves documents for a query and then summarizes those documents.

<a name="standard_pipelines.SearchSummarizationPipeline.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(summarizer: BaseSummarizer, retriever: BaseRetriever, return_in_answer_format: bool = False)
```

**Arguments**:

- `summarizer`: Summarizer instance
- `retriever`: Retriever instance
- `return_in_answer_format`: Whether the results should be returned as documents (False) or in the answer
                                format used in other QA pipelines (True). With the latter, you can use this
                                pipeline as a "drop-in replacement" for other QA pipelines.

<a name="standard_pipelines.SearchSummarizationPipeline.run"></a>
#### run

```python
 | run(query: str, params: Optional[dict] = None, debug: Optional[bool] = None)
```

**Arguments**:

- `query`: the query string.
- `params`: params for the `retriever` and `summarizer`. For instance,
               params={"Retriever": {"top_k": 10}, "Summarizer": {"generate_single_summary": True}}
- `debug`: Whether the pipeline should instruct nodes to collect debug information
      about their execution. By default these include the input parameters
      they received and the output they generated.
      All debug information can then be found in the dict returned
      by this method under the key "_debug"

<a name="standard_pipelines.FAQPipeline"></a>
## FAQPipeline

```python
class FAQPipeline(BaseStandardPipeline)
```

Pipeline for finding similar FAQs using semantic document search.

<a name="standard_pipelines.FAQPipeline.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(retriever: BaseRetriever)
```

**Arguments**:

- `retriever`: Retriever instance

<a name="standard_pipelines.FAQPipeline.run"></a>
#### run

```python
 | run(query: str, params: Optional[dict] = None, debug: Optional[bool] = None)
```

**Arguments**:

- `query`: the query string.
- `params`: params for the `retriever`. For instance, params={"Retriever": {"top_k": 10}}
- `debug`: Whether the pipeline should instruct nodes to collect debug information
      about their execution. By default these include the input parameters
      they received and the output they generated.
      All debug information can then be found in the dict returned
      by this method under the key "_debug"

<a name="standard_pipelines.TranslationWrapperPipeline"></a>
## TranslationWrapperPipeline

```python
class TranslationWrapperPipeline(BaseStandardPipeline)
```

Takes an existing search pipeline and adds one "input translation node" after the Query and one
"output translation" node just before returning the results

<a name="standard_pipelines.TranslationWrapperPipeline.__init__"></a>
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

<a name="standard_pipelines.QuestionGenerationPipeline"></a>
## QuestionGenerationPipeline

```python
class QuestionGenerationPipeline(BaseStandardPipeline)
```

A simple pipeline that takes documents as input and generates
questions that it thinks can be answered by the documents.

<a name="standard_pipelines.RetrieverQuestionGenerationPipeline"></a>
## RetrieverQuestionGenerationPipeline

```python
class RetrieverQuestionGenerationPipeline(BaseStandardPipeline)
```

A simple pipeline that takes a query as input, performs retrieval, and then generates
questions that it thinks can be answered by the retrieved documents.

<a name="standard_pipelines.QuestionAnswerGenerationPipeline"></a>
## QuestionAnswerGenerationPipeline

```python
class QuestionAnswerGenerationPipeline(BaseStandardPipeline)
```

This is a pipeline which takes a document as input, generates questions that the model thinks can be answered by
this document, and then performs question answering of this questions using that single document.

<a name="standard_pipelines.MostSimilarDocumentsPipeline"></a>
## MostSimilarDocumentsPipeline

```python
class MostSimilarDocumentsPipeline(BaseStandardPipeline)
```

<a name="standard_pipelines.MostSimilarDocumentsPipeline.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(document_store: BaseDocumentStore)
```

Initialize a Pipeline for finding the most similar documents to a given document.
This pipeline can be helpful if you already show a relevant document to your end users and they want to search for just similar ones.

**Arguments**:

- `document_store`: Document Store instance with already stored embeddings.

<a name="standard_pipelines.MostSimilarDocumentsPipeline.run"></a>
#### run

```python
 | run(document_ids: List[str], top_k: int = 5)
```

**Arguments**:

- `document_ids`: document ids
- `top_k`: How many documents id to return against single document

