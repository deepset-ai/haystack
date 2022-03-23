<a id="base"></a>

# Module base

<a id="base.RootNode"></a>

## RootNode

```python
class RootNode(BaseComponent)
```

RootNode feeds inputs together with corresponding params to a Pipeline.

<a id="base.BasePipeline"></a>

## BasePipeline

```python
class BasePipeline(ABC)
```

Base class for pipelines, providing the most basic methods to load and save them in different ways.
See also the `Pipeline` class for the actual pipeline logic.

<a id="base.BasePipeline.get_config"></a>

#### get\_config

```python
@abstractmethod
def get_config(return_defaults: bool = False) -> dict
```

Returns a configuration for the Pipeline that can be used with `Pipeline.load_from_config()`.

**Arguments**:

- `return_defaults`: whether to output parameters that have the default values.

<a id="base.BasePipeline.to_code"></a>

#### to\_code

```python
def to_code(pipeline_variable_name: str = "pipeline", generate_imports: bool = True, add_comment: bool = False) -> str
```

Returns the code to create this pipeline as string.

**Arguments**:

- `pipeline_variable_name`: The variable name of the generated pipeline.
Default value is 'pipeline'.
- `generate_imports`: Whether to include the required import statements into the code.
Default value is True.
- `add_comment`: Whether to add a preceding comment that this code has been generated.
Default value is False.

<a id="base.BasePipeline.to_notebook_cell"></a>

#### to\_notebook\_cell

```python
def to_notebook_cell(pipeline_variable_name: str = "pipeline", generate_imports: bool = True, add_comment: bool = True)
```

Creates a new notebook cell with the code to create this pipeline.

**Arguments**:

- `pipeline_variable_name`: The variable name of the generated pipeline.
Default value is 'pipeline'.
- `generate_imports`: Whether to include the required import statements into the code.
Default value is True.
- `add_comment`: Whether to add a preceding comment that this code has been generated.
Default value is True.

<a id="base.BasePipeline.load_from_config"></a>

#### load\_from\_config

```python
@classmethod
@abstractmethod
def load_from_config(cls, pipeline_config: Dict, pipeline_name: Optional[str] = None, overwrite_with_env_variables: bool = True)
```

Load Pipeline from a config dict defining the individual components and how they're tied together to form

a Pipeline. A single config can declare multiple Pipelines, in which case an explicit `pipeline_name` must
be passed.

Here's a sample configuration:

    ```python
    |   {
    |       "version": "1.0",
    |       "components": [
    |           {  # define all the building-blocks for Pipeline
    |               "name": "MyReader",  # custom-name for the component; helpful for visualization & debugging
    |               "type": "FARMReader",  # Haystack Class name for the component
    |               "params": {"no_ans_boost": -10, "model_name_or_path": "deepset/roberta-base-squad2"},
    |           },
    |           {
    |               "name": "MyESRetriever",
    |               "type": "ElasticsearchRetriever",
    |               "params": {
    |                   "document_store": "MyDocumentStore",  # params can reference other components defined in the YAML
    |                   "custom_query": None,
    |               },
    |           },
    |           {"name": "MyDocumentStore", "type": "ElasticsearchDocumentStore", "params": {"index": "haystack_test"}},
    |       ],
    |       "pipelines": [
    |           {  # multiple Pipelines can be defined using the components from above
    |               "name": "my_query_pipeline",  # a simple extractive-qa Pipeline
    |               "nodes": [
    |                   {"name": "MyESRetriever", "inputs": ["Query"]},
    |                   {"name": "MyReader", "inputs": ["MyESRetriever"]},
    |               ],
    |           }
    |       ],
    |   }
    ```

**Arguments**:

- `pipeline_config`: the pipeline config as dict
- `pipeline_name`: if the config contains multiple pipelines, the pipeline_name to load must be set.
- `overwrite_with_env_variables`: Overwrite the configuration with environment variables. For example,
to change index name param for an ElasticsearchDocumentStore, an env
variable 'MYDOCSTORE_PARAMS_INDEX=documents-2021' can be set. Note that an
`_` sign must be used to specify nested hierarchical properties.

<a id="base.BasePipeline.load_from_yaml"></a>

#### load\_from\_yaml

```python
@classmethod
@abstractmethod
def load_from_yaml(cls, path: Path, pipeline_name: Optional[str] = None, overwrite_with_env_variables: bool = True)
```

Load Pipeline from a YAML file defining the individual components and how they're tied together to form

a Pipeline. A single YAML can declare multiple Pipelines, in which case an explicit `pipeline_name` must
be passed.

Here's a sample configuration:

    ```yaml
    |   version: '1.0'
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

Note that, in case of a mismatch in version between Haystack and the YAML, a warning will be printed.
If the pipeline loads correctly regardless, save again the pipeline using `Pipeline.save_to_yaml()` to remove the warning.

**Arguments**:

- `path`: path of the YAML file.
- `pipeline_name`: if the YAML contains multiple pipelines, the pipeline_name to load must be set.
- `overwrite_with_env_variables`: Overwrite the YAML configuration with environment variables. For example,
to change index name param for an ElasticsearchDocumentStore, an env
variable 'MYDOCSTORE_PARAMS_INDEX=documents-2021' can be set. Note that an
`_` sign must be used to specify nested hierarchical properties.

<a id="base.BasePipeline.load_from_deepset_cloud"></a>

#### load\_from\_deepset\_cloud

```python
@classmethod
def load_from_deepset_cloud(cls, pipeline_config_name: str, pipeline_name: str = "query", workspace: str = "default", api_key: Optional[str] = None, api_endpoint: Optional[str] = None, overwrite_with_env_variables: bool = False)
```

Load Pipeline from Deepset Cloud defining the individual components and how they're tied together to form

a Pipeline. A single config can declare multiple Pipelines, in which case an explicit `pipeline_name` must
be passed.

In order to get a list of all available pipeline_config_names, call `list_pipelines_on_deepset_cloud()`.
Use the returned `name` as `pipeline_config_name`.

**Arguments**:

- `pipeline_config_name`: name of the config file inside the Deepset Cloud workspace.
To get a list of all available pipeline_config_names, call `list_pipelines_on_deepset_cloud()`.
- `pipeline_name`: specifies which pipeline to load from config.
Deepset Cloud typically provides a 'query' and a 'index' pipeline per config.
- `workspace`: workspace in Deepset Cloud
- `api_key`: Secret value of the API key.
If not specified, will be read from DEEPSET_CLOUD_API_KEY environment variable.
- `api_endpoint`: The URL of the Deepset Cloud API.
If not specified, will be read from DEEPSET_CLOUD_API_ENDPOINT environment variable.
- `overwrite_with_env_variables`: Overwrite the config with environment variables. For example,
to change return_no_answer param for a FARMReader, an env
variable 'READER_PARAMS_RETURN_NO_ANSWER=False' can be set. Note that an
`_` sign must be used to specify nested hierarchical properties.

<a id="base.BasePipeline.list_pipelines_on_deepset_cloud"></a>

#### list\_pipelines\_on\_deepset\_cloud

```python
@classmethod
def list_pipelines_on_deepset_cloud(cls, workspace: str = "default", api_key: Optional[str] = None, api_endpoint: Optional[str] = None) -> List[dict]
```

Lists all pipeline configs available on Deepset Cloud.

**Arguments**:

- `workspace`: workspace in Deepset Cloud
- `api_key`: Secret value of the API key.
If not specified, will be read from DEEPSET_CLOUD_API_KEY environment variable.
- `api_endpoint`: The URL of the Deepset Cloud API.
If not specified, will be read from DEEPSET_CLOUD_API_ENDPOINT environment variable.

Returns:
    list of dictionaries: List[dict]
    each dictionary: {
                "name": str -> `pipeline_config_name` to be used in `load_from_deepset_cloud()`,
                "..." -> additional pipeline meta information
                }
    example:
            [{'name': 'my_super_nice_pipeline_config',
                'pipeline_id': '2184e0c1-c6ec-40a1-9b28-5d2768e5efa2',
                'status': 'DEPLOYED',
                'created_at': '2022-02-01T09:57:03.803991+00:00',
                'deleted': False,
                'is_default': False,
                'indexing': {'status': 'IN_PROGRESS',
                'pending_file_count': 3,
                'total_file_count': 31}}]

<a id="base.BasePipeline.save_to_deepset_cloud"></a>

#### save\_to\_deepset\_cloud

```python
@classmethod
def save_to_deepset_cloud(cls, query_pipeline: BasePipeline, index_pipeline: BasePipeline, pipeline_config_name: str, workspace: str = "default", api_key: Optional[str] = None, api_endpoint: Optional[str] = None, overwrite: bool = False)
```

Saves a Pipeline config to Deepset Cloud defining the individual components and how they're tied together to form

a Pipeline. A single config must declare a query pipeline and a index pipeline.

**Arguments**:

- `query_pipeline`: the query pipeline to save.
- `index_pipeline`: the index pipeline to save.
- `pipeline_config_name`: name of the config file inside the Deepset Cloud workspace.
- `workspace`: workspace in Deepset Cloud
- `api_key`: Secret value of the API key.
If not specified, will be read from DEEPSET_CLOUD_API_KEY environment variable.
- `api_endpoint`: The URL of the Deepset Cloud API.
If not specified, will be read from DEEPSET_CLOUD_API_ENDPOINT environment variable.
- `overwrite`: Whether to overwrite the config if it already exists. Otherwise an error is being raised.

<a id="base.BasePipeline.deploy_on_deepset_cloud"></a>

#### deploy\_on\_deepset\_cloud

```python
@classmethod
def deploy_on_deepset_cloud(cls, pipeline_config_name: str, workspace: str = "default", api_key: Optional[str] = None, api_endpoint: Optional[str] = None, timeout: int = 60)
```

Deploys the pipelines of a pipeline config on Deepset Cloud.

Blocks until pipelines are successfully deployed, deployment failed or timeout exceeds.
If pipelines are already deployed no action will be taken and an info will be logged.
If timeout exceeds a TimeoutError will be raised.
If deployment fails a DeepsetCloudError will be raised.

Pipeline config must be present on Deepset Cloud. See save_to_deepset_cloud() for more information.

**Arguments**:

- `pipeline_config_name`: name of the config file inside the Deepset Cloud workspace.
- `workspace`: workspace in Deepset Cloud
- `api_key`: Secret value of the API key.
If not specified, will be read from DEEPSET_CLOUD_API_KEY environment variable.
- `api_endpoint`: The URL of the Deepset Cloud API.
If not specified, will be read from DEEPSET_CLOUD_API_ENDPOINT environment variable.
- `timeout`: The time in seconds to wait until deployment completes.
If the timeout is exceeded an error will be raised.

<a id="base.BasePipeline.undeploy_on_deepset_cloud"></a>

#### undeploy\_on\_deepset\_cloud

```python
@classmethod
def undeploy_on_deepset_cloud(cls, pipeline_config_name: str, workspace: str = "default", api_key: Optional[str] = None, api_endpoint: Optional[str] = None, timeout: int = 60)
```

Undeploys the pipelines of a pipeline config on Deepset Cloud.

Blocks until pipelines are successfully undeployed, undeployment failed or timeout exceeds.
If pipelines are already undeployed no action will be taken and an info will be logged.
If timeout exceeds a TimeoutError will be raised.
If deployment fails a DeepsetCloudError will be raised.

Pipeline config must be present on Deepset Cloud. See save_to_deepset_cloud() for more information.

**Arguments**:

- `pipeline_config_name`: name of the config file inside the Deepset Cloud workspace.
- `workspace`: workspace in Deepset Cloud
- `api_key`: Secret value of the API key.
If not specified, will be read from DEEPSET_CLOUD_API_KEY environment variable.
- `api_endpoint`: The URL of the Deepset Cloud API.
If not specified, will be read from DEEPSET_CLOUD_API_ENDPOINT environment variable.
- `timeout`: The time in seconds to wait until undeployment completes.
If the timeout is exceeded an error will be raised.

<a id="base.Pipeline"></a>

## Pipeline

```python
class Pipeline(BasePipeline)
```

Pipeline brings together building blocks to build a complex search pipeline with Haystack & user-defined components.

Under-the-hood, a pipeline is represented as a directed acyclic graph of component nodes. It enables custom query
flows with options to branch queries(eg, extractive qa vs keyword match query), merge candidate documents for a
Reader from multiple Retrievers, or re-ranking of candidate documents.

<a id="base.Pipeline.add_node"></a>

#### add\_node

```python
def add_node(component: BaseComponent, name: str, inputs: List[str])
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

<a id="base.Pipeline.get_node"></a>

#### get\_node

```python
def get_node(name: str) -> Optional[BaseComponent]
```

Get a node from the Pipeline.

**Arguments**:

- `name`: The name of the node.

<a id="base.Pipeline.set_node"></a>

#### set\_node

```python
def set_node(name: str, component)
```

Set the component for a node in the Pipeline.

**Arguments**:

- `name`: The name of the node.
- `component`: The component object to be set at the node.

<a id="base.Pipeline.run"></a>

#### run

```python
def run(query: Optional[str] = None, file_paths: Optional[List[str]] = None, labels: Optional[MultiLabel] = None, documents: Optional[List[Document]] = None, meta: Optional[Union[dict, List[dict]]] = None, params: Optional[dict] = None, debug: Optional[bool] = None)
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

<a id="base.Pipeline.eval_beir"></a>

#### eval\_beir

```python
@classmethod
def eval_beir(cls, index_pipeline: Pipeline, query_pipeline: Pipeline, index_params: dict = {}, query_params: dict = {}, dataset: str = "scifact", dataset_dir: Path = Path("."), top_k_values: List[int] = [1, 3, 5, 10, 100, 1000], keep_index: bool = False) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]
```

Runs information retrieval evaluation of a pipeline using BEIR on a specified BEIR dataset.

See https://github.com/beir-cellar/beir for more information.

**Arguments**:

- `index_pipeline`: The indexing pipeline to use.
- `query_pipeline`: The query pipeline to evaluate.
- `index_params`: The params to use during indexing (see pipeline.run's params).
- `query_params`: The params to use during querying (see pipeline.run's params).
- `dataset`: The BEIR dataset to use.
- `dataset_dir`: The directory to store the dataset to.
- `top_k_values`: The top_k values each metric will be calculated for.
- `keep_index`: Whether to keep the index after evaluation.
If True the index will be kept after beir evaluation. Otherwise it will be deleted immediately afterwards.
                   Defaults to False.

Returns a tuple containing the ncdg, map, recall and precision scores.
Each metric is represented by a dictionary containing the scores for each top_k value.

<a id="base.Pipeline.eval"></a>

#### eval

```python
@send_event
def eval(labels: List[MultiLabel], documents: Optional[List[List[Document]]] = None, params: Optional[dict] = None, sas_model_name_or_path: str = None, sas_batch_size: int = 32, sas_use_gpu: bool = True, add_isolated_node_eval: bool = False) -> EvaluationResult
```

Evaluates the pipeline by running the pipeline once per query in debug mode

and putting together all data that is needed for evaluation, e.g. calculating metrics.

**Arguments**:

- `labels`: The labels to evaluate on
- `documents`: List of List of Document that the first node in the pipeline should get as input per multilabel. Can be used to evaluate a pipeline that consists of a reader without a retriever.
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
- `sas_batch_size`: Number of prediction label pairs to encode at once by CrossEncoder or SentenceTransformer while calculating SAS.
- `sas_use_gpu`: Whether to use a GPU or the CPU for calculating semantic answer similarity.
Falls back to CPU if no GPU is available.
- `add_isolated_node_eval`: If set to True, in addition to the integrated evaluation of the pipeline, each node is evaluated in isolated evaluation mode.
This mode helps to understand the bottlenecks of a pipeline in terms of output quality of each individual node.
If a node performs much better in the isolated evaluation than in the integrated evaluation, the previous node needs to be optimized to improve the pipeline's performance.
If a node's performance is similar in both modes, this node itself needs to be optimized to improve the pipeline's performance.
The isolated evaluation calculates the upper bound of each node's evaluation metrics under the assumption that it received perfect inputs from the previous node.
To this end, labels are used as input to the node instead of the output of the previous node in the pipeline.
The generated dataframes in the EvaluationResult then contain additional rows, which can be distinguished from the integrated evaluation results based on the
values "integrated" or "isolated" in the column "eval_mode" and the evaluation report then additionally lists the upper bound of each node's evaluation metrics.

<a id="base.Pipeline.get_nodes_by_class"></a>

#### get\_nodes\_by\_class

```python
def get_nodes_by_class(class_type) -> List[Any]
```

Gets all nodes in the pipeline that are an instance of a certain class (incl. subclasses).

This is for example helpful if you loaded a pipeline and then want to interact directly with the document store.
Example:
| from haystack.document_stores.base import BaseDocumentStore
| INDEXING_PIPELINE = Pipeline.load_from_yaml(Path(PIPELINE_YAML_PATH), pipeline_name=INDEXING_PIPELINE_NAME)
| res = INDEXING_PIPELINE.get_nodes_by_class(class_type=BaseDocumentStore)

**Returns**:

List of components that are an instance the requested class

<a id="base.Pipeline.get_document_store"></a>

#### get\_document\_store

```python
def get_document_store() -> Optional[BaseDocumentStore]
```

Return the document store object used in the current pipeline.

**Returns**:

Instance of DocumentStore or None

<a id="base.Pipeline.draw"></a>

#### draw

```python
def draw(path: Path = Path("pipeline.png"))
```

Create a Graphviz visualization of the pipeline.

**Arguments**:

- `path`: the path to save the image.

<a id="base.Pipeline.load_from_yaml"></a>

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
    |   version: '1.0'
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

Note that, in case of a mismatch in version between Haystack and the YAML, a warning will be printed.
If the pipeline loads correctly regardless, save again the pipeline using `Pipeline.save_to_yaml()` to remove the warning.

**Arguments**:

- `path`: path of the YAML file.
- `pipeline_name`: if the YAML contains multiple pipelines, the pipeline_name to load must be set.
- `overwrite_with_env_variables`: Overwrite the YAML configuration with environment variables. For example,
to change index name param for an ElasticsearchDocumentStore, an env
variable 'MYDOCSTORE_PARAMS_INDEX=documents-2021' can be set. Note that an
`_` sign must be used to specify nested hierarchical properties.

<a id="base.Pipeline.load_from_config"></a>

#### load\_from\_config

```python
@classmethod
def load_from_config(cls, pipeline_config: Dict, pipeline_name: Optional[str] = None, overwrite_with_env_variables: bool = True)
```

Load Pipeline from a config dict defining the individual components and how they're tied together to form

a Pipeline. A single config can declare multiple Pipelines, in which case an explicit `pipeline_name` must
be passed.

Here's a sample configuration:

    ```python
    |   {
    |       "version": "0.9",
    |       "components": [
    |           {  # define all the building-blocks for Pipeline
    |               "name": "MyReader",  # custom-name for the component; helpful for visualization & debugging
    |               "type": "FARMReader",  # Haystack Class name for the component
    |               "params": {"no_ans_boost": -10, "model_name_or_path": "deepset/roberta-base-squad2"},
    |           },
    |           {
    |               "name": "MyESRetriever",
    |               "type": "ElasticsearchRetriever",
    |               "params": {
    |                   "document_store": "MyDocumentStore",  # params can reference other components defined in the YAML
    |                   "custom_query": None,
    |               },
    |           },
    |           {"name": "MyDocumentStore", "type": "ElasticsearchDocumentStore", "params": {"index": "haystack_test"}},
    |       ],
    |       "pipelines": [
    |           {  # multiple Pipelines can be defined using the components from above
    |               "name": "my_query_pipeline",  # a simple extractive-qa Pipeline
    |               "nodes": [
    |                   {"name": "MyESRetriever", "inputs": ["Query"]},
    |                   {"name": "MyReader", "inputs": ["MyESRetriever"]},
    |               ],
    |           }
    |       ],
    |   }
    ```

**Arguments**:

- `pipeline_config`: the pipeline config as dict
- `pipeline_name`: if the config contains multiple pipelines, the pipeline_name to load must be set.
- `overwrite_with_env_variables`: Overwrite the configuration with environment variables. For example,
to change index name param for an ElasticsearchDocumentStore, an env
variable 'MYDOCSTORE_PARAMS_INDEX=documents-2021' can be set. Note that an
`_` sign must be used to specify nested hierarchical properties.

<a id="base.Pipeline.save_to_yaml"></a>

#### save\_to\_yaml

```python
def save_to_yaml(path: Path, return_defaults: bool = False)
```

Save a YAML configuration for the Pipeline that can be used with `Pipeline.load_from_yaml()`.

**Arguments**:

- `path`: path of the output YAML file.
- `return_defaults`: whether to output parameters that have the default values.

<a id="base.Pipeline.get_config"></a>

#### get\_config

```python
def get_config(return_defaults: bool = False) -> dict
```

Returns a configuration for the Pipeline that can be used with `Pipeline.load_from_config()`.

**Arguments**:

- `return_defaults`: whether to output parameters that have the default values.

<a id="base.Pipeline.print_eval_report"></a>

#### print\_eval\_report

```python
def print_eval_report(eval_result: EvaluationResult, n_wrong_examples: int = 3, metrics_filter: Optional[Dict[str, List[str]]] = None)
```

Prints evaluation report containing a metrics funnel and worst queries for further analysis.

**Arguments**:

- `eval_result`: The evaluation result, can be obtained by running eval().
- `n_wrong_examples`: The number of worst queries to show.
- `metrics_filter`: The metrics to show per node. If None all metrics will be shown.

<a id="base.RayPipeline"></a>

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

<a id="base.RayPipeline.__init__"></a>

#### \_\_init\_\_

```python
def __init__(address: str = None, **kwargs)
```

**Arguments**:

- `address`: The IP address for the Ray cluster. If set to None, a local Ray instance is started.
- `kwargs`: Optional parameters for initializing Ray.

<a id="base.RayPipeline.load_from_yaml"></a>

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
    |   version: '0.9'
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
    |      type: RayPipeline
    |      nodes:
    |      - name: MyESRetriever
    |        inputs: [Query]
    |        replicas: 2    # number of replicas to create on the Ray cluster
    |      - name: MyReader
    |        inputs: [MyESRetriever]
    ```


Note that, in case of a mismatch in version between Haystack and the YAML, a warning will be printed.
If the pipeline loads correctly regardless, save again the pipeline using `RayPipeline.save_to_yaml()` to remove the warning.

**Arguments**:

- `path`: path of the YAML file.
- `pipeline_name`: if the YAML contains multiple pipelines, the pipeline_name to load must be set.
- `overwrite_with_env_variables`: Overwrite the YAML configuration with environment variables. For example,
to change index name param for an ElasticsearchDocumentStore, an env
variable 'MYDOCSTORE_PARAMS_INDEX=documents-2021' can be set. Note that an
`_` sign must be used to specify nested hierarchical properties.
- `address`: The IP address for the Ray cluster. If set to None, a local Ray instance is started.

<a id="base._RayDeploymentWrapper"></a>

## \_RayDeploymentWrapper

```python
class _RayDeploymentWrapper()
```

Ray Serve supports calling of __init__ methods on the Classes to create "deployment" instances.

In case of Haystack, some Components like Retrievers have complex init methods that needs objects
like Document Stores.

This wrapper class encapsulates the initialization of Components. Given a Component Class
name, it creates an instance using the YAML Pipeline config.

<a id="base._RayDeploymentWrapper.__init__"></a>

#### \_\_init\_\_

```python
def __init__(pipeline_config: dict, component_name: str)
```

Create an instance of Component.

**Arguments**:

- `pipeline_config`: Pipeline YAML parsed as a dict.
- `component_name`: Component Class name.

<a id="base._RayDeploymentWrapper.__call__"></a>

#### \_\_call\_\_

```python
def __call__(*args, **kwargs)
```

Ray calls this method which is then re-directed to the corresponding component's run().

<a id="base._HaystackBeirRetrieverAdapter"></a>

## \_HaystackBeirRetrieverAdapter

```python
class _HaystackBeirRetrieverAdapter()
```

<a id="base._HaystackBeirRetrieverAdapter.__init__"></a>

#### \_\_init\_\_

```python
def __init__(index_pipeline: Pipeline, query_pipeline: Pipeline, index_params: dict, query_params: dict)
```

Adapter mimicking a BEIR retriever used by BEIR's EvaluateRetrieval class to run BEIR evaluations on Haystack Pipelines.

This has nothing to do with Haystack's retriever classes.
See https://github.com/beir-cellar/beir/blob/main/beir/retrieval/evaluation.py.

**Arguments**:

- `index_pipeline`: The indexing pipeline to use.
- `query_pipeline`: The query pipeline to evaluate.
- `index_params`: The params to use during indexing (see pipeline.run's params).
- `query_params`: The params to use during querying (see pipeline.run's params).

<a id="standard_pipelines"></a>

# Module standard\_pipelines

<a id="standard_pipelines.BaseStandardPipeline"></a>

## BaseStandardPipeline

```python
class BaseStandardPipeline(ABC)
```

Base class for pre-made standard Haystack pipelines.
This class does not inherit from Pipeline.

<a id="standard_pipelines.BaseStandardPipeline.add_node"></a>

#### add\_node

```python
def add_node(component, name: str, inputs: List[str])
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

<a id="standard_pipelines.BaseStandardPipeline.get_node"></a>

#### get\_node

```python
def get_node(name: str)
```

Get a node from the Pipeline.

**Arguments**:

- `name`: The name of the node.

<a id="standard_pipelines.BaseStandardPipeline.set_node"></a>

#### set\_node

```python
def set_node(name: str, component)
```

Set the component for a node in the Pipeline.

**Arguments**:

- `name`: The name of the node.
- `component`: The component object to be set at the node.

<a id="standard_pipelines.BaseStandardPipeline.draw"></a>

#### draw

```python
def draw(path: Path = Path("pipeline.png"))
```

Create a Graphviz visualization of the pipeline.

**Arguments**:

- `path`: the path to save the image.

<a id="standard_pipelines.BaseStandardPipeline.save_to_yaml"></a>

#### save\_to\_yaml

```python
def save_to_yaml(path: Path, return_defaults: bool = False)
```

Save a YAML configuration for the Pipeline that can be used with `Pipeline.load_from_yaml()`.

**Arguments**:

- `path`: path of the output YAML file.
- `return_defaults`: whether to output parameters that have the default values.

<a id="standard_pipelines.BaseStandardPipeline.load_from_yaml"></a>

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

- `path`: path of the YAML file.
- `pipeline_name`: if the YAML contains multiple pipelines, the pipeline_name to load must be set.
- `overwrite_with_env_variables`: Overwrite the YAML configuration with environment variables. For example,
to change index name param for an ElasticsearchDocumentStore, an env
variable 'MYDOCSTORE_PARAMS_INDEX=documents-2021' can be set. Note that an
`_` sign must be used to specify nested hierarchical properties.

<a id="standard_pipelines.BaseStandardPipeline.get_nodes_by_class"></a>

#### get\_nodes\_by\_class

```python
def get_nodes_by_class(class_type) -> List[Any]
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

<a id="standard_pipelines.BaseStandardPipeline.get_document_store"></a>

#### get\_document\_store

```python
def get_document_store() -> Optional[BaseDocumentStore]
```

Return the document store object used in the current pipeline.

**Returns**:

Instance of DocumentStore or None

<a id="standard_pipelines.BaseStandardPipeline.eval"></a>

#### eval

```python
def eval(labels: List[MultiLabel], params: Optional[dict] = None, sas_model_name_or_path: Optional[str] = None, add_isolated_node_eval: bool = False) -> EvaluationResult
```

Evaluates the pipeline by running the pipeline once per query in debug mode

and putting together all data that is needed for evaluation, e.g. calculating metrics.

**Arguments**:

- `labels`: The labels to evaluate on
- `params`: Params for the `retriever` and `reader`. For instance,
params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
- `sas_model_name_or_path`: SentenceTransformers semantic textual similarity model to be used for sas value calculation,
should be path or string pointing to downloadable models.
- `add_isolated_node_eval`: Whether to additionally evaluate the reader based on labels as input instead of output of previous node in pipeline

<a id="standard_pipelines.ExtractiveQAPipeline"></a>

## ExtractiveQAPipeline

```python
class ExtractiveQAPipeline(BaseStandardPipeline)
```

Pipeline for Extractive Question Answering.

<a id="standard_pipelines.ExtractiveQAPipeline.__init__"></a>

#### \_\_init\_\_

```python
def __init__(reader: BaseReader, retriever: BaseRetriever)
```

**Arguments**:

- `reader`: Reader instance
- `retriever`: Retriever instance

<a id="standard_pipelines.ExtractiveQAPipeline.run"></a>

#### run

```python
def run(query: str, params: Optional[dict] = None, debug: Optional[bool] = None)
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

<a id="standard_pipelines.DocumentSearchPipeline"></a>

## DocumentSearchPipeline

```python
class DocumentSearchPipeline(BaseStandardPipeline)
```

Pipeline for semantic document search.

<a id="standard_pipelines.DocumentSearchPipeline.__init__"></a>

#### \_\_init\_\_

```python
def __init__(retriever: BaseRetriever)
```

**Arguments**:

- `retriever`: Retriever instance

<a id="standard_pipelines.DocumentSearchPipeline.run"></a>

#### run

```python
def run(query: str, params: Optional[dict] = None, debug: Optional[bool] = None)
```

**Arguments**:

- `query`: the query string.
- `params`: params for the `retriever` and `reader`. For instance, params={"Retriever": {"top_k": 10}}
- `debug`: Whether the pipeline should instruct nodes to collect debug information
about their execution. By default these include the input parameters
they received and the output they generated.
All debug information can then be found in the dict returned
by this method under the key "_debug"

<a id="standard_pipelines.GenerativeQAPipeline"></a>

## GenerativeQAPipeline

```python
class GenerativeQAPipeline(BaseStandardPipeline)
```

Pipeline for Generative Question Answering.

<a id="standard_pipelines.GenerativeQAPipeline.__init__"></a>

#### \_\_init\_\_

```python
def __init__(generator: BaseGenerator, retriever: BaseRetriever)
```

**Arguments**:

- `generator`: Generator instance
- `retriever`: Retriever instance

<a id="standard_pipelines.GenerativeQAPipeline.run"></a>

#### run

```python
def run(query: str, params: Optional[dict] = None, debug: Optional[bool] = None)
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

<a id="standard_pipelines.SearchSummarizationPipeline"></a>

## SearchSummarizationPipeline

```python
class SearchSummarizationPipeline(BaseStandardPipeline)
```

Pipeline that retrieves documents for a query and then summarizes those documents.

<a id="standard_pipelines.SearchSummarizationPipeline.__init__"></a>

#### \_\_init\_\_

```python
def __init__(summarizer: BaseSummarizer, retriever: BaseRetriever, return_in_answer_format: bool = False)
```

**Arguments**:

- `summarizer`: Summarizer instance
- `retriever`: Retriever instance
- `return_in_answer_format`: Whether the results should be returned as documents (False) or in the answer
format used in other QA pipelines (True). With the latter, you can use this
pipeline as a "drop-in replacement" for other QA pipelines.

<a id="standard_pipelines.SearchSummarizationPipeline.run"></a>

#### run

```python
def run(query: str, params: Optional[dict] = None, debug: Optional[bool] = None)
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

<a id="standard_pipelines.FAQPipeline"></a>

## FAQPipeline

```python
class FAQPipeline(BaseStandardPipeline)
```

Pipeline for finding similar FAQs using semantic document search.

<a id="standard_pipelines.FAQPipeline.__init__"></a>

#### \_\_init\_\_

```python
def __init__(retriever: BaseRetriever)
```

**Arguments**:

- `retriever`: Retriever instance

<a id="standard_pipelines.FAQPipeline.run"></a>

#### run

```python
def run(query: str, params: Optional[dict] = None, debug: Optional[bool] = None)
```

**Arguments**:

- `query`: the query string.
- `params`: params for the `retriever`. For instance, params={"Retriever": {"top_k": 10}}
- `debug`: Whether the pipeline should instruct nodes to collect debug information
about their execution. By default these include the input parameters
they received and the output they generated.
All debug information can then be found in the dict returned
by this method under the key "_debug"

<a id="standard_pipelines.TranslationWrapperPipeline"></a>

## TranslationWrapperPipeline

```python
class TranslationWrapperPipeline(BaseStandardPipeline)
```

Takes an existing search pipeline and adds one "input translation node" after the Query and one
"output translation" node just before returning the results

<a id="standard_pipelines.TranslationWrapperPipeline.__init__"></a>

#### \_\_init\_\_

```python
def __init__(input_translator: BaseTranslator, output_translator: BaseTranslator, pipeline: BaseStandardPipeline)
```

Wrap a given `pipeline` with the `input_translator` and `output_translator`.

**Arguments**:

- `input_translator`: A Translator node that shall translate the input query from language A to B
- `output_translator`: A Translator node that shall translate the pipeline results from language B to A
- `pipeline`: The pipeline object (e.g. ExtractiveQAPipeline) you want to "wrap".
Note that pipelines with split or merge nodes are currently not supported.

<a id="standard_pipelines.QuestionGenerationPipeline"></a>

## QuestionGenerationPipeline

```python
class QuestionGenerationPipeline(BaseStandardPipeline)
```

A simple pipeline that takes documents as input and generates
questions that it thinks can be answered by the documents.

<a id="standard_pipelines.RetrieverQuestionGenerationPipeline"></a>

## RetrieverQuestionGenerationPipeline

```python
class RetrieverQuestionGenerationPipeline(BaseStandardPipeline)
```

A simple pipeline that takes a query as input, performs retrieval, and then generates
questions that it thinks can be answered by the retrieved documents.

<a id="standard_pipelines.QuestionAnswerGenerationPipeline"></a>

## QuestionAnswerGenerationPipeline

```python
class QuestionAnswerGenerationPipeline(BaseStandardPipeline)
```

This is a pipeline which takes a document as input, generates questions that the model thinks can be answered by
this document, and then performs question answering of this questions using that single document.

<a id="standard_pipelines.MostSimilarDocumentsPipeline"></a>

## MostSimilarDocumentsPipeline

```python
class MostSimilarDocumentsPipeline(BaseStandardPipeline)
```

<a id="standard_pipelines.MostSimilarDocumentsPipeline.__init__"></a>

#### \_\_init\_\_

```python
def __init__(document_store: BaseDocumentStore)
```

Initialize a Pipeline for finding the most similar documents to a given document.

This pipeline can be helpful if you already show a relevant document to your end users and they want to search for just similar ones.

**Arguments**:

- `document_store`: Document Store instance with already stored embeddings.

<a id="standard_pipelines.MostSimilarDocumentsPipeline.run"></a>

#### run

```python
def run(document_ids: List[str], top_k: int = 5)
```

**Arguments**:

- `document_ids`: document ids
- `top_k`: How many documents id to return against single document

