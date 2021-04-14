<a name="pipeline"></a>
# Module pipeline

<a name="pipeline.Pipeline"></a>
## Pipeline Objects

```python
class Pipeline(ABC)
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
    |   version: '0.7'
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

<a name="pipeline.SearchSummarizationPipeline"></a>
## SearchSummarizationPipeline Objects

```python
class SearchSummarizationPipeline(BaseStandardPipeline)
```

<a name="pipeline.SearchSummarizationPipeline.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(summarizer: BaseSummarizer, retriever: BaseRetriever)
```

Initialize a Pipeline that retrieves documents for a query and then summarizes those documents.

**Arguments**:

- `summarizer`: Summarizer instance
- `retriever`: Retriever instance

<a name="pipeline.SearchSummarizationPipeline.run"></a>
#### run

```python
 | run(query: str, filters: Optional[Dict] = None, top_k_retriever: Optional[int] = None, generate_single_summary: Optional[bool] = None, return_in_answer_format: bool = False)
```

**Arguments**:

- `query`: Your search query
- `filters`: 
- `top_k_retriever`: Number of top docs the retriever should pass to the summarizer.
                        The higher this value, the slower your pipeline.
- `generate_single_summary`: Whether to generate single summary from all retrieved docs (True) or one per doc (False).
- `return_in_answer_format`: Whether the results should be returned as documents (False) or in the answer format used in other QA pipelines (True).
                                With the latter, you can use this pipeline as a "drop-in replacement" for other QA pipelines.

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

