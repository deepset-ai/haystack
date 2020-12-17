<a name="pipeline"></a>
# Module pipeline

<a name="pipeline.Pipeline"></a>
## Pipeline Objects

```python
class Pipeline()
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
 | get_node(name: str)
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

<a name="pipeline.BaseStandardPipeline"></a>
## BaseStandardPipeline Objects

```python
class BaseStandardPipeline()
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

<a name="pipeline.JoinDocuments"></a>
## JoinDocuments Objects

```python
class JoinDocuments()
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

