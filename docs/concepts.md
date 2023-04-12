# Basic Concepts

Canals is a **component orchestration engine**. It can be used to connect a group of smaller objects, called Components,
that  perform well-defined tasks into a network, called Pipeline, that achieves a larger goal.

Components are Python objects that can execute a task, like reading a file, performing calculations, or making API
calls. Canals connects these objects together: it builds a graph of components and takes care of managing their
execution order, making sure that each object receives the input it expects from the other components of the pipeline.


Canals relies on two main concepts: Components and Pipelines.

## What is a Component?

A Component is a Python class that performs a well-defined task: for example a REST API call, a mathematical operation,
a data trasformation, writing something to a file or a database, and so on.

To be recognized as Components by Canals, a Python class needs to respect these rules:

1. Must be decorated with the `@component` decorator
2. Must have a `run()` method with a specific signature
3. Must return output that Canals can interpret.

We will see the details of all of these requirements below.

## What is a Pipeline?

A Pipeline is a network of Components. Pipelines define what components receive and send output to which other, makes
sure all the connections are valid, and takes care of calling the component's `run()` method in the right order.

Pipeline connects compoonents together through so-called connections, which are the edges of the pipeline graph.
Each component should declare which inputs it expects, which output it will generate, and Pipeline is going
to make sure that all the connections are valid based on these two elements.

For example, if a component produces a value called `document`, among others, and another component expects an input called `document`, among others, Pipeline will be able to connect them. Otherwise, it will raise an exception.

## Example

This is an example of a Pipeline that performs some mathematical operations combining two components.

```python
from typing import Dict, Any, List, Tuple
from canals import Pipeline, component

@component
class AddValue:
    def __init__(
        self,
        add: int = 1,
        input_name: str = "value",
        output_name: str = "value"
    ):
        self.add = add
        self.init_parameters = {"add": add}
        self.inputs = [input_name]
        self.outputs = [output_name]

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        my_parameters = parameters.get(name, {})
        add = my_parameters.get("add", self.add)

        for _, value in data:
            value += add

        return ({self.outputs[0]: value}, parameters)


@node
class Double:
    def __init__(self, input_connection: str = "value"):
        self.init_parameters = {"input_connection": input_connection}
        self.inputs = [input_connection]
        self.outputs = [input_connection]

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        for _, value in data:
            value *= 2

        return ({self.outputs[0]: value}, parameters)


pipeline = Pipeline()

# Components can be initialized as standalone objects.
# These instances can be added to the Pipeline in several places.
addition = AddValue(add=1)

# Components are added with a name and an node. Note the lack of references to
# any other node. Components can store default parameters per node.
pipeline.add_component("first_addition", addition, parameters={"add": 3})
pipeline.add_component("second_addition", addition)  # Instances can be reused
pipeline.add_component("double", Double())

# Components are the connected as input node: [list of output components]
pipeline.connect(connect_from="first_addition", connect_to="double")
pipeline.connect(connect_from="double", connect_to="second_addition")

pipeline.draw("pipeline.png")

# Pipeline.run() accepts 'data' and 'parameters' only. Such dictionaries can
# contain anything, depending on what the first component(s) of the pipeline
# requires. Pipeline does not validate the input: every component(s) should
# do so.
results = pipeline.run(
    data={"value": 1},
    # Parameters can be passed at this stage as well
    parameters = {"second_addition": {"add": 10}}
)
assert results == {"value": 18}
```

The result of `Pipeline.draw()`:

```mermaid
graph TD
I((input)) --> A(first addition)
A --> B(double)
B --> C(second addition)
C --> O((output))
```

## How do I create a Component?

In order to be recognized as components and work in a Pipeline, Components must follow the contract below.

### Decorator

All component classes must be decorated with the `@component` decorator. This allows Canals to discover them.

### __init__

```python
def __init__(self, [... components init parameters ...]):
```

The constructor is a mandatory method for Canals components.

In their `__init__`, Components must define:

- `self.inputs = [<expected_input_connection_name(s)>]`:
    A list with all the connections they can possibly receive input from

- `self.outputs = [<expected_output_connection_name(s)>]`:
    A list with the connections they might possibly produce as output

- `self.init_parameters = {<init parameters>}`:
    Any state they wish to be persisted when they are marshalled.
    These values will be given to the `__init__` method of a new instance
    when the pipeline is unmarshalled.

If components want to let users customize their input and output connections (be it
the connection name, the connection count, etc...) they should provide properly
named init parameters:

- `input: str` or `inputs: List[str]` (always with proper defaults)
- `output: str` or `outputs: List[str]` (always with proper defaults)

All the rest is going to be interpreted as a regular init parameter that
has nothing to do with the component connections.

The `__init__` must be extrememly lightweight, because it's a frequent
operation during the construction and validation of the pipeline. If a component
has some heavy state to initialize (models, backends, etc...) refer to the
`warm_up()` method.

### `warm_up()`

```python
def warm_up(self):
```

Optional method. If it's defined, this method is called by Pipeline before the graph execution.
Make sure to avoid double-initializations, because Pipeline will not keep track of which components it called
`warm_up()` on.

### `run()`

```python
def run(
    self,
    name: str,
    data: List[Tuple[str, Any]],
    parameters: Dict[str, Dict[str, Any]],
):
```

This is the method that is called by `Pipeline.run()`. When calling it, Pipeline passes the following parameters to it:

- `name: str`: the name of the component. Allows the component to find its own parameters in
    the `parameters` dictionary (see below).

- `data: List[Tuple[str, Any]]`: the input data.
    Pipeline guarantees that the following assert always passes:

    `assert self.inputs == [name for name, value in data]`

    which means that:
    - `data` is of the same length as `self.inputs`.
    - `data` contains one tuple for each string stored in `self.inputs`.
    - no guarantee is given on the values of these tuples: notably, if there was a
        decision component upstream, some values might be `None`.

    For example, if a component declares `self.inputs = ["value", "value"]` (think of a
    `Sum` component), `data` might look like:

    `[("value", 1), ("value", 10)]`

    `[("value", None), ("value", 10)]`

    `[("value", None), ("value", None)]`

    `[("value", 1), ("value", ["something", "unexpected"])]`

    but it will never look like:

    `[("value", 1), ("value", 10), ("value", 100)]`

    `[("value": 15)]`

    `[("value": 15), ("unexpected", 10)]`

- `parameters: Dict[str, Dict[str, Any]]`: a dictionary of dictionaries with all
    the parameters for all components in the Pipeline.
    Note that all components have access to all parameters for all other components: this
    might come handy to components that want to influence the behavior
    of other components downstream.
    Components can access their own parameters using `name`, but they must **not** assume
    their name is present in the dictionary.
    Therefore, the best way to get the parameters is with
    `my_parameters = parameters.get(name, {})`

Pipeline expect the output of this function to be a tuple of two dictionaries.
The first item is a dictionary that represents the output and it should always
abide to the following format:

`{output_name: output_value for output_name in <subset of self.expected_output>}`

Which means that:
- Components are not forced to produce output on all the expected outputs: for example,
    components taking a decision, like classifiers, can produce output on a subset of
    the expected output connections and Pipeline will figure out the rest.
- Components must not add any key in the data dictionary that is not present in `self.outputs`.

The second item of the tuple is the `parameters` dictionary. This allows component to
propagate downstream any change they might have done to the `parameters` dictionary.


## Topologies

Canals supports a variety of different pipeline topologies. Check the pipeline's test suite for some examples:
these are only representations of the graphs that those pipelines generate.

TODO

## Validation

Pipeline performs validation on the connection name level: when calling `Pipeline.connect()`, it uses the values of the components' `self.inputs` and `self.outputs` to make sure that the connection is possible.

Components are required, by contract, to explicitly define their inputs and outputs, and these values are used by the connect method to validate the connection, and by the run method to route values.

For example, let's imagine we have two components with the following I/O declared:

```python
@component
class ComponentA:

    def __init__(self):
        self.inputs = ["input"]
        self.outputs = ["intermediate_value"]

    def run(self):
        pass

@component
class ComponentB:

    def __init__(self):
        self.inputs = ["intermediate_value"]
        self.outputs = ["output"]

    def run(self):
        pass
```

This is the behavior of `Pipeline.connect()`:

```python
pipeline.connect('component_a', 'component_b')
# Succeeds: no output

pipeline.connect('component_a', 'component_a')
# Traceback (most recent call last):
#   File "/home/me/projects/canals/example.py", line 29, in <module>
#     pipeline.connect('component_a', 'component_a')
#   File "/home/me/projects/canals/canals/pipeline/pipeline.py", line 224, in connect
#     raise PipelineConnectError(
# haystack.pipeline._utils.PipelineConnectError: Cannot connect 'component_a' with 'component_a' with a connection named 'intermediate_value': their declared inputs and outputs do not match.
# Upstream component 'component_a' declared these outputs:
#  - intermediate_value (free)
# Downstream component 'component_a' declared these inputs:
#  - input (free)

pipeline.connect('component_b', 'component_a')
# Traceback (most recent call last):
#   File "/home/me/projects/canals/example.py", line 29, in <module>
#     pipeline.connect('component_b', 'node_a')
#   File "/home/me/projects/canals/canals/pipeline/pipeline.py", line 224, in connect
#     raise PipelineConnectError(
# haystack.pipeline._utils.PipelineConnectError: Cannot connect 'component_b' with 'component_a' with an edge named 'output': their declared inputs and outputs do not match.
# Upstream component 'component_b' declared these outputs:
#  - output (free)
# Downstream component 'component_a' declared these inputs:
#  - input (free)
```

This type of error reporting was found especially useful for components that declare a variable number and name of inputs and outputs depending on their initialization parameters (think of classifiers, for example).

One shortcoming is that currently Pipeline "trusts" the components to respect their own declarations. So if a component states that it will output `intermediate_value`, but outputs something else once run, Pipeline will fail. We accept this failure as a "contract breach": the node should fix its behavior and Pipeline should not try to prevent such scenarios.


## Parameters hierarchy

Parameters can be passed to components at several stages, and they have different priorities. Here they're listed from least priority to top priority.

- Components's default `__init__` parameters: components's `__init__` can provide defaults. Those are used only if no other parameters are passed at any stage.

- Components's `__init__` parameters: at initialization, nodes might be given values for their parameters. These are stored within the component instance and, if the instance is reused in the pipeline several times, they will be the same on all of them.

- Pipeline's `add_component()`: When added to the pipeline, users can specify some parameters that have to be given only to that component specifically. They will override the component instance's parameters, but they will be applied only in that specific location of the pipeline and not be applied to other instances of the same component anywhere else in the graph.

- Pipeline's `run()`: `run()` also accepts a dictionary of parameters that will override all conflicting parameters set at any level below.

Example:

```python
@component
class Component:
    def __init__(self, value_1: int = 1, value_2: int = 1, value_3: int = 1, value_4: int = 1):
        ...

component = Component(value_2=2, value_3=2, value_4=2)
pipeline = Pipeline()
pipeline.add_component("component", component, parameters={"value_3": 3, "value_4": 3})
...
pipeline.run(data={...}, parameters={"node": {"value_4": 4}})

# Component will receive {"value_1": 1, "value_2": 2, "value_3": 3,"value_4": 4}
```


## Pipeline save and load

Pipelines can be serialized to Python dictionaries, that can be then dumped to JSON or to any other suitable format, like YAML, TOML, HCL, etc.

These pipelines can then be loaded back.

Here is an example of Pipeline marshalling and unmarshalling:

```python
from haystack.pipelines import Pipeline, save_pipelines, load_pipelines

query_pipeline = Pipeline()
indexing_pipeline = Pipeline()
# .. assemble the pipelines ...

# Save the pipelines
save_pipelines(
    pipelines={
        "query": query_pipeline,
        "indexing": indexing_pipeline,
    },
    path="my_pipelines.json",
    _writer=json.dumps
)

# Load the pipelines
new_pipelines = load_pipelines(
    path="my_pipelines.json",
    _reader=json.loads
)

assert new_pipelines["query"] == query_pipeline
assert new_pipelines["indexing"] == indexing_pipeline
```

Note how the save/load functions accept a `_writer`/`_reader` function: this choice frees us from committing strongly to a specific template language, and although a default will be set (be it YAML, TOML, HCL or anything else) the decision can be overridden by passing another explicit reader/writer function to the `save_pipelines`/`load_pipelines` functions.

This is how the resulting file will look like, assuming a JSON writer was chosen.

`my_pipeline.json`

```python
{
    # A list of "dependencies" for the application.
    # Used to ensure all external nodes are present when loading.
    "dependencies" : [
        "haystack == 2.0.0",
        "my_custom_node_module == 0.0.1",
    ],

    # Stores are defined here, outside single pipeline graphs.
    # All pipelines have access to all these docstores.
    "stores": {
        # Nodes will be able to access them by the name defined here,
        # in this case `my_first_store` (see the retrievers below).
        "my_first_store": {
            # class_name is mandatory
            "class_name": "InMemoryDocumentStore",
            # Then come all the additional parameters for the store
            "use_bm25": true
        },
        "my_second_store": {
            "class_name": "InMemoryDocumentStore",
            "use_bm25": false
        }
    },

    # Nodes are defined here, outside single pipeline graphs as well.
    # All pipelines can use these nodes. Instances are re-used across
    # Pipelines if they happen to share a node.
    "nodes": {
        # In order to reuse an instance across multiple nodes, instead
        # of a `class_name` there should be a pointer to another node.
        "my_sparse_retriever": {
            # class_name is mandatory, unless it's a pointer to another node.
            "class_name": "BM25Retriever",
            # Then come all the additional init parameters for the node
            "store_name": "my_first_store",
            "top_k": 5
        },
        "my_dense_retriever": {
            "class_name": "EmbeddingRetriever",
            "model_name": "deepset-ai/a-model-name",
            "store_name": "my_second_store",
            "top_k": 5
        },
        "my_ranker": {
            "class_name": "Ranker",
            "inputs": ["documents", "documents"],
            "outputs": ["documents"],
        },
        "my_reader": {
            "class_name": "Reader",
            "model_name": "deepset-ai/another-model-name",
            "top_k": 3
        }
    },

    # Pipelines are defined here. They can reference all nodes above.
    # All pipelines will get access to all docstores
    "pipelines": {
        "sparse_question_answering": {
            # Mandatory list of edges. Same syntax as for `Pipeline.connect()`
            "edges": [
                ("my_sparse_retriever", ["reader"])
            ],
            # To pass some parameters at the `Pipeline.add_node()` stage, add them here.
            "parameters": {
                "my_sparse_retriever": {
                    "top_k": 10
                }
            },
            # Metadata can be very valuable for dC and to organize larger Applications
            "metadata": {
                "type": "question_answering",
                "description": "A test pipeline to evaluate Sparse QA.",
                "author": "ZanSara"
            },
            # Other `Pipeline.__init__()` parameters
            "max_allowed_loops": 10,
        },
        "dense_question_answering": {
            "edges": [
                ("my_dense_retriever", ["reader"])
            ],
            "metadata": {
                "type": "question_answering",
                "description": "A test pipeline to evaluate Sparse QA.",
                "author": "an_intern"
            }
        },
        "hybrid_question_answering": {
            "edges": [
                ("my_sparse_retriever", ["ranker"]),
                ("my_dense_retriever", ["ranker"]),
                ("ranker", ["reader"]),
            ],
            "metadata": {
                "type": "question_answering",
                "description": "A test pipeline to evaluate Hybrid QA.",
                "author": "the_boss"
            }
        }
    }
}
```
