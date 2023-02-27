- Title: Drop `BaseComponent` and reimplement `Pipeline`.
- Decision driver: @ZanSara
- Start Date: 27/02/2023
- Proposal PR: #4284
- Github Issue or Discussion: #2807

# Summary

Haystack Pipelines are very powerful objects, but they still have a number of unnecessary limitations, by design and by implementation.

This proposal aims to address most of the implementation issues, some fundamental assumptions like the need for DAGs and the `BaseComponent` class, and proposes a solution for the question of `DocumentStore`'s status with respect to the `Pipeline`.


# Motivation

Pipelines are the fundamental component of Haystack and one of its most powerful concepts. At its core, a Pipeline is a DAG (Directed Acyclic Graph) of objects called Nodes, or Components, each of whom executes a specific transformation on the data flowing along the pipeline. In this way, users can combine powerful libraries, NLP models, and simple Python snippets to connect a herd of tools into a one single, coherent object that can fulfill an infinite variety of tasks.

However, as it currently stands, the `Pipeline` object is also imposing a number of limitations on its use, most of which are likely to be unnecessary. Some of these include:

- DAGs. DAGs are safe, but loops could enable many more usecases, like `Agents`.

- `Pipeline` can select among branches, but cannot run such branches in parallel, except for some specific and inconsistent corner cases. For further reference and discussions on the topic, see:
    - https://github.com/deepset-ai/haystack/pull/2593
    - https://github.com/deepset-ai/haystack/pull/2981#issuecomment-1207850632
    - https://github.com/deepset-ai/haystack/issues/2999#issuecomment-1210382151

- `Pipeline`s are forced to have one single input and one single output node, and the input node has to be called either `Query` or `Indexing`, which softly forbids any other type of pipeline.

- The fixed set of allowed inputs (`query`, `file_paths`, `labels`, `documents`, `meta`, `params` and `debug`) blocks several usecases, like summarization pipelines, translation pipelines, even some sort of generative pipelines.

- `Pipeline`s are often required to have a `DocumentStore` _somewhere_ (see below), even in situation where it wouldn't be needed.
  - For example, `Pipeline` has a `get_document_store()` method which iterates over all nodes looking for a `Retriever`.

- The redundant concept of `run()` and `run_batch()`: nodes should take care of this distinction internally if it's important, otherwise run in batches by default.

- The distinction between a `Pipeline` and its YAML representation is confusing: YAMLs can contain several pipelines, but `Pipeline.save_to_yaml()` can only save a single pipeline.

In addition, there are a number of known bugs that makes the current Pipeline implementation hard to work with. Some of these include:

- Branching and merging logic is known to be buggy even where it's supported.
- Nodes can't be added twice to the same pipeline in different locations, limiting their reusability.
- Pipeline YAML validation needs to happen with a YAML schema because `Pipeline`s can only be loaded along with all their nodes, which is a very heavy operation. Shallow or lazy loading of nodes doesn't exist.
- Being forced to use a schema for YAML validation makes impossible to validate the graph in advance.

On top of these issues, there is the tangential issue of `DocumentStore`s and their uncertain relationship with `Pipeline`s. This problem has to be taken into account during a redesign of `Pipeline` and, if necessary, `DocumentStore`s should also be partially impacted. Some of these issues include:

- `DocumentStore`s are nodes in theory, but in practice they can be added to `Pipeline`s only to receive documents to be stored. On the other hand, `DocumentStore`'s most prominent usecase is as a _source_ of documents, and currently they are not suited for this task without going through an intermediary, most often a `Retriever` class.
  - The relationship between `DocumentStore` and `Retriever` should be left as a topic for a separate proposal but kept in mind, because `Retriever`s currently act as the main interface for `DocumentStore`s into `Pipeline`s.

This proposal tries to adress all the above point by taking a radical stance with:

- A full reimplementation of the `Pipeline` class that does not limit itself to DAGs, can run branches in parallel, can skip branches and can process loops safely.

- Dropping the concept of `BaseComponent` and introducing the much lighter concept of `Node` in its place.

- Define a clear contract between `Pipeline` and the `Node`s.

- Define a clear place for `DocumentStore`s with respect to `Pipeline`s that doesn't forcefully involve `Retriever`s.

- Redesign the YAML representation of `Pipeline`s.

# Basic example

A simple example of how the new Pipeline could look like is shown here. This example does not address `DocumentStore`s or YAML serialization, but rather focuses on the shift between `BaseComponent` and `Node`s.

For the detailed explanation behind the design choices and all open questions, see the "Detailed Design" section and the draft implementation here: https://github.com/ZanSara/haystack-2.0-draft

## Simple example

This is a very simplified example that performs some mathematical operations. See below for more realistic examples.

```python
from typing import Dict, Any, List, Tuple
from new_haystack.pipeline import Pipeline
from new_haystack.nodes import haystack_node

# A Haystack Node. See below for details about this contract.
# Crucial components are the @haystack_node decorator and the `run()` method
@haystack_node
class AddValue:
    def __init__(self, add: int = 1, input_name: str = "value", output_name: str = "value"):
        self.add = add
        self.init_parameters = {"add": add}
        self.expected_inputs = [input_name]
        self.expected_outputs = [output_name]

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

        return {"value": value}


@haystack_node
class Double:
    def __init__(self, input_edge: str = "value"):
        self.init_parameters = {"input_edge": input_edge}
        self.expected_inputs = [input_edge]
        self.expected_outputs = [input_edge]

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        for _, value in data:
            value *= 2

        return {self.expected_outputs[0]: value}


pipeline = Pipeline()

# Nodes can be initialized as standalone objects.
# These instances can be added to the Pipeline in several places.
addition = AddValue(add=1)

# Nodes are added with a name and an node. Note the lack of references to any other node.
pipeline.add_node("first_addition", addition, parameters={"add": 3})  # Nodes can store default parameters per node.
pipeline.add_node("second_addition", addition)  # Note that instances can be reused
pipeline.add_node("double", Double())

# Nodes are the connected in a chain with a separate call to Pipeline.connect()
pipeline.connect(["first_addition", "double", "second_addition"])

pipeline.draw("pipeline.png")

# Pipeline.run() accepts 'data' and 'parameters' only. Such dictionaries can contain
# anything, depending on what the first node(s) of the pipeline requires.
# Pipeline does not validate the input: the first node(s) should do so.
results = pipeline.run(
    data={"value": 1},
    parameters = {"second_addition": {"add": 10}}   # Parameters can be passed at this stage as well
)
assert results == {"value": 18}
```

The result of `Pipeline.draw()`:

![image](images/4284-drop-basecomponent/pipeline.png)


## Realistic Query Pipeline

**TODO**

## Realistic Indexing Pipeline

**TODO**

# Detailed design

This section focuses on the concept rather than the implementation strategy. For a discussion on the implementation, see the draft here: https://github.com/ZanSara/haystack-2.0-draft

## The Pipeline API

These are the core features that drove the design of the revised Pipeline API:

- An execution graph that is more flexible than a DAG.
- A clear place for `DocumentStore`s

Therefore, the revised Pipeline object has the following API:

- Core functions:
    - `run(data, parameters)`: the core of the class. Relies on `networkx` for most of the heavy-lifting. Check out the implementation (https://github.com/ZanSara/haystack-2.0-draft/blob/main/new-haystack/new_haystack/pipeline/pipeline.py) for details: the code is heavily commented on the main loop and on the handling of non-trivial execution paths like branch selection, parallel branch execution, loops handling, multiple input/output and so on.
    - `draw(path)`: as in the old Pipeline object. Based on `pygraphviz` (which requires `graphviz`), but we might need to look for pure Python alternatives based on Matplotlib to reduce our dependencies.
- Graph building:
    - `add_node(name, node, parameters)`: adds a disconnected node to the graph. It expects Haystack nodes in the `node` parameter and will fail if they aren't respecting the contract. See below for a more detailed discussion of the Nodes' contract.
    - `get_node(name)`: returns the node's information stored in the graph
    - `connect(nodes)`: chains a series of nodes together. It will fail if the nodes inputs and outputs do not match: see the Nodes' contract to understand how Nodes can declare their I/O.
- Docstore management:
    - `connect_store(name, store)`: adds a DocumentStore to the stores that are passed down to the nodes through the `stores` variable.
    - `list_stores()`: returns all connected stores.
    - `get_store(name)`: returns a specific document store by name.
    - `disconnect_store(name, store)`: removes a store from the registry.
- Serialization and validation:
    - `__init__(path=None)`: if a path is given, loads the pipeline from the YAML found at that path. Note that at this stage `Pipeline` will collect nodes from all imported modules (see the implementation - the search can be scoped down to selected modules) and **all nodes' `__init__` method is called**. Therefore, `__init__` must be lightweight. See the Node's contract to understand how heavy nodes should design their initialization.
    - `save(path)`: serializes and saves the pipeline as a YAML at the given path.

Example pipeline topologies supported by the new implementation (images taken from the test suite):

<details>
<summary>Merging pipeline</summary>

![image](images/4284-drop-basecomponent/merging_pipeline.png)

In this pipeline, several nodes send their input into a single output node. Note that this pipeline has several starting nodes, something that is currently not supported by Haystack's `Pipeline`.

</details>

<details>
<summary>Branching pipeline with branch skipping</summary>

![image](images/4284-drop-basecomponent/decision_pipeline.png)

In this pipeline, only one edge will run depending on the decision taken by the `remainder` node. Note that this pipeline has several terminal nodes, something that is currently not supported by Haystack's `Pipeline`.

</details>

<details>
<summary>Branching pipeline with parallel branch execution</summary>

![image](images/4284-drop-basecomponent/parallel_branches_pipeline.png)

In this pipeline, all the edges that leave `enumerate` are run by `Pipeline`. Note that this usecase is currently not supported by Haystack's `Pipeline`.

</details>

<details>
<summary>Branching pipeline with branch skipping and merge</summary>

![image](images/4284-drop-basecomponent/decision_and_merge_pipeline.png)

In this pipeline, the merge node can understand that some of its upstream nodes will never run (`remainder` selects only one output edge) and waits only for the inputs that it can receive, so one from `remainder`, plus `no-op`.

</details>

<details>
<summary>Looping pipeline</summary>

![image](images/4284-drop-basecomponent/looping_pipeline.png)

This is a pipeline with a loop and a counter that statefully counts how many times it has been called.

Note that the new `Pipeline` can set a maximum number of allowed visits to nodes, so that loops are eventually stopped if they get stuck.

</details>

<details>
<summary>Looping pipeline with merge</summary>

![image](images/4284-drop-basecomponent/looping_and_merge_pipeline.png)

This is a pipeline with a loop and a counter that statefully counts how many times it has been called. There is also a merge node at the bottom, which shows how Pipeline can wait for the entire loop to exit before running `sum`.

</details>

<details>
<summary>Arbitrarily complex pipeline</summary>

![image](images/4284-drop-basecomponent/complex_pipeline.png)

This is an example of how complex Pipelines the new objects can support. This pipeline combines all cases above:
- Multiple inputs
- Multiple outputs
- Decision nodes and branches skipped due to a selection
- Distribution nodes and branches executed in parallel
- Merge nodes where it's unclear how many edges will actually carry output
- Merge nodes with repeated inputs ('sum' takes three `value` edges) or distinct inputs ('diff' takes `value` and `sum`)
- Loops along a branch
</details>

NOTE: the draft implementation supports all of these topologies already. You can find the code for each of these pipelines under https://github.com/ZanSara/new-haystack-pipeline-draft/tree/main/new-haystack/tests/integration

## The Node contract

A Haystack node is any class that abides the following contract:

```python
# This decorator does very little, but is necessary for Pipelines to recognize
# this class as a Haystack node. Check its implementation for details.
@haystack_node
class MyNode:

    def __init__(self, model_name: str: "deepset-ai/a-model-name"):
        """
        Haystack nodes should have an `__init__` method where they define:

        - `self.expected_inputs = [<expected_input_edge_name(s)>]`:
            A list with all the edges they can possibly receive input from

        - `self.expected_outputs = [<expected_output_edge_name(s)>]`:
            A list with the edges they might possibly produce as output

        - `self.init_parameters = {<init parameters>}`:
            Any state they wish to be persisted in their YAML serialization.
            These values will be given to the `__init__` method of a new instance
            when the pipeline is deserialized.

        The `__init__` must be extrememly lightweight, because it's a frequent
        operation during the construction and validation of the pipeline. If a node
        has some heavy state to initialize (models, backends, etc...) refer to the
        `warm_up()` method.
        """
        # Lightweight state can be initialized here, for example storing the model name
        # to be loaded later. See self.warm_up()
        self.model = None
        self.model_name = model_name
        self.how_many_times_have_I_been_called = 0

        # Contract - all three are mandatory.
        self.init_parameters = {"model_name": model_name}
        self.expected_inputs = ["expected_input_edge_name"]
        self.expected_outputs = ["expected_output_edge_name"]

    def warm_up(self):
        """
        Optional method.

        This method is called by Pipeline before the graph execution.
        Make sure to avoid double-initializations, because Pipeline will not keep
        track of which nodes it called `warm_up` on.
        """
        if not self.model:
            self.model = AutoModel.load_from_pretrained(self.model_name)

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        """
        Mandatory method.

        This is the method where the main functionality of the node should be carried out.
        It's called by `Pipeline.run()`, which passes the following parameters to it:

        - `name: str`: the name of the node. Allows the node to find its own parameters in the `parameters` dictionary (see below).

        - `data: List[Tuple[str, Any]]`: the input data.
            Pipeline guarantees that the following assert always passes: `assert self.expected_inputs == [name for name, value in data]`,
            which means that:
            - `data` is of the same length as `self.expected_inputs`.
            - `data` contains one tuple for each string stored in `self.expected_inputs`.
            - no guarantee is given on the values of these tuples: notably, if there was a decision node upstream, some values might be `None`.
            For example, if a node declares `self.expected_inputs = ["value", "value"]` (think of a Sum node), `data` might look like:
            - `[("value", 1), ("value", 10)]`
            - `[("value", None), ("value", 10)]`
            - `[("value", None), ("value", None)]`, or even
            - `[("value", 1), ("value", ["something", "unexpected"])]`
            but it will never look like:
            - `[("value", 1), ("value", 10), ("value", 100)]`,
            - `[("value": 15)]` or
            - `[("value": 15), ("unexpected", 10)]`.

        - `parameters: Dict[str, Dict[str, Any]]`: a dictionary of dictionaries with all the parameters for all nodes.
            Note that all nodes have access to all parameters for all other nodes: this might come handy to nodes like `Agent`s, that
            want to influence the behavior of nodes downstream.
            Nodes can access their own parameters using `name`, but they must not assume their name is present in the dictionary.
            Therefore the best way to get the parameters is with `my_parameters = parameters.get(name, {})`

        - `stores`: a dictionary of all the (Document)Stores connected to this pipeline.

        Pipeline expect the output of this function to be either a dictionary or a tuple.
        If it's a dictionary, it should always abide to the following format:

        `{output_name: output_value for output_name in <subset of self.expected_output>}`

        Which means that:
        - Nodes are not forced to produce output on all the expected outputs: for example nodes taking a decision, like classifiers,
            can produce output on a subset of the expected output edges and Pipeline will figure out the rest.
        - Nodes must not add any key in the data dictionary that is not present in `self.expected_outputs`,

        Nodes may also want to return a tuple when they altered the content of `parameters` and want their changes to propagate
        downstream. In that case, the format is `(data, parameters)` where `data` follows the contract above and `parameters` should
        match the same format as it had in input, so `{"node_name": {"parameter_name": parameter_value, ...}, ...}`

        """
        self.how_many_times_have_I_been_called += 1

        value = data[0][1]
        print(f"Hello I'm {name}! This instance have been called {self.how_many_times_have_I_been_called} times and this is the value I received: {value}")

        return {"expected_output_edge_name": value}
```

This contract is stored in the docstring of `@haystack_node` and acts as the single source of truth.

### Nodes discovery logic

When pipelines are loaded from YAML, Pipeline needs to find the classes definition somewhere in the imported modules. Currently, at initialization `Pipeline` looks for classes which is decorated with the `@haystack_node` decorator under `haystack`, however such search can be extended (or narrowed) by setting the `search_nodes_in` init parameter of `Pipeline`. Note that it will try to import any module that is not imported yet.

Search might fail in narrow corner cases: for example, inner classes are not discovered (often the case in tests). For these scenarios, `Pipeline` also accepts an `extra_nodes` init parameter that allows users to explicitly provide a dictionary of nodes to merge with the other discovered nodes.

Name collisions are handled by prefixing the node name with the name of the module it was imported from.

## Pipeline TOML representation

_(Disclaimer: no draft implementation available yet)_

Instead of YAML, which is prone to indentation issues, we select TOML as the pipeline serialization format.

Pipeline TOMLs have the following layout:

```toml
# A list of "dependencies" for the pipeline.
# Used to ensure all external nodes are present when loading.
dependencies = [
    "haystack == 2.0.0",
    "my_custom_node_module == 0.0.1",
]

# Stores are defined each in a `[stores.<store_name>]` block.
# Nodes will be able to access them by the name defined here,
# in this case `my_first_store` (see the retrievers below).
[stores.my_first_store]
# class_name is mandatory
class_name = "InMemoryDocumentStore"
# Then come all the additional parameters for the store
use_bm25 = True

[stores.my_second_store]
class_name = "InMemoryDocumentStore"
use_bm25 = False

# Nodes are defined each in a `[node.<node_name>]` block.
# In order to reuse an instance across multiple nodes, instead
# of a `class_name` there should be a pointer to another node.
# TODO: check if TOML has pointer syntax.
[nodes.my_sparse_retriever]
# class_name is mandatory, unless there is a pointer to another node.
class_name = "BM25Retriever"
# Then come all the additional init parameters for the node
store_name = "my_first_store"
top_k = 5

[nodes.my_dense_retriever]
class_name = "EmbeddingRetriever"
model_name = 'deepset-ai/a-model-name'
store_name = "my_second_store"
top_k = 5

[nodes.my_ranker]
class_name = "Ranker"
expected_inputs = ["documents", "documents"]
expected_outputs = ["documents"]

[nodes.my_reader]
class_name = "Reader"
model_name = 'deepset-ai/a-model-name'
top_k = 10

# All the Pipeline parameters, notably:
# - the list of edges is defined here
# - Other init parameters like `max_allowed_loops`, etc..
[pipeline]
edges = [
    ("my_sparse_retriever", "ranker"),
    ("my_dense_retriever", "ranker"),
    ("ranker", "reader"),
]
max_allowed_loops = 10

```

Note that: **1 TOML = 1 Pipeline**

## Haystack Pipeline vs Haystack Project

_(Disclaimer: no draft implementation available yet)_

Haystack Projects are wrappers on top of a set of pipelines. Their advantage is that they can contain nodes and stores that are shared across different pipelines.

In code, they look like this:

```python
class Project:

    def __init__(self, path):
        ... loads the Project TOML ...

    def list_pipelines(self):
        return self.pipelines

    def list_nodes(self):
        return self.nodes

    def list_stores(self):
        return self.stores

    ... CRUD operations for Pipelines, Nodes and Stores ...

    def save(self, path):
        ... serializes down to TOML ...
```

A Project's TOML looks very similar to the Pipeline's, with the difference that each Pipeline is named.

```toml
dependencies = [
    "haystack == 2.0.0",
    "my_custom_node_module == 0.0.1",
]

[stores.my_first_store]
class_name = "InMemoryDocumentStore"
use_bm25 = True

[stores.my_second_store]
class_name = "InMemoryDocumentStore"
use_bm25 = False

[nodes.my_sparse_retriever]
class_name = "BM25Retriever"
store_name = "my_first_store"
top_k = 5

[nodes.my_dense_retriever]
class_name = "EmbeddingRetriever"
model_name = 'deepset-ai/a-model-name'
store_name = "my_second_store"
top_k = 5

[nodes.my_ranker]
class_name = "Ranker"
expected_inputs = ["documents", "documents"]
expected_outputs = ["documents"]

[nodes.my_reader]
class_name = "Reader"
model_name = 'deepset-ai/a-model-name'
top_k = 10

# Note how this Pipeline is named, unlike in Pipeline's TOMLs
[pipeline.hybrid_question_answering]
edges = [
    ("my_sparse_retriever", "ranker"),
    ("my_dense_retriever", "ranker"),
    ("ranker", "reader"),
]
max_allowed_loops = 10

[pipeline.sparse_search]
edges = [
    ("my_sparse_retriever", "reader"),
]
max_allowed_loops = 10

[pipeline.dense_search]
edges = [
    ("my_dense_retriever", "reader"),
]
max_allowed_loops = 10

```

# Open questions

### Choice of TOML vs YAML or HCL

I choose TOML because it looks declarative and quite readable while not suffering from typical YAML issues like sensitivity to whitespace and indentation. However there are many pros and cons of TOML, not least the fact that it needs an external package for serialization, unlike YAML.

### Naming of "Haystack Project"

Better naming is welcome.

### At which level to serialize?

Pipeline and Project's TOML definitions are extremely similar. We might want to keep both, or we might want to take a radical stance and decide that **Pipelines cannot be serialized: only Projects can**.

There are clearly pros and cons and the point surely needs further discussion.

# Drawbacks

There are a number of drawbacks about the proposed approach:

- Migration is going to be far from straightforward for us. Although many nodes can probably work with minor adaptations into the new system, it would be beneficial for most of them to be reduced to their `run()` method, especially indexing nodes. This means that nodes need, at least, to be migrated one by one to the new system and code copied over.

- Migration is going to be far from straightforward for the users: see "Adoption strategy".

- This system allows for pipelines with more complex topologies, which brings the risk of more corner cases. `Pipeline.run()` must be made very solid in order to avoid this scenario.

- Nodes might break more easily while running due to unexpected inputs. While well designed nodes should internally check and deal with such situations, we might face larger amount of bugs due to our failure at noticing the lack of checks at review time.

- The entire system work on the assumption that nodes are well behaving and "polite" to other nodes, for example not touching their parameters unless necessary, etc. Malicious or otherwise "rude" nodes can wreak havoc in `Pipeline`s very easily by messing with other node's parameters and inputs.

# Adoption strategy

Old and new `Pipeline` and nodes are going to be fully incompatible.

We must provide a migration script that can convert their existing pipeline YAMLs into the new ones.

This proposal is best thought as part of the design of Haystack 2.0, where we can afford drastic API changes such as this.

Adoption for dC: still an open question.
