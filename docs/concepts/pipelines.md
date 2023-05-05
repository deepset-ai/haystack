# Pipelines

Canals aims to support pipelines of (close to) arbitrary complexity. It currently supports a variety of different topologies, such as:

- Simple linear pipelines
- Branching pipelines where all or only some branches are executed
- Pipelines merging a variable number of inputs, depending on decisions taken upstream
- Simple loops
- Multiple entry components, either alternative or parallel
- Multiple exit components, either alternative or parallel

Check the pipeline's test suite for some examples.

## Validation

Pipeline performs validation on the connection type level: when calling `Pipeline.connect()`, it uses the values of the
components' `run()` method signature and the `Output` dataclass (or equivalent dataclass returned by
`self.output_type()`) to make sure that the connection is possible.

On top of this, specific connections can be specified with the syntax `component_name.input_or_output_name`.

For example, let's imagine we have two components with the following I/O declared:

```python
@component
class ComponentA:

    @dataclass
    class Output:
        intermediate_value: str

    def run(self, input_value: int) -> Output:
        return ComponentA.output(intermediate_value="hello")

@component
class ComponentB:

    @dataclass
    class Output:
        output_value: List[int]

    def run(self, intermediate_value: str) -> Output:
        return ComponentB.output(output_value=[1, 2, 3])
```

This is the behavior of `Pipeline.connect()`:

```python
pipeline.add_component('component_a', ComponentA())
pipeline.add_component('component_b', ComponentB())

# All of these succeeds
pipeline.connect('component_a', 'component_b')
pipeline.connect('component_a.intermediate_value', 'component_b')
pipeline.connect('component_a', 'component_b.intermediate_value')
pipeline.connect('component_a.intermediate_value', 'component_b.intermediate_value')
```

These, instead, fail:

```python
pipeline.connect('component_a', 'component_a')
# canals.errors.PipelineConnectError: Cannot connect 'component_a' with 'component_a': no matching connections available.
# 'component_a':
#  - intermediate_value (str)
# 'component_a':
#  - input_value (int, available)

pipeline.connect('component_b', 'component_a')
# canals.errors.PipelineConnectError: Cannot connect 'component_b' with 'component_a': no matching connections available.
# 'component_b':
#  - output_value (List)
# 'component_a':
#  - input_value (int, available)
```

In addition, components names are validated:

```python
pipeline.connect('component_a', 'component_c')
# ValueError: Component named component_c not found in the pipeline.
```

Just like input and output names, when stated:

```python
pipeline.connect('component_a.input', 'component_b')
# canals.errors.PipelineConnectError: 'component_a.input does not exist. Output connections of component_a are: intermediate_value (type str)

pipeline.connect('component_a.output', 'component_b')
# canals.errors.PipelineConnectError: 'component_a.output does not exist. Output connections of component_a are: intermediate_value (type str)

pipeline.connect('component_a', 'component_b.input')
# canals.errors.PipelineConnectError: 'component_b.input does not exist. Input connections of component_b are: intermediate_value (type str)

pipeline.connect('component_a', 'component_b.output')
# canals.errors.PipelineConnectError: 'component_b.output does not exist. Input connections of component_b are: intermediate_value (type str)
```

## Save and Load

Pipelines can be serialized to Python dictionaries, that can be then dumped to JSON or to any other suitable format, like YAML, TOML, HCL, etc. These pipelines can then be loaded back.

Here is an example of Pipeline saving and loading:

```python
from haystack.pipelines import Pipeline, save_pipelines, load_pipelines

pipe1 = Pipeline()
pipe2 = Pipeline()

# .. assemble the pipelines ...

# Save the pipelines
save_pipelines(
    pipelines={
        "pipe1": pipe1,
        "pipe2": pipe2,
    },
    path="my_pipelines.json",
    _writer=json.dumps
)

# Load the pipelines
new_pipelines = load_pipelines(
    path="my_pipelines.json",
    _reader=json.loads
)

assert new_pipelines["pipe1"] == pipe1
assert new_pipelines["pipe2"] == pipe2
```

Note how the save/load functions accept a `_writer`/`_reader` function: this choice frees us from committing strongly to a specific template language, and although a default will be set (be it YAML, TOML, HCL or anything else) the decision can be overridden by passing another explicit reader/writer function to the `save_pipelines`/`load_pipelines` functions.

This is how the resulting file will look like, assuming a JSON writer was chosen.

`my_pipeline.json`

```python
{
    "pipelines": {
        "pipe1": {
            # All the components that would be added with a
            # Pipeline.add_component() call
            "components": {
                "first_addition": {
                    "type": "AddValue",
                    "init_parameters": {
                        "add": 1
                    },
                },
                "double": {
                    "type": "Double",
                    "init_parameters": {}
                },
                "second_addition": {
                    "type": "AddValue",
                    "init_parameters": {
                        "add": 1
                    },
                },
                # This is how instances of the same component are reused
                "third_addition": {
                    "refer_to": "pipe1.first_addition"
                },
            },
            # All the components that would be made with a
            # Pipeline.connect() call
            "connections": [
                ("first_addition", "double", "value/value"),
                ("double", "second_addition", "value/value"),
                ("second_addition", "third_addition", "value/value"),
            ],
            # All other Pipeline.__init__() parameters go here.
            "metadata": {"type": "test pipeline", "author": "me"},
            "max_loops_allowed": 100,
        },
        "pipe2": {
            "components": {
                "first_addition": {
                    # We can reference components from other pipelines too!
                    "refer_to": "pipe1.first_addition",
                },
                "double": {
                    "type": "Double",
                    "init_parameters": {}
                },
                "second_addition": {
                    "refer_to": "pipe1.second_addition"
                },
            },
            "connections": [
                ("first_addition", "double", "value/value"),
                ("double", "second_addition", "value/value"),
            ],
            "metadata": {"type": "another test pipeline", "author": "you"},
            "max_loops_allowed": 100,
        },
    },
    # A list of "dependencies" for the application.
    # Used to ensure all external components are present when loading.
    "dependencies": ["my_custom_components_module"],
}
```
