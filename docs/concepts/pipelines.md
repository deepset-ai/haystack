# Advanced Pipelines

Canals aims to support pipelines of (close to) arbitrary complexity.

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

## Topologies

Canals supports a variety of different pipeline topologies:

- Simple linear pipelines
- Branching pipelines where all or only some branches are executed
- Pipelines merging a variable number of inputs, depending on decisions taken upstream
- Simple loops
- Multiple entry components, either alternative or parallel
- Multiple exit components, either alternative or parallel

Check the pipeline's test suite for some examples: these are only representations of the graphs that those pipelines generate.
