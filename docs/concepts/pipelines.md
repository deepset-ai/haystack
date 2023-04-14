# Advanced Pipelines

Canals aims to support pipelines of arbitrary complexity.

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
#     pipeline.connect('component_b', 'component_a')
#   File "/home/me/projects/canals/canals/pipeline/pipeline.py", line 224, in connect
#     raise PipelineConnectError(
# haystack.pipeline._utils.PipelineConnectError: Cannot connect 'component_b' with 'component_a' with a connection named 'output': their declared inputs and outputs do not match.
# Upstream component 'component_b' declared these outputs:
#  - output (free)
# Downstream component 'component_a' declared these inputs:
#  - input (free)
```

This type of error reporting was found especially useful for components that declare a variable number and name of inputs and outputs depending on their initialization parameters (think of classifiers, for example).

One shortcoming is that currently Pipeline "trusts" the components to respect their own declarations. So if a component states that it will output `intermediate_value`, but outputs something else once run, Pipeline will fail. We accept this failure as a "contract breach": the component should fix its behavior and Pipeline should not try to prevent such scenarios.


## Topologies

Canals supports a variety of different pipeline topologies. Check the pipeline's test suite for some examples:
these are only representations of the graphs that those pipelines generate.

TODO
