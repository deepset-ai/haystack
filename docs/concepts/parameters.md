# Parameters

Parameters are additional values that can be sent directly from the input to each component.
Components will receive them through the `parameters` dictionary of their `run()` method.

Parameters can be provided at different stages with different degrees of importance.
Their purpose is to help users to configure their components dynamically.

## Parameters hierarchy

Parameters can be passed to components at several stages, and they have different priorities. Here they're listed from least priority to top priority.

- Components's default `__init__` parameters: components's `__init__` can provide defaults. Those are used only if no other parameters are passed at any stage.

- Components's `__init__` parameters: at initialization, components might be given values for their parameters. These are stored within the component instance and, if the instance is reused in the pipeline several times, they will be the same on all of them.

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
pipeline.run(data={...}, parameters={"component": {"value_4": 4}})

# Component will receive {"value_1": 1, "value_2": 2, "value_3": 3,"value_4": 4}
```
