# Components

In order to be recognized as components and work in a Pipeline, Components must follow the contract below.

## Requirements

### `@component` decorator

All component classes must be decorated with the `@component` decorator. This allows Canals to discover them.

### `@component.input`

All components must decorate one single method with the `@component.input` decorator. This method must return a dataclass, which will be used as structure of the input of the component.

For example, if the node is expecting a list of Documents, the fields of the returned dataclass should be `documents: List[Document]`. Note that you don't need to decorate the dataclass youself: `@component.input` will add the decorator for you.

Here is an example of such method:

```python
@component.input
def input(self):
    class Input:
        value: int
        add: int

    return Input
```

Defaults are allowed, as much as default factories and other dataclass properties.

By default `@component.input` sets `None` as default for all fields, regardless of their definition: this gives you the
possibility of passing a part of the input to the pipeline without defining every field of the component. For example,
using the above definition, you can create an Input dataclass as:

```python
self.input(add=3)
```

and the resulting dataclass will look like `Input(value=None, add=3)`.

However, if you don't explicitly define them as Optionals, Pipeline will make sure to collect all the values of this
dataclass before calling the `run()` method, making them in practice non-optional.

If you instead define a specific field as Optional in the dataclass, then Pipeline will **not** wait for them, and will
run the component as soon as all the non-optional fields have received a value or, if all fields are optional, if at
least one of them received it.

This behavior allows Canals to define loops by not waiting on both incoming inputs of the entry component of the loop,
and instead running as soon as at least one of them receives a value.

### `@component.output`

All components must decorate one single method with the `@component.output` decorator. This method must return a dataclass, which will be used as structure of the output of the component.

For example, if the node is producing a list of Documents, the fields of the returned dataclass should be `documents: List[Document]`. Note that you don't need to decorate the dataclass youself: `@component.output` will add the decorator for you.

Here is an example of such method:

```python
@component.output
def output(self):
    class Output:
        value: int

    return Output
```

Defaults are allowed, as much as default factories and other dataclass properties.

### `__init__(self, **kwargs)`

Optional method.

Components may have an `__init__` method where they define:

- `self.defaults = {parameter_name: parameter_default_value, ...}`:
    All values defined here will be sent to the `run()` method when the Pipeline calls it.
    If any of these parameters is also receiving input from other components, those have precedence.
    This collection of values is supposed to replace the need for default values in `run()` and make them
    dynamically configurable. Keep in mind that only these defaults will count at runtime: defaults given to
    the `Input` dataclass (see above) will be ignored.

- `self.init_parameters = {same parameters that the __init__ method received}`:
    In this dictionary you can store any state the components wish to be persisted when they are saved.
    These values will be given to the `__init__` method of a new instance when the pipeline is loaded.
    Note that by default the `@component` decorator saves the arguments automatically.
    However, if a component sets their own `init_parameters` manually in `__init__()`, that will be used instead.
    Note: all of the values contained here **must be JSON serializable**. Serialize them manually if needed.

Components should take only "basic" Python types as parameters of their `__init__` function, or iterables and
dictionaries containing only such values. Anything else (objects, functions, etc) will raise an exception at init
time. If there's the need for such values, consider serializing them to a string.

_(TODO explain how to use classes and functions in init. In the meantime see `test/components/test_accumulate.py`)_

The `__init__` must be extrememly lightweight, because it's a frequent operation during the construction and
validation of the pipeline. If a component has some heavy state to initialize (models, backends, etc...) refer to
the `warm_up()` method.


### `warm_up(self)`

Optional method.

This method is called by Pipeline before the graph execution. Make sure to avoid double-initializations,
because Pipeline will not keep track of which components it called `warm_up()` on.


### `run(self, data)`

Mandatory method.

This is the method where the main functionality of the component should be carried out. It's called by
`Pipeline.run()`.

When the component should run, Pipeline will call this method with an instance of the dataclass returned by the method decorated with `@component.input`. This dataclass contains:

- all the input values coming from other components connected to it,
- if any is missing, the corresponding value defined in `self.defaults`, if it exists.

`run()` must return a single instance of the dataclass declared through the method decorated with `@component.output`.


## Example components

Here is an example of a simple component that adds a fixed value to its input and returns their sum.

```python
from typing import Optional
from canals.component import component

@component
class AddFixedValue:
    """
    Adds the value of `add` to `value`. If not given, `add` defaults to 1.
    """

    @component.input  # type: ignore
    def input(self):
        class Input:
            value: int
            add: int

        return Input

    @component.output  # type: ignore
    def output(self):
        class Output:
            value: int

        return Output

    def __init__(self, add: Optional[int] = 1):
        if add:
            self.defaults = {"add": add}

    def run(self, data):
        return self.output(value=data.value + data.add)
```

See `tests/sample_components` for examples of more complex components with variable inputs and output, and so on.
