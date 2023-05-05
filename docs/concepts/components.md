# Components

In order to be recognized as components and work in a Pipeline, Components must follow the contract below.

## Requirements

### `@component` decorator

All component classes must be decorated with the `@component` decorator. This allows Canals to discover them.

### `__init__()`

```python
def __init__(self, [... components init parameters ...]):
```
Optional method.

Components may have an `__init__` method where they define:

- `self.defaults = {parameter_name: parameter_default_value, ...}`:
    All values defined here will be sent to the `run()` method when the Pipeline calls it.
    If any of these parameters is also receiving input from other components, those have precedence.
    This collection of values is supposed to replace the need for default values in `run()` and make them
    dynamically configurable.

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


### `warm_up()`

```python
def warm_up(self):
```
Optional method.

This method is called by Pipeline before the graph execution. Make sure to avoid double-initializations,
because Pipeline will not keep track of which components it called `warm_up()` on.


### `Output`

```python
@dataclass
class Output:
    <expected output fields>
```
Semi-mandatory method (either this or `self.output_types(self)`).

This inner class defines how the output of this component looks like. For example, if the node is producing
a list of Documents, the fields of the class should be `documents: List[Document]`

Defaults are allowed, however `Optional`, `Union` and similar "generic" types are not. This is necessary to allow
proper validation of the connections, which rely on the type of these fields.

Some components may need more dynamic output: for example, your component accepts a list of file extensions at
init time and wants to have one output field for each of those. For these scenarios, refer to `self.output_type()`.

Every component should define **either** `Output` or `self.output_types`.


### `output_types()`

```python
def output_types(self) -> dataclass:
```
Semi-mandatory method (either this or `class Output`).

This method defines how the output of this component looks like. For example, if the node is producing
a list of Documents, this method should return a dataclass with such fields, for example:
`return make_dataclass("Output", [(f"documents", List[Document], None)])`

Defaults are allowed, however `Optional`, `Union` and similar "generic" types are not. This is necessary to allow
proper validation of the connections, which rely on the type of these fields.

If the output is static, normally the `Output` dataclass is preferred, as it provides autocompletion for the users.

Every component should define **either** `Output` or `self.output_types`.


### `run()`

```python
def run(self, <parameters, typed>) -> Output:
```
Mandatory method.

This is the method where the main functionality of the component should be carried out. It's called by
`Pipeline.run()`.

When the component should run, Pipeline will call this method with:

- all the input values coming from "upstream" components connected to it,
- if any is missing, the corresponding value defined in `self.defaults`, if it exists.

All parameters of `run()` **must be typed**. The types are used by `Pipeline.connect()` to make sure the two
components agree on the type being passed, to try ensure the connection will be successful.
Defaults are allowed, however `Optional`, `Union` and similar "generic" types are not, just as for the outputs.

`run()` must return a single instance of the dataclass declared through either `Output` or `self.output_types()`.

A variadic `run()` method is allowed if it respects the following rules:

- It can take **either** regular parameters, or a single variadic positional (`*args`), NOT BOTH.
- `**kwargs` are not supported
- The variadic `*args` must be typed, for example `*args: int` if the component accepts any number of integers.

Args:
    class_: the class that Canals should use as a component.
    serializable: whether to check, at init time, if the component can be saved with
    `save_pipelines()`.


## Example components

### Basic
Here is an example of a simple component that adds two values together and returns their sum.

```python
from dataclasses import dataclass
from canals import component

@component
class AddTwoValues:
    """
    Adds the value of `add` to `value`. If not given, `add` defaults to 1.
    """

    @dataclass
    class Output:
        value: int

    def __init__(self, add: int = 1):
        self.defaults = {"add": add}

    def run(self, value: int, add: int) -> Output:
        return AddTwoValues.Output(value=value + add)
```

### Variadic

Here is an example of a variadic component that adds all the incoming values together and returns their sum.

```python
from dataclasses import dataclass
from canals import component

@component
class Sum:
    """
    Sums the values of all the input connections together.
    """

    @dataclass
    class Output:
        total: int

    def run(self, *value: int) -> Output:
        total = sum(value)
        return Sum.Output(total=total)
```

### Dynamic output

Here is an example of a component that returns the incoming value on a different edge depending on its remainder.

This is an example of how to use `self.output_type()` in practice.

```python
from dataclasses import make_dataclass
from canals import component

@component
class Remainder:
    """
    Redirects the value, unchanged, along the connection corresponding to the remainder of a division.
    For example, if `divisor=3`, the value `5` would be sent along the second output connection.
    """

    def __init__(self, divisor: int = 2):
        self.divisor = divisor
        self._output_type = make_dataclass("Output", [(f"remainder_is_{val}", int, None) for val in range(divisor)])

    @property
    def output_type(self):
        return self._output_type

    def run(self, value: int):
        """
        :param value: the value to check the remainder of.
        """
        remainder = value % self.divisor
        output = self.output_type()
        setattr(output, f"remainder_is_{remainder}", value)
        return output
```
