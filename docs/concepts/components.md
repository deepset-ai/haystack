# Components

In order to be recognized as components and work in a Pipeline, Components must follow the contract below.

## Requirements

### `@component` decorator

All component classes must be decorated with the `@component` decorator. This allows Canals to discover them.

### `Input`

```python
@dataclass
class Input(ComponentInput / VariadicComponentInput):
    <expected input fields, typed, with no defaults>
```
Semi-mandatory method (either this or `self.input_type(self)`).

This inner class defines how the input of this component looks like. For example, if the node is expecting
a list of Documents, the fields of the class should be `documents: List[Document]`

Defaults are allowed, however `Optional`, `Union` and similar "generic" types are not. This is necessary to allow
proper validation of the connections, which rely on the type of these fields.

If your node expects variadic input, use `VariadicComponentInput`. In all other scenarios, use `ComponentInput`
as your base class.

Some components may need more dynamic input. For these scenarios, refer to `self.input_type()`.

Every component should define **either** `Input` or `self.input_type()`.


### `input_type()`

```python
@property
def input_type(self) -> ComponentInput / VariadicComponentInput:
```
Semi-mandatory method (either this or `class Input`).

This method defines how the input of this component looks like. For example, if the node is expecting
a list of Documents, this method should return a dataclass, subclass of either `ComponentInput` or
`VariadicComponentInput`, with such fields. For example, it could build the dataclass as
`make_dataclass("Input", fields=[(f"documents", List[Document], None)], bases=(ComponentInput, ))` and return it.

Defaults are allowed, however `Optional`, `Union` and similar "generic" types are not. This is necessary to allow
proper validation of the connections, which rely on the type of these fields.

Normally the `Input` dataclass is preferred, as it provides autocompletion for the users and is much easier to use.

Every component should define **either** `Input` or `self.input_type()`.


### `Output`

```python
@dataclass
class Output(ComponentOutput):
    <expected output fields, typed>
```
Semi-mandatory method (either this or `self.output_type()`).

This inner class defines how the output of this component looks like. For example, if the node is producing
a list of Documents, the fields of the class should be `documents: List[Document]`

Defaults are allowed, however `Optional`, `Union` and similar "generic" types are not. This is necessary to allow
proper validation of the connections, which rely on the type of these fields.

Some components may need more dynamic output: for example, your component accepts a list of file extensions at
init time and wants to have one output field for each of those. For these scenarios, refer to `self.output_type()`.

Every component should define **either** `Output` or `self.output_type()`.


### `output_type()`

```python
@property
def output_type(self) -> ComponentOutput:
```
Semi-mandatory method (either this or `class Output`).

This method defines how the output of this component looks like. For example, if the node is producing
a list of Documents, this method should return a dataclass with such fields, for example:
`return make_dataclass("Output", fields=[(f"documents", List[Document], None)], bases=(ComponentOutput, ))`

Defaults are allowed, however `Optional`, `Union` and similar "generic" types are not. This is necessary to allow
proper validation of the connections, which rely on the type of these fields.

If the output is static, normally the `Output` dataclass is preferred, as it provides autocompletion for the users.

Every component should define **either** `Output` or `self.output_type`.


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


### `warm_up()`

```python
def warm_up(self):
```
Optional method.

This method is called by Pipeline before the graph execution. Make sure to avoid double-initializations,
because Pipeline will not keep track of which components it called `warm_up()` on.


### `run()`

```python
def run(self, data: <Input if defined, otherwise untyped>) -> <Output if defined, otherwise untyped>:
```
Mandatory method.

This is the method where the main functionality of the component should be carried out. It's called by
`Pipeline.run()`.

When the component should run, Pipeline will call this method with:

- all the input values coming from other components connected to it,
- if any is missing, the corresponding value defined in `self.defaults`, if it exists.

`run()` must return a single instance of the dataclass declared through either `Output` or `self.output_type()`.


## Example components

### Basic
Here is an example of a simple component that adds two values together and returns their sum.

```python
from dataclasses import dataclass
from canals.component import component, ComponentInput, ComponentOutput

@component
class AddFixedValue:
    """
    Adds the value of `add` to `value`. If not given, `add` defaults to 1.
    """

    @dataclass
    class Input(ComponentInput):
        value: int
        add: int

    @dataclass
    class Output(ComponentOutput):
        value: int

    def __init__(self, add: Optional[int] = 1):
        if add:
            self.defaults = {"add": add}

    def run(self, data: Input) -> Output:
        return AddFixedValue.Output(value=data.value + data.add)

```

### Variadic

Here is an example of a variadic component that adds all the incoming values together and returns their sum.

```python
from dataclasses import dataclass
from canals.component import component, VariadicComponentInput, ComponentOutput

@component
class Sum:
    """
    Sums the values of all the input connections together.
    """

    @dataclass
    class Input(VariadicComponentInput):
        values: List[int]

    @dataclass
    class Output(ComponentOutput):
        total: int

    def run(self, data: Input) -> Output:
        return Sum.Output(total=sum(data.values))

```

### Dynamic output

Here is an example of a component that returns the incoming value on a different edge depending on its remainder.

This is an example of how to use `self.output_type()` in practice.

```python
from dataclasses import make_dataclass
from canals.component import component, ComponentInput, ComponentOutput


@component
class Remainder:
    """
    Redirects the value, unchanged, along the connection corresponding to the remainder
    of a division. For example, if `divisor=3`, the value `5` would be sent along
    the second output connection.
    """

    @dataclass
    class Input(ComponentInput):
        value: int
        add: int = 1

    def __init__(self, divisor: int = 2):
        if divisor == 0:
            raise ValueError("Can't divide by zero")
        self.divisor = divisor

        self._output_type = make_dataclass(
            "Output", fields=[(f"remainder_is_{val}", int, None) for val in range(divisor)], bases=(ComponentOutput,)
        )

    @property
    def output_type(self):
        return self._output_type

    def run(self, data: Input):
        """
        :param value: the value to check the remainder of.
        """
        remainder = data.value % self.divisor
        output = self.output_type()
        setattr(output, f"remainder_is_{remainder}", data.value)
        return output

```
