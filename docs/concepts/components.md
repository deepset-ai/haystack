# Creating Components

In order to be recognized as components and work in a Pipeline, Components must follow the contract below.

### Decorator

All component classes must be decorated with the `@component` decorator. This allows Canals to discover them.

### `__init__()`

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
