# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging
import inspect

from canals.errors import ComponentError
from canals.component.decorators import save_init_params, init_defaults


logger = logging.getLogger(__name__)


def component(class_):
    """
    Marks a class as a component. Any class decorated with `@component` can be used by a Pipeline.

    All components must follow the contract below. This docstring is the source of truth for components contract.

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

    Args:
        class_: the class that Canals should use as a component.
        serializable: whether to check, at init time, if the component can be saved with
        `save_pipelines()`.

    Returns:
        A class that can be recognized as a component.

    Raises:
        ComponentError: if the class provided has no `run()` method or otherwise doesn't respect the component contract.
    """
    logger.debug("Registering %s as a component", class_)

    # '__canals_component__' is used to distinguish components from regular classes.
    # Its value is set to the desired component name: normally it is the class name, but it can technically be customized.
    class_.__canals_component__ = class_.__name__

    # Check for Input
    if not hasattr(class_, "Input") and not hasattr(class_, "input_type"):
        raise ComponentError(
            "Components must either have an Input dataclass or a 'input_type' property that returns such dataclass"
        )
    if hasattr(class_, "Input") and not hasattr(class_.Input, "_component_input"):
        raise ComponentError(f"{class_.__name__}.Input must inherit from ComponentInput")

    # Check for Output
    if not hasattr(class_, "Output") and not hasattr(class_, "output_type"):
        raise ComponentError(
            "Components must either have an Output dataclass or a 'output_type' property that returns such dataclass"
        )
    if hasattr(class_, "Output") and not hasattr(class_.Output, "_component_output"):
        raise ComponentError(f"{class_.__name__}.Output must inherit from ComponentOutput")

    # Check for run()
    if not hasattr(class_, "run"):
        raise ComponentError(f"{class_.__name__} must have a 'run()' method. See the docs for more information.")
    run_signature = inspect.signature(class_.run)

    # run() must take a single input param
    if len(run_signature.parameters) != 2:
        raise ComponentError("run() must accept only a single parameter called 'data'.")

    # The input param must be called data
    if not "data" in run_signature.parameters:
        raise ComponentError("run() must accept a parameter called 'data'.")

    # Either give a self.input_type function or type 'data' with the Input dataclass
    if not hasattr(class_, "input_type") and run_signature.parameters["data"].annotation != class_.Input:
        raise ComponentError(f"'data' must be typed and the type must be {class_.__name__}.Input.")

    # Check for the return types
    if not hasattr(class_, "output_type") and run_signature.return_annotation == inspect.Parameter.empty:
        raise ComponentError(f"{class_.__name__}.run() must declare the type of its return value.")

    # Automatically registers all the init parameters in an instance attribute called `init_parameters`.
    # See `save_init_params()`.
    class_.__init__ = save_init_params(class_.__init__)

    # Makes sure the self.defaults dictionary is always present
    class_.__init__ = init_defaults(class_.__init__)

    return class_
