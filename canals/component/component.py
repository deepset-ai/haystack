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

    ```
    def warm_up(self):
    ```
    Optional method.

    This method is called by Pipeline before the graph execution. Make sure to avoid double-initializations,
    because Pipeline will not keep track of which components it called `warm_up()` on.

    ```
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

    ```
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

    ```
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

    # Automatically registers all the init parameters in an instance attribute called `_init_parameters`.
    # See `save_init_params()`.
    class_.__init__ = save_init_params(class_.__init__)

    # Makes sure the self.defaults dictionary is always present
    class_.__init__ = init_defaults(class_.__init__)

    return class_
