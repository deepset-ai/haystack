# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging
import inspect
from typing import Protocol, Union, List, Any, get_origin, get_args
from dataclasses import fields, Field
from functools import wraps

from canals.errors import ComponentError
from canals.component.input_output import Connection, _input, _output


logger = logging.getLogger(__name__)


# We ignore too-few-public-methods Pylint error as this is only meant to be
# the definition of the Component interface.
# A concrete Component will have more than method in any case.
class Component(Protocol):  # pylint: disable=too-few-public-methods
    """
    Abstract interface of a Component.
    This is only used by type checking tools.

    If you want to create a new Component use the @component decorator.
    """

    def run(self, data: Any) -> Any:
        """
        Takes the Component input and returns its output.
        Input and output dataclasses types must be defined in separate methods
        decorated with @component.input and @component.output respectively.

        We use Any both as data and return types since dataclasses don't have a specific type.
        """

    @property
    def __canals_input__(self) -> type:
        pass

    @property
    def __canals_output__(self) -> type:
        pass

    @property
    def __canals_optional_inputs__(self) -> List[str]:
        pass

    @property
    def __canals_mandatory_inputs__(self) -> List[str]:
        pass


class _Component:
    """
    Marks a class as a component. Any class decorated with `@component` can be used by a Pipeline.

    All components must follow the contract below. This docstring is the source of truth for components contract.

    ### `@component` decorator

    All component classes must be decorated with the `@component` decorator. This allows Canals to discover them.

    ### `@component.input`

    All components must decorate one single method with the `@component.input` decorator. This method must return a
    dataclass, which will be used as structure of the input of the component.

    For example, if the node is expecting a list of Documents, the fields of the returned dataclass should be
    `documents: List[Document]`. Note that you don't need to decorate the dataclass youself: `@component.input` will
    add the decorator for you.

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

    By default `@component.input` sets `None` as default for all fields, regardless of their definition: this gives you
    the possibility of passing a part of the input to the pipeline without defining every field of the component.
    For example, using the above definition, you can create an Input dataclass as:

    ```python
    self.input(add=3)
    ```

    and the resulting dataclass will look like `Input(value=None, add=3)`.

    However, if you don't explicitly define them as Optionals, Pipeline will make sure to collect all the values of
    this dataclass before calling the `run()` method, making them in practice non-optional.

    If you instead define a specific field as Optional in the dataclass, then Pipeline will **not** wait for them, and
    will run the component as soon as all the non-optional fields have received a value or, if all fields are optional,
    if at least one of them received it.

    This behavior allows Canals to define loops by not waiting on both incoming inputs of the entry component of the
    loop, and instead running as soon as at least one of them receives a value.

    ### `@component.output`

    All components must decorate one single method with the `@component.output` decorator. This method must return a
    dataclass, which will be used as structure of the output of the component.

    For example, if the node is producing a list of Documents, the fields of the returned dataclass should be
    `documents: List[Document]`. Note that you don't need to decorate the dataclass youself: `@component.output` will
    add the decorator for you.

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

    When the component should run, Pipeline will call this method with an instance of the dataclass returned by the
    method decorated with `@component.input`. This dataclass contains:

    - all the input values coming from other components connected to it,
    - if any is missing, the corresponding value defined in `self.defaults`, if it exists.

    `run()` must return a single instance of the dataclass declared through the method decorated with
    `@component.output`.

    Args:
        class_: the class that Canals should use as a component.
        serializable: whether to check, at init time, if the component can be saved with
        `save_pipelines()`.

    Returns:
        A class that can be recognized as a component.

    Raises:
        ComponentError: if the class provided has no `run()` method or otherwise doesn't respect the component contract.
    """

    def __init__(self):
        self.registry = {}

    @property
    def input(self):
        """
        TODO: Documentation
        """
        return _input

    @property
    def output(self):
        """
        TODO: Documentation
        """
        return _output

    def _decorate(self, class_):
        # '__canals_component__' is used to distinguish components from regular classes.
        # Its value is set to the desired component name: normally it is the class name, but it can technically be customized.
        class_.__canals_component__ = class_.__name__

        # Find input and output properties
        (input_, output) = _find_input_output(class_)

        # Save the input and output properties so it's easier to find them when running the Component since we won't
        # need to search the exact property name each time
        class_.__canals_input__ = input_
        class_.__canals_output__ = output

        # Save optional inputs, optionals inputs are those fields for the __canals_input__ dataclass
        # that have an Optional type.
        # Those are necessary to implement Components that can run with partial input, this gives us
        # the possibility to have cycles in Pipelines.
        class_.__canals_optional_inputs__ = property(_optional_inputs)
        class_.__canals_mandatory_inputs__ = property(_mandatory_inputs)

        # Check that the run method respects all constraints
        _check_run_signature(class_)

        # Makes sure the self.defaults and self.init_parameters dictionaries are always present
        class_.init_parameters = {}
        class_.defaults = {}

        # Automatically registers all the init parameters in an instance attribute called `init_parameters`.
        class_.__init__ = _save_init_params(class_.__init__)

        if class_.__name__ in self.registry:
            logger.error(
                "Component %s is already registered. Previous imported from '%s', new imported from '%s'",
                class_.__name__,
                self.registry[class_.__name__],
                class_,
            )

        self.registry[class_.__name__] = class_
        logger.debug("Registered Component %s", class_)

        return class_

    def __call__(self, class_=None):
        if class_:
            return self._decorate(class_)

        return self._decorate


component = _Component()


def _find_input_output(class_):
    """
    Finds the input and the output definitions for class_ and returns them.

    There must be only a single definition of input and output for class_, if either
    none or more than one are found raise ConnectionError.
    """
    inputs_found = []
    outputs_found = []

    # Get all properties of class_
    properties = inspect.getmembers(class_, predicate=lambda m: isinstance(m, property))
    for _, prop in properties:
        if not hasattr(prop, "fget") or not hasattr(prop.fget, "__canals_connection__"):
            continue

        # Field __canals_connection__ is set by _input and _output decorators
        if prop.fget.__canals_connection__ == Connection.INPUT:
            inputs_found.append(prop)
        elif prop.fget.__canals_connection__ == Connection.OUTPUT:
            outputs_found.append(prop)

    if (in_len := len(inputs_found)) != 1:
        # Raise if we don't find only a single input definition
        if in_len == 0:
            raise ComponentError(
                f"No input definition found in Component {class_.__name__}. "
                "Create a method that returns a dataclass defining the input and "
                "decorate it with @component.input() to fix the error."
            )
        raise ComponentError(f"Multiple input definitions found for Component {class_.__name__}.")

    if (in_len := len(outputs_found)) != 1:
        # Raise if we don't find only a single output definition
        if in_len == 0:
            raise ComponentError(
                f"No output definition found in Component {class_.__name__}. "
                "Create a method that returns a dataclass defining the output and "
                "decorate it with @component.output() to fix the error."
            )
        raise ComponentError(f"Multiple output definitions found for Component {class_.__name__}.")

    return (inputs_found[0], outputs_found[0])


def _check_run_signature(class_):
    """
    Check that the component's run() method exists and respects all constraints
    """
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


def _is_optional(field: Field) -> bool:
    """
    Utility method that returns whether a field has an Optional type or not.
    """
    return get_origin(field.type) is Union and type(None) in get_args(field.type)


def _optional_inputs(self) -> List[str]:
    """
    Return all field names of self that have an Optional type.
    This is meant to be set as a property in a Component.
    """
    return [f.name for f in fields(self.__canals_input__) if _is_optional(f)]


def _mandatory_inputs(self) -> List[str]:
    """
    Return all field names of self that don't have an Optional type.
    This is meant to be set as a property in a Component.
    """
    return [f.name for f in fields(self.__canals_input__) if not _is_optional(f)]


def _save_init_params(init_func):
    """
    Decorator that saves the init parameters of a component in `self.init_parameters`
    """

    @wraps(init_func)
    def wrapper(self, *args, **kwargs):
        # Call the actual __init__ function with the arguments
        init_func(self, *args, **kwargs)

        # Collect and store all the init parameters, preserving whatever the components might have already added there
        self.init_parameters = {**kwargs, **self.init_parameters}

    return wrapper
