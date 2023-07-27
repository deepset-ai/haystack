# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import inspect
from typing import Protocol, Union, Dict, Any, get_origin, get_args
from functools import wraps

from canals.errors import ComponentError
from canals.type_checking import _types_are_compatible


logger = logging.getLogger(__name__)


# We ignore too-few-public-methods Pylint error as this is only meant to be
# the definition of the Component interface.
class Component(Protocol):  # pylint: disable=too-few-public-methods
    """
    Abstract interface of a Component.
    This is only used by type checking tools.
    If you want to create a new Component use the @component decorator.
    """

    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Takes the Component input and returns its output.
        Inputs are defined explicitly by the run method's signature or with `component.set_input_types()` if dynamic.
        Outputs are defined by decorating the run method with `@component.output_types()`
        or with `component.set_output_types()` if dynamic.
        """


def _prepare_init_params_and_sockets(init_func):
    """
    Decorator that saves the init parameters of a component in `self.init_parameters`
    """

    @wraps(init_func)
    def wrapper(self, *args, **kwargs):
        # Call the actual __init__ function with the arguments
        init_func(self, *args, **kwargs)

        # Collect and store all the init parameters, preserving whatever the components might have already added there
        self.init_parameters = {**kwargs, **getattr(self, "init_parameters", {})}

        if not hasattr(self.run, "__canals_io__"):
            raise ComponentError("This component seems to have neither inputs nor outputs.")

    return wrapper


class _Component:
    """
    Marks a class as a component. Any class decorated with `@component` can be used by a Pipeline.

    All components must follow the contract below. This docstring is the source of truth for components contract.

    ### `@component` decorator

    All component classes must be decorated with the `@component` decorator. This allows Canals to discover them.

    ### `__init__(self, **kwargs)`

    Optional method.

    Components may have an `__init__` method where they define:

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

    def set_input_types(self, instance, **types):
        """
        Method that validates the input kwargs of the run method.

        Use as:

        ```python
        @component
        class MyComponent:

            def __init__(self, value: int):
                component.set_input_types(value_1=str, value_2=str)
                ...

            @component.output_types(output_1=int, output_2=str)
            def run(self, **kwargs):
                return {"output_1": kwargs["value_1"], "output_2": ""}
        ```
        """
        run_method = instance.run

        def wrapper(**kwargs):
            """
            Adds a check that validates the input kwargs of the run method.
            """
            # Check input types
            for key, value in kwargs.items():
                if key not in types:
                    raise ComponentError(f"Input value '{key}' not declared in component.set_input_types()")
                if _types_are_compatible(value, types[key]):
                    raise ComponentError(
                        f"Input type {type(value)} for value '{key}' doesn't match the one declared in "
                        f"component.set_input_types() ({types[key]}))"
                    )
            return run_method(**kwargs)

        # Store the input types in the run method
        wrapper.__canals_io__ = getattr(instance.run, "__canals_io__", {})
        wrapper.__canals_io__["input_types"] = {
            name: {"name": name, "type": type_, "is_optional": _is_optional(type_)} for name, type_ in types.items()
        }

        # Assigns the wrapped method to the instance's run()
        instance.run = wrapper

    def set_output_types(self, instance, **types):
        """
        Method that validates the output dictionary of the run method.

        Use as:

        ```python
        @component
        class MyComponent:

            def __init__(self, value: int):
                component.set_output_types(output_1=int, output_2=str)
                ...

            def run(self, value: int):
                return {"output_1": 1, "output_2": "2"}
        ```
        """
        if not types:
            return

        run_method = instance.run

        def wrapper(*args, **kwargs):
            """
            Adds a check that validates the output dictionary of the run method.
            """
            result = run_method(*args, **kwargs)
            # Check output types
            for key in result:
                if key not in types:
                    raise ComponentError(f"Return value '{key}' not declared in component.set_output_types()")
                if _types_are_compatible(types[key], result[key]):
                    raise ComponentError(
                        f"Return type {type(result[key])} for value '{key}' doesn't match the one declared in "
                        f"component.set_output_types() ({types[key]}))"
                    )
            return result

        # Store the output types in the run method
        wrapper.__canals_io__ = getattr(instance.run, "__canals_io__", {})
        wrapper.__canals_io__["output_types"] = {name: {"name": name, "type": type_} for name, type_ in types.items()}

        # Assigns the wrapped method to the instance's run()
        instance.run = wrapper

    def output_types(self, **types):
        """
        Decorator factory that validates the output dictionary of the run method.

        Use as:

        ```python
        @component
        class MyComponent:
            @component.output_types(output_1=int, output_2=str)
            def run(self, value: int):
                return {"output_1": 1, "output_2": "2"}
        ```
        """

        def output_types_decorator(run_method):
            """
            Decorator that validates the output dictionary of the run method.
            """
            # Store the output types in the run method - used by the pipeline to build the sockets.
            if not hasattr(run_method, "__canals_io__"):
                run_method.__canals_io__ = {}
            run_method.__canals_io__["output_types"] = {
                name: {"name": name, "type": type_} for name, type_ in types.items()
            }

            @wraps(run_method)
            def output_types_impl(self, *args, **kwargs):
                """
                Adds a check that validates the output dictionary of the run method.
                """
                result = run_method(self, *args, **kwargs)

                # Check output types
                for key in result:
                    if key not in types:
                        raise ComponentError(f"Return value '{key}' not declared in @output_types decorator")
                    if _types_are_compatible(types[key], result[key]):
                        raise ComponentError(
                            f"Return type {type(result[key])} for value '{key}' doesn't match the one declared in "
                            f"@output_types decorator ({types[key]}))"
                        )
                return result

            return output_types_impl

        return output_types_decorator

    def _component(self, class_):
        """
        Decorator validating the structure of the component and registering it in the components registry.
        """
        logger.debug("Registering %s as a component", class_)

        # Check for run()
        if not hasattr(class_, "run"):
            raise ComponentError(f"{class_.__name__} must have a 'run()' method. See the docs for more information.")
        run_signature = inspect.signature(class_.run)

        # Create the input sockets
        if not hasattr(class_.run, "__canals_io__"):
            class_.run.__canals_io__ = {}
        class_.run.__canals_io__["input_types"] = {
            param: {
                "name": param,
                "type": run_signature.parameters[param].annotation,
                "is_optional": _is_optional(run_signature.parameters[param].annotation),
            }
            for param in list(run_signature.parameters)[1:]  # First is 'self' and it doesn't matter.
        }

        # Automatically registers all the init parameters in an instance attribute called `_init_parameters`.
        # See `save_init_parameters()`.
        class_.__init__ = _prepare_init_params_and_sockets(class_.__init__)

        # Save the component in the class registry (for deserialization)
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
        """Allows us to use this decorator with parenthesis and without."""
        if class_:
            return self._component(class_)

        return self._component


component = _Component()


def _is_optional(type_: type) -> bool:
    """
    Utility method that returns whether a type is Optional.
    """
    return get_origin(type_) is Union and type(None) in get_args(type_)
