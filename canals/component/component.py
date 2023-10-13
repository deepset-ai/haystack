# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
"""
    Attributes:

        component: Marks a class as a component. Any class decorated with `@component` can be used by a Pipeline.

    All components must follow the contract below. This docstring is the source of truth for components contract.

    <hr>

    `@component` decorator

    All component classes must be decorated with the `@component` decorator. This allows Canals to discover them.

    <hr>

    `__init__(self, **kwargs)`

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

    <hr>

    `warm_up(self)`

    Optional method.

    This method is called by Pipeline before the graph execution. Make sure to avoid double-initializations,
    because Pipeline will not keep track of which components it called `warm_up()` on.

    <hr>

    `run(self, data)`

    Mandatory method.

    This is the method where the main functionality of the component should be carried out. It's called by
    `Pipeline.run()`.

    When the component should run, Pipeline will call this method with an instance of the dataclass returned by the
    method decorated with `@component.input`. This dataclass contains:

    - all the input values coming from other components connected to it,
    - if any is missing, the corresponding value defined in `self.defaults`, if it exists.

    `run()` must return a single instance of the dataclass declared through the method decorated with
    `@component.output`.

"""

import logging
import inspect
from typing import Protocol, Dict, Any
from functools import wraps

from canals.errors import ComponentError
from canals.type_utils import _is_optional


logger = logging.getLogger(__name__)


class Component(Protocol):
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


class _Component:
    """
    See module's docstring.

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
            return run_method(**kwargs)

        # Store the input types in the run method
        wrapper.__canals_input__ = {
            name: {"name": name, "type": type_, "is_optional": _is_optional(type_)} for name, type_ in types.items()
        }
        wrapper.__canals_output__ = getattr(run_method, "__canals_output__", {})

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
            return run_method(*args, **kwargs)

        # Store the output types in the run method
        wrapper.__canals_input__ = getattr(run_method, "__canals_input__", {})
        wrapper.__canals_output__ = {name: {"name": name, "type": type_} for name, type_ in types.items()}

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

            @wraps(run_method)
            def wrapper(self, *args, **kwargs):
                return run_method(self, *args, **kwargs)

            wrapper.__canals_input__ = getattr(run_method, "__canals_input__", {})
            wrapper.__canals_output__ = {name: {"name": name, "type": type_} for name, type_ in types.items()}

            return wrapper

        return output_types_decorator

    def _component(self, class_):
        """
        Decorator validating the structure of the component and registering it in the components registry.
        """
        logger.debug("Registering %s as a component", class_)

        # Check for required methods
        if not hasattr(class_, "run"):
            raise ComponentError(f"{class_.__name__} must have a 'run()' method. See the docs for more information.")
        run_signature = inspect.signature(class_.run)

        # Create the input sockets
        class_.run.__canals_input__ = {
            param: {
                "name": param,
                "type": run_signature.parameters[param].annotation,
                "is_optional": _is_optional(run_signature.parameters[param].annotation),
            }
            for param in list(run_signature.parameters)[1:]  # First is 'self' and it doesn't matter.
        }

        # Save the component in the class registry (for deserialization)
        if class_.__name__ in self.registry:
            # It may occur easily in notebooks by re-running cells.
            logger.debug(
                "Component %s is already registered. Previous imported from '%s', new imported from '%s'",
                class_.__name__,
                self.registry[class_.__name__],
                class_,
            )
        self.registry[class_.__name__] = class_
        logger.debug("Registered Component %s", class_)

        setattr(class_, "__canals_component__", True)

        return class_

    def __call__(self, class_=None):
        """Allows us to use this decorator with parenthesis and without."""
        if class_:
            return self._component(class_)

        return self._component


component = _Component()
