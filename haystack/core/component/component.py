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

    The `__init__` must be extremely lightweight, because it's a frequent operation during the construction and
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
from typing import Protocol, runtime_checkable, Any
from types import new_class
from copy import deepcopy

from haystack.core.component.sockets import InputSocket, OutputSocket, _empty
from haystack.core.errors import ComponentError

logger = logging.getLogger(__name__)


@runtime_checkable
class Component(Protocol):
    """
    Note this is only used by type checking tools.

    In order to implement the `Component` protocol, custom components need to
    have a `run` method. The signature of the method and its return value
    won't be checked, i.e. classes with the following methods:

        def run(self, param: str) -> Dict[str, Any]:
            ...

    and

        def run(self, **kwargs):
            ...

    will be both considered as respecting the protocol. This makes the type
    checking much weaker, but we have other places where we ensure code is
    dealing with actual Components.

    The protocol is runtime checkable so it'll be possible to assert:

        isinstance(MyComponent, Component)
    """

    def run(self, *args: Any, **kwargs: Any):  # pylint: disable=missing-function-docstring
        ...


class ComponentMeta(type):
    def __call__(cls, *args, **kwargs):
        """
        This method is called when clients instantiate a Component and
        runs before __new__ and __init__.
        """
        # This will call __new__ then __init__, giving us back the Component instance
        instance = super().__call__(*args, **kwargs)

        # Before returning, we have the chance to modify the newly created
        # Component instance, so we take the chance and set up the I/O sockets

        # If `component.set_output_types()` was called in the component constructor,
        # `__canals_output__` is already populated, no need to do anything.
        if not hasattr(instance, "__canals_output__"):
            # If that's not the case, we need to populate `__canals_output__`
            #
            # If the `run` method was decorated, it has a `_output_types_cache` field assigned
            # that stores the output specification.
            # We deepcopy the content of the cache to transfer ownership from the class method
            # to the actual instance, so that different instances of the same class won't share this data.
            instance.__canals_output__ = deepcopy(getattr(instance.run, "_output_types_cache", {}))

        # Create the sockets if set_input_types() wasn't called in the constructor.
        # If it was called and there are some parameters also in the `run()` method, these take precedence.
        if not hasattr(instance, "__canals_input__"):
            instance.__canals_input__ = {}
        run_signature = inspect.signature(getattr(cls, "run"))
        for param in list(run_signature.parameters)[1:]:  # First is 'self' and it doesn't matter.
            if run_signature.parameters[param].kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):  # ignore variable args
                socket_kwargs = {"name": param, "type": run_signature.parameters[param].annotation}
                if run_signature.parameters[param].default != inspect.Parameter.empty:
                    socket_kwargs["default_value"] = run_signature.parameters[param].default
                instance.__canals_input__[param] = InputSocket(**socket_kwargs)
        return instance


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

    def set_input_type(self, instance, name: str, type: Any, default: Any = _empty):
        """
        Add a single input socket to the component instance.

        :param instance: Component instance where the input type will be added.
        :param name: name of the input socket.
        :param type: type of the input socket.
        :param default: default value of the input socket, defaults to _empty
        """
        if not hasattr(instance, "__canals_input__"):
            instance.__canals_input__ = {}
        instance.__canals_input__[name] = InputSocket(name=name, type=type, default_value=default)

    def set_input_types(self, instance, **types):
        """
        Method that specifies the input types when 'kwargs' is passed to the run method.

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

        Note that if the `run()` method also specifies some parameters, those will take precedence.

        For example:

        ```python
        @component
        class MyComponent:

            def __init__(self, value: int):
                component.set_input_types(value_1=str, value_2=str)
                ...

            @component.output_types(output_1=int, output_2=str)
            def run(self, value_0: str, value_1: Optional[str] = None, **kwargs):
                return {"output_1": kwargs["value_1"], "output_2": ""}
        ```

        would add a mandatory `value_0` parameters, make the `value_1`
        parameter optional with a default None, and keep the `value_2`
        parameter mandatory as specified in `set_input_types`.

        """
        instance.__canals_input__ = {name: InputSocket(name=name, type=type_) for name, type_ in types.items()}

    def set_output_types(self, instance, **types):
        """
        Method that specifies the output types when the 'run' method is not decorated
        with 'component.output_types'.

        Use as:

        ```python
        @component
        class MyComponent:

            def __init__(self, value: int):
                component.set_output_types(output_1=int, output_2=str)
                ...

            # no decorators here
            def run(self, value: int):
                return {"output_1": 1, "output_2": "2"}
        ```
        """
        instance.__canals_output__ = {name: OutputSocket(name=name, type=type_) for name, type_ in types.items()}

    def output_types(self, **types):
        """
        Decorator factory that specifies the output types of a component.

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
            This happens at class creation time, and since we don't have the decorated
            class available here, we temporarily store the output types as an attribute of
            the decorated method. The ComponentMeta metaclass will use this data to create
            sockets at instance creation time.
            """
            setattr(
                run_method,
                "_output_types_cache",
                {name: OutputSocket(name=name, type=type_) for name, type_ in types.items()},
            )
            return run_method

        return output_types_decorator

    def _component(self, class_):
        """
        Decorator validating the structure of the component and registering it in the components registry.
        """
        logger.debug("Registering %s as a component", class_)

        # Check for required methods and fail as soon as possible
        if not hasattr(class_, "run"):
            raise ComponentError(f"{class_.__name__} must have a 'run()' method. See the docs for more information.")

        def copy_class_namespace(namespace):
            """
            This is the callback that `typing.new_class` will use
            to populate the newly created class. We just copy
            the whole namespace from the decorated class.
            """
            for key, val in dict(class_.__dict__).items():
                # __dict__ and __weakref__ are class-bound, we should let Python recreate them.
                if key in ("__dict__", "__weakref__"):
                    continue
                namespace[key] = val

        # Recreate the decorated component class so it uses our metaclass
        class_ = new_class(class_.__name__, class_.__bases__, {"metaclass": ComponentMeta}, copy_class_namespace)

        # Save the component in the class registry (for deserialization)
        class_path = f"{class_.__module__}.{class_.__name__}"
        if class_path in self.registry:
            # Corner case, but it may occur easily in notebooks when re-running cells.
            logger.debug(
                "Component %s is already registered. Previous imported from '%s', new imported from '%s'",
                class_path,
                self.registry[class_path],
                class_,
            )
        self.registry[class_path] = class_
        logger.debug("Registered Component %s", class_)

        return class_

    def __call__(self, class_):
        return self._component(class_)


component = _Component()
