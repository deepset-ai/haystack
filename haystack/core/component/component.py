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

import inspect
import sys
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from types import new_class
from typing import Any, Dict, Optional, Protocol, Type, runtime_checkable

from haystack import logging
from haystack.core.errors import ComponentError

from .sockets import Sockets
from .types import InputSocket, OutputSocket, _empty

logger = logging.getLogger(__name__)


@dataclass
class PreInitHookPayload:
    """
    Payload for the hook called before a component instance is initialized.

    :param callback:
        Receives the following inputs: component class and init parameter keyword args.
    :param in_progress:
        Flag to indicate if the hook is currently being executed.
        Used to prevent it from being called recursively (if the component's constructor
        instantiates another component).
    """

    callback: Callable
    in_progress: bool = False


_COMPONENT_PRE_INIT_HOOK: ContextVar[Optional[PreInitHookPayload]] = ContextVar("component_pre_init_hook", default=None)


@contextmanager
def _hook_component_init(callback: Callable):
    """
    Context manager to set a callback that will be invoked before a component's constructor is called.

    The callback receives the component class and the init parameters (as keyword arguments) and can modify the init
    parameters in place.

    :param callback:
        Callback function to invoke.
    """
    token = _COMPONENT_PRE_INIT_HOOK.set(PreInitHookPayload(callback))
    try:
        yield
    finally:
        _COMPONENT_PRE_INIT_HOOK.reset(token)


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

    # This is the most reliable way to define the protocol for the `run` method.
    # Defining a method doesn't work as different Components will have different
    # arguments. Even defining here a method with `**kwargs` doesn't work as the
    # expected signature must be identical.
    # This makes most Language Servers and type checkers happy and shows less errors.
    # NOTE: This check can be removed when we drop Python 3.8 support.
    if sys.version_info >= (3, 9):
        run: Callable[..., Dict[str, Any]]
    else:
        run: Callable


class ComponentMeta(type):
    @staticmethod
    def _positional_to_kwargs(cls_type, args) -> Dict[str, Any]:
        """
        Convert positional arguments to keyword arguments based on the signature of the `__init__` method.
        """
        init_signature = inspect.signature(cls_type.__init__)
        init_params = {name: info for name, info in init_signature.parameters.items() if name != "self"}

        out = {}
        for arg, (name, info) in zip(args, init_params.items()):
            if info.kind == inspect.Parameter.VAR_POSITIONAL:
                raise ComponentError(
                    "Pre-init hooks do not support components with variadic positional args in their init method"
                )

            assert info.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY)
            out[name] = arg
        return out

    @staticmethod
    def _parse_and_set_output_sockets(instance: Any):
        has_async_run = hasattr(instance, "run_async")

        # If `component.set_output_types()` was called in the component constructor,
        # `__haystack_output__` is already populated, no need to do anything.
        if not hasattr(instance, "__haystack_output__"):
            # If that's not the case, we need to populate `__haystack_output__`
            #
            # If either of the run methods were decorated, they'll have a field assigned that
            # stores the output specification. If both run methods were decorated, we ensure that
            # outputs are the same. We deepcopy the content of the cache to transfer ownership from
            # the class method to the actual instance, so that different instances of the same class
            # won't share this data.

            run_output_types = getattr(instance.run, "_output_types_cache", {})
            async_run_output_types = getattr(instance.run_async, "_output_types_cache", {}) if has_async_run else {}

            if has_async_run and run_output_types != async_run_output_types:
                raise ComponentError("Output type specifications of 'run' and 'run_async' methods must be the same")
            output_types_cache = run_output_types

            instance.__haystack_output__ = Sockets(instance, deepcopy(output_types_cache), OutputSocket)

    @staticmethod
    def _parse_and_set_input_sockets(component_cls: Type, instance: Any):
        def inner(method, sockets):
            from inspect import Parameter

            run_signature = inspect.signature(method)

            for param_name, param_info in run_signature.parameters.items():
                if param_name == "self" or param_info.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
                    continue

                socket_kwargs = {"name": param_name, "type": param_info.annotation}
                if param_info.default != Parameter.empty:
                    socket_kwargs["default_value"] = param_info.default

                new_socket = InputSocket(**socket_kwargs)

                # Also ensure that new sockets don't override existing ones.
                existing_socket = sockets.get(param_name)
                if existing_socket is not None and existing_socket != new_socket:
                    raise ComponentError(
                        "set_input_types()/set_input_type() cannot override the parameters of the 'run' method"
                    )

                sockets[param_name] = new_socket

            return run_signature

        # Create the sockets if set_input_types() wasn't called in the constructor.
        if not hasattr(instance, "__haystack_input__"):
            instance.__haystack_input__ = Sockets(instance, {}, InputSocket)

        inner(getattr(component_cls, "run"), instance.__haystack_input__)

        # Ensure that the sockets are the same for the async method, if it exists.
        async_run = getattr(component_cls, "run_async", None)
        if async_run is not None:
            run_sockets = Sockets(instance, {}, InputSocket)
            async_run_sockets = Sockets(instance, {}, InputSocket)

            # Can't use the sockets from above as they might contain
            # values set with set_input_types().
            run_sig = inner(getattr(component_cls, "run"), run_sockets)
            async_run_sig = inner(async_run, async_run_sockets)

            if async_run_sockets != run_sockets or run_sig != async_run_sig:
                raise ComponentError("Parameters of 'run' and 'run_async' methods must be the same")

    def __call__(cls, *args, **kwargs):
        """
        This method is called when clients instantiate a Component and runs before __new__ and __init__.
        """
        # This will call __new__ then __init__, giving us back the Component instance
        pre_init_hook = _COMPONENT_PRE_INIT_HOOK.get()
        if pre_init_hook is None or pre_init_hook.in_progress:
            instance = super().__call__(*args, **kwargs)
        else:
            try:
                pre_init_hook.in_progress = True
                named_positional_args = ComponentMeta._positional_to_kwargs(cls, args)
                assert (
                    set(named_positional_args.keys()).intersection(kwargs.keys()) == set()
                ), "positional and keyword arguments overlap"
                kwargs.update(named_positional_args)
                pre_init_hook.callback(cls, kwargs)
                instance = super().__call__(**kwargs)
            finally:
                pre_init_hook.in_progress = False

        # Before returning, we have the chance to modify the newly created
        # Component instance, so we take the chance and set up the I/O sockets
        has_async_run = hasattr(instance, "run_async")
        if has_async_run and not inspect.iscoroutinefunction(instance.run_async):
            raise ComponentError(f"Method 'run_async' of component '{cls.__name__}' must be a coroutine")
        instance.__haystack_supports_async__ = has_async_run

        ComponentMeta._parse_and_set_input_sockets(cls, instance)
        ComponentMeta._parse_and_set_output_sockets(instance)

        # Since a Component can't be used in multiple Pipelines at the same time
        # we need to know if it's already owned by a Pipeline when adding it to one.
        # We use this flag to check that.
        instance.__haystack_added_to_pipeline__ = None

        return instance


def _component_repr(component: Component) -> str:
    """
    All Components override their __repr__ method with this one.

    It prints the component name and the input/output sockets.
    """
    result = object.__repr__(component)
    if pipeline := getattr(component, "__haystack_added_to_pipeline__", None):
        # This Component has been added in a Pipeline, let's get the name from there.
        result += f"\n{pipeline.get_component_name(component)}"

    # We're explicitly ignoring the type here because we're sure that the component
    # has the __haystack_input__ and __haystack_output__ attributes at this point
    return (
        f'{result}\n{getattr(component, "__haystack_input__", "<invalid_input_sockets>")}'
        f'\n{getattr(component, "__haystack_output__", "<invalid_output_sockets>")}'
    )


def _component_run_has_kwargs(component_cls: Type) -> bool:
    run_method = getattr(component_cls, "run", None)
    if run_method is None:
        return False
    else:
        return any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in inspect.signature(run_method).parameters.values()
        )


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

    def set_input_type(
        self,
        instance,
        name: str,
        type: Any,  # noqa: A002
        default: Any = _empty,
    ):
        """
        Add a single input socket to the component instance.

        Replaces any existing input socket with the same name.

        :param instance: Component instance where the input type will be added.
        :param name: name of the input socket.
        :param type: type of the input socket.
        :param default: default value of the input socket, defaults to _empty
        """
        if not _component_run_has_kwargs(instance.__class__):
            raise ComponentError(
                "Cannot set input types on a component that doesn't have a kwargs parameter in the 'run' method"
            )

        if not hasattr(instance, "__haystack_input__"):
            instance.__haystack_input__ = Sockets(instance, {}, InputSocket)
        instance.__haystack_input__[name] = InputSocket(name=name, type=type, default_value=default)

    def set_input_types(self, instance, **types):
        """
        Method that specifies the input types when 'kwargs' is passed to the run method.

        Use as:

        ```python
        @component
        class MyComponent:

            def __init__(self, value: int):
                component.set_input_types(self, value_1=str, value_2=str)
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
                component.set_input_types(self, value_1=str, value_2=str)
                ...

            @component.output_types(output_1=int, output_2=str)
            def run(self, value_0: str, value_1: Optional[str] = None, **kwargs):
                return {"output_1": kwargs["value_1"], "output_2": ""}
        ```

        would add a mandatory `value_0` parameters, make the `value_1`
        parameter optional with a default None, and keep the `value_2`
        parameter mandatory as specified in `set_input_types`.

        """
        if not _component_run_has_kwargs(instance.__class__):
            raise ComponentError(
                "Cannot set input types on a component that doesn't have a kwargs parameter in the 'run' method"
            )

        instance.__haystack_input__ = Sockets(
            instance, {name: InputSocket(name=name, type=type_) for name, type_ in types.items()}, InputSocket
        )

    def set_output_types(self, instance, **types):
        """
        Method that specifies the output types when the 'run' method is not decorated with 'component.output_types'.

        Use as:

        ```python
        @component
        class MyComponent:

            def __init__(self, value: int):
                component.set_output_types(self, output_1=int, output_2=str)
                ...

            # no decorators here
            def run(self, value: int):
                return {"output_1": 1, "output_2": "2"}
        ```
        """
        has_decorator = hasattr(instance.run, "_output_types_cache")
        if has_decorator:
            raise ComponentError(
                "Cannot call `set_output_types` on a component that already has "
                "the 'output_types' decorator on its `run` method"
            )

        instance.__haystack_output__ = Sockets(
            instance, {name: OutputSocket(name=name, type=type_) for name, type_ in types.items()}, OutputSocket
        )

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
            Decorator that sets the output types of the decorated method.

            This happens at class creation time, and since we don't have the decorated
            class available here, we temporarily store the output types as an attribute of
            the decorated method. The ComponentMeta metaclass will use this data to create
            sockets at instance creation time.
            """
            method_name = run_method.__name__
            if method_name not in ("run", "run_async"):
                raise ComponentError("'output_types' decorator can only be used on 'run' and 'run_async' methods")

            setattr(
                run_method,
                "_output_types_cache",
                {name: OutputSocket(name=name, type=type_) for name, type_ in types.items()},
            )
            return run_method

        return output_types_decorator

    def _component(self, cls: Any):
        """
        Decorator validating the structure of the component and registering it in the components registry.
        """
        logger.debug("Registering {component} as a component", component=cls)

        # Check for required methods and fail as soon as possible
        if not hasattr(cls, "run"):
            raise ComponentError(f"{cls.__name__} must have a 'run()' method. See the docs for more information.")

        def copy_class_namespace(namespace):
            """
            This is the callback that `typing.new_class` will use to populate the newly created class.

            Simply copy the whole namespace from the decorated class.
            """
            for key, val in dict(cls.__dict__).items():
                # __dict__ and __weakref__ are class-bound, we should let Python recreate them.
                if key in ("__dict__", "__weakref__"):
                    continue
                namespace[key] = val

        # Recreate the decorated component class so it uses our metaclass.
        # We must explicitly redefine the type of the class to make sure language servers
        # and type checkers understand that the class is of the correct type.
        # mypy doesn't like that we do this though so we explicitly ignore the type check.
        cls: cls.__name__ = new_class(cls.__name__, cls.__bases__, {"metaclass": ComponentMeta}, copy_class_namespace)  # type: ignore[no-redef]

        # Save the component in the class registry (for deserialization)
        class_path = f"{cls.__module__}.{cls.__name__}"
        if class_path in self.registry:
            # Corner case, but it may occur easily in notebooks when re-running cells.
            logger.debug(
                "Component {component} is already registered. Previous imported from '{module_name}', \
                new imported from '{new_module_name}'",
                component=class_path,
                module_name=self.registry[class_path],
                new_module_name=cls,
            )
        self.registry[class_path] = cls
        logger.debug("Registered Component {component}", component=cls)

        # Override the __repr__ method with a default one
        cls.__repr__ = _component_repr

        return cls

    def __call__(self, cls: Optional[type] = None):
        # We must wrap the call to the decorator in a function for it to work
        # correctly with or without parens
        def wrap(cls):
            return self._component(cls)

        if cls:
            # Decorator is called without parens
            return wrap(cls)

        # Decorator is called with parens
        return wrap


component = _Component()
