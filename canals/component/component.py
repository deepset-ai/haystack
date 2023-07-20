# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging
import inspect
from functools import wraps

from canals.errors import ComponentError
from canals.pipeline.connections import _types_are_compatible


logger = logging.getLogger(__name__)


def _prepare_for_serialization(init_func):
    """
    Decorator that saves the init parameters of a component in `self.init_parameters`
    """

    @wraps(init_func)
    def wrapper(self, *args, **kwargs):
        # Call the actual __init__ function with the arguments
        init_func(self, *args, **kwargs)

        # Collect and store all the init parameters, preserving whatever the components might have already added there
        self.init_parameters = {**kwargs, **getattr(self, "init_parameters", {})}

    return wrapper


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

    def return_types(self, **return_types):
        """
        Decorator that checks the return types of the `run()` method.
        """

        def return_types_decorator(run_method):
            run_method.__return_types__ = return_types

            @wraps(run_method)
            def return_types_impl(*args, **kwargs):
                result = run_method(*args, **kwargs)

                # Check output types
                for key in result:
                    if key not in return_types:
                        raise ComponentError(f"Return value '{key}' not declared in @return_types decorator")
                    if _types_are_compatible(return_types[key], result[key]):
                        raise ComponentError(
                            f"Return type {type(result[key])} for value '{key}' doesn't match the one declared in "
                            f"@return_types decorator ({return_types[key]}))"
                        )
                return result

            return return_types_impl

        return return_types_decorator

    def _decorator(self, class_):
        """
        Decorator validating the structure of the component and registering it in the components registry.
        """
        logger.debug("Registering %s as a component", class_)

        # '__canals_component__' is used to distinguish components from regular classes.
        # Its value is set to the desired component name: normally it is the class name, but it can be customized.
        class_.__canals_component__ = class_.__name__

        # Check for run()
        if not hasattr(class_, "run"):
            raise ComponentError(f"{class_.__name__} must have a 'run()' method. See the docs for more information.")
        run_signature = inspect.signature(class_.run)

        # Check the run() signature for keyword variadic arguments
        if any(
            run_signature.parameters[param].kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
            for param in run_signature.parameters
        ):
            raise ComponentError(
                f"{class_.__name__} can't have variadic keyword arguments like *args or **kwargs in its 'run()' method."
            )

        # Check the run() signature for missing types
        missing_types = [
            parameter
            for parameter in list(run_signature.parameters)[1:]  # First is 'self' and it doesn't matter.
            if run_signature.parameters[parameter].annotation == inspect.Parameter.empty
        ]
        if missing_types:
            raise ComponentError(
                f"{class_.__name__}.run() must declare types for all its parameters, "
                f"but these parameters are not typed: {', '.join(missing_types)}."
            )

        #
        # TODO create the input sockets
        #

        # # Check the run() signature for the return_types wrapper
        if not hasattr(class_.run, "__return_types__"):
            raise ComponentError(
                f"{class_.__name__}.run() must have a @return_types decorator. See the docs for more information."
            )
        #
        # TODO Create the output sockets
        #

        # Automatically registers all the init parameters in an instance attribute called `_init_parameters`.
        # See `save_init_parameters()`.
        class_.__init__ = _prepare_for_serialization(class_.__init__)

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
            return self._decorator(class_)

        return self._decorator


component = _Component()
