# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import inspect
from functools import wraps

from canals.errors import ComponentError
from canals.type_checking import _types_are_compatible
from canals.sockets import InputSocket, OutputSocket


logger = logging.getLogger(__name__)


def _prepare_init_params_and_sockets(init_func):
    """
    Decorator that saves the init parameters of a component in `self.init_parameters`
    """

    @wraps(init_func)
    def wrapper(self, *args, **kwargs):
        # Call the actual __init__ function with the arguments
        init_func(self, *args, **kwargs)

        # Create the component's input and output sockets
        self.__input_sockets__ = {name: InputSocket(**data) for name, data in self.run.__run_method_types__.items()}
        self.__output_sockets__ = {name: OutputSocket(**data) for name, data in self.run.__return_types__.items()}

        # Collect and store all the init parameters, preserving whatever the components might have already added there
        self.init_parameters = {**kwargs, **getattr(self, "init_parameters", {})}

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

    def set_input_types(self, instance, **input_types):
        """
        TODO
        """
        unwrapped = instance.run

        def wrapper(**kwargs):
            for key, value in kwargs.items():
                if key not in input_types:
                    raise ComponentError(f"Input value '{key}' not declared in @run_method_types decorator")
                if _types_are_compatible(value, input_types[key]):
                    raise ComponentError(
                        f"Input type {type(value)} for value '{key}' doesn't match the one declared in "
                        f"@run_method_types decorator ({input_types[key]}))"
                    )

            return unwrapped(**kwargs)

        if hasattr(instance.run, "__return_types__"):
            wrapper.__return_types__ = instance.run.__return_types__
        wrapper.__run_method_types__ = {name: {"name": name, "type": type_} for name, type_ in input_types.items()}
        instance.run = wrapper

    def set_output_types(self, instance, **output_types):
        """
        TODO
        """
        unwrapped = instance.run

        def wrapper(*args, **kwargs):
            result = unwrapped(*args, **kwargs)
            # Check output types
            for key in result:
                if key not in output_types:
                    raise ComponentError(f"Return value '{key}' not declared in @output_types decorator")
                if _types_are_compatible(output_types[key], result[key]):
                    raise ComponentError(
                        f"Return type {type(result[key])} for value '{key}' doesn't match the one declared in "
                        f"@return_types decorator ({output_types[key]}))"
                    )
            return result

        if hasattr(instance.run, "__run_method_types__"):
            wrapper.__run_method_types__ = instance.run.__run_method_types__
        wrapper.__return_types__ = {name: {"name": name, "type": type_} for name, type_ in output_types.items()}
        instance.run = wrapper

    def return_types(self, **return_types):
        """
        Decorator that checks the return types of the `run()` method.
        """
        print(self)

        def return_types_decorator(run_method):
            run_method.__return_types__ = {name: {"name": name, "type": type_} for name, type_ in return_types.items()}

            @wraps(run_method)
            def return_types_impl(self, *args, **kwargs):
                result = run_method(self, *args, **kwargs)

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

        if not hasattr(class_.run, "__run_method_types__"):
            # Check the run() signature for keyword variadic arguments
            # if any(
            #     run_signature.parameters[param].kind
            #     in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
            #     for param in run_signature.parameters
            # ):
            #     raise ComponentError(
            #         f"{class_.__name__} can't have arguments like *args or **kwargs in its 'run()' method."
            #     )

            # Check the run() signature for missing types
            # missing_types = [
            #     parameter
            #     for parameter in list(run_signature.parameters)[1:]  # First is 'self' and it doesn't matter.
            #     if run_signature.parameters[parameter].annotation == inspect.Parameter.empty
            # ]
            # if missing_types:
            #     raise ComponentError(
            #         f"{class_.__name__}.run() must declare types for all its parameters, "
            #         f"but these parameters are not typed: {', '.join(missing_types)}."
            #     )

            # Create the input sockets
            class_.run.__run_method_types__ = {
                param: {
                    "name": param,
                    "type": run_signature.parameters[param].annotation,
                    "default": run_signature.parameters[param].default
                    if run_signature.parameters[param].default is not inspect.Parameter.empty
                    else None,
                }
                for param in list(run_signature.parameters)[1:]  # First is 'self' and it doesn't matter.
            }

        # # Check the run() signature for the return_types wrapper
        # if not hasattr(class_.run, "__return_types__"):
        #     raise ComponentError(
        #         f"{class_.__name__}.run() must have a @return_types decorator. See the docs for more information."
        #     )

        # Automatically registers all the init parameters in an instance attribute called `_init_parameters`.
        # See `save_init_parameters()`.
        class_.__init__ = _prepare_init_params_and_sockets(class_.__init__)

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
