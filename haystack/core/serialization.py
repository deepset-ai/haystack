# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Type

from haystack import logging
from haystack.core.component.component import _hook_component_init
from haystack.core.errors import DeserializationError, SerializationError
from haystack.utils.type_serialization import thread_safe_import

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeserializationCallbacks:
    """
    Callback functions that are invoked in specific stages of the pipeline deserialization process.

    :param component_pre_init:
        Invoked just before a component instance is
        initialized. Receives the following inputs:
        `component_name` (`str`), `component_class` (`Type`), `init_params` (`Dict[str, Any]`).

        The callback is allowed to modify the `init_params`
        dictionary, which contains all the parameters that
        are passed to the component's constructor.
    """

    component_pre_init: Optional[Callable] = None


def component_to_dict(obj: Any, name: str) -> Dict[str, Any]:
    """
    Converts a component instance into a dictionary.

    If a `to_dict` method is present in the component instance, that will be used instead of the default method.

    :param obj:
        The component to be serialized.
    :param name:
        The name of the component.
    :returns:
        A dictionary representation of the component.

    :raises SerializationError:
        If the component doesn't have a `to_dict` method.
        If the values of the init parameters can't be determined.
        If a non-basic Python type is used in the serialized data.
    """
    if hasattr(obj, "to_dict"):
        data = obj.to_dict()
    else:
        init_parameters = {}
        for param_name, param in inspect.signature(obj.__init__).parameters.items():
            # Ignore `args` and `kwargs`, used by the default constructor
            if param_name in ("args", "kwargs"):
                continue
            try:
                # This only works if the Component constructor assigns the init
                # parameter to an instance variable or property with the same name
                param_value = getattr(obj, param_name)
            except AttributeError as e:
                # If the parameter doesn't have a default value, raise an error
                if param.default is param.empty:
                    raise SerializationError(
                        f"Cannot determine the value of the init parameter '{param_name}' "
                        f"for the class {obj.__class__.__name__}."
                        f"You can fix this error by assigning 'self.{param_name} = {param_name}' or adding a "
                        f"custom serialization method 'to_dict' to the class."
                    ) from e
                # In case the init parameter was not assigned, we use the default value
                param_value = param.default
            init_parameters[param_name] = param_value

        data = default_to_dict(obj, **init_parameters)

    _validate_component_to_dict_output(obj, name, data)
    return data


def _validate_component_to_dict_output(component: Any, name: str, data: Dict[str, Any]) -> None:
    # Ensure that only basic Python types are used in the serde data.
    def is_allowed_type(obj: Any) -> bool:
        return isinstance(obj, (str, int, float, bool, list, dict, set, tuple, type(None)))

    def check_iterable(l: Iterable[Any]):
        for v in l:
            if not is_allowed_type(v):
                raise SerializationError(
                    f"Component '{name}' of type '{type(component).__name__}' has an unsupported value "
                    f"of type '{type(v).__name__}' in the serialized data."
                )
            if isinstance(v, (list, set, tuple)):
                check_iterable(v)
            elif isinstance(v, dict):
                check_dict(v)

    def check_dict(d: Dict[str, Any]):
        if any(not isinstance(k, str) for k in data.keys()):
            raise SerializationError(
                f"Component '{name}' of type '{type(component).__name__}' has a non-string key in the serialized data."
            )

        for k, v in d.items():
            if not is_allowed_type(v):
                raise SerializationError(
                    f"Component '{name}' of type '{type(component).__name__}' has an unsupported value "
                    f"of type '{type(v).__name__}' in the serialized data under key '{k}'."
                )
            if isinstance(v, (list, set, tuple)):
                check_iterable(v)
            elif isinstance(v, dict):
                check_dict(v)

    check_dict(data)


def generate_qualified_class_name(cls: Type[object]) -> str:
    """
    Generates a qualified class name for a class.

    :param cls:
        The class whose qualified name is to be generated.
    :returns:
        The qualified name of the class.
    """
    return f"{cls.__module__}.{cls.__name__}"


def component_from_dict(
    cls: Type[object], data: Dict[str, Any], name: str, callbacks: Optional[DeserializationCallbacks] = None
) -> Any:
    """
    Creates a component instance from a dictionary.

    If a `from_dict` method is present in the component class, that will be used instead of the default method.

    :param cls:
        The class to be used for deserialization.
    :param data:
        The serialized data.
    :param name:
        The name of the component.
    :param callbacks:
        Callbacks to invoke during deserialization.
    :returns:
        The deserialized component.
    """

    def component_pre_init_callback(component_cls, init_params):
        assert callbacks is not None
        assert callbacks.component_pre_init is not None
        callbacks.component_pre_init(name, component_cls, init_params)

    def do_from_dict():
        if hasattr(cls, "from_dict"):
            return cls.from_dict(data)

        return default_from_dict(cls, data)

    if callbacks is None or callbacks.component_pre_init is None:
        return do_from_dict()

    with _hook_component_init(component_pre_init_callback):
        return do_from_dict()


def default_to_dict(obj: Any, **init_parameters) -> Dict[str, Any]:
    """
    Utility function to serialize an object to a dictionary.

    This is mostly necessary for components but can be used by any object.
    `init_parameters` are parameters passed to the object class `__init__`.
    They must be defined explicitly as they'll be used when creating a new
    instance of `obj` with `from_dict`. Omitting them might cause deserialisation
    errors or unexpected behaviours later, when calling `from_dict`.

    An example usage:

    ```python
    class MyClass:
        def __init__(self, my_param: int = 10):
            self.my_param = my_param

        def to_dict(self):
            return default_to_dict(self, my_param=self.my_param)


    obj = MyClass(my_param=1000)
    data = obj.to_dict()
    assert data == {
        "type": "MyClass",
        "init_parameters": {
            "my_param": 1000,
        },
    }
    ```

    :param obj:
        The object to be serialized.
    :param init_parameters:
        The parameters used to create a new instance of the class.
    :returns:
        A dictionary representation of the instance.
    """
    return {"type": generate_qualified_class_name(type(obj)), "init_parameters": init_parameters}


def default_from_dict(cls: Type[object], data: Dict[str, Any]) -> Any:
    """
    Utility function to deserialize a dictionary to an object.

    This is mostly necessary for components but can be used by any object.

    The function will raise a `DeserializationError` if the `type` field in `data` is
    missing or it doesn't match the type of `cls`.

    If `data` contains an `init_parameters` field it will be used as parameters to create
    a new instance of `cls`.

    :param cls:
        The class to be used for deserialization.
    :param data:
        The serialized data.
    :returns:
        The deserialized object.

    :raises DeserializationError:
        If the `type` field in `data` is missing or it doesn't match the type of `cls`.
    """
    init_params = data.get("init_parameters", {})
    if "type" not in data:
        raise DeserializationError("Missing 'type' in serialization data")
    if data["type"] != generate_qualified_class_name(cls):
        raise DeserializationError(f"Class '{data['type']}' can't be deserialized as '{cls.__name__}'")
    return cls(**init_params)


def import_class_by_name(fully_qualified_name: str) -> Type[object]:
    """
    Utility function to import (load) a class object based on its fully qualified class name.

    This function dynamically imports a class based on its string name.
    It splits the name into module path and class name, imports the module,
    and returns the class object.

    :param fully_qualified_name: the fully qualified class name as a string
    :returns: the class object.
    :raises ImportError: If the class cannot be imported or found.
    """
    try:
        module_path, class_name = fully_qualified_name.rsplit(".", 1)
        logger.debug(
            "Attempting to import class '{cls_name}' from module '{md_path}'", cls_name=class_name, md_path=module_path
        )
        module = thread_safe_import(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as error:
        logger.error("Failed to import class '{full_name}'", full_name=fully_qualified_name)
        raise ImportError(f"Could not import class '{fully_qualified_name}'") from error
