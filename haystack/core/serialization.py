# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Type, Union

from haystack import logging
from haystack.core.component.component import _hook_component_init
from haystack.core.errors import DeserializationError, SerializationError
from haystack.utils import Secret, deserialize_callable, deserialize_secrets_inplace, serialize_callable
from haystack.utils.type_serialization import thread_safe_import
from haystack.tools.serde_utils import serialize_tools_or_toolset, deserialize_tools_or_toolset_inplace

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
    # Get the to_dict method from the instance
    to_dict_method = getattr(obj, "to_dict", None)

    # If there's no to_dict method or if we're being called from the default to_dict method
    if (
        to_dict_method is None
        or to_dict_method.__name__ == "<lambda>"
        or getattr(to_dict_method, "__module__", None) == "haystack.core.component.component"
    ):
        init_parameters = {}
        for param_name, param in inspect.signature(obj.__init__).parameters.items():
            # Ignore `args` and `kwargs`, used by the default constructor
            if param_name in ("args", "kwargs", "self"):
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
    else:
        data = to_dict_method()

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
        # Get the from_dict method from the class
        from_dict_method = getattr(cls, "from_dict", None)

        # If there's no from_dict method or if we're being called from the default from_dict method
        if (
            from_dict_method is None
            or getattr(from_dict_method, "__module__", None) == "haystack.core.component.component"
        ):
            return default_from_dict(cls, data)
        return from_dict_method(data)

    if callbacks is None or callbacks.component_pre_init is None:
        return do_from_dict()

    with _hook_component_init(component_pre_init_callback):
        return do_from_dict()


def _is_list_of_tools(annotation: Any) -> bool:
    """Check if the annotation is a List[Tool]."""
    from haystack.tools import Tool  # Import here to avoid circular dependency

    if not hasattr(annotation, "__origin__"):
        return False
    if annotation.__origin__ is not list:
        return False
    if len(annotation.__args__) != 1:
        return False
    return annotation.__args__[0] is Tool


def _is_union_with_tools(annotation: Any) -> bool:
    """Check if the annotation is a Union containing List[Tool] or Toolset."""
    from haystack.tools import Toolset  # Import here to avoid circular dependency

    if not hasattr(annotation, "__origin__"):
        return False
    if annotation.__origin__ is not Union:
        return False

    # Check for List[Tool] in Union args
    for arg in annotation.__args__:
        if hasattr(arg, "__origin__") and _is_list_of_tools(arg):
            return True

    # Check for Toolset in Union args
    return Toolset in annotation.__args__


def default_to_dict(obj: Any, **init_parameters) -> Dict[str, Any]:
    """
    Utility function to serialize an object to a dictionary.

    This is mostly necessary for components but can be used by any object.
    `init_parameters` are parameters passed to the object class `__init__`.
    They must be defined explicitly as they'll be used when creating a new
    instance of `obj` with `from_dict`. Omitting them might cause deserialisation
    errors or unexpected behaviours later, when calling `from_dict`.

    Special handling for:
    - Secret objects: calls to_dict() on them
    - Union[List[Tool], Toolset]: calls serialize_tools_or_toolset
    - StreamingCallbackT: calls serialize_callable

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
    # Handle special types in init_parameters
    processed_params = {}
    for key, value in init_parameters.items():
        if isinstance(value, Secret):
            processed_params[key] = value.to_dict()
        elif value is not None:
            from haystack.tools import Tool, Toolset  # Import here to avoid circular dependency

            if (isinstance(value, list) and all(isinstance(t, Tool) for t in value)) or isinstance(value, Toolset):
                processed_params[key] = serialize_tools_or_toolset(value)
            elif callable(value) and hasattr(value, "__annotations__") and "return" in value.__annotations__:
                # Check if it's a streaming callback by looking at its return type annotation
                processed_params[key] = serialize_callable(value)
            else:
                processed_params[key] = value
        else:
            processed_params[key] = value

    return {"type": generate_qualified_class_name(type(obj)), "init_parameters": processed_params}


def default_from_dict(cls: Type[object], data: Dict[str, Any]) -> Any:
    """
    Utility function to deserialize a dictionary to an object.

    This is mostly necessary for components but can be used by any object.

    The function will raise a `DeserializationError` if the `type` field in `data` is
    missing or it doesn't match the type of `cls`.

    If `data` contains an `init_parameters` field it will be used as parameters to create
    a new instance of `cls`.

    Special handling for:
    - Secret objects: calls deserialize_secrets_inplace
    - Union[List[Tool], Toolset]: calls deserialize_tools_or_toolset_inplace
    - StreamingCallbackT: calls deserialize_callable

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

    # Find which init parameters are of type Secret
    secret_params = []
    tool_params = []
    for param_name, param in inspect.signature(cls.__init__).parameters.items():
        if param.annotation == Secret or (
            hasattr(param.annotation, "__origin__")
            and param.annotation.__origin__ is Optional
            and param.annotation.__args__[0] == Secret
        ):  # pylint: disable=too-many-boolean-expressions
            secret_params.append(param_name)
        elif hasattr(param.annotation, "__origin__") and (
            _is_list_of_tools(param.annotation) or _is_union_with_tools(param.annotation)
        ):
            tool_params.append(param_name)

    # Handle special types in init_parameters
    deserialize_secrets_inplace(init_params, keys=secret_params)

    for tool_param in tool_params:
        deserialize_tools_or_toolset_inplace(init_params, key=tool_param)

    # Handle streaming callbacks
    for key, value in init_params.items():
        if isinstance(value, str) and key == "streaming_callback":
            init_params[key] = deserialize_callable(value)

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
