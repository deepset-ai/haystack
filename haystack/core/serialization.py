# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

from haystack.core.component.component import _hook_component_init
from haystack.core.errors import DeserializationError, SerializationError


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


def component_to_dict(obj: Any) -> Dict[str, Any]:
    """
    Converts a component instance into a dictionary.

    If a `to_dict` method is present in the component instance, that will be used instead of the default method.

    :param obj:
        The component to be serialized.
    :returns:
        A dictionary representation of the component.

    :raises SerializationError:
        If the component doesn't have a `to_dict` method and the values of the init parameters can't be determined.
    """
    if hasattr(obj, "to_dict"):
        return obj.to_dict()

    init_parameters = {}
    for name, param in inspect.signature(obj.__init__).parameters.items():
        # Ignore `args` and `kwargs`, used by the default constructor
        if name in ("args", "kwargs"):
            continue
        try:
            # This only works if the Component constructor assigns the init
            # parameter to an instance variable or property with the same name
            param_value = getattr(obj, name)
        except AttributeError as e:
            # If the parameter doesn't have a default value, raise an error
            if param.default is param.empty:
                raise SerializationError(
                    f"Cannot determine the value of the init parameter '{name}' for the class {obj.__class__.__name__}."
                    f"You can fix this error by assigning 'self.{name} = {name}' or adding a "
                    f"custom serialization method 'to_dict' to the class."
                ) from e
            # In case the init parameter was not assigned, we use the default value
            param_value = param.default
        init_parameters[name] = param_value

    return default_to_dict(obj, **init_parameters)


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
