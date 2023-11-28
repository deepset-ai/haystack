# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import inspect
from typing import Type, Dict, Any

from haystack.core.errors import DeserializationError, SerializationError


def component_to_dict(obj: Any) -> Dict[str, Any]:
    """
    The marshaller used by the Pipeline. If a `to_dict` method is present in the
    component instance, that will be used instead of the default method.
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


def component_from_dict(cls: Type[object], data: Dict[str, Any]) -> Any:
    """
    The unmarshaller used by the Pipeline. If a `from_dict` method is present in the
    component instance, that will be used instead of the default method.
    """
    if hasattr(cls, "from_dict"):
        return cls.from_dict(data)

    return default_from_dict(cls, data)


def default_to_dict(obj: Any, **init_parameters) -> Dict[str, Any]:
    """
    Utility function to serialize an object to a dictionary.
    This is mostly necessary for Components but it can be used by any object.

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
    """
    return {"type": f"{obj.__class__.__module__}.{obj.__class__.__name__}", "init_parameters": init_parameters}


def default_from_dict(cls: Type[object], data: Dict[str, Any]) -> Any:
    """
    Utility function to deserialize a dictionary to an object.
    This is mostly necessary for Components but it can be used by any object.

    The function will raise a `DeserializationError` if the `type` field in `data` is
    missing or it doesn't match the type of `cls`.

    If `data` contains an `init_parameters` field it will be used as parameters to create
    a new instance of `cls`.
    """
    init_params = data.get("init_parameters", {})
    if "type" not in data:
        raise DeserializationError("Missing 'type' in serialization data")
    if data["type"] != f"{cls.__module__}.{cls.__name__}":
        raise DeserializationError(f"Class '{data['type']}' can't be deserialized as '{cls.__name__}'")
    return cls(**init_params)
