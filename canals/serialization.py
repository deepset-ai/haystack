# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Type, Dict, Any

from canals.errors import DeserializationError


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
    return {
        "type": obj.__class__.__name__,
        "init_parameters": init_parameters,
    }


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
    if data["type"] != cls.__name__:
        raise DeserializationError(f"Class '{data['type']}' can't be deserialized as '{cls.__name__}'")
    return cls(**init_params)
