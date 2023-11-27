# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, Optional, Tuple, Type

from canals import component, Component
from canals.serialization import default_to_dict, default_from_dict


def component_class(
    name: str,
    input_types: Optional[Dict[str, Any]] = None,
    output_types: Optional[Dict[str, Any]] = None,
    output: Optional[Dict[str, Any]] = None,
    bases: Optional[Tuple[type, ...]] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Type[Component]:
    """
    Utility class to create a Component class with the given name and input and output types.

    If `output` is set but `output_types` is not, `output_types` will be set to the types of the values in `output`.
    Though if `output_types` is set but `output` is not the component's `run` method will return a dictionary
    of the same keys as `output_types` all with a value of None.

    ### Usage

    Create a component class with default input and output types:
    ```python
    MyFakeComponent = component_class_factory("MyFakeComponent")
    component = MyFakeComponent()
    output = component.run(value=1)
    assert output == {"value": None}
    ```

    Create a component class with an "value" input of type `int` and with a "value" output of `10`:
    ```python
    MyFakeComponent = component_class_factory(
        "MyFakeComponent",
        input_types={"value": int},
        output={"value": 10}
    )
    component = MyFakeComponent()
    output = component.run(value=1)
    assert output == {"value": 10}
    ```

    Create a component class with a custom base class:
    ```python
    MyFakeComponent = component_class_factory(
        "MyFakeComponent",
        bases=(MyBaseClass,)
    )
    component = MyFakeComponent()
    assert isinstance(component, MyBaseClass)
    ```

    Create a component class with an extra field `my_field`:
    ```python
    MyFakeComponent = component_class_factory(
        "MyFakeComponent",
        extra_fields={"my_field": 10}
    )
    component = MyFakeComponent()
    assert component.my_field == 10
    ```

    Args:
    name: Name of the component class
    input_types: Dictionary of string and type that defines the inputs of the component,
        if set to None created component will expect a single input "value" of Any type.
        Defaults to None.
    output_types: Dictionary of string and type that defines the outputs of the component,
        if set to None created component will return a single output "value" of NoneType and None value.
        Defaults to None.
    output: Actual output dictionary returned by the created component run,
        is set to None it will return a dictionary of string and None values.
        Keys will be the same as the keys of output_types. Defaults to None.
    bases: Base classes for this component, if set to None only base is object. Defaults to None.
    extra_fields: Extra fields for the Component, defaults to None.

    :return: A class definition that can be used as a component.
    """
    if input_types is None:
        input_types = {"value": Any}
    if output_types is None and output is not None:
        output_types = {key: type(value) for key, value in output.items()}
    elif output_types is None:
        output_types = {"value": type(None)}

    def init(self):
        component.set_input_types(self, **input_types)
        component.set_output_types(self, **output_types)

    # Both arguments are necessary to correctly define
    # run but pylint doesn't like that we don't use them.
    # It's fine ignoring the warning here.
    def run(self, **kwargs):  # pylint: disable=unused-argument
        if output is not None:
            return output
        return {name: None for name in output_types.keys()}

    def to_dict(self):
        return default_to_dict(self)

    def from_dict(cls, data: Dict[str, Any]):
        return default_from_dict(cls, data)

    fields = {
        "__init__": init,
        "run": run,
        "to_dict": to_dict,
        "from_dict": classmethod(from_dict),
    }
    if extra_fields is not None:
        fields = {**fields, **extra_fields}

    if bases is None:
        bases = (object,)

    cls = type(name, bases, fields)
    return component(cls)
