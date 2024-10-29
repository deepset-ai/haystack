# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import datetime

import sys
from unittest.mock import Mock

import pytest

from haystack.core.pipeline import Pipeline
from haystack.core.component import component
from haystack.core.errors import DeserializationError, SerializationError
from haystack.testing import factory
from haystack.core.serialization import (
    default_to_dict,
    default_from_dict,
    generate_qualified_class_name,
    import_class_by_name,
    component_to_dict,
)


def test_default_component_to_dict():
    MyComponent = factory.component_class("MyComponent")
    comp = MyComponent()
    res = default_to_dict(comp)
    assert res == {"type": "haystack.testing.factory.MyComponent", "init_parameters": {}}


def test_default_component_to_dict_with_init_parameters():
    MyComponent = factory.component_class("MyComponent")
    comp = MyComponent()
    res = default_to_dict(comp, some_key="some_value")
    assert res == {"type": "haystack.testing.factory.MyComponent", "init_parameters": {"some_key": "some_value"}}


def test_default_component_from_dict():
    def custom_init(self, some_param):
        self.some_param = some_param

    extra_fields = {"__init__": custom_init}
    MyComponent = factory.component_class("MyComponent", extra_fields=extra_fields)
    comp = default_from_dict(
        MyComponent, {"type": "haystack.testing.factory.MyComponent", "init_parameters": {"some_param": 10}}
    )
    assert isinstance(comp, MyComponent)
    assert comp.some_param == 10


def test_default_component_from_dict_without_type():
    with pytest.raises(DeserializationError, match="Missing 'type' in serialization data"):
        default_from_dict(Mock, {})


def test_default_component_from_dict_unregistered_component(request):
    # We use the test function name as component name to make sure it's not registered.
    # Since the registry is global we risk to have a component with the same name registered in another test.
    component_name = request.node.name

    with pytest.raises(DeserializationError, match=f"Class '{component_name}' can't be deserialized as 'Mock'"):
        default_from_dict(Mock, {"type": component_name})


def test_from_dict_import_type():
    pipeline_dict = {
        "metadata": {},
        "components": {
            "greeter": {
                "type": "haystack.testing.sample_components.greet.Greet",
                "init_parameters": {
                    "message": "\nGreeting component says: Hi! The value is {value}\n",
                    "log_level": "INFO",
                },
            }
        },
        "connections": [],
    }

    # remove the target component from the registry if already there
    component.registry.pop("haystack.testing.sample_components.greet.Greet", None)
    # remove the module from sys.modules if already there
    sys.modules.pop("haystack.testing.sample_components.greet", None)

    p = Pipeline.from_dict(pipeline_dict)

    from haystack.testing.sample_components.greet import Greet

    assert type(p.get_component("greeter")) == Greet


def test_get_qualified_class_name():
    MyComponent = factory.component_class("MyComponent")
    comp = MyComponent()
    res = generate_qualified_class_name(type(comp))
    assert res == "haystack.testing.factory.MyComponent"


def test_import_class_by_name():
    data = "haystack.core.pipeline.Pipeline"
    class_object = import_class_by_name(data)
    class_instance = class_object()
    assert isinstance(class_instance, Pipeline)


def test_import_class_by_name_no_valid_class():
    data = "some.invalid.class"
    with pytest.raises(ImportError):
        import_class_by_name(data)


class CustomData:
    def __init__(self, a: int, b: str) -> None:
        self.a = a
        self.b = b


@component()
class UnserializableClass:
    def __init__(self, a: int, b: str, c: CustomData) -> None:
        self.a = a
        self.b = b
        self.c = c

    def run(self):
        pass


def test_component_to_dict_invalid_type():
    with pytest.raises(SerializationError, match="unsupported value of type 'CustomData'"):
        component_to_dict(UnserializableClass(1, "s", CustomData(99, "aa")), "invalid_component")
