# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
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
from haystack.utils import Secret
from haystack.tools import Tool, Toolset
from haystack.dataclasses.streaming_chunk import StreamingChunk, StreamingCallbackT


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


def test_to_dict_with_secret():
    MyComponent = factory.component_class("MyComponent")
    comp = MyComponent()
    secret = Secret.from_env_var("TEST_SECRET")
    res = default_to_dict(comp, api_key=secret)
    assert res == {
        "type": "haystack.testing.factory.MyComponent",
        "init_parameters": {"api_key": {"env_var": "TEST_SECRET", "strict": True}},
    }


def test_from_dict_with_secret():
    def custom_init(self, api_key):
        self.api_key = api_key

    extra_fields = {"__init__": custom_init}
    MyComponent = factory.component_class("MyComponent", extra_fields=extra_fields)
    data = {
        "type": "haystack.testing.factory.MyComponent",
        "init_parameters": {"api_key": {"env_var": "TEST_SECRET", "strict": True}},
    }
    comp = default_from_dict(MyComponent, data)
    assert isinstance(comp, MyComponent)
    assert isinstance(comp.api_key, Secret)
    assert comp.api_key.env_var == "TEST_SECRET"


def test_to_dict_with_tools():
    MyComponent = factory.component_class("MyComponent")
    comp = MyComponent()
    tool = Tool(name="test_tool", description="A test tool", function=lambda x: x)
    tools = [tool]
    res = default_to_dict(comp, tools=tools)
    assert res == {
        "type": "haystack.testing.factory.MyComponent",
        "init_parameters": {
            "tools": [{"name": "test_tool", "description": "A test tool", "function": "test_serialization.<lambda>"}]
        },
    }


def test_from_dict_with_tools():
    def custom_init(self, tools):
        self.tools = tools

    extra_fields = {"__init__": custom_init}
    MyComponent = factory.component_class("MyComponent", extra_fields=extra_fields)
    data = {
        "type": "haystack.testing.factory.MyComponent",
        "init_parameters": {
            "tools": [{"name": "test_tool", "description": "A test tool", "function": "test_serialization.<lambda>"}]
        },
    }
    comp = default_from_dict(MyComponent, data)
    assert isinstance(comp, MyComponent)
    assert isinstance(comp.tools, list)
    assert len(comp.tools) == 1
    assert isinstance(comp.tools[0], Tool)
    assert comp.tools[0].name == "test_tool"


def test_to_dict_with_streaming_callback():
    MyComponent = factory.component_class("MyComponent")
    comp = MyComponent()

    def streaming_callback(chunk: StreamingChunk) -> None:
        pass

    res = default_to_dict(comp, streaming_callback=streaming_callback)
    assert res == {
        "type": "haystack.testing.factory.MyComponent",
        "init_parameters": {"streaming_callback": "test_serialization.streaming_callback"},
    }


def test_from_dict_with_streaming_callback():
    def custom_init(self, streaming_callback):
        self.streaming_callback = streaming_callback

    extra_fields = {"__init__": custom_init}
    MyComponent = factory.component_class("MyComponent", extra_fields=extra_fields)
    data = {
        "type": "haystack.testing.factory.MyComponent",
        "init_parameters": {"streaming_callback": "test_serialization.streaming_callback"},
    }
    comp = default_from_dict(MyComponent, data)
    assert isinstance(comp, MyComponent)
    assert callable(comp.streaming_callback)
    assert comp.streaming_callback.__name__ == "streaming_callback"


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
