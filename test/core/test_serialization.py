# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from unittest.mock import Mock

import pytest

from haystack.core.component import component
from haystack.core.errors import DeserializationError, SerializationError
from haystack.core.pipeline import Pipeline
from haystack.core.serialization import (
    component_from_dict,
    component_to_dict,
    default_from_dict,
    default_to_dict,
    generate_qualified_class_name,
    import_class_by_name,
)
from haystack.testing import factory
from haystack.utils import Secret


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


@component
class CustomComponentWithSecrets:
    def __init__(self, api_key: Secret | None = None, token: Secret | None = None, regular_param: str | None = None):
        self.api_key = api_key
        self.token = token
        self.regular_param = regular_param

    @component.output_types(value=str)
    def run(self, value: str):
        return {"value": value}


def test_component_to_dict_with_secret():
    """Test that Secret instances are automatically serialized in component_to_dict."""
    # Test with EnvVarSecret (serializable)
    env_secret = Secret.from_env_var("TEST_API_KEY")
    comp = CustomComponentWithSecrets(api_key=env_secret)
    res = component_to_dict(comp, "test_component")
    assert res["init_parameters"]["api_key"] == env_secret.to_dict()
    assert res["init_parameters"]["api_key"]["type"] == "env_var"

    # Test with None
    comp = CustomComponentWithSecrets(api_key=None)
    res = component_to_dict(comp, "test_component")
    assert res["init_parameters"]["api_key"] is None

    # Test with regular value (not a Secret)
    comp = CustomComponentWithSecrets(regular_param="regular_string")
    res = component_to_dict(comp, "test_component")
    assert res["init_parameters"]["regular_param"] == "regular_string"

    # Test with multiple secrets
    env_secret1 = Secret.from_env_var("TEST_API_KEY1")
    env_secret2 = Secret.from_env_var("TEST_API_KEY2")
    comp = CustomComponentWithSecrets(api_key=env_secret1, token=env_secret2, regular_param="test")
    res = component_to_dict(comp, "test_component")
    assert res["init_parameters"]["api_key"] == env_secret1.to_dict()
    assert res["init_parameters"]["api_key"]["type"] == "env_var"
    assert res["init_parameters"]["token"] == env_secret2.to_dict()
    assert res["init_parameters"]["token"]["type"] == "env_var"
    assert res["init_parameters"]["regular_param"] == "test"


def test_component_from_dict_with_secret():
    """Test that serialized Secret dictionaries are automatically deserialized in component_from_dict."""
    # Test with EnvVarSecret
    env_secret = Secret.from_env_var("TEST_API_KEY")
    serialized_secret = env_secret.to_dict()
    data = {
        "type": generate_qualified_class_name(CustomComponentWithSecrets),
        "init_parameters": {"api_key": serialized_secret, "regular_param": "test"},
    }
    comp = component_from_dict(CustomComponentWithSecrets, data, "test_component")
    assert isinstance(comp, CustomComponentWithSecrets)
    assert isinstance(comp.api_key, Secret)
    assert comp.api_key.type.value == "env_var"
    assert comp.regular_param == "test"

    # Test with None
    data = {
        "type": generate_qualified_class_name(CustomComponentWithSecrets),
        "init_parameters": {"api_key": None, "regular_param": "test"},
    }
    comp = component_from_dict(CustomComponentWithSecrets, data, "test_component")
    assert comp.api_key is None
    assert comp.regular_param == "test"

    # Test with regular dict (not a Secret)
    data = {
        "type": generate_qualified_class_name(CustomComponentWithSecrets),
        "init_parameters": {"api_key": {"some": "dict"}, "regular_param": "test"},
    }
    comp = component_from_dict(CustomComponentWithSecrets, data, "test_component")
    assert comp.api_key == {"some": "dict"}
    assert comp.regular_param == "test"

    # Test with multiple secrets
    env_secret1 = Secret.from_env_var("TEST_API_KEY1")
    env_secret2 = Secret.from_env_var("TEST_API_KEY2")
    data = {
        "type": generate_qualified_class_name(CustomComponentWithSecrets),
        "init_parameters": {"api_key": env_secret1.to_dict(), "token": env_secret2.to_dict(), "regular_param": "test"},
    }
    comp = component_from_dict(CustomComponentWithSecrets, data, "test_component")
    assert isinstance(comp.api_key, Secret)
    assert isinstance(comp.token, Secret)
    assert comp.regular_param == "test"


def test_component_to_dict_and_from_dict_roundtrip_with_secret():
    """Test that serialization and deserialization work together for Secrets."""
    # Test roundtrip with EnvVarSecret
    original_secret = Secret.from_env_var("TEST_API_KEY")
    comp = CustomComponentWithSecrets(api_key=original_secret)

    serialized = component_to_dict(comp, "test_component")
    assert serialized["init_parameters"]["api_key"]["type"] == "env_var"

    deserialized_comp = component_from_dict(CustomComponentWithSecrets, serialized, "test_component")
    assert isinstance(deserialized_comp.api_key, Secret)
    assert deserialized_comp.api_key.type.value == "env_var"
    assert deserialized_comp.api_key._env_vars == original_secret._env_vars

    # Test roundtrip with multiple secrets
    env_secret1 = Secret.from_env_var("TEST_API_KEY1")
    env_secret2 = Secret.from_env_var("TEST_API_KEY2")
    comp = CustomComponentWithSecrets(api_key=env_secret1, token=env_secret2, regular_param="test")

    serialized = component_to_dict(comp, "test_component")
    assert serialized["init_parameters"]["api_key"]["type"] == "env_var"
    assert serialized["init_parameters"]["token"]["type"] == "env_var"
    assert serialized["init_parameters"]["regular_param"] == "test"

    deserialized_comp = component_from_dict(CustomComponentWithSecrets, serialized, "test_component")
    assert isinstance(deserialized_comp.api_key, Secret)
    assert isinstance(deserialized_comp.token, Secret)
    assert deserialized_comp.api_key.type.value == "env_var"
    assert deserialized_comp.token.type.value == "env_var"
    assert deserialized_comp.regular_param == "test"
    assert deserialized_comp.api_key._env_vars == env_secret1._env_vars
    assert deserialized_comp.token._env_vars == env_secret2._env_vars
