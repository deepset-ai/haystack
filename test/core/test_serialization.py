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
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.testing import factory
from haystack.utils import ComponentDevice, Secret
from haystack.utils.device import Device, DeviceMap


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
    def run(self, value: str) -> dict[str, str]:
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


@component
class CustomComponentWithDevice:
    def __init__(
        self,
        device: ComponentDevice | None = None,
        other_device: ComponentDevice | None = None,
        name: str | None = None,
    ):
        self.device = device
        self.other_device = other_device
        self.name = name

    @component.output_types(value=str)
    def run(self, value: str) -> dict[str, str]:
        return {"value": value}


def test_component_to_dict_with_component_device():
    """Test that ComponentDevice instances are automatically serialized in component_to_dict."""
    # Test with single device (CPU)
    device = ComponentDevice.from_single(Device.cpu())
    comp = CustomComponentWithDevice(device=device)
    res = component_to_dict(comp, "test_component")
    assert res["init_parameters"]["device"] == {"type": "single", "device": "cpu"}

    # Test with single device (GPU with id)
    device = ComponentDevice.from_single(Device.gpu(1))
    comp = CustomComponentWithDevice(device=device)
    res = component_to_dict(comp, "test_component")
    assert res["init_parameters"]["device"] == {"type": "single", "device": "cuda:1"}

    # Test with None
    comp = CustomComponentWithDevice(device=None)
    res = component_to_dict(comp, "test_component")
    assert res["init_parameters"]["device"] is None

    # Test with multiple devices (device map)
    device_map = DeviceMap({"layer1": Device.gpu(0), "layer2": Device.gpu(1)})
    device = ComponentDevice.from_multiple(device_map)
    comp = CustomComponentWithDevice(device=device)
    res = component_to_dict(comp, "test_component")
    assert res["init_parameters"]["device"] == {
        "type": "multiple",
        "device_map": {"layer1": "cuda:0", "layer2": "cuda:1"},
    }

    # Test with multiple ComponentDevice params
    device1 = ComponentDevice.from_single(Device.cpu())
    device2 = ComponentDevice.from_single(Device.gpu(0))
    comp = CustomComponentWithDevice(device=device1, other_device=device2, name="test")
    res = component_to_dict(comp, "test_component")
    assert res["init_parameters"]["device"] == {"type": "single", "device": "cpu"}
    assert res["init_parameters"]["other_device"] == {"type": "single", "device": "cuda:0"}
    assert res["init_parameters"]["name"] == "test"


def test_component_from_dict_with_component_device():
    """Test that serialized ComponentDevice dictionaries are automatically deserialized in component_from_dict."""
    # Test with single device (CPU)
    data = {
        "type": generate_qualified_class_name(CustomComponentWithDevice),
        "init_parameters": {"device": {"type": "single", "device": "cpu"}, "name": "test"},
    }
    comp = component_from_dict(CustomComponentWithDevice, data, "test_component")
    assert isinstance(comp, CustomComponentWithDevice)
    assert isinstance(comp.device, ComponentDevice)
    assert comp.device.to_torch_str() == "cpu"
    assert comp.name == "test"

    # Test with single device (GPU with id)
    data = {
        "type": generate_qualified_class_name(CustomComponentWithDevice),
        "init_parameters": {"device": {"type": "single", "device": "cuda:1"}, "name": "test"},
    }
    comp = component_from_dict(CustomComponentWithDevice, data, "test_component")
    assert isinstance(comp.device, ComponentDevice)
    assert comp.device.to_torch_str() == "cuda:1"

    # Test with None
    data = {
        "type": generate_qualified_class_name(CustomComponentWithDevice),
        "init_parameters": {"device": None, "name": "test"},
    }
    comp = component_from_dict(CustomComponentWithDevice, data, "test_component")
    assert comp.device is None
    assert comp.name == "test"

    # Test with multiple devices (device map)
    data = {
        "type": generate_qualified_class_name(CustomComponentWithDevice),
        "init_parameters": {"device": {"type": "multiple", "device_map": {"layer1": "cuda:0", "layer2": "cuda:1"}}},
    }
    comp = component_from_dict(CustomComponentWithDevice, data, "test_component")
    assert isinstance(comp.device, ComponentDevice)
    assert comp.device.has_multiple_devices

    # Test with regular dict (not a ComponentDevice - different structure)
    data = {
        "type": generate_qualified_class_name(CustomComponentWithDevice),
        "init_parameters": {"device": {"some": "dict"}, "name": "test"},
    }
    comp = component_from_dict(CustomComponentWithDevice, data, "test_component")
    assert comp.device == {"some": "dict"}
    assert comp.name == "test"

    # Test with multiple ComponentDevice params
    data = {
        "type": generate_qualified_class_name(CustomComponentWithDevice),
        "init_parameters": {
            "device": {"type": "single", "device": "cpu"},
            "other_device": {"type": "single", "device": "cuda:0"},
            "name": "test",
        },
    }
    comp = component_from_dict(CustomComponentWithDevice, data, "test_component")
    assert isinstance(comp.device, ComponentDevice)
    assert isinstance(comp.other_device, ComponentDevice)
    assert comp.device.to_torch_str() == "cpu"
    assert comp.other_device.to_torch_str() == "cuda:0"
    assert comp.name == "test"


def test_component_to_dict_and_from_dict_roundtrip_with_component_device():
    """Test that serialization and deserialization work together for ComponentDevice."""
    # Test roundtrip with single device
    original_device = ComponentDevice.from_single(Device.cpu())
    comp = CustomComponentWithDevice(device=original_device)

    serialized = component_to_dict(comp, "test_component")
    assert serialized["init_parameters"]["device"]["type"] == "single"

    deserialized_comp = component_from_dict(CustomComponentWithDevice, serialized, "test_component")
    assert isinstance(deserialized_comp.device, ComponentDevice)
    assert deserialized_comp.device.to_torch_str() == original_device.to_torch_str()

    # Test roundtrip with GPU device
    original_device = ComponentDevice.from_single(Device.gpu(2))
    comp = CustomComponentWithDevice(device=original_device)

    serialized = component_to_dict(comp, "test_component")
    deserialized_comp = component_from_dict(CustomComponentWithDevice, serialized, "test_component")
    assert deserialized_comp.device.to_torch_str() == "cuda:2"

    # Test roundtrip with device map
    device_map = DeviceMap({"layer1": Device.gpu(0), "layer2": Device.cpu()})
    original_device = ComponentDevice.from_multiple(device_map)
    comp = CustomComponentWithDevice(device=original_device)

    serialized = component_to_dict(comp, "test_component")
    assert serialized["init_parameters"]["device"]["type"] == "multiple"

    deserialized_comp = component_from_dict(CustomComponentWithDevice, serialized, "test_component")
    assert isinstance(deserialized_comp.device, ComponentDevice)
    assert deserialized_comp.device.has_multiple_devices

    # Test roundtrip with multiple ComponentDevice params
    device1 = ComponentDevice.from_single(Device.cpu())
    device2 = ComponentDevice.from_single(Device.gpu(0))
    comp = CustomComponentWithDevice(device=device1, other_device=device2, name="test")

    serialized = component_to_dict(comp, "test_component")
    assert serialized["init_parameters"]["device"]["type"] == "single"
    assert serialized["init_parameters"]["other_device"]["type"] == "single"
    assert serialized["init_parameters"]["name"] == "test"

    deserialized_comp = component_from_dict(CustomComponentWithDevice, serialized, "test_component")
    assert isinstance(deserialized_comp.device, ComponentDevice)
    assert isinstance(deserialized_comp.other_device, ComponentDevice)
    assert deserialized_comp.device.to_torch_str() == "cpu"
    assert deserialized_comp.other_device.to_torch_str() == "cuda:0"
    assert deserialized_comp.name == "test"


@component
class CustomComponentWithDocumentStore:
    def __init__(self, document_store: InMemoryDocumentStore | None = None, name: str | None = None):
        self.document_store = document_store
        self.name = name

    @component.output_types(value=str)
    def run(self, value: str) -> dict[str, str]:
        return {"value": value}


def test_component_to_dict_with_document_store():
    """Test that DocumentStore instances are automatically serialized in component_to_dict."""
    # Test with InMemoryDocumentStore
    doc_store = InMemoryDocumentStore()
    comp = CustomComponentWithDocumentStore(document_store=doc_store)
    res = component_to_dict(comp, "test_component")
    assert "type" in res["init_parameters"]["document_store"]
    assert "init_parameters" in res["init_parameters"]["document_store"]
    assert (
        res["init_parameters"]["document_store"]["type"]
        == "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore"
    )

    # Test with None
    comp = CustomComponentWithDocumentStore(document_store=None)
    res = component_to_dict(comp, "test_component")
    assert res["init_parameters"]["document_store"] is None


def test_component_from_dict_with_document_store():
    """Test that serialized DocumentStore dictionaries are automatically deserialized in component_from_dict."""
    # Test with InMemoryDocumentStore
    doc_store = InMemoryDocumentStore()
    serialized_doc_store = doc_store.to_dict()
    data = {
        "type": generate_qualified_class_name(CustomComponentWithDocumentStore),
        "init_parameters": {"document_store": serialized_doc_store, "name": "test"},
    }
    comp = component_from_dict(CustomComponentWithDocumentStore, data, "test_component")
    assert isinstance(comp, CustomComponentWithDocumentStore)
    assert isinstance(comp.document_store, InMemoryDocumentStore)
    assert comp.name == "test"

    # Test with None
    data = {
        "type": generate_qualified_class_name(CustomComponentWithDocumentStore),
        "init_parameters": {"document_store": None, "name": "test"},
    }
    comp = component_from_dict(CustomComponentWithDocumentStore, data, "test_component")
    assert comp.document_store is None
    assert comp.name == "test"


def test_component_to_dict_and_from_dict_roundtrip_with_document_store():
    """Test that serialization and deserialization work together for DocumentStore."""
    # Test roundtrip with InMemoryDocumentStore
    original_doc_store = InMemoryDocumentStore()
    comp = CustomComponentWithDocumentStore(document_store=original_doc_store)

    serialized = component_to_dict(comp, "test_component")
    assert "type" in serialized["init_parameters"]["document_store"]
    assert (
        serialized["init_parameters"]["document_store"]["type"]
        == "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore"
    )

    deserialized_comp = component_from_dict(CustomComponentWithDocumentStore, serialized, "test_component")
    assert isinstance(deserialized_comp.document_store, InMemoryDocumentStore)
    assert deserialized_comp.document_store.bm25_algorithm == original_doc_store.bm25_algorithm
    assert (
        deserialized_comp.document_store.embedding_similarity_function
        == original_doc_store.embedding_similarity_function
    )

    # Test roundtrip with custom parameters
    original_doc_store = InMemoryDocumentStore(
        bm25_algorithm="BM25Okapi", embedding_similarity_function="cosine", return_embedding=False
    )
    comp = CustomComponentWithDocumentStore(document_store=original_doc_store)

    serialized = component_to_dict(comp, "test_component")
    deserialized_comp = component_from_dict(CustomComponentWithDocumentStore, serialized, "test_component")
    assert isinstance(deserialized_comp.document_store, InMemoryDocumentStore)
    assert deserialized_comp.document_store.bm25_algorithm == "BM25Okapi"
    assert deserialized_comp.document_store.embedding_similarity_function == "cosine"
    assert deserialized_comp.document_store.return_embedding is False


def test_default_to_dict_with_document_store():
    """Test that DocumentStore instances are automatically serialized in default_to_dict."""
    doc_store = InMemoryDocumentStore()
    res = default_to_dict(doc_store)
    assert res["type"] == "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore"
    assert "init_parameters" in res

    # Test that DocumentStore is serialized when passed as a parameter
    doc_store = InMemoryDocumentStore()
    comp = CustomComponentWithDocumentStore(document_store=doc_store)
    res = default_to_dict(comp, document_store=doc_store, name="test")
    assert "type" in res["init_parameters"]["document_store"]
    assert res["init_parameters"]["name"] == "test"


def test_default_from_dict_with_document_store():
    """Test that serialized DocumentStore dictionaries are automatically deserialized in default_from_dict."""
    doc_store = InMemoryDocumentStore()
    serialized = doc_store.to_dict()

    # Test direct deserialization
    deserialized = default_from_dict(InMemoryDocumentStore, serialized)
    assert isinstance(deserialized, InMemoryDocumentStore)
    assert deserialized.bm25_algorithm == doc_store.bm25_algorithm

    # Test deserialization when DocumentStore is in init_parameters
    data = {
        "type": generate_qualified_class_name(CustomComponentWithDocumentStore),
        "init_parameters": {"document_store": serialized, "name": "test"},
    }
    comp = default_from_dict(CustomComponentWithDocumentStore, data)
    assert isinstance(comp.document_store, InMemoryDocumentStore)
    assert comp.name == "test"


def test_default_from_dict_with_invalid_class_name():
    """Test that deserialization raises ImportError with improved message when class cannot be imported."""
    data = {
        "type": generate_qualified_class_name(CustomComponentWithDocumentStore),
        "init_parameters": {
            "document_store": {"type": "nonexistent.module.Class", "init_parameters": {}},
            "name": "test",
        },
    }
    # Verify the error message includes the parameter key and original error
    with pytest.raises(ImportError, match=r"Failed to deserialize 'document_store':.*nonexistent\.module\.Class"):
        default_from_dict(CustomComponentWithDocumentStore, data)
