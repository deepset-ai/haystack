# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
from typing import Any, Dict, List

from haystack import component
from haystack.core.serialization import default_to_dict, default_from_dict
from haystack.tools import Tool, Toolset
from haystack.utils import Secret


def mock_tool_function(x: str) -> str:
    """A mock function for testing tool serialization."""
    return x


def test_default_serialization_methods_added():
    """Test that default to_dict and from_dict methods are added when not present."""

    @component
    class SimpleComponent:
        def __init__(self, value: int = 42):
            self.value = value

        def run(self, input_value: int) -> Dict[str, Any]:
            return {"output": input_value + self.value}

    # Create an instance
    comp = SimpleComponent(value=10)

    # Test to_dict
    data = comp.to_dict()
    assert data == {"type": "test_component_serialization.SimpleComponent", "init_parameters": {"value": 10}}

    # Test from_dict
    new_comp = SimpleComponent.from_dict(data)
    assert isinstance(new_comp, SimpleComponent)
    assert new_comp.value == 10


def test_custom_serialization_methods_preserved():
    """Test that custom to_dict and from_dict methods are preserved."""

    @component
    class CustomComponent:
        def __init__(self, value: int = 42):
            self.value = value

        def run(self, input_value: int) -> Dict[str, Any]:
            return {"output": input_value + self.value}

        def to_dict(self) -> Dict[str, Any]:
            return {"type": "custom", "value": self.value}

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> "CustomComponent":
            return cls(value=data["value"])

    # Create an instance
    comp = CustomComponent(value=10)

    # Test custom to_dict
    data = comp.to_dict()
    assert data == {"type": "custom", "value": 10}

    # Test custom from_dict
    new_comp = CustomComponent.from_dict(data)
    assert isinstance(new_comp, CustomComponent)
    assert new_comp.value == 10


def test_inherited_serialization_methods_preserved():
    """Test that inherited to_dict and from_dict methods are preserved."""

    class BaseComponent:
        def __init__(self, value: int = 42):
            self.value = value

        def to_dict(self) -> Dict[str, Any]:
            return {"type": "base", "value": self.value}

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> "BaseComponent":
            return cls(value=data["value"])

    @component
    class InheritedComponent(BaseComponent):
        def run(self, input_value: int) -> Dict[str, Any]:
            return {"output": input_value + self.value}

    # Create an instance
    comp = InheritedComponent(value=10)

    # Test inherited to_dict
    data = comp.to_dict()
    assert data == {"type": "base", "value": 10}

    # Test inherited from_dict
    new_comp = InheritedComponent.from_dict(data)
    assert isinstance(new_comp, InheritedComponent)
    assert new_comp.value == 10


def test_serialization_with_complex_init_parameters():
    """Test serialization with complex initialization parameters."""

    @component
    class ComplexComponent:
        def __init__(self, value: int = 42, name: str = "test", enabled: bool = True):
            self.value = value
            self.name = name
            self.enabled = enabled

        def run(self, input_value: int) -> Dict[str, Any]:
            return {"output": input_value + self.value}

    # Create an instance with multiple parameters
    comp = ComplexComponent(value=10, name="complex", enabled=False)

    # Test to_dict
    data = comp.to_dict()
    assert data == {
        "type": "test_component_serialization.ComplexComponent",
        "init_parameters": {"value": 10, "name": "complex", "enabled": False},
    }

    # Test from_dict
    new_comp = ComplexComponent.from_dict(data)
    assert isinstance(new_comp, ComplexComponent)
    assert new_comp.value == 10
    assert new_comp.name == "complex"
    assert new_comp.enabled is False


def test_serialization_with_tools():
    """Test serialization of components with tools using default serialization methods."""

    @component
    class ToolComponent:
        def __init__(self, tools: List[Tool]):
            self.tools = tools

        def run(self, input_value: str) -> Dict[str, Any]:
            return {"output": input_value}

    # Create a tool
    tool = Tool(
        name="test_tool",
        description="A test tool",
        parameters={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
        function=mock_tool_function,
    )

    # Create an instance with tools
    comp = ToolComponent(tools=[tool])

    # Test to_dict
    data = comp.to_dict()
    assert data["type"] == "test_component_serialization.ToolComponent"
    assert "tools" in data["init_parameters"]
    assert len(data["init_parameters"]["tools"]) == 1
    assert data["init_parameters"]["tools"][0]["type"] == "haystack.tools.tool.Tool"
    assert data["init_parameters"]["tools"][0]["data"]["name"] == "test_tool"

    # Test from_dict
    new_comp = ToolComponent.from_dict(data)
    assert isinstance(new_comp, ToolComponent)
    assert len(new_comp.tools) == 1
    assert isinstance(new_comp.tools[0], Tool)
    assert new_comp.tools[0].name == "test_tool"


def test_serialization_with_toolset():
    """Test serialization of components with Toolset using default serialization methods."""

    @component
    class ToolsetComponent:
        def __init__(self, tools: Toolset):
            self.tools = tools

        def run(self, input_value: str) -> Dict[str, Any]:
            return {"output": input_value}

    # Create a tool
    tool = Tool(
        name="test_tool",
        description="A test tool",
        parameters={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
        function=mock_tool_function,
    )

    # Create a toolset
    toolset = Toolset(tools=[tool])

    # Create an instance with toolset
    comp = ToolsetComponent(tools=toolset)

    # Test to_dict
    data = comp.to_dict()
    assert data["type"] == "test_component_serialization.ToolsetComponent"
    assert "tools" in data["init_parameters"]
    assert data["init_parameters"]["tools"]["type"] == "haystack.tools.toolset.Toolset"
    assert len(data["init_parameters"]["tools"]["data"]) == 1
    assert data["init_parameters"]["tools"]["data"][0]["data"]["name"] == "test_tool"

    # Test from_dict
    new_comp = ToolsetComponent.from_dict(data)
    assert isinstance(new_comp, ToolsetComponent)
    # The tools should be a dictionary in the serialized format
    assert isinstance(new_comp.tools, dict)
    assert new_comp.tools["type"] == "haystack.tools.toolset.Toolset"
    assert len(new_comp.tools["data"]) == 1
    assert new_comp.tools["data"][0]["data"]["name"] == "test_tool"


def test_serialization_with_secrets(monkeypatch):
    """Test serialization of components with secrets using default serialization methods."""
    monkeypatch.setenv("TEST_API_KEY", "test-api-key")

    @component
    class SecretComponent:
        def __init__(self, api_key: Secret):
            self.api_key = api_key

        def run(self, input_value: str) -> Dict[str, Any]:
            return {"output": input_value}

    secret = Secret.from_env_var("TEST_API_KEY")
    comp = SecretComponent(api_key=secret)

    # Test to_dict
    data = comp.to_dict()
    assert data["type"] == "test_component_serialization.SecretComponent"
    assert "api_key" in data["init_parameters"]
    assert data["init_parameters"]["api_key"]["type"] == "env_var"
    assert "TEST_API_KEY" in data["init_parameters"]["api_key"]["env_vars"]

    # Test from_dict
    new_comp = SecretComponent.from_dict(data)
    assert isinstance(new_comp, SecretComponent)
    assert isinstance(new_comp.api_key, Secret)
    assert new_comp.api_key.resolve_value() == "test-api-key"
