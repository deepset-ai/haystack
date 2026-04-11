# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.components.generators.chat.minimax import MiniMaxChatGenerator
from haystack.components.generators.utils import print_streaming_chunk
from haystack.tools import Tool
from haystack.utils.auth import Secret


class TestMiniMaxChatGenerator:
    def test_supported_models(self) -> None:
        """SUPPORTED_MODELS contains exactly the two MiniMax models."""
        models = MiniMaxChatGenerator.SUPPORTED_MODELS
        assert isinstance(models, list)
        assert "MiniMax-M2.7" in models
        assert "MiniMax-M2.7-highspeed" in models

    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-key")
        component = MiniMaxChatGenerator()
        assert component.client.api_key == "test-minimax-key"
        assert component.model == "MiniMax-M2.7"
        assert component.streaming_callback is None
        assert not component.generation_kwargs
        assert component.tools is None
        assert not component.tools_strict

    def test_default_base_url(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-key")
        component = MiniMaxChatGenerator()
        assert str(component.client.base_url).rstrip("/") == "https://api.minimax.io/v1"

    def test_init_fail_without_api_key(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        with pytest.raises(ValueError):
            MiniMaxChatGenerator()

    def test_init_with_parameters(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-key")
        tool = Tool(name="name", description="description", parameters={"x": {"type": "string"}}, function=lambda x: x)
        component = MiniMaxChatGenerator(
            api_key=Secret.from_token("test-minimax-key"),
            model="MiniMax-M2.7-highspeed",
            streaming_callback=print_streaming_chunk,
            generation_kwargs={"temperature": 0.7},
            tools=[tool],
            tools_strict=True,
        )
        assert component.client.api_key == "test-minimax-key"
        assert component.model == "MiniMax-M2.7-highspeed"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"temperature": 0.7}
        assert component.tools == [tool]
        assert component.tools_strict

    def test_custom_base_url(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-key")
        component = MiniMaxChatGenerator(api_base_url="https://custom.minimax.io/v1")
        assert str(component.client.base_url).rstrip("/") == "https://custom.minimax.io/v1"

    def test_to_dict_default(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-key")
        component = MiniMaxChatGenerator()
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.generators.chat.minimax.MiniMaxChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["MINIMAX_API_KEY"], "strict": True, "type": "env_var"},
                "model": "MiniMax-M2.7",
                "streaming_callback": None,
                "api_base_url": None,
                "organization": None,
                "generation_kwargs": {},
                "timeout": None,
                "max_retries": None,
                "tools": None,
                "tools_strict": False,
                "http_client_kwargs": None,
            },
        }

    def test_to_dict_with_parameters(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-key")
        component = MiniMaxChatGenerator(
            model="MiniMax-M2.7-highspeed",
            api_base_url="https://custom.minimax.io/v1",
            generation_kwargs={"temperature": 0.8},
        )
        data = component.to_dict()
        assert data["init_parameters"]["model"] == "MiniMax-M2.7-highspeed"
        assert data["init_parameters"]["api_base_url"] == "https://custom.minimax.io/v1"
        assert data["init_parameters"]["generation_kwargs"] == {"temperature": 0.8}

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-key")
        data = {
            "type": "haystack.components.generators.chat.minimax.MiniMaxChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["MINIMAX_API_KEY"], "strict": True, "type": "env_var"},
                "model": "MiniMax-M2.7",
                "streaming_callback": None,
                "api_base_url": None,
                "organization": None,
                "generation_kwargs": {},
                "timeout": None,
                "max_retries": None,
                "tools": None,
                "tools_strict": False,
                "http_client_kwargs": None,
            },
        }
        component = MiniMaxChatGenerator.from_dict(data)
        assert component.model == "MiniMax-M2.7"
        assert component.client.api_key == "test-minimax-key"
        assert str(component.client.base_url).rstrip("/") == "https://api.minimax.io/v1"

    def test_serialization_round_trip(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-key")
        component = MiniMaxChatGenerator(
            model="MiniMax-M2.7-highspeed",
            generation_kwargs={"temperature": 0.9},
        )
        data = component.to_dict()
        restored = MiniMaxChatGenerator.from_dict(data)
        assert restored.model == component.model
        assert restored.generation_kwargs == component.generation_kwargs
