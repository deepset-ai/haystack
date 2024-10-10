# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.utils import Secret

from haystack.components.generators.openai_dalle import DALLEImageGenerator


class TestDALLEImageGenerator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        component = DALLEImageGenerator()
        assert component.model == "dall-e-3"
        assert component.quality == "standard"
        assert component.size == "1024x1024"
        assert component.response_format == "url"
        assert component.api_key == Secret.from_env_var("OPENAI_API_KEY")
        assert component.api_base_url is None
        assert component.organization is None
        assert pytest.approx(component.timeout) == 30.0
        assert component.max_retries is 5

    def test_warm_up(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        component = DALLEImageGenerator()
        component.warm_up()
        assert component.client.api_key == "test-api-key"
        assert component.client.timeout == 30
        assert component.client.max_retries == 5

    def test_to_dict(self) -> None:
        generator = DALLEImageGenerator()
        data = generator.to_dict()
        assert data == {
            "type": "haystack.components.generators.openai_dalle.DALLEImageGenerator",
            "init_parameters": {
                "model": "dall-e-3",
                "quality": "standard",
                "size": "1024x1024",
                "response_format": "url",
                "api_key": {"type": "env_var", "env_vars": ["OPENAI_API_KEY"], "strict": True},
                "api_base_url": None,
                "organization": None,
            },
        }

    def test_from_dict(self) -> None:
        data = {
            "type": "haystack.components.generators.openai_dalle.DALLEImageGenerator",
            "init_parameters": {
                "model": "dall-e-3",
                "quality": "standard",
                "size": "1024x1024",
                "response_format": "url",
                "api_key": {"type": "env_var", "env_vars": ["OPENAI_API_KEY"], "strict": True},
                "api_base_url": None,
                "organization": None,
            },
        }
        generator = DALLEImageGenerator.from_dict(data)
        assert generator.model == "dall-e-3"
        assert generator.quality == "standard"
        assert generator.size == "1024x1024"
        assert generator.response_format == "url"
        assert generator.api_key.to_dict() == {"type": "env_var", "env_vars": ["OPENAI_API_KEY"], "strict": True}
