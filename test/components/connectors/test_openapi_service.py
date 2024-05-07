# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import json
import os
from unittest.mock import patch

import pytest

from haystack.components.connectors import OpenAPIServiceConnector
from haystack.dataclasses import ChatMessage


class TestOpenAPIServiceConnector:
    @pytest.fixture
    def setup_mock(self):
        with patch("haystack.components.connectors.openapi_service.OpenAPIServiceClient") as mock_client:
            mock_client_instance = mock_client.return_value
            mock_client_instance.invoke.return_value = {"service_response": "Yes, he was fired and rehired"}
            yield mock_client_instance

    def test_init(self):
        service_connector = OpenAPIServiceConnector()
        assert service_connector is not None
        assert service_connector.provider_map is not None
        assert service_connector.default_provider == "openai"

    def test_init_with_anthropic_provider(self):
        service_connector = OpenAPIServiceConnector(default_provider="anthropic")
        assert service_connector is not None
        assert service_connector.provider_map is not None
        assert service_connector.default_provider == "anthropic"

    def test_run_with_mock(self, setup_mock, test_files_path):
        fc_payload = [
            {
                "function": {"arguments": '{"q": "Why was Sam Altman ousted from OpenAI?"}', "name": "search"},
                "id": "call_PmEBYvZ7mGrQP5PUASA5m9wO",
                "type": "function",
            }
        ]
        with open(os.path.join(test_files_path, "json/serperdev_openapi_spec.json"), "r") as file:
            serperdev_openapi_spec = json.load(file)

        service_connector = OpenAPIServiceConnector()
        result = service_connector.run(
            messages=[ChatMessage.from_assistant(json.dumps(fc_payload))],
            service_openapi_spec=serperdev_openapi_spec,
            service_credentials="fake_api_key",
        )

        assert "service_response" in result
        assert len(result["service_response"]) == 1
        assert isinstance(result["service_response"][0], ChatMessage)
        response_content = json.loads(result["service_response"][0].content)
        assert response_content == {"service_response": "Yes, he was fired and rehired"}

        # verify invocation payload
        setup_mock.invoke.assert_called_once()
        invocation_payload = [
            {
                "function": {"arguments": '{"q": "Why was Sam Altman ousted from OpenAI?"}', "name": "search"},
                "id": "call_PmEBYvZ7mGrQP5PUASA5m9wO",
                "type": "function",
            }
        ]
        setup_mock.invoke.assert_called_with(invocation_payload)

    @pytest.mark.integration
    @pytest.mark.skipif("SERPERDEV_API_KEY" not in os.environ, reason="SerperDev API key is not available")
    def test_run(self, test_files_path):
        fc_payload = [
            {
                "function": {"arguments": '{"q": "Why was Sam Altman ousted from OpenAI?"}', "name": "search"},
                "id": "call_PmEBYvZ7mGrQP5PUASA5m9wO",
                "type": "function",
            }
        ]

        with open(os.path.join(test_files_path, "json/serperdev_openapi_spec.json"), "r") as file:
            serperdev_openapi_spec = json.load(file)

        service_connector = OpenAPIServiceConnector()
        result = service_connector.run(
            messages=[ChatMessage.from_assistant(json.dumps(fc_payload))],
            service_openapi_spec=serperdev_openapi_spec,
            service_credentials=os.environ["SERPERDEV_API_KEY"],
        )
        assert "service_response" in result
        assert len(result["service_response"]) == 1
        assert isinstance(result["service_response"][0], ChatMessage)
        response_text = result["service_response"][0].content
        assert "Sam" in response_text or "Altman" in response_text

    @pytest.mark.integration
    def test_run_no_credentials(self, test_files_path):
        fc_payload = [
            {
                "function": {"arguments": '{"q": "Why was Sam Altman ousted from OpenAI?"}', "name": "search"},
                "id": "call_PmEBYvZ7mGrQP5PUASA5m9wO",
                "type": "function",
            }
        ]

        with open(os.path.join(test_files_path, "json/serperdev_openapi_spec.json"), "r") as file:
            serperdev_openapi_spec = json.load(file)

        service_connector = OpenAPIServiceConnector()
        result = service_connector.run(
            messages=[ChatMessage.from_assistant(json.dumps(fc_payload))], service_openapi_spec=serperdev_openapi_spec
        )
        assert "service_response" in result
        assert len(result["service_response"]) == 1
        assert isinstance(result["service_response"][0], ChatMessage)
        response_text = result["service_response"][0].content
        assert "403" in response_text
