# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import json
import os
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import pytest
import requests
from openapi_service_client import OpenAPIServiceClient

from haystack.components.connectors import OpenAPIServiceConnector
from haystack.dataclasses import ChatMessage


@pytest.fixture
def openapi_service_mock():
    return MagicMock(spec=OpenAPIServiceClient)


class TestOpenAPIServiceConnector:
    @pytest.fixture
    def connector(self):
        return OpenAPIServiceConnector()

    def test_parse_message_invalid_json(self, connector):
        # Test invalid JSON content
        with pytest.raises(ValueError):
            connector._parse_message(ChatMessage.from_assistant("invalid json"))

    def test_parse_valid_json_message(self):
        connector = OpenAPIServiceConnector()

        # The content format here is OpenAI function calling descriptor
        content = (
            '[{"function":{"name": "compare_branches","arguments": "{\\n  \\"parameters\\": {\\n   '
            ' \\"basehead\\": \\"main...openapi_container_v5\\",\\n   '
            ' \\"owner\\": \\"deepset-ai\\",\\n    \\"repo\\": \\"haystack\\"\\n  }\\n}"}, "type": "function"}]'
        )
        descriptors = connector._parse_message(ChatMessage.from_assistant(content))

        # Assert that the descriptor contains the expected method name and arguments
        assert descriptors[0]["name"] == "compare_branches"
        assert descriptors[0]["arguments"]["parameters"] == {
            "basehead": "main...openapi_container_v5",
            "owner": "deepset-ai",
            "repo": "haystack",
        }
        # but not the requestBody
        assert "requestBody" not in descriptors[0]["arguments"]

        # The content format here is OpenAI function calling descriptor
        content = '[{"function": {"name": "search","arguments": "{\\n  \\"requestBody\\": {\\n    \\"q\\": \\"haystack\\"\\n  }\\n}"}, "type": "function"}]'
        descriptors = connector._parse_message(ChatMessage.from_assistant(content))
        assert descriptors[0]["name"] == "search"
        assert descriptors[0]["arguments"]["requestBody"] == {"q": "haystack"}

        # but not the parameters
        assert "parameters" not in descriptors[0]["arguments"]

    def test_parse_message_missing_fields(self, connector):
        # Test JSON content with missing fields
        with pytest.raises(ValueError):
            connector._parse_message(ChatMessage.from_assistant('[{"function": {"name": "test_method"}}]'))

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
