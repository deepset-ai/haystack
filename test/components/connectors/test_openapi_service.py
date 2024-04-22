# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import json
import os
from unittest.mock import MagicMock

import pytest
from openapi_service_client import OpenAPIServiceClient

from haystack.components.connectors import OpenAPIServiceConnector
from haystack.dataclasses import ChatMessage


class TestOpenAPIServiceConnector:
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
