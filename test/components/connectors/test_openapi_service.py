import json
import os

import pytest
from unittest.mock import MagicMock, Mock

import requests
from openapi3 import OpenAPI
from openapi3.schemas import Model
from haystack.components.connectors import OpenAPIServiceConnector
from haystack.dataclasses import ChatMessage


@pytest.fixture
def openapi_service_mock():
    return MagicMock(spec=OpenAPI)


@pytest.fixture
def random_open_pull_request_head_branch() -> str:
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github.v3+json", "Authorization": f"token {token}"}
    response = requests.get("https://api.github.com/repos/deepset-ai/haystack/pulls?state=open", headers=headers)

    if response.status_code == 200:
        pull_requests = response.json()
        for pr in pull_requests:
            if pr["base"]["ref"] == "main":
                return pr["head"]["ref"]
    else:
        raise Exception(f"Failed to fetch pull requests. Status code: {response.status_code}")


@pytest.fixture
def genuine_fc_message(random_open_pull_request_head_branch):
    basehead = "main..." + random_open_pull_request_head_branch
    return f"""[{{"id": "call_NJr1NBz2Th7iUWJpRIJZoJIA", "function": {{"arguments": "{{\\n  \\"parameters\\": {{\\n    \\"basehead\\": \\"{basehead}\\",\\n    \\"owner\\": \\"deepset-ai\\",\\n    \\"repo\\": \\"haystack\\"\\n  }}\\n}}", "name": "compare_branches"}}, "type": "function"}}]"""


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

    def test_authenticate_service_missing_authentication(self, connector, openapi_service_mock):
        # Test missing authentication when it is required
        openapi_service_mock.components.securitySchemes = {"apiKey": {}}
        with pytest.raises(ValueError):
            connector._authenticate_service(openapi_service_mock)

    def test_authenticate_service_having_authentication(self, connector, openapi_service_mock):
        # Test authentication when it is required and provided
        openapi_service_mock.components.securitySchemes = {"apiKey": {}}
        connector._authenticate_service(openapi_service_mock, "some_token")

    def test_invoke_method_valid(self, connector, openapi_service_mock):
        # Test valid method invocation
        method_invocation_descriptor = {"name": "test_method", "arguments": {}}
        openapi_service_mock.call_test_method = Mock(return_value="response")
        result = connector._invoke_method(openapi_service_mock, method_invocation_descriptor)
        assert result == "response"

    def test_invoke_method_invalid(self, connector, openapi_service_mock):
        # Test invalid method invocation
        method_invocation_descriptor = {"name": "invalid_method", "arguments": {}}
        with pytest.raises(RuntimeError):
            connector._invoke_method(openapi_service_mock, method_invocation_descriptor)

    def test_for_internal_raw_data_field(self):
        # see https://github.com/deepset-ai/haystack/pull/6772 for details
        model = Model(data={}, schema={})
        assert hasattr(model, "_raw_data"), (
            "openapi3 changed. Model should have a _raw_data field, we rely on it in OpenAPIServiceConnector"
            " to get the raw data from the service response"
        )

    @pytest.mark.integration
    def test_run(self, genuine_fc_message):
        openapi_service = OpenAPIServiceConnector()
        github_compare_schema = requests.get("https://bit.ly/github_compare").json()
        messages = [ChatMessage.from_assistant(genuine_fc_message)]

        # genuine call to the GitHub OpenAPI service
        result = openapi_service.run(messages, github_compare_schema, os.getenv("GITHUB_TOKEN"))
        assert result

        # load json from the service response
        service_payload = json.loads(result["service_response"][0].content)

        # verify that the service response contains the expected fields
        assert "url" in service_payload and "files" in service_payload
