# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import json
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import pytest
from openapi3 import OpenAPI
from openapi3.schemas import Model

from haystack.components.connectors import OpenAPIServiceConnector
from haystack.dataclasses import ChatMessage


@pytest.fixture
def openapi_service_mock():
    return MagicMock(spec=OpenAPI)


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

    def test_authenticate_service_missing_authentication_token(self, connector, openapi_service_mock):
        security_schemes_dict = {
            "components": {"securitySchemes": {"apiKey": {"in": "header", "name": "x-api-key", "type": "apiKey"}}}
        }
        openapi_service_mock.raw_element = security_schemes_dict

        with pytest.raises(ValueError):
            connector._authenticate_service(openapi_service_mock)

    def test_authenticate_service_having_authentication_token(self, connector, openapi_service_mock):
        security_schemes_dict = {
            "components": {"securitySchemes": {"apiKey": {"in": "header", "name": "x-api-key", "type": "apiKey"}}}
        }
        openapi_service_mock.raw_element = security_schemes_dict
        openapi_service_mock.components.securitySchemes.raw_element = {
            "apiKey": {"in": "header", "name": "x-api-key", "type": "apiKey"}
        }
        connector._authenticate_service(openapi_service_mock, "some_fake_token")

    def test_authenticate_service_having_authentication_dict(self, connector, openapi_service_mock):
        security_schemes_dict = {
            "components": {"securitySchemes": {"apiKey": {"in": "header", "name": "x-api-key", "type": "apiKey"}}}
        }
        openapi_service_mock.raw_element = security_schemes_dict
        openapi_service_mock.components.securitySchemes.raw_element = {
            "apiKey": {"in": "header", "name": "x-api-key", "type": "apiKey"}
        }
        connector._authenticate_service(openapi_service_mock, {"apiKey": "some_fake_token"})

    def test_authenticate_service_having_authentication_dict_but_unsupported_auth(
        self, connector, openapi_service_mock
    ):
        security_schemes_dict = {"components": {"securitySchemes": {"oauth2": {"type": "oauth2"}}}}
        openapi_service_mock.raw_element = security_schemes_dict
        openapi_service_mock.components.securitySchemes.raw_element = {"oauth2": {"type": "oauth2"}}
        with pytest.raises(ValueError):
            connector._authenticate_service(openapi_service_mock, {"apiKey": "some_fake_token"})

    def test_for_internal_raw_data_field(self):
        # see https://github.com/deepset-ai/haystack/pull/6772 for details
        model = Model(data={}, schema={})
        assert hasattr(model, "_raw_data"), (
            "openapi3 changed. Model should have a _raw_data field, we rely on it in OpenAPIServiceConnector"
            " to get the raw data from the service response"
        )

    @patch("haystack.components.connectors.openapi_service.OpenAPI")
    def test_run(self, openapi_mock, test_files_path):
        connector = OpenAPIServiceConnector()
        spec_path = test_files_path / "json" / "github_compare_branch_openapi_spec.json"
        spec = json.loads((spec_path).read_text())

        mock_message = json.dumps(
            [
                {
                    "id": "call_NJr1NBz2Th7iUWJpRIJZoJIA",
                    "function": {
                        "arguments": '{"basehead": "main...some_branch", "owner": "deepset-ai", "repo": "haystack"}',
                        "name": "compare_branches",
                    },
                    "type": "function",
                }
            ]
        )
        messages = [ChatMessage.from_assistant(mock_message)]
        call_compare_branches = Mock(return_value=Mock(_raw_data="some_data"))
        call_compare_branches.operation.__self__ = Mock()
        call_compare_branches.operation.__self__.raw_element = {
            "parameters": [{"name": "basehead"}, {"name": "owner"}, {"name": "repo"}]
        }
        mock_service = Mock(
            call_compare_branches=call_compare_branches,
            components=Mock(securitySchemes=Mock(raw_element={"apikey": {"type": "apiKey"}})),
        )
        openapi_mock.return_value = mock_service

        connector.run(messages=messages, service_openapi_spec=spec, service_credentials="fake_key")

        openapi_mock.assert_called_once_with(spec, ssl_verify=None)
        mock_service.authenticate.assert_called_once_with("apikey", "fake_key")

        # verify call went through on the wire with the correct parameters
        mock_service.call_compare_branches.assert_called_once_with(
            parameters={"basehead": "main...some_branch", "owner": "deepset-ai", "repo": "haystack"}
        )

    @patch("haystack.components.connectors.openapi_service.OpenAPI")
    def test_run_with_mix_params_request_body(self, openapi_mock, test_files_path):
        connector = OpenAPIServiceConnector()
        spec_path = test_files_path / "yaml" / "openapi_greeting_service.yml"
        with open(spec_path, "r") as file:
            spec = json.loads(file.read())
        mock_message = json.dumps(
            [
                {
                    "id": "call_NJr1NBz2Th7iUWJpRIJZoJIA",
                    "function": {"arguments": '{"name": "John", "message": "Hello"}', "name": "greet"},
                    "type": "function",
                }
            ]
        )
        call_greet = Mock(return_value=Mock(_raw_data="Hello, John"))
        call_greet.operation.__self__ = Mock()
        call_greet.operation.__self__.raw_element = {
            "parameters": [{"name": "name"}],
            "requestBody": {
                "content": {"application/json": {"schema": {"properties": {"message": {"type": "string"}}}}}
            },
        }

        mock_service = Mock(call_greet=call_greet)
        mock_service.raw_element = {}
        openapi_mock.return_value = mock_service

        messages = [ChatMessage.from_assistant(mock_message)]
        result = connector.run(messages=messages, service_openapi_spec=spec)

        # verify call went through on the wire
        mock_service.call_greet.assert_called_once_with(parameters={"name": "John"}, data={"message": "Hello"})

        response = json.loads(result["service_response"][0].text)
        assert response == "Hello, John"

    @patch("haystack.components.connectors.openapi_service.OpenAPI")
    def test_run_with_complex_types(self, openapi_mock, test_files_path):
        connector = OpenAPIServiceConnector()
        spec_path = test_files_path / "json" / "complex_types_openapi_service.json"
        with open(spec_path, "r") as file:
            spec = json.loads(file.read())
        mock_message = json.dumps(
            [
                {
                    "id": "call_NJr1NBz2Th7iUWJpRIJZoJIA",
                    "function": {
                        "arguments": '{"transaction_amount": 150.75, "description": "Monthly subscription fee", "payment_method_id": "visa_ending_in_1234", "payer": {"name": "Alex Smith", "email": "alex.smith@example.com", "identification": {"type": "Driver\'s License", "number": "D12345678"}}}',
                        "name": "processPayment",
                    },
                    "type": "function",
                }
            ]
        )

        call_processPayment = Mock(return_value=Mock(_raw_data={"result": "accepted"}))
        call_processPayment.operation.__self__ = Mock()
        call_processPayment.operation.__self__.raw_element = {
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "properties": {
                                "transaction_amount": {"type": "number", "example": 150.75},
                                "description": {"type": "string", "example": "Monthly subscription fee"},
                                "payment_method_id": {"type": "string", "example": "visa_ending_in_1234"},
                                "payer": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string", "example": "Alex Smith"},
                                        "email": {"type": "string", "example": "alex.smith@example.com"},
                                        "identification": {
                                            "type": "object",
                                            "properties": {
                                                "type": {"type": "string", "example": "Driver's License"},
                                                "number": {"type": "string", "example": "D12345678"},
                                            },
                                            "required": ["type", "number"],
                                        },
                                    },
                                    "required": ["name", "email", "identification"],
                                },
                            },
                            "required": ["transaction_amount", "description", "payment_method_id", "payer"],
                        }
                    }
                }
            }
        }
        mock_service = Mock(call_processPayment=call_processPayment)
        mock_service.raw_element = {}
        openapi_mock.return_value = mock_service

        messages = [ChatMessage.from_assistant(mock_message)]
        result = connector.run(messages=messages, service_openapi_spec=spec)

        # verify call went through on the wire
        mock_service.call_processPayment.assert_called_once_with(
            data={
                "transaction_amount": 150.75,
                "description": "Monthly subscription fee",
                "payment_method_id": "visa_ending_in_1234",
                "payer": {
                    "name": "Alex Smith",
                    "email": "alex.smith@example.com",
                    "identification": {"type": "Driver's License", "number": "D12345678"},
                },
            }
        )

        response = json.loads(result["service_response"][0].text)
        assert response == {"result": "accepted"}

    @patch("haystack.components.connectors.openapi_service.OpenAPI")
    def test_run_with_request_params_missing_in_invocation_args(self, openapi_mock, test_files_path):
        connector = OpenAPIServiceConnector()
        spec_path = test_files_path / "yaml" / "openapi_greeting_service.yml"
        with open(spec_path, "r") as file:
            spec = json.loads(file.read())
        mock_message = json.dumps(
            [
                {
                    "id": "call_NJr1NBz2Th7iUWJpRIJZoJIA",
                    "function": {"arguments": '{"message": "Hello"}', "name": "greet"},
                    "type": "function",
                }
            ]
        )
        call_greet = Mock(return_value=Mock(_raw_data="Hello, John"))
        call_greet.operation.__self__ = Mock()
        call_greet.operation.__self__.raw_element = {
            "parameters": [{"name": "name", "required": True}],
            "requestBody": {
                "content": {"application/json": {"schema": {"properties": {"message": {"type": "string"}}}}}
            },
        }

        mock_service = Mock(call_greet=call_greet)
        mock_service.raw_element = {}
        openapi_mock.return_value = mock_service

        messages = [ChatMessage.from_assistant(mock_message)]
        with pytest.raises(ValueError, match="Missing parameter: 'name' required for the 'greet' operation."):
            connector.run(messages=messages, service_openapi_spec=spec)

    @patch("haystack.components.connectors.openapi_service.OpenAPI")
    def test_run_with_body_properties_missing_in_invocation_args(self, openapi_mock, test_files_path):
        connector = OpenAPIServiceConnector()
        spec_path = test_files_path / "yaml" / "openapi_greeting_service.yml"
        with open(spec_path, "r") as file:
            spec = json.loads(file.read())
        mock_message = json.dumps(
            [
                {
                    "id": "call_NJr1NBz2Th7iUWJpRIJZoJIA",
                    "function": {"arguments": '{"name": "John"}', "name": "greet"},
                    "type": "function",
                }
            ]
        )
        call_greet = Mock(return_value=Mock(_raw_data="Hello, John"))
        call_greet.operation.__self__ = Mock()
        call_greet.operation.__self__.raw_element = {
            "parameters": [{"name": "name"}],
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {"properties": {"message": {"type": "string"}}, "required": ["message"]}
                    }
                }
            },
        }

        mock_service = Mock(call_greet=call_greet)
        mock_service.raw_element = {}
        openapi_mock.return_value = mock_service

        messages = [ChatMessage.from_assistant(mock_message)]
        with pytest.raises(
            ValueError, match="Missing requestBody parameter: 'message' required for the 'greet' operation."
        ):
            connector.run(messages=messages, service_openapi_spec=spec)

    def test_serialization(self):
        for test_val in ("myvalue", True, None):
            openapi_service_connector = OpenAPIServiceConnector(test_val)
            serialized = openapi_service_connector.to_dict()
            assert serialized["init_parameters"]["ssl_verify"] == test_val
            deserialized = OpenAPIServiceConnector.from_dict(serialized)
            assert deserialized.ssl_verify == test_val
