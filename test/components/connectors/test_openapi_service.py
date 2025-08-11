# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests
from openapi3 import OpenAPI

from haystack import Pipeline
from haystack.components.connectors import OpenAPIServiceConnector
from haystack.components.converters.openapi_functions import OpenAPIServiceToFunctions
from haystack.components.converters.output_adapter import OutputAdapter
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.dataclasses.byte_stream import ByteStream


@pytest.fixture
def openapi_service_mock():
    return MagicMock(spec=OpenAPI)


class TestOpenAPIServiceConnector:
    @pytest.fixture
    def connector(self):
        return OpenAPIServiceConnector()

    def test_run_without_tool_calls(self, connector):
        message = ChatMessage.from_assistant(text="Just a regular message")
        with pytest.raises(ValueError, match="has no tool calls"):
            connector.run(messages=[message], service_openapi_spec={})

    def test_run_with_non_assistant_message(self, connector):
        message = ChatMessage.from_user(text="User message")
        with pytest.raises(ValueError, match="is not from the assistant"):
            connector.run(messages=[message], service_openapi_spec={})

    def test_authenticate_service_missing_authentication_token(self, connector, openapi_service_mock):
        security_schemes_dict = {
            "components": {"securitySchemes": {"apiKey": {"in": "header", "name": "x-api-key", "type": "apiKey"}}}
        }
        openapi_service_mock.raw_element = security_schemes_dict

        with pytest.raises(ValueError, match="requires authentication but no credentials were provided"):
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
        openapi_service_mock.authenticate.assert_called_once_with("apiKey", "some_fake_token")

    def test_authenticate_service_having_authentication_dict(self, connector, openapi_service_mock):
        security_schemes_dict = {
            "components": {"securitySchemes": {"apiKey": {"in": "header", "name": "x-api-key", "type": "apiKey"}}}
        }
        openapi_service_mock.raw_element = security_schemes_dict
        openapi_service_mock.components.securitySchemes.raw_element = {
            "apiKey": {"in": "header", "name": "x-api-key", "type": "apiKey"}
        }
        connector._authenticate_service(openapi_service_mock, {"apiKey": "some_fake_token"})
        openapi_service_mock.authenticate.assert_called_once_with("apiKey", "some_fake_token")

    def test_authenticate_service_having_unsupported_auth(self, connector, openapi_service_mock):
        security_schemes_dict = {"components": {"securitySchemes": {"oauth2": {"type": "oauth2"}}}}
        openapi_service_mock.raw_element = security_schemes_dict
        openapi_service_mock.components.securitySchemes.raw_element = {"oauth2": {"type": "oauth2"}}
        with pytest.raises(ValueError, match="Check the service configuration and credentials"):
            connector._authenticate_service(openapi_service_mock, {"apiKey": "some_fake_token"})

    @patch("haystack.components.connectors.openapi_service.OpenAPI")
    def test_run_with_parameters(self, openapi_mock):
        connector = OpenAPIServiceConnector()
        tool_call = ToolCall(
            tool_name="compare_branches",
            arguments={"basehead": "main...some_branch", "owner": "deepset-ai", "repo": "haystack"},
        )
        message = ChatMessage.from_assistant(tool_calls=[tool_call])

        # Mock the OpenAPI service
        call_compare_branches = Mock(return_value={"status": "success"})
        call_compare_branches.operation.__self__ = Mock()
        call_compare_branches.operation.__self__.raw_element = {
            "parameters": [{"name": "basehead"}, {"name": "owner"}, {"name": "repo"}]
        }
        mock_service = Mock(call_compare_branches=call_compare_branches, raw_element={})
        openapi_mock.return_value = mock_service

        result = connector.run(messages=[message], service_openapi_spec={})

        # Verify the service call
        mock_service.call_compare_branches.assert_called_once_with(
            parameters={"basehead": "main...some_branch", "owner": "deepset-ai", "repo": "haystack"}, raw_response=True
        )
        assert json.loads(result["service_response"][0].text) == {"status": "success"}

    @patch("haystack.components.connectors.openapi_service.OpenAPI")
    def test_run_with_request_body(self, openapi_mock):
        connector = OpenAPIServiceConnector()
        tool_call = ToolCall(tool_name="greet", arguments={"message": "Hello", "name": "John"})
        message = ChatMessage.from_assistant(tool_calls=[tool_call])

        # Mock the OpenAPI service
        call_greet = Mock(return_value="Hello, John")
        call_greet.operation.__self__ = Mock()
        call_greet.operation.__self__.raw_element = {
            "parameters": [{"name": "name"}],
            "requestBody": {
                "content": {"application/json": {"schema": {"properties": {"message": {"type": "string"}}}}}
            },
        }
        mock_service = Mock(call_greet=call_greet, raw_element={})
        openapi_mock.return_value = mock_service

        result = connector.run(messages=[message], service_openapi_spec={})

        # Verify the service call
        mock_service.call_greet.assert_called_once_with(
            parameters={"name": "John"}, data={"message": "Hello"}, raw_response=True
        )
        assert json.loads(result["service_response"][0].text) == "Hello, John"

    @patch("haystack.components.connectors.openapi_service.OpenAPI")
    def test_run_with_missing_required_parameter(self, openapi_mock):
        connector = OpenAPIServiceConnector()
        tool_call = ToolCall(
            tool_name="greet",
            arguments={"message": "Hello"},  # missing required 'name' parameter
        )
        message = ChatMessage.from_assistant(tool_calls=[tool_call])

        # Mock the OpenAPI service
        call_greet = Mock()
        call_greet.operation.__self__ = Mock()
        call_greet.operation.__self__.raw_element = {
            "parameters": [{"name": "name", "required": True}],
            "requestBody": {
                "content": {"application/json": {"schema": {"properties": {"message": {"type": "string"}}}}}
            },
        }
        mock_service = Mock(call_greet=call_greet, raw_element={})
        openapi_mock.return_value = mock_service

        with pytest.raises(ValueError, match="Missing parameter: 'name' required for the 'greet' operation"):
            connector.run(messages=[message], service_openapi_spec={})

    @patch("haystack.components.connectors.openapi_service.OpenAPI")
    def test_run_with_missing_required_parameters_in_request_body(self, openapi_mock):
        """
        Test that the connector raises a ValueError when the request body is missing required parameters.
        """
        connector = OpenAPIServiceConnector()
        tool_call = ToolCall(
            tool_name="post_message",
            arguments={"recipient": "John"},  # only providing URL parameter, no request body data
        )
        message = ChatMessage.from_assistant(tool_calls=[tool_call])

        # Mock the OpenAPI service
        call_post_message = Mock()
        call_post_message.operation.__self__ = Mock()
        call_post_message.operation.__self__.raw_element = {
            "parameters": [{"name": "recipient"}],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "required": ["message"],  # Mark message as required in schema
                            "properties": {"message": {"type": "string"}},
                        }
                    }
                },
            },
        }
        mock_service = Mock(call_post_message=call_post_message, raw_element={})
        openapi_mock.return_value = mock_service

        with pytest.raises(
            ValueError, match="Missing requestBody parameter: 'message' required for the 'post_message' operation"
        ):
            connector.run(messages=[message], service_openapi_spec={})

        # Verify that the service was never called since validation failed
        call_post_message.assert_not_called()

    def test_serialization(self):
        for test_val in ("myvalue", True, None):
            connector = OpenAPIServiceConnector(test_val)
            serialized = connector.to_dict()
            assert serialized["init_parameters"]["ssl_verify"] == test_val
            deserialized = OpenAPIServiceConnector.from_dict(serialized)
            assert deserialized.ssl_verify == test_val

    def test_serde_in_pipeline(self):
        """
        Test serialization/deserialization of OpenAPIServiceConnector in a Pipeline,
        including YAML conversion and detailed dictionary validation
        """
        connector = OpenAPIServiceConnector(ssl_verify=True)

        pipeline = Pipeline()
        pipeline.add_component("connector", connector)

        pipeline_dict = pipeline.to_dict()
        assert pipeline_dict == {
            "metadata": {},
            "max_runs_per_component": 100,
            "connection_type_validation": True,
            "components": {
                "connector": {
                    "type": "haystack.components.connectors.openapi_service.OpenAPIServiceConnector",
                    "init_parameters": {"ssl_verify": True},
                }
            },
            "connections": [],
        }

        pipeline_yaml = pipeline.dumps()
        new_pipeline = Pipeline.loads(pipeline_yaml)
        assert new_pipeline == pipeline

    @pytest.mark.skipif(not os.getenv("SERPERDEV_API_KEY"), reason="SERPERDEV_API_KEY is not set")
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY is not set")
    @pytest.mark.integration
    def test_run_live(self):
        # An OutputAdapter filter we'll use to setup function calling
        def prepare_fc_params(openai_functions_schema: dict[str, Any]) -> dict[str, Any]:
            return {
                "tools": [{"type": "function", "function": openai_functions_schema}],
                "tool_choice": {"type": "function", "function": {"name": openai_functions_schema["name"]}},
            }

        pipe = Pipeline()
        pipe.add_component("spec_to_functions", OpenAPIServiceToFunctions())
        pipe.add_component("functions_llm", OpenAIChatGenerator(model="gpt-4o-mini"))

        pipe.add_component("openapi_container", OpenAPIServiceConnector())
        pipe.add_component(
            "prepare_fc_adapter",
            OutputAdapter("{{functions[0] | prepare_fc}}", dict[str, Any], {"prepare_fc": prepare_fc_params}),
        )
        pipe.add_component("openapi_spec_adapter", OutputAdapter("{{specs[0]}}", dict[str, Any], unsafe=True))
        pipe.add_component(
            "final_prompt_adapter",
            OutputAdapter("{{system_message + service_response}}", list[ChatMessage], unsafe=True),
        )
        pipe.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini", streaming_callback=print_streaming_chunk))

        pipe.connect("spec_to_functions.functions", "prepare_fc_adapter.functions")
        pipe.connect("spec_to_functions.openapi_specs", "openapi_spec_adapter.specs")
        pipe.connect("prepare_fc_adapter", "functions_llm.generation_kwargs")
        pipe.connect("functions_llm.replies", "openapi_container.messages")
        pipe.connect("openapi_spec_adapter", "openapi_container.service_openapi_spec")
        pipe.connect("openapi_container.service_response", "final_prompt_adapter.service_response")
        pipe.connect("final_prompt_adapter", "llm.messages")

        serperdev_spec = requests.get(
            "https://gist.githubusercontent.com/vblagoje/241a000f2a77c76be6efba71d49e2856/raw/722ccc7fe6170a744afce3e3fb3a30fdd095c184/serper.json"
        ).json()
        system_prompt = requests.get("https://bit.ly/serper_dev_system").text

        query = "Why did Elon Musk sue OpenAI?"

        result = pipe.run(
            data={
                "functions_llm": {
                    "messages": [ChatMessage.from_system("Only do tool/function calling"), ChatMessage.from_user(query)]
                },
                "openapi_container": {"service_credentials": os.getenv("SERPERDEV_API_KEY")},
                "spec_to_functions": {"sources": [ByteStream.from_string(json.dumps(serperdev_spec))]},
                "final_prompt_adapter": {"system_message": [ChatMessage.from_system(system_prompt)]},
            }
        )
        assert isinstance(result["llm"]["replies"][0], ChatMessage)
        assert "Elon" in result["llm"]["replies"][0].text
        assert "OpenAI" in result["llm"]["replies"][0].text
