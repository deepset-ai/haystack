# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import Mock, patch

import pytest
from haystack import Pipeline
from haystack.utils import Secret
from haystack.components.connectors.openapi import OpenAPIConnector

# Mock OpenAPI spec for testing
MOCK_OPENAPI_SPEC = """
openapi: 3.0.0
info:
  title: Test API
  version: 1.0.0
paths:
  /search:
    get:
      operationId: search
      parameters:
        - name: q
          in: query
          required: true
          schema:
            type: string
"""


@pytest.fixture
def mock_client():
    with patch("haystack.components.connectors.openapi.OpenAPIClient") as mock:
        client_instance = Mock()
        mock.from_spec.return_value = client_instance
        yield client_instance


class TestOpenAPIConnector:
    def test_init(self, mock_client):
        # Test initialization with credentials and service_kwargs
        service_kwargs = {"allowed_operations": ["search"]}
        connector = OpenAPIConnector(
            openapi_spec=MOCK_OPENAPI_SPEC, credentials=Secret.from_token("test-token"), service_kwargs=service_kwargs
        )
        assert connector.openapi_spec == MOCK_OPENAPI_SPEC
        assert connector.credentials.resolve_value() == "test-token"
        assert connector.service_kwargs == service_kwargs

        # Test initialization without credentials and service_kwargs
        connector = OpenAPIConnector(openapi_spec=MOCK_OPENAPI_SPEC)
        assert connector.credentials is None
        assert connector.service_kwargs == {}

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-api-key")
        service_kwargs = {"allowed_operations": ["search"]}
        connector = OpenAPIConnector(
            openapi_spec=MOCK_OPENAPI_SPEC, credentials=Secret.from_env_var("ENV_VAR"), service_kwargs=service_kwargs
        )
        serialized = connector.to_dict()
        assert serialized == {
            "type": "haystack.components.connectors.openapi.OpenAPIConnector",
            "init_parameters": {
                "openapi_spec": MOCK_OPENAPI_SPEC,
                "credentials": {"env_vars": ["ENV_VAR"], "type": "env_var", "strict": True},
                "service_kwargs": service_kwargs,
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-api-key")
        service_kwargs = {"allowed_operations": ["search"]}
        data = {
            "type": "haystack.components.connectors.openapi.OpenAPIConnector",
            "init_parameters": {
                "openapi_spec": MOCK_OPENAPI_SPEC,
                "credentials": {"env_vars": ["ENV_VAR"], "type": "env_var", "strict": True},
                "service_kwargs": service_kwargs,
            },
        }
        connector = OpenAPIConnector.from_dict(data)
        assert connector.openapi_spec == MOCK_OPENAPI_SPEC
        assert connector.credentials == Secret.from_env_var("ENV_VAR")
        assert connector.service_kwargs == service_kwargs

    def test_run(self, mock_client):
        service_kwargs = {"allowed_operations": ["search"]}
        connector = OpenAPIConnector(
            openapi_spec=MOCK_OPENAPI_SPEC, credentials=Secret.from_token("test-token"), service_kwargs=service_kwargs
        )

        # Mock the response from the client
        mock_client.invoke.return_value = {"results": ["test result"]}

        # Test with arguments
        response = connector.run(operation_id="search", arguments={"q": "test query"})
        mock_client.invoke.assert_called_with({"name": "search", "arguments": {"q": "test query"}})
        assert response == {"response": {"results": ["test result"]}}

        # Test without arguments
        response = connector.run(operation_id="search")
        mock_client.invoke.assert_called_with({"name": "search", "arguments": {}})

    def test_in_pipeline(self, mock_client):
        mock_client.invoke.return_value = {"results": ["test result"]}

        connector = OpenAPIConnector(openapi_spec=MOCK_OPENAPI_SPEC, credentials=Secret.from_token("test-token"))

        pipe = Pipeline()
        pipe.add_component("api", connector)

        # Test pipeline execution
        results = pipe.run(data={"api": {"operation_id": "search", "arguments": {"q": "test query"}}})

        assert results == {"api": {"response": {"results": ["test result"]}}}

    def test_from_dict_fail_wo_env_var(self, monkeypatch):
        monkeypatch.delenv("ENV_VAR", raising=False)
        data = {
            "type": "haystack.components.connectors.openapi.OpenAPIConnector",
            "init_parameters": {
                "openapi_spec": MOCK_OPENAPI_SPEC,
                "credentials": {"env_vars": ["ENV_VAR"], "type": "env_var", "strict": True},
            },
        }
        with pytest.raises(ValueError, match="None of the .* environment variables are set"):
            OpenAPIConnector.from_dict(data)

    def test_serde_in_pipeline(self, monkeypatch):
        """
        Test serialization/deserialization of OpenAPIConnector in a Pipeline,
        including detailed dictionary validation
        """
        monkeypatch.setenv("API_KEY", "test-api-key")

        # Create connector with specific configuration
        connector = OpenAPIConnector(
            openapi_spec=MOCK_OPENAPI_SPEC,
            credentials=Secret.from_env_var("API_KEY"),
            service_kwargs={"allowed_operations": ["search"]},
        )

        # Create and configure pipeline
        pipeline = Pipeline()
        pipeline.add_component("api", connector)

        # Get pipeline dictionary and verify its structure
        pipeline_dict = pipeline.to_dict()
        assert pipeline_dict == {
            "metadata": {},
            "max_runs_per_component": 100,
            "components": {
                "api": {
                    "type": "haystack.components.connectors.openapi.OpenAPIConnector",
                    "init_parameters": {
                        "openapi_spec": MOCK_OPENAPI_SPEC,
                        "credentials": {"env_vars": ["API_KEY"], "type": "env_var", "strict": True},
                        "service_kwargs": {"allowed_operations": ["search"]},
                    },
                }
            },
            "connections": [],
        }

        # Test YAML serialization/deserialization
        pipeline_yaml = pipeline.dumps()
        new_pipeline = Pipeline.loads(pipeline_yaml)
        assert new_pipeline == pipeline

        # Verify the loaded pipeline's connector has the same configuration
        loaded_connector = new_pipeline.get_component("api")
        assert loaded_connector.openapi_spec == connector.openapi_spec
        assert loaded_connector.credentials == connector.credentials
        assert loaded_connector.service_kwargs == connector.service_kwargs


@pytest.mark.integration
class TestOpenAPIConnectorIntegration:
    @pytest.mark.skipif(
        not os.environ.get("SERPERDEV_API_KEY", None),
        reason="Export an env var called SERPERDEV_API_KEY to run this test.",
    )
    @pytest.mark.integration
    def test_serper_dev_integration(self):
        component = OpenAPIConnector(
            openapi_spec="https://bit.ly/serperdev_openapi", credentials=Secret.from_env_var("SERPERDEV_API_KEY")
        )
        response = component.run(operation_id="search", arguments={"q": "Who was Nikola Tesla?"})
        assert isinstance(response, dict)
        assert "response" in response

    @pytest.mark.skipif(
        not os.environ.get("GITHUB_TOKEN", None), reason="Export an env var called GITHUB_TOKEN to run this test."
    )
    @pytest.mark.integration
    def test_github_api_integration(self):
        component = OpenAPIConnector(
            openapi_spec="https://raw.githubusercontent.com/github/rest-api-description/main/descriptions/api.github.com/api.github.com.json",
            credentials=Secret.from_env_var("GITHUB_TOKEN"),
        )
        response = component.run(operation_id="search_repos", arguments={"q": "deepset-ai"})
        assert isinstance(response, dict)
        assert "response" in response
