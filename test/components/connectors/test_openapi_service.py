import json
import os
import time
from typing import Dict, Any
from unittest.mock import MagicMock, patch
from urllib.parse import urlparse

import flaky
import pytest
import requests
from flask import Flask, jsonify, request
from flask.testing import FlaskClient
from openapi3 import OpenAPI
from openapi3.schemas import Model
from requests import Response
from requests.adapters import HTTPAdapter

from haystack import Pipeline
from haystack.components.connectors import OpenAPIServiceConnector
from haystack.components.converters import OpenAPIServiceToFunctions, OutputAdapter
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
    # arguments, see below, are always passed as a string representation of a JSON object
    params = '{"basehead": "' + basehead + '", "owner": "deepset-ai", "repo": "haystack"}'
    payload_json = [
        {
            "id": "call_NJr1NBz2Th7iUWJpRIJZoJIA",
            "function": {"arguments": params, "name": "compare_branches"},
            "type": "function",
        }
    ]

    return json.dumps(payload_json)


"""
In OpenAPI Spec, REST parameters are defined either as URL parameters (path and query) under the parameters section
or as JSON payload in the requestBody section of the spec. URL parameters include path parameters, defined within curly
braces in the URL path (in: path), and query parameters, appended after a ? (in: query). JSON payload, suitable
for complex data, is specified in the requestBody for methods like POST, PUT, PATCH, or DELETE. OAS also
supports mixed parameters, allowing both URL parameters and JSON payload in a single request.

Let's prepare three Flask apps for testing the OpenAPI service connector with different types of parameters:

- greet_mix_params_body: a Flask app with a single POST endpoint /greet/<name> that accepts a JSON payload with a message
    and returns a greeting message with the name from the URL path and the message from the JSON payload.

- greet_params_only: a Flask app with a single GET endpoint /greet-params/<name> that accepts a URL parameter and returns
    a greeting message with the name from the URL path.

- greet_request_body_only: a Flask app with a single POST endpoint /greet-body that accepts a JSON payload with a name
    and a message and returns a greeting message with the name and the message from the JSON payload.


The corresponding OpenAPI specs for these endpoints are defined in openapi_greeting_service.yml in this directory.
"""


def greet_mix_params_body():
    app = Flask(__name__)

    @app.route("/greet/<name>", methods=["POST"])
    def greet(name):
        data = request.get_json()
        message = data.get("message")
        greeting = f"{message}, {name} from mix_params_body!"
        return jsonify(greeting=greeting)

    return app


def greet_params_only():
    app = Flask(__name__)

    @app.route("/greet-params/<name>", methods=["GET"])
    def greet_params(name):
        # Use the URL parameter for the greeting
        greeting = f"Hello, {name} from params_only!"
        return jsonify(greeting=greeting)

    return app


def greet_request_body_only():
    app = Flask(__name__)

    @app.route("/greet-body", methods=["POST"])
    def greet_request_body():
        data = request.get_json()
        name = data.get("name")
        message = data.get("message")
        greeting = f"{message}, {name} from request_body_only!"
        return jsonify(greeting=greeting)

    return app


@pytest.fixture
def flask_client(request):
    app_factory = request.param
    app = app_factory()
    app.config.update(TESTING=True)
    with app.test_client() as client:
        yield client


class FlaskTestClientAdapter(HTTPAdapter):
    """
    This class is a custom adapter for the requests library that allows us to use the Flask test client
    to simulate HTTP requests. Requests made with this adapter will be handled by the Flask test client
    and sent to the Flask application, instead of being sent over the network.
    """

    def __init__(self, flask_test_client, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flask_test_client = flask_test_client

    def send(self, request, **kwargs):
        url_details = urlparse(request.url)
        path = url_details.path
        if url_details.query:
            path += "?" + url_details.query

        # Determine the method (GET, POST, etc.) and perform the request with the Flask test client
        method = request.method.lower()
        content_type = request.headers.get("Content-Type", "")

        body = None
        if "application/json" in content_type and request.body:
            body = json.loads(request.body)

        # Now we invoke Flask test client instead of the actual HTTP request
        if body is not None:
            response = getattr(self.flask_test_client, method)(path, json=body)
        else:
            response = getattr(self.flask_test_client, method)(path, data=request.body)

        # Copy Flask test client response in requests.Response object
        req_response = Response()
        req_response.status_code = response.status_code
        req_response.headers = response.headers

        # Handle JSON content specifically, otherwise, transfer the data as-is
        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type:
            req_response._content = json.dumps(response.get_json()).encode("utf-8")
        else:
            req_response._content = response.data

        return req_response


@pytest.fixture
def patched_requests_session(flask_client):
    session_with_adapter = requests.Session()
    session_with_adapter.mount("http://localhost", FlaskTestClientAdapter(flask_test_client=flask_client))

    # Patch requests.Session to return the custom session
    with patch("requests.Session", return_value=session_with_adapter) as mock_session:
        yield mock_session


@pytest.fixture
def pipeline_with_openapi_service():
    cf = {"json_loads": lambda s: json.loads(s) if isinstance(s, str) else json.loads(str(s))}

    pipe = Pipeline()
    pipe.add_component("spec_to_functions", OpenAPIServiceToFunctions())
    pipe.add_component("openapi_container", OpenAPIServiceConnector())
    pipe.add_component("a1", OutputAdapter("{{documents[0].content | json_loads}}", Dict[str, Any], cf))
    pipe.add_component("a2", OutputAdapter("{{documents[0].meta['spec']}}", Dict[str, Any], cf))

    pipe.connect("spec_to_functions.documents", "a1")
    pipe.connect("spec_to_functions.documents", "a2")
    pipe.connect("a2", "openapi_container.service_openapi_spec")

    return pipe


@pytest.fixture
def openapi_fc_message(request):
    function_name = request.param
    msg = [
        {
            "id": "call_UOEigRO94Kryftz3MQJ9CWF2",
            "function": {"arguments": json.dumps({"message": "Hello", "name": "John"}), "name": function_name},
            "type": "function",
            "role": "assistant",
        }
    ]
    return msg


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

    @flaky.flaky(max_runs=5, rerun_filter=lambda *_: time.sleep(5))
    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("GITHUB_TOKEN"), reason="GITHUB_TOKEN is not set")
    def test_run(self, genuine_fc_message, test_files_path):
        openapi_service = OpenAPIServiceConnector()

        open_api_spec_path = test_files_path / "json" / "github_compare_branch_openapi_spec.json"
        with open(open_api_spec_path, "r") as file:
            github_compare_schema = json.load(file)
        messages = [ChatMessage.from_assistant(genuine_fc_message)]

        # genuine call to the GitHub OpenAPI service
        result = openapi_service.run(messages, github_compare_schema, os.getenv("GITHUB_TOKEN"))
        assert result

        # load json from the service response
        service_payload = json.loads(result["service_response"][0].content)

        # verify that the service response contains the expected fields
        assert "url" in service_payload and "files" in service_payload

    @pytest.mark.parametrize("flask_client", [greet_mix_params_body], indirect=True)
    @pytest.mark.parametrize("openapi_fc_message", ["greet"], indirect=True)
    def test_mix_body_params(
        self,
        flask_client: FlaskClient,
        patched_requests_session,
        pipeline_with_openapi_service,
        openapi_fc_message,
        test_files_path,
    ):
        # We pretend FC message openapi_fc_message was generated by LLM, we just need to set the function
        # name to greet using fixture openapi_fc_message
        # do a real invoke into the Flask app via the pipeline
        result = pipeline_with_openapi_service.run(
            data={
                "openapi_container": {"messages": [ChatMessage.from_assistant(json.dumps(openapi_fc_message))]},
                "spec_to_functions": {"sources": [test_files_path / "yaml" / "openapi_greeting_service.yml"]},
            }
        )

        response = json.loads(result["openapi_container"]["service_response"][0].content)
        assert response["greeting"] == "Hello, John from mix_params_body!"

    @pytest.mark.parametrize("flask_client", [greet_params_only], indirect=True)
    @pytest.mark.parametrize("openapi_fc_message", ["greetParams"], indirect=True)
    def test_params(
        self,
        flask_client: FlaskClient,
        patched_requests_session,
        pipeline_with_openapi_service,
        openapi_fc_message,
        test_files_path,
    ):
        # We pretend FC message openapi_fc_message was generated by LLM, we just need to set the function
        # name to greetParams using fixture openapi_fc_message
        # do a real invoke into the Flask app via the pipeline
        result = pipeline_with_openapi_service.run(
            data={
                "openapi_container": {"messages": [ChatMessage.from_assistant(json.dumps(openapi_fc_message))]},
                "spec_to_functions": {"sources": [test_files_path / "yaml" / "openapi_greeting_service.yml"]},
            }
        )

        response = json.loads(result["openapi_container"]["service_response"][0].content)
        assert response["greeting"] == "Hello, John from params_only!"

    @pytest.mark.parametrize("flask_client", [greet_request_body_only], indirect=True)
    @pytest.mark.parametrize("openapi_fc_message", ["greetBody"], indirect=True)
    def test_request_body(
        self,
        flask_client: FlaskClient,
        patched_requests_session,
        pipeline_with_openapi_service,
        openapi_fc_message,
        test_files_path,
    ):
        # We pretend FC message openapi_fc_message was generated by LLM, we just need to set the function
        # name to greetBody using fixture openapi_fc_message
        # do a real invoke into the Flask app via the pipeline
        result = pipeline_with_openapi_service.run(
            data={
                "openapi_container": {"messages": [ChatMessage.from_assistant(json.dumps(openapi_fc_message))]},
                "spec_to_functions": {"sources": [test_files_path / "yaml" / "openapi_greeting_service.yml"]},
            }
        )

        response = json.loads(result["openapi_container"]["service_response"][0].content)
        assert response["greeting"] == "Hello, John from request_body_only!"
