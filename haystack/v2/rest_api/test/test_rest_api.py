import json

import pytest
from fastapi.testclient import TestClient

from haystack import __version__
from haystack.v2.rest_api.app import get_app


@pytest.fixture
def client():
    app = get_app()
    client = TestClient(app)
    return client


def test_ready(client):
    response = client.get(url="/ready")
    assert 200 == response.status_code
    response_json = response.json()
    assert response_json == True


def test_version(client):
    response = client.get(url="/version")
    assert 200 == response.status_code
    response_json = response.json()
    assert response_json == {"haystack": __version__}


def test_list_pipelines(client):
    response = client.get(url="/pipelines")
    assert 200 == response.status_code
    response_json = response.json()
    assert response_json == {
        "haystack_pipeline_1": {"type": "test pipeline", "author": "me"},
        "haystack_pipeline_2": {"type": "another test pipeline", "author": "you"},
    }


def test_run_pipeline_generic_wrong_name(client):
    response = client.post(
        url="/pipelines/run/non_existing", data=json.dumps({"data": {"value": "test"}, "parameters": {}})
    )
    assert 404 == response.status_code
    response_json = response.json()
    assert response_json == {
        "errors": [
            "Pipeline named 'non_existing' not found. Available pipelines: haystack_pipeline_1, haystack_pipeline_2"
        ]
    }


def test_run_pipeline_generic_wrong_payload_structure(client):
    response = client.post(url="/pipelines/run/haystack_pipeline_1", data=json.dumps({"not": "correct"}))
    assert 422 == response.status_code


def test_run_pipeline_generic_wrong_payload_content(client):
    response = client.post(
        url="/pipelines/run/haystack_pipeline_1", data=json.dumps({"data": {"unexpected": "test"}, "parameters": {}})
    )
    assert 500 == response.status_code
    response_json = response.json()
    assert response_json["errors"][0].startswith("Pipeline 'haystack_pipeline_1' failed. Exception: ")
