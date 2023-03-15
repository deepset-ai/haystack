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


def test_run_pipeline_generic_correct_payload_data_only(client):
    response = client.post(
        url="/pipelines/run/haystack_pipeline_1", data=json.dumps({"data": {"value": 1}, "parameters": {}})
    )
    assert 200 == response.status_code
    response_json = response.json()
    assert response_json == {"value": 16}

    response = client.post(
        url="/pipelines/run/haystack_pipeline_1", data=json.dumps({"data": {"value": 10}, "parameters": {}})
    )
    assert 200 == response.status_code
    response_json = response.json()
    assert response_json == {"value": 34}


def test_run_pipeline_generic_correct_payload_data_and_params_for_nonexisting_node(client):
    response = client.post(
        url="/pipelines/run/haystack_pipeline_1",
        data=json.dumps({"data": {"value": 1}, "parameters": {"non_existing": {"wrong": "stuff"}}}),
    )
    assert 404 == response.status_code
    response_json = response.json()
    assert response_json["errors"][0].startswith("Node named 'non_existing' not found.")


def test_run_pipeline_generic_correct_payload_data_and_params(client):
    response = client.post(
        url="/pipelines/run/haystack_pipeline_1",
        data=json.dumps({"data": {"value": 1}, "parameters": {"third_addition": {"add": 2}}}),
    )
    assert 200 == response.status_code
    response_json = response.json()
    assert response_json == {"value": 17}

    response = client.post(
        url="/pipelines/run/haystack_pipeline_1",
        data=json.dumps(
            {"data": {"value": 1}, "parameters": {"first_addition": {"add": 10}, "third_addition": {"add": 100}}}
        ),
    )
    assert 200 == response.status_code
    response_json = response.json()
    assert response_json == {"value": 123}
