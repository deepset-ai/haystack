from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def get_test_client_and_override_dependencies():
    import os
    os.environ["PIPELINE_YAML_PATH"] = "samples/pipeline/test_pipeline.yaml"
    os.environ["QUERY_PIPELINE_NAME"] = "query_pipeline"
    os.environ["INDEXING_PIPELINE_NAME"] = "indexing_pipeline"

    from rest_api.application import app
    return TestClient(app)


@pytest.mark.slow
@pytest.mark.elasticsearch
@pytest.mark.parametrize("reader", ["farm"], indirect=True)
@pytest.mark.parametrize("document_store", ["elasticsearch"], indirect=True)
def test_api(reader, document_store):
    client = get_test_client_and_override_dependencies()

    # test file upload API
    file_to_upload = {'file': Path("samples/pdf/sample_pdf_1.pdf").open('rb')}
    response = client.post(url="/file-upload", files=file_to_upload, data={"meta": '{"meta_key": "meta_value"}'})
    assert 200 == response.status_code

    # test query API
    query_with_no_filter_value = {"query": "Who made the PDF specification?"}
    response = client.post(url="/query", json=query_with_no_filter_value)
    assert 200 == response.status_code
    response_json = response.json()
    assert response_json["answers"][0]["answer"] == "Adobe Systems"
    document_id = response_json["answers"][0]["document_id"]

    query_with_filter = {"query": "Who made the PDF specification?", "filters": {"meta_key": "meta_value"}}
    response = client.post(url="/query", json=query_with_filter)
    assert 200 == response.status_code
    response_json = response.json()
    assert response_json["answers"][0]["answer"] == "Adobe Systems"

    query_with_filter_list = {"query": "Who made the PDF specification?", "filters": {"meta_key": ["meta_value", "another_value"]}}
    response = client.post(url="/query", json=query_with_filter_list)
    assert 200 == response.status_code
    response_json = response.json()
    assert response_json["answers"][0]["answer"] == "Adobe Systems"

    query_with_invalid_filter = {"query": "Who made the PDF specification?", "filters": {"meta_key": "invalid_value"}}
    response = client.post(url="/query", json=query_with_invalid_filter)
    assert 200 == response.status_code
    response_json = response.json()
    assert len(response_json["answers"]) == 0

    # test write feedback
    feedback = {
        "question": "Who made the PDF specification?",
        "is_correct_answer": True,
        "document_id": document_id,
        "is_correct_document": True,
        "answer": "Adobe Systems",
        "offset_start_in_doc": 60
    }
    response = client.post(url="/feedback", json=feedback)
    assert 200 == response.status_code

    # test export feedback
    feedback_urls = [
        "/export-feedback?full_document_context=true",
        "/export-feedback?full_document_context=false&context_size=50",
        "/export-feedback?full_document_context=false&context_size=50000",
    ]
    for url in feedback_urls:
        response = client.get(url=url, json=feedback)
        response_json = response.json()
        context = response_json["data"][0]["paragraphs"][0]["context"]
        answer_start = response_json["data"][0]["paragraphs"][0]["qas"][0]["answers"][0]["answer_start"]
        answer = response_json["data"][0]["paragraphs"][0]["qas"][0]["answers"][0]["text"]
        assert context[answer_start:answer_start+len(answer)] == answer

