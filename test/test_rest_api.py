import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from haystack import Answer, Span, Label

from rest_api.application import app
from rest_api.application import DOCUMENT_STORE


@pytest.fixture(scope="session")
def client() -> TestClient:
    os.environ["PIPELINE_YAML_PATH"] = str((Path(__file__).parent / "samples"/"pipeline"/"test_pipeline.yaml").absolute())
    os.environ["QUERY_PIPELINE_NAME"] = "query_pipeline"
    os.environ["INDEXING_PIPELINE_NAME"] = "indexing_pipeline"
    return TestClient(app)

@pytest.fixture(scope="session")
def populated_client(client: TestClient) -> TestClient:
    file_to_upload = {'files': (Path(__file__).parent / "samples"/"pdf"/"sample_pdf_1.pdf").open('rb')}
    client.post(url="/file-upload", files=file_to_upload, data={"meta": '{"meta_key": "meta_value"}'})
    yield client
    client.post(url="/documents/delete_by_filters", data={"meta_key": ["meta_value"]})



def test_file_upload(client: TestClient):
    file_to_upload = {'files': (Path(__file__).parent / "samples"/"pdf"/"sample_pdf_1.pdf").open('rb')}
    response = client.post(url="/file-upload", files=file_to_upload, data={"meta": '{"meta_key": "meta_value"}'})
    assert 200 == response.status_code

def test_delete_documents(client: TestClient):
    file_to_upload = {'files': (Path(__file__).parent / "samples"/"pdf"/"sample_pdf_1.pdf").open('rb')}
    response = client.post(url="/file-upload", files=file_to_upload, data={"meta": '{"meta_key": "meta_value"}'})

    client.post(url="/documents/delete_by_filters", data={"meta_key": ["meta_value"]})
    assert 200 == response.status_code

def test_query_with_no_filter(populated_client: TestClient):
    query_with_no_filter_value = {"query": "Who made the PDF specification?"}
    response = populated_client.post(url="/query", json=query_with_no_filter_value)
    assert 200 == response.status_code
    response_json = response.json()
    assert response_json["answers"][0]["answer"] == "Adobe Systems"

def test_query_with_one_filter(populated_client: TestClient):
    query_with_filter = {"query": "Who made the PDF specification?", "params": {"filters": {"meta_key": "meta_value"}}}
    response = populated_client.post(url="/query", json=query_with_filter)
    assert 200 == response.status_code
    response_json = response.json()
    assert response_json["answers"][0]["answer"] == "Adobe Systems"

def test_query_with_filter_list(populated_client: TestClient):
    query_with_filter_list = {
        "query": "Who made the PDF specification?",
        "params": {"filters": {"meta_key": ["meta_value", "another_value"]}}
    }
    response = populated_client.post(url="/query", json=query_with_filter_list)
    assert 200 == response.status_code
    response_json = response.json()
    assert response_json["answers"][0]["answer"] == "Adobe Systems"

def test_query_with_invalid_filter(populated_client: TestClient):
    query_with_invalid_filter = {
        "query": "Who made the PDF specification?", "params": {"filters": {"meta_key": "invalid_value"}}
    }
    response = populated_client.post(url="/query", json=query_with_invalid_filter)
    assert 200 == response.status_code
    response_json = response.json()
    assert len(response_json["answers"]) == 0

def test_write_feedback(populated_client: TestClient):
    response = populated_client.post(url="/query", json={"query": "Who made the PDF specification?"})
    response_json = response.json()
    document_id = response_json["answers"][0]["document_id"]

    # feedback = {
    #     "question": "Who made the PDF specification?",
    #     "is_correct_answer": True,
    #     "document_id": document_id,
    #     "is_correct_document": True,
    #     "answer": "Adobe Systems",
    #     "offset_start_in_doc": 60
    # }

    # test write feedback
    feedback = {
        "id": "abc",
        "query": "Who made the PDF specification?",
        "document": {
            "content": "A sample PDF file\n\nHistory and standardization\nFormat (PDF) Adobe Systems made the PDF specification available free of charge in 1993. In the early years PDF was popular mainly in desktop publishing workflows, and competed with a variety of formats such as DjVu, Envoy, Common Ground Digital Paper, Farallon Replica and even Adobe's own PostScript format. PDF was a proprietary format controlled by Adobe until it was released as an open standard on July 1, 2008, and published by the International Organization for Standardization as ISO 32000-1:2008, at which time control of the specification passed to an ISO Committee of volunteer industry experts. In 2008, Adobe published a Public Patent License to ISO 32000-1 granting royalty-free rights for all patents owned by Adobe that are necessary to make, use, sell, and distribute PDF-compliant implementations. PDF 1.7, the sixth edition of the PDF specification that became ISO 32000-1, includes some proprietary technologies defined only by Adobe, such as Adobe XML Forms Architecture (XFA) and JavaScript extension for Acrobat, which are referenced by ISO 32000-1 as normative and indispensable for the full implementation of the ISO 32000-1 specification. These proprietary technologies are not standardized and their specification is published only on Adobes website. Many of them are also not supported by popular third-party implementations of PDF. Column 1",
            "content_type": "text",
            "score": None,
            "id": "fc18c987a8312e72a47fb1524f230bb0",
            "meta": {}
        },
        "answer":
            {
                "answer": "Adobe Systems",
                "type": "extractive",
                "context": "A sample PDF file\n\nHistory and standardization\nFormat (PDF) Adobe Systems made the PDF specification available free of charge in 1993. In the early ye",
                "offsets_in_context": [{"start": 60, "end": 73}],
                "offsets_in_document": [{"start": 60, "end": 73}],
                "document_id": "fc18c987a8312e72a47fb1524f230bb0"
            },
        "is_correct_answer": True,
        "is_correct_document": True,
        "origin": "user-feedback",
        "pipeline_id": "some-123",
    }

    response = populated_client.post(url="/feedback", json=feedback)
    assert 200 == response.status_code

def test_export_feedback(populated_client: TestClient):
    response = populated_client.post(url="/query", json={"query": "Who made the PDF specification?"})
    response_json = response.json()
    document_id = response_json["answers"][0]["document_id"]

    feedback = {
        "question": "Who made the PDF specification?",
        "is_correct_answer": True,
        "document_id": document_id,
        "is_correct_document": True,
        "answer": "Adobe Systems",
        "offset_start_in_doc": 60
    }
    feedback_urls = [
        "/export-feedback?full_document_context=true",
        "/export-feedback?full_document_context=false&context_size=50",
        "/export-feedback?full_document_context=false&context_size=50000",
    ]
    for url in feedback_urls:
        response = populated_client.get(url=url, json=feedback)
        response_json = response.json()
        context = response_json["data"][0]["paragraphs"][0]["context"]
        answer_start = response_json["data"][0]["paragraphs"][0]["qas"][0]["answers"][0]["answer_start"]
        answer = response_json["data"][0]["paragraphs"][0]["qas"][0]["answers"][0]["text"]
        assert context[answer_start:answer_start+len(answer)] == answer

