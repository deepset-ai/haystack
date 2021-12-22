import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


from rest_api.application import app

FEEDBACK={
        "id": "123",
        "query": "Who made the PDF specification?",
        "document": {
            "content": "A sample PDF file\n\nHistory and standardization\nFormat (PDF) Adobe Systems made the PDF specification available free of charge in 1993. In the early years PDF was popular mainly in desktop publishing workflows, and competed with a variety of formats such as DjVu, Envoy, Common Ground Digital Paper, Farallon Replica and even Adobe's own PostScript format. PDF was a proprietary format controlled by Adobe until it was released as an open standard on July 1, 2008, and published by the International Organization for Standardization as ISO 32000-1:2008, at which time control of the specification passed to an ISO Committee of volunteer industry experts. In 2008, Adobe published a Public Patent License to ISO 32000-1 granting royalty-free rights for all patents owned by Adobe that are necessary to make, use, sell, and distribute PDF-compliant implementations. PDF 1.7, the sixth edition of the PDF specification that became ISO 32000-1, includes some proprietary technologies defined only by Adobe, such as Adobe XML Forms Architecture (XFA) and JavaScript extension for Acrobat, which are referenced by ISO 32000-1 as normative and indispensable for the full implementation of the ISO 32000-1 specification. These proprietary technologies are not standardized and their specification is published only on Adobes website. Many of them are also not supported by popular third-party implementations of PDF. Column 1",
            "content_type": "text",
            "score": None,
            "id": "fc18c987a8312e72a47fb1524f230bb0",
            "meta": {},
            "embedding": None,
            "id_hash_keys": None
        },
        "answer":
            {
                "answer": "Adobe Systems",
                "type": "extractive",
                "context": "A sample PDF file\n\nHistory and standardization\nFormat (PDF) Adobe Systems made the PDF specification available free of charge in 1993. In the early ye",
                "offsets_in_context": [{"start": 60, "end": 73}],
                "offsets_in_document": [{"start": 60, "end": 73}],
                "document_id": "fc18c987a8312e72a47fb1524f230bb0",
                "meta": {},
                "score": None
            },
        "is_correct_answer": True,
        "is_correct_document": True,
        "origin": "user-feedback",
        "pipeline_id": "some-123",
    }


def exclude_no_answer(responses):
    responses["answers"] = [response for response in responses["answers"] if response.get("answer", None)]
    return responses


@pytest.mark.elasticsearch
@pytest.fixture(scope="session")
def client() -> TestClient:
    os.environ["PIPELINE_YAML_PATH"] = str((Path(__file__).parent / "samples"/"pipeline"/"test_pipeline.yaml").absolute())
    os.environ["INDEXING_PIPELINE_NAME"] = "indexing_text_pipeline"
    client = TestClient(app)
    yield client
    # Clean up
    client.post(url="/documents/delete_by_filters", data='{"filters": {}}')


@pytest.mark.elasticsearch
@pytest.fixture(scope="session")
def populated_client(client: TestClient) -> TestClient:
    client.post(url="/documents/delete_by_filters", data='{"filters": {}}')
    files_to_upload = [
        {'files': (Path(__file__).parent / "samples"/"pdf"/"sample_pdf_1.pdf").open('rb')},
        {'files': (Path(__file__).parent / "samples"/"pdf"/"sample_pdf_2.pdf").open('rb')}
    ]
    for index, fi in enumerate(files_to_upload):
        response = client.post(url="/file-upload", files=fi, data={"meta": f'{{"meta_key": "meta_value", "meta_index": "{index}"}}'})
        assert 200 == response.status_code
    yield client
    client.post(url="/documents/delete_by_filters", data='{"filters": {}}')


def test_get_documents():
    os.environ["PIPELINE_YAML_PATH"] = str((Path(__file__).parent / "samples"/"pipeline"/"test_pipeline.yaml").absolute())
    os.environ["INDEXING_PIPELINE_NAME"] = "indexing_text_pipeline"
    client = TestClient(app)

    # Clean up to make sure the docstore is empty
    client.post(url="/documents/delete_by_filters", data='{"filters": {}}')

    # Upload the files
    files_to_upload = [
        {'files': (Path(__file__).parent / "samples"/"docs"/"doc_1.txt").open('rb')},
        {'files': (Path(__file__).parent / "samples"/"docs"/"doc_2.txt").open('rb')}
    ]
    for index, fi in enumerate(files_to_upload):
        response = client.post(url="/file-upload", files=fi, data={"meta": f'{{"meta_key": "meta_value_get"}}'})
        assert 200 == response.status_code

    # Get the documents
    response = client.post(url="/documents/get_by_filters", data='{"filters": {"meta_key": ["meta_value_get"]}}')
    assert 200 == response.status_code
    response_json = response.json()

    # Make sure the right docs are found
    assert len(response_json) == 2
    names = [doc["meta"]["name"] for doc in response_json]
    assert "doc_1.txt" in names
    assert "doc_2.txt" in names
    meta_keys = [doc["meta"]["meta_key"] for doc in response_json]
    assert all("meta_value_get"==meta_key for meta_key in meta_keys)


def test_delete_documents():
    os.environ["PIPELINE_YAML_PATH"] = str((Path(__file__).parent / "samples"/"pipeline"/"test_pipeline.yaml").absolute())
    os.environ["INDEXING_PIPELINE_NAME"] = "indexing_text_pipeline"
    client = TestClient(app)

    # Clean up to make sure the docstore is empty
    client.post(url="/documents/delete_by_filters", data='{"filters": {}}')

    # Upload the files
    files_to_upload = [
        {'files': (Path(__file__).parent / "samples"/"docs"/"doc_1.txt").open('rb')},
        {'files': (Path(__file__).parent / "samples"/"docs"/"doc_2.txt").open('rb')}
    ]
    for index, fi in enumerate(files_to_upload):
        response = client.post(url="/file-upload", files=fi, data={"meta": f'{{"meta_key": "meta_value_del", "meta_index": "{index}"}}'})
        assert 200 == response.status_code

    # Make sure there are two docs
    response = client.post(url="/documents/get_by_filters", data='{"filters": {"meta_key": ["meta_value_del"]}}')
    assert 200 == response.status_code
    response_json = response.json()
    assert len(response_json) == 2

    # Delete one doc
    response = client.post(url="/documents/delete_by_filters", data='{"filters": {"meta_index": ["0"]}}')
    assert 200 == response.status_code

    # Now there should be only one doc
    response = client.post(url="/documents/get_by_filters", data='{"filters": {"meta_key": ["meta_value_del"]}}')
    assert 200 == response.status_code
    response_json = response.json()
    assert len(response_json) == 1

    # Make sure the right doc was deleted
    response = client.post(url="/documents/get_by_filters", data='{"filters": {"meta_index": ["0"]}}')
    assert 200 == response.status_code
    response_json = response.json()
    assert len(response_json) == 0
    response = client.post(url="/documents/get_by_filters", data='{"filters": {"meta_index": ["1"]}}')
    assert 200 == response.status_code
    response_json = response.json()
    assert len(response_json) == 1
    

def test_file_upload(client: TestClient):
    file_to_upload = {'files': (Path(__file__).parent / "samples"/"pdf"/"sample_pdf_1.pdf").open('rb')}
    response = client.post(url="/file-upload", files=file_to_upload, data={"meta": '{"meta_key": "meta_value", "non-existing-field": "wrong-value"}'})
    assert 200 == response.status_code


def test_query_with_no_filter(populated_client: TestClient):
    query_with_no_filter_value = {"query": "Who made the PDF specification?"}
    response = populated_client.post(url="/query", json=query_with_no_filter_value)
    assert 200 == response.status_code
    response_json = response.json()
    response_json = exclude_no_answer(response_json)
    assert response_json["answers"][0]["answer"] == "Adobe Systems"


def test_query_with_one_filter(populated_client: TestClient):
    query_with_filter = {"query": "Who made the PDF specification?", "params": {"Retriever": {"filters": {"meta_key": "meta_value"}}}}
    response = populated_client.post(url="/query", json=query_with_filter)
    assert 200 == response.status_code
    response_json = response.json()
    response_json = exclude_no_answer(response_json)
    assert response_json["answers"][0]["answer"] == "Adobe Systems"


def test_query_with_one_global_filter(populated_client: TestClient):
    query_with_filter = {"query": "Who made the PDF specification?", "params": {"filters": {"meta_key": "meta_value"}}}
    response = populated_client.post(url="/query", json=query_with_filter)
    assert 200 == response.status_code
    response_json = response.json()
    response_json = exclude_no_answer(response_json)
    assert response_json["answers"][0]["answer"] == "Adobe Systems"


def test_query_with_filter_list(populated_client: TestClient):
    query_with_filter_list = {
        "query": "Who made the PDF specification?",
        "params": {"Retriever": {"filters": {"meta_key": ["meta_value", "another_value"]}}}
    }
    response = populated_client.post(url="/query", json=query_with_filter_list)
    assert 200 == response.status_code
    response_json = response.json()
    response_json = exclude_no_answer(response_json)
    assert response_json["answers"][0]["answer"] == "Adobe Systems"


def test_query_with_invalid_filter(populated_client: TestClient):
    query_with_invalid_filter = {
        "query": "Who made the PDF specification?", "params": {"Retriever": {"filters": {"meta_key": "invalid_value"}}}
    }
    response = populated_client.post(url="/query", json=query_with_invalid_filter)
    assert 200 == response.status_code
    response_json = response.json()
    response_json = exclude_no_answer(response_json)
    assert len(response_json["answers"]) == 0


def test_write_feedback(populated_client: TestClient):
    response = populated_client.post(url="/feedback", json=FEEDBACK)
    assert 200 == response.status_code


def test_get_feedback(client: TestClient):
    response = client.post(url="/feedback", json=FEEDBACK)
    assert response.status_code == 200
    response = client.get(url="/feedback")
    assert response.status_code == 200
    json_response = response.json()
    for response_item, expected_item in [(json_response[0][key], value) for key, value in FEEDBACK.items()]:
        assert response_item == expected_item


def test_export_feedback(client: TestClient):
    response = client.post(url="/feedback", json=FEEDBACK)
    assert 200 == response.status_code

    feedback_urls = [
        "/export-feedback?full_document_context=true",
        "/export-feedback?full_document_context=false&context_size=50",
        "/export-feedback?full_document_context=false&context_size=50000",
    ]
    for url in feedback_urls:
        response = client.get(url=url, json=FEEDBACK)
        response_json = response.json()
        context = response_json["data"][0]["paragraphs"][0]["context"]
        answer_start = response_json["data"][0]["paragraphs"][0]["qas"][0]["answers"][0]["answer_start"]
        answer = response_json["data"][0]["paragraphs"][0]["qas"][0]["answers"][0]["text"]
        assert context[answer_start:answer_start+len(answer)] == answer


def test_get_feedback_malformed_query(client: TestClient):
    feedback = FEEDBACK.copy()
    feedback["unexpected_field"] = "misplaced-value"
    response = client.post(url="/feedback", json=feedback)
    assert response.status_code == 422
