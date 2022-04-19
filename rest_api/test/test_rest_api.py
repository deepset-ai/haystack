from typing import Dict, List, Optional

import os
from copy import deepcopy
from pathlib import Path
from textwrap import dedent

import pytest
from fastapi.testclient import TestClient
from haystack import Document, Answer
from haystack.nodes import BaseReader, BaseRetriever
from haystack.document_stores import BaseDocumentStore
from haystack.schema import Label

from rest_api.utils import get_app, get_pipelines


FEEDBACK = {
    "id": "123",
    "query": "Who made the PDF specification?",
    "document": {
        "content": "A sample PDF file\n\nHistory and standardization\nFormat (PDF) Adobe Systems made the PDF specification available free of charge in 1993. In the early years PDF was popular mainly in desktop publishing workflows, and competed with a variety of formats such as DjVu, Envoy, Common Ground Digital Paper, Farallon Replica and even Adobe's own PostScript format. PDF was a proprietary format controlled by Adobe until it was released as an open standard on July 1, 2008, and published by the International Organization for Standardization as ISO 32000-1:2008, at which time control of the specification passed to an ISO Committee of volunteer industry experts. In 2008, Adobe published a Public Patent License to ISO 32000-1 granting royalty-free rights for all patents owned by Adobe that are necessary to make, use, sell, and distribute PDF-compliant implementations. PDF 1.7, the sixth edition of the PDF specification that became ISO 32000-1, includes some proprietary technologies defined only by Adobe, such as Adobe XML Forms Architecture (XFA) and JavaScript extension for Acrobat, which are referenced by ISO 32000-1 as normative and indispensable for the full implementation of the ISO 32000-1 specification. These proprietary technologies are not standardized and their specification is published only on Adobes website. Many of them are also not supported by popular third-party implementations of PDF. Column 1",
        "content_type": "text",
        "score": None,
        "id": "fc18c987a8312e72a47fb1524f230bb0",
        "meta": {},
        "embedding": None,
    },
    "answer": {
        "answer": "Adobe Systems",
        "type": "extractive",
        "context": "A sample PDF file\n\nHistory and standardization\nFormat (PDF) Adobe Systems made the PDF specification available free of charge in 1993. In the early ye",
        "offsets_in_context": [{"start": 60, "end": 73}],
        "offsets_in_document": [{"start": 60, "end": 73}],
        "document_id": "fc18c987a8312e72a47fb1524f230bb0",
        "meta": {},
        "score": None,
    },
    "is_correct_answer": True,
    "is_correct_document": True,
    "origin": "user-feedback",
    "pipeline_id": "some-123",
}


def exclude_no_answer(responses):
    responses["answers"] = [response for response in responses["answers"] if response.get("answer", None)]
    return responses


class MockReader(BaseReader):
    outgoing_edges = 1

    def predict(self, query: str, documents: List[Document], top_k: Optional[int] = None):
        return {"query": query, "no_ans_gap": None, "answers": [Answer(answer="Adobe Systems")]}

    def predict_batch(self, query_doc_list: List[dict], top_k: Optional[int] = None, batch_size: Optional[int] = None):
        pass


class MockRetriever(BaseRetriever):
    outgoing_edges = 1

    def __init__(self, document_store: BaseDocumentStore):
        super().__init__()
        self.document_store = document_store

    def retrieve(
        self,
        query: str,
        filters: dict = None,
        top_k: Optional[int] = None,
        index: str = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        if filters and not isinstance(filters, dict):
            raise ValueError("You can't do this!")
        return self.document_store.get_all_documents(filters=filters)


@pytest.fixture(scope="session")
def yaml_pipeline_path(tmp_path_factory):
    root_temp = tmp_path_factory.mktemp("tests")
    pipeline_path = root_temp / "test.haystack-pipeline.yml"
    with open(pipeline_path, "w") as pipeline_file:
        pipeline_file.write(
            f"""
version: 'unstable'

components:
  - name: TestReader
    type: MockReader
  - name: TestRetriever
    type: MockRetriever
    params:
      document_store: TestDocumentStore
  - name: TestDocumentStore
    type: SQLDocumentStore
    params:
      url: sqlite:///{root_temp.absolute()}/test_docstore.db
  - name: TestPreprocessor
    type: PreProcessor
    params:
      clean_whitespace: true
  - name: TestPDFConverter
    type: PDFToTextConverter
    params:
      remove_numeric_tables: false


pipelines:
  - name: test-query
    nodes:
      - name: TestRetriever
        inputs: [Query]
      - name: TestReader
        inputs: [TestRetriever]

  - name: test-indexing
    nodes:
      - name: TestPDFConverter
        inputs: [File]
      - name: TestPreprocessor
        inputs: [TestPDFConverter]
      - name: TestDocumentStore
        inputs: [TestPreprocessor]
    """
        )
    return pipeline_path


@pytest.fixture
def client(yaml_pipeline_path):
    os.environ["PIPELINE_YAML_PATH"] = str(yaml_pipeline_path)
    os.environ["INDEXING_PIPELINE_NAME"] = "test-indexing"
    os.environ["QUERY_PIPELINE_NAME"] = "test-query"

    app = get_app()
    client = TestClient(app)

    pipelines = get_pipelines()
    document_store: BaseDocumentStore = pipelines["document_store"]
    document_store.delete_documents()
    document_store.delete_labels()

    yield client

    document_store.delete_documents()
    document_store.delete_labels()


@pytest.fixture
def populated_client(client: TestClient):
    pipelines = get_pipelines()
    document_store: BaseDocumentStore = pipelines["document_store"]
    document_store.write_documents(
        [
            Document(
                content=dedent(
                    """\
            History and standardization
            Format (PDF) Adobe Systems made the PDF specification available free of
            charge in 1993. In the early years PDF was popular mainly in desktop
            publishing workflows, and competed with a variety of formats such as DjVu,
            Envoy, Common Ground Digital Paper, Farallon Replica and even Adobe's
            own PostScript format. PDF was a proprietary format controlled by Adobe
            until it was released as an open standard on July 1, 2008, and published by
            the International Organization for Standardization as ISO 32000-1:2008, at
            which time control of the specification passed to an ISO Committee of
            volunteer industry experts."""
                ),
                meta={"name": "test.txt", "test_key": "test_value", "test_index": "1"},
            ),
            Document(
                content=dedent(
                    """\
            In 2008, Adobe published a Public Patent License
            to ISO 32000-1 granting royalty-free rights for all patents owned by Adobe
            that are necessary to make, use, sell, and distribute PDF-compliant
            implementations. PDF 1.7, the sixth edition of the PDF specification that
            became ISO 32000-1, includes some proprietary technologies defined only by
            Adobe, such as Adobe XML Forms Architecture (XFA) and JavaScript
            extension for Acrobat, which are referenced by ISO 32000-1 as normative
            and indispensable for the full implementation of the ISO 32000-1
            specification. These proprietary technologies are not standardized and their
            specification is published only on Adobe's website. Many of them are also not
            supported by popular third-party implementations of PDF."""
                ),
                meta={"name": "test.txt", "test_key": "test_value", "test_index": "2"},
            ),
        ]
    )
    yield client


@pytest.fixture
def populated_client_with_feedback(populated_client: TestClient):
    pipelines = get_pipelines()
    document_store: BaseDocumentStore = pipelines["document_store"]
    document_store.write_labels([FEEDBACK])
    yield populated_client


@pytest.fixture
def api_document_store():
    pipelines = get_pipelines()
    yield pipelines["document_store"]


def test_get_all_documents(populated_client: TestClient):
    response = populated_client.post(url="/documents/get_by_filters", data='{"filters": {}}')
    assert 200 == response.status_code
    response_json = response.json()

    assert len(response_json) == 2


def test_get_documents_with_filters(populated_client: TestClient):
    response = populated_client.post(url="/documents/get_by_filters", data='{"filters": {"test_index": ["2"]}}')
    assert 200 == response.status_code
    response_json = response.json()

    assert len(response_json) == 1
    assert response_json[0]["meta"]["test_index"] == "2"


def test_delete_all_documents(populated_client: TestClient, api_document_store: BaseDocumentStore):
    response = populated_client.post(url="/documents/delete_by_filters", data='{"filters": {}}')
    assert 200 == response.status_code

    remaining_docs = api_document_store.get_all_documents()
    assert len(remaining_docs) == 0


def test_delete_documents_with_filters(populated_client: TestClient, api_document_store: BaseDocumentStore):
    response = populated_client.post(url="/documents/delete_by_filters", data='{"filters": {"test_index": ["1"]}}')
    assert 200 == response.status_code

    remaining_docs = api_document_store.get_all_documents()
    assert len(remaining_docs) == 1
    assert remaining_docs[0].meta["test_index"] == "2"


def test_file_upload(client: TestClient, api_document_store: BaseDocumentStore):
    file_to_upload = {"files": (Path(__file__).parent / "samples" / "pdf" / "sample_pdf_1.pdf").open("rb")}
    response = client.post(url="/file-upload", files=file_to_upload, data={"meta": '{"test_key": "test_value"}'})
    assert 200 == response.status_code

    documents = api_document_store.get_all_documents()
    assert len(documents) > 0
    for doc in documents:
        assert doc.meta["name"] == "sample_pdf_1.pdf"
        assert doc.meta["test_key"] == "test_value"


def test_file_upload_with_no_meta(client: TestClient, api_document_store: BaseDocumentStore):
    file_to_upload = {"files": (Path(__file__).parent / "samples" / "pdf" / "sample_pdf_1.pdf").open("rb")}
    response = client.post(url="/file-upload", files=file_to_upload, data={"meta": ""})
    assert 200 == response.status_code

    documents = api_document_store.get_all_documents()
    assert len(documents) > 0
    for doc in documents:
        assert doc.meta["name"] == "sample_pdf_1.pdf"


def test_file_upload_with_wrong_meta(client: TestClient, api_document_store: BaseDocumentStore):
    file_to_upload = {"files": (Path(__file__).parent / "samples" / "pdf" / "sample_pdf_1.pdf").open("rb")}
    response = client.post(url="/file-upload", files=file_to_upload, data={"meta": "1"})
    assert 500 == response.status_code
    assert api_document_store.get_document_count() == 0


def test_query_with_no_filter(populated_client: TestClient):
    query_with_no_filter_value = {"query": "Who made the PDF specification?"}
    response = populated_client.post(url="/query", json=query_with_no_filter_value)
    assert 200 == response.status_code
    response_json = response.json()
    response_json = exclude_no_answer(response_json)
    assert response_json["answers"][0]["answer"] == "Adobe Systems"


def test_query_with_one_filter(populated_client: TestClient):
    query_with_filter = {
        "query": "Who made the PDF specification?",
        "params": {"TestRetriever": {"filters": {"test_key": ["test_value"]}}},
    }
    response = populated_client.post(url="/query", json=query_with_filter)
    assert 200 == response.status_code
    response_json = response.json()
    response_json = exclude_no_answer(response_json)
    assert response_json["answers"][0]["answer"] == "Adobe Systems"


def test_query_with_one_global_filter(populated_client: TestClient):
    query_with_filter = {
        "query": "Who made the PDF specification?",
        "params": {"filters": {"test_key": ["test_value"]}},
    }
    response = populated_client.post(url="/query", json=query_with_filter)
    assert 200 == response.status_code
    response_json = response.json()
    response_json = exclude_no_answer(response_json)
    assert response_json["answers"][0]["answer"] == "Adobe Systems"


def test_query_with_filter_list(populated_client: TestClient):
    query_with_filter_list = {
        "query": "Who made the PDF specification?",
        "params": {"TestRetriever": {"filters": {"test_key": ["test_value", "another_value"]}}},
    }
    response = populated_client.post(url="/query", json=query_with_filter_list)
    assert 200 == response.status_code
    response_json = response.json()
    response_json = exclude_no_answer(response_json)
    assert response_json["answers"][0]["answer"] == "Adobe Systems"


def test_query_with_invalid_filter(populated_client: TestClient):
    query_with_invalid_filter = {
        "query": "Who made the PDF specification?",
        "params": {"TestRetriever": {"filters": {"test_key": "invalid_value"}}},
    }
    response = populated_client.post(url="/query", json=query_with_invalid_filter)
    assert 200 == response.status_code
    response_json = response.json()
    response_json = exclude_no_answer(response_json)
    assert len(response_json["answers"]) == 0


def test_query_with_no_documents_and_no_answers(client: TestClient):
    query = {"query": "Who made the PDF specification?"}
    response = client.post(url="/query", json=query)
    assert 200 == response.status_code
    response_json = response.json()
    assert response_json["documents"] == []
    assert response_json["answers"] == []


def test_write_feedback(populated_client: TestClient, api_document_store: BaseDocumentStore):
    response = populated_client.post(url="/feedback", json=FEEDBACK)
    assert 200 == response.status_code
    assert api_document_store.get_label_count() == 1

    label: Label = api_document_store.get_all_labels()[0]
    label_values = label.to_dict()
    for actual_item, expected_item in [(label_values[key], value) for key, value in FEEDBACK.items()]:
        assert actual_item == expected_item


def test_write_feedback_without_id(populated_client: TestClient, api_document_store: BaseDocumentStore):
    feedback = deepcopy(FEEDBACK)
    del feedback["id"]
    response = populated_client.post(url="/feedback", json=feedback)
    assert 200 == response.status_code
    assert api_document_store.get_label_count() == 1

    label: Label = api_document_store.get_all_labels()[0]
    label_values = label.to_dict()
    for actual_item, expected_item in [(label_values[key], value) for key, value in FEEDBACK.items() if key != "id"]:
        assert actual_item == expected_item


def test_get_feedback(populated_client_with_feedback: TestClient):
    response = populated_client_with_feedback.get(url="/feedback")
    assert response.status_code == 200
    json_response = response.json()
    for response_item, expected_item in [(json_response[0][key], value) for key, value in FEEDBACK.items()]:
        assert response_item == expected_item


def test_delete_feedback(populated_client_with_feedback: TestClient, api_document_store: BaseDocumentStore):
    response = populated_client_with_feedback.delete(url="/feedback")
    assert 200 == response.status_code
    assert api_document_store.get_label_count() == 0


def test_do_not_delete_gold_labels(populated_client_with_feedback: TestClient, api_document_store: BaseDocumentStore):
    feedback = deepcopy(FEEDBACK)
    feedback["id"] = "456"
    feedback["origin"] = "gold-label"
    api_document_store.write_labels([feedback])

    response = populated_client_with_feedback.delete(url="/feedback")
    assert 200 == response.status_code

    assert api_document_store.get_label_count() == 1

    label: Label = api_document_store.get_all_labels()[0]
    label_values = label.to_dict()
    for actual_item, expected_item in [(label_values[key], value) for key, value in feedback.items()]:
        assert actual_item == expected_item


def test_export_feedback(populated_client_with_feedback: TestClient):
    feedback_urls = [
        "/export-feedback?full_document_context=true",
        "/export-feedback?full_document_context=false&context_size=50",
        "/export-feedback?full_document_context=false&context_size=50000",
    ]
    for url in feedback_urls:
        response = populated_client_with_feedback.get(url=url, json=FEEDBACK)
        response_json = response.json()
        context = response_json["data"][0]["paragraphs"][0]["context"]
        answer_start = response_json["data"][0]["paragraphs"][0]["qas"][0]["answers"][0]["answer_start"]
        answer = response_json["data"][0]["paragraphs"][0]["qas"][0]["answers"][0]["text"]
        assert context[answer_start : answer_start + len(answer)] == answer


def test_get_feedback_malformed_query(populated_client_with_feedback: TestClient):
    feedback = deepcopy(FEEDBACK)
    feedback["unexpected_field"] = "misplaced-value"
    response = populated_client_with_feedback.post(url="/feedback", json=feedback)
    assert response.status_code == 422
