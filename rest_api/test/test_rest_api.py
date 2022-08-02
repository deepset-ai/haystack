from typing import Dict, List, Optional, Union, Generator

import os
from pathlib import Path
from textwrap import dedent
from unittest import mock
from unittest.mock import MagicMock
import numpy as np
import pandas as pd

import pytest
from fastapi.testclient import TestClient
from haystack import Document, Answer
from haystack.nodes import BaseReader, BaseRetriever
from haystack.document_stores import BaseDocumentStore
from haystack.schema import Label
from haystack.nodes.file_converter import BaseConverter

from rest_api.utils import get_app


TEST_QUERY = "Who made the PDF specification?"


class MockReader(BaseReader):
    outgoing_edges = 1

    def predict(self, query: str, documents: List[Document], top_k: Optional[int] = None):
        return {"query": query, "no_ans_gap": None, "answers": [Answer(answer="Adobe Systems")]}

    def predict_batch(
        self,
        queries: List[str],
        documents: Union[List[Document], List[List[Document]]],
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
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
        scale_score=True,
    ) -> List[Document]:
        if filters and not isinstance(filters, dict):
            raise ValueError("You can't do this!")
        return self.document_store.get_all_documents(filters=filters)

    def retrieve_batch(
        self,
        queries: List[str],
        filters: dict = None,
        top_k: Optional[int] = None,
        index: str = None,
        headers: Optional[Dict[str, str]] = None,
        batch_size: Optional[int] = None,
        scale_score=True,
    ):
        pass


class MockPDFToTextConverter(BaseConverter):
    mocker = MagicMock()

    def convert(self, *args, **kwargs):
        self.mocker.convert(*args, **kwargs)
        return []


class MockDocumentStore(BaseDocumentStore):
    mocker = MagicMock()

    def write_documents(self, *args, **kwargs):
        pass

    def get_all_documents(self, *args, **kwargs) -> List[Document]:
        self.mocker.get_all_documents(*args, **kwargs)
        return [
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

    def get_all_documents_generator(self, *args, **kwargs) -> Generator[Document, None, None]:
        pass

    def get_all_labels(self, *args, **kwargs) -> List[Label]:
        return self.mocker.get_all_labels(*args, **kwargs)

    def get_document_by_id(self, *args, **kwargs) -> Optional[Document]:
        pass

    def get_document_count(self, *args, **kwargs) -> int:
        pass

    def query_by_embedding(self, *args, **kwargs) -> List[Document]:
        pass

    def get_label_count(self, *args, **kwargs) -> int:
        pass

    def write_labels(self, *args, **kwargs):
        self.mocker.write_labels(*args, **kwargs)

    def delete_documents(self, *args, **kwargs):
        self.mocker.delete_documents(*args, **kwargs)

    def delete_labels(self, *args, **kwargs):
        self.mocker.delete_labels(*args, **kwargs)

    def delete_index(self, index: str):
        pass

    def _create_document_field_map(self) -> Dict:
        pass

    def get_documents_by_id(self, *args, **kwargs) -> List[Document]:
        pass

    def update_document_meta(self, *args, **kwargs):
        pass


@pytest.fixture(scope="function")
def feedback():
    """
    Some test functions change the content of the `feedback` dictionary, let's keep
    the default "function" scope so we don't need to deepcopy the dict each time
    """
    return {
        "id": "123",
        "query": "Who made the PDF specification?",
        "document": {
            "content": "A sample PDF file\n\nHistory and standardization\nFormat (PDF) Adobe Systems made the PDF specification available free of charge in 1993. In the early years PDF was popular mainly in desktop publishing workflows, and competed with a variety of formats such as DjVu, Envoy, Common Ground Digital Paper, Farallon Replica and even Adobe's own PostScript format. PDF was a proprietary format controlled by Adobe until it was released as an open standard on July 1, 2008, and published by the International Organization for Standardization as ISO 32000-1:2008, at which time control of the specification passed to an ISO Committee of volunteer industry experts. In 2008, Adobe published a Public Patent License to ISO 32000-1 granting royalty-free rights for all patents owned by Adobe that are necessary to make, use, sell, and distribute PDF-compliant implementations. PDF 1.7, the sixth edition of the PDF specification that became ISO 32000-1, includes some proprietary technologies defined only by Adobe, such as Adobe XML Forms Architecture (XFA) and JavaScript extension for Acrobat, which are referenced by ISO 32000-1 as normative and indispensable for the full implementation of the ISO 32000-1 specification. These proprietary technologies are not standardized and their specification is published only on Adobes website. Many of them are also not supported by popular third-party implementations of PDF. Column 1",
            "content_type": "text",
            "score": None,
            "id": "fc18c987a8312e72a47fb1524f230bb0",
            "meta": {},
            "embedding": [0.1, 0.2, 0.3],
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


@pytest.fixture
def client():
    yaml_pipeline_path = Path(__file__).parent.resolve() / "samples" / "test.haystack-pipeline.yml"
    os.environ["PIPELINE_YAML_PATH"] = str(yaml_pipeline_path)
    os.environ["INDEXING_PIPELINE_NAME"] = "test-indexing"
    os.environ["QUERY_PIPELINE_NAME"] = "test-query"

    app = get_app()
    client = TestClient(app)

    MockDocumentStore.mocker.reset_mock()
    MockPDFToTextConverter.mocker.reset_mock()

    return client


def test_get_all_documents(client):
    response = client.post(url="/documents/get_by_filters", data='{"filters": {}}')
    assert 200 == response.status_code
    # Ensure `get_all_documents` was called with the expected `filters` param
    MockDocumentStore.mocker.get_all_documents.assert_called_with(filters={})
    # Ensure results are part of the response body
    response_json = response.json()
    assert len(response_json) == 2


def test_get_documents_with_filters(client):
    response = client.post(url="/documents/get_by_filters", data='{"filters": {"test_index": ["2"]}}')
    assert 200 == response.status_code
    # Ensure `get_all_documents` was called with the expected `filters` param
    MockDocumentStore.mocker.get_all_documents.assert_called_with(filters={"test_index": ["2"]})


def test_delete_all_documents(client):
    response = client.post(url="/documents/delete_by_filters", data='{"filters": {}}')
    assert 200 == response.status_code
    # Ensure `delete_documents` was called on the Document Store instance
    MockDocumentStore.mocker.delete_documents.assert_called_with(filters={})


def test_delete_documents_with_filters(client):
    response = client.post(url="/documents/delete_by_filters", data='{"filters": {"test_index": ["1"]}}')
    assert 200 == response.status_code
    # Ensure `delete_documents` was called on the Document Store instance with the same params
    MockDocumentStore.mocker.delete_documents.assert_called_with(filters={"test_index": ["1"]})


def test_file_upload(client):
    file_to_upload = {"files": (Path(__file__).parent / "samples" / "pdf" / "sample_pdf_1.pdf").open("rb")}
    response = client.post(url="/file-upload", files=file_to_upload, data={"meta": '{"test_key": "test_value"}'})
    assert 200 == response.status_code
    # Ensure the `convert` method was called with the right keyword params
    _, kwargs = MockPDFToTextConverter.mocker.convert.call_args
    # Files are renamed with random prefix like 83f4c1f5b2bd43f2af35923b9408076b_sample_pdf_1.pdf
    # so we just ensure the original file name is contained in the converted file name
    assert "sample_pdf_1.pdf" in str(kwargs["file_path"])
    assert kwargs["meta"]["test_key"] == "test_value"


def test_file_upload_with_no_meta(client):
    file_to_upload = {"files": (Path(__file__).parent / "samples" / "pdf" / "sample_pdf_1.pdf").open("rb")}
    response = client.post(url="/file-upload", files=file_to_upload, data={})
    assert 200 == response.status_code
    # Ensure the `convert` method was called with the right keyword params
    _, kwargs = MockPDFToTextConverter.mocker.convert.call_args
    assert kwargs["meta"] == {"name": "sample_pdf_1.pdf"}


def test_file_upload_with_empty_meta(client):
    file_to_upload = {"files": (Path(__file__).parent / "samples" / "pdf" / "sample_pdf_1.pdf").open("rb")}
    response = client.post(url="/file-upload", files=file_to_upload, data={"meta": ""})
    assert 200 == response.status_code
    # Ensure the `convert` method was called with the right keyword params
    _, kwargs = MockPDFToTextConverter.mocker.convert.call_args
    assert kwargs["meta"] == {"name": "sample_pdf_1.pdf"}


def test_file_upload_with_wrong_meta(client):
    file_to_upload = {"files": (Path(__file__).parent / "samples" / "pdf" / "sample_pdf_1.pdf").open("rb")}
    response = client.post(url="/file-upload", files=file_to_upload, data={"meta": "1"})
    assert 500 == response.status_code
    # Ensure the `convert` method was never called
    MockPDFToTextConverter.mocker.convert.assert_not_called()


def test_query_with_no_filter(client):
    with mock.patch("rest_api.controller.search.query_pipeline") as mocked_pipeline:
        # `run` must return a dictionary containing a `query` key
        mocked_pipeline.run.return_value = {"query": TEST_QUERY}
        response = client.post(url="/query", json={"query": TEST_QUERY})
        assert 200 == response.status_code
        # Ensure `run` was called with the expected parameters
        mocked_pipeline.run.assert_called_with(query=TEST_QUERY, params={}, debug=False)


def test_query_with_one_filter(client):
    params = {"TestRetriever": {"filters": {"test_key": ["test_value"]}}}
    with mock.patch("rest_api.controller.search.query_pipeline") as mocked_pipeline:
        # `run` must return a dictionary containing a `query` key
        mocked_pipeline.run.return_value = {"query": TEST_QUERY}
        response = client.post(url="/query", json={"query": TEST_QUERY, "params": params})
        assert 200 == response.status_code
        # Ensure `run` was called with the expected parameters
        mocked_pipeline.run.assert_called_with(query=TEST_QUERY, params=params, debug=False)


def test_query_with_one_global_filter(client):
    params = {"filters": {"test_key": ["test_value"]}}
    with mock.patch("rest_api.controller.search.query_pipeline") as mocked_pipeline:
        # `run` must return a dictionary containing a `query` key
        mocked_pipeline.run.return_value = {"query": TEST_QUERY}
        response = client.post(url="/query", json={"query": TEST_QUERY, "params": params})
        assert 200 == response.status_code
        # Ensure `run` was called with the expected parameters
        mocked_pipeline.run.assert_called_with(query=TEST_QUERY, params=params, debug=False)


def test_query_with_filter_list(client):
    params = {"TestRetriever": {"filters": {"test_key": ["test_value", "another_value"]}}}
    with mock.patch("rest_api.controller.search.query_pipeline") as mocked_pipeline:
        # `run` must return a dictionary containing a `query` key
        mocked_pipeline.run.return_value = {"query": TEST_QUERY}
        response = client.post(url="/query", json={"query": TEST_QUERY, "params": params})
        assert 200 == response.status_code
        # Ensure `run` was called with the expected parameters
        mocked_pipeline.run.assert_called_with(query=TEST_QUERY, params=params, debug=False)


def test_query_with_deprecated_filter_format(client):
    request_params = {"TestRetriever": {"filters": {"test_key": "i_should_be_a_list"}}}
    expected_params = {"TestRetriever": {"filters": {"test_key": ["i_should_be_a_list"]}}}
    with mock.patch("rest_api.controller.search.query_pipeline") as mocked_pipeline:
        # `run` must return a dictionary containing a `query` key
        mocked_pipeline.run.return_value = {"query": TEST_QUERY}
        response = client.post(url="/query", json={"query": TEST_QUERY, "params": request_params})
        assert 200 == response.status_code
        # Ensure `run` was called with the expected parameters. In this case,
        # `_format_filters` will fix the `filters` format within the params
        mocked_pipeline.run.assert_called_with(query=TEST_QUERY, params=expected_params, debug=False)


def test_query_with_no_documents_and_no_answers(client):
    with mock.patch("rest_api.controller.search.query_pipeline") as mocked_pipeline:
        # `run` must return a dictionary containing a `query` key
        mocked_pipeline.run.return_value = {"query": TEST_QUERY}
        response = client.post(url="/query", json={"query": TEST_QUERY})
        assert 200 == response.status_code
        response_json = response.json()
        assert response_json["documents"] == []
        assert response_json["answers"] == []


def test_query_with_bool_in_params(client):
    """
    Ensure items of params can be other types than dictionary, see
    https://github.com/deepset-ai/haystack/issues/2656
    """
    with mock.patch("rest_api.controller.search.query_pipeline") as mocked_pipeline:
        # `run` must return a dictionary containing a `query` key
        mocked_pipeline.run.return_value = {"query": TEST_QUERY}
        request_body = {
            "query": TEST_QUERY,
            "params": {"debug": True, "Retriever": {"top_k": 5}, "Reader": {"top_k": 3}},
        }
        response = client.post(url="/query", json=request_body)
        assert 200 == response.status_code
        response_json = response.json()
        assert response_json["documents"] == []
        assert response_json["answers"] == []


def test_query_with_embeddings(client):
    with mock.patch("rest_api.controller.search.query_pipeline") as mocked_pipeline:
        # `run` must return a dictionary containing a `query` key
        mocked_pipeline.run.return_value = {
            "query": TEST_QUERY,
            "documents": [
                Document(
                    content="test",
                    content_type="text",
                    score=0.9,
                    meta={"test_key": "test_value"},
                    embedding=np.array([0.1, 0.2, 0.3]),
                )
            ],
        }
        response = client.post(url="/query", json={"query": TEST_QUERY})
        assert 200 == response.status_code
        assert len(response.json()["documents"]) == 1
        assert response.json()["documents"][0]["content"] == "test"
        assert response.json()["documents"][0]["content_type"] == "text"
        assert response.json()["documents"][0]["embedding"] == [0.1, 0.2, 0.3]
        # Ensure `run` was called with the expected parameters
        mocked_pipeline.run.assert_called_with(query=TEST_QUERY, params={}, debug=False)


def test_query_with_dataframe(client):
    with mock.patch("rest_api.controller.search.query_pipeline") as mocked_pipeline:
        # `run` must return a dictionary containing a `query` key
        mocked_pipeline.run.return_value = {
            "query": TEST_QUERY,
            "documents": [
                Document(
                    content=pd.DataFrame.from_records([{"col1": "text_1", "col2": 1}, {"col1": "text_2", "col2": 2}]),
                    content_type="table",
                    score=0.9,
                    meta={"test_key": "test_value"},
                )
            ],
        }
        response = client.post(url="/query", json={"query": TEST_QUERY})
        assert 200 == response.status_code
        assert len(response.json()["documents"]) == 1
        assert response.json()["documents"][0]["content"] == [
            {"col1": "text_1", "col2": 1},
            {"col1": "text_2", "col2": 2},
        ]
        assert response.json()["documents"][0]["content_type"] == "table"
        # Ensure `run` was called with the expected parameters
        mocked_pipeline.run.assert_called_with(query=TEST_QUERY, params={}, debug=False)


def test_write_feedback(client, feedback):
    response = client.post(url="/feedback", json=feedback)
    assert 200 == response.status_code
    # Ensure `write_labels` was called on the Document Store instance passing a list
    # containing only one label
    args, _ = MockDocumentStore.mocker.write_labels.call_args
    labels = args[0]
    assert len(labels) == 1
    # Ensure all the items that were in `feedback` are also part of
    # the stored label (which has several more keys)
    label = labels[0]
    assert label == Label.from_dict(feedback)


def test_write_feedback_without_id(client, feedback):
    del feedback["id"]
    response = client.post(url="/feedback", json=feedback)
    assert 200 == response.status_code
    # Ensure `write_labels` was called on the Document Store instance passing a list
    # containing only one label
    args, _ = MockDocumentStore.mocker.write_labels.call_args
    labels = args[0]
    assert len(labels) == 1
    # Ensure the `id` was automatically set before storing the label
    label = labels[0].to_dict()
    assert label["id"]


def test_get_feedback(client, feedback):
    MockDocumentStore.mocker.get_all_labels.return_value = [Label.from_dict(feedback)]
    response = client.get("/feedback")
    assert response.status_code == 200
    assert Label.from_dict(response.json()[0]) == Label.from_dict(feedback)
    MockDocumentStore.mocker.get_all_labels.assert_called_once()


def test_delete_feedback(client, monkeypatch, feedback):
    # This label contains `origin=user-feedback` and should be deleted
    label_to_delete = Label.from_dict(feedback)
    # This other label has a different origin and should NOT be deleted
    label_to_keep = Label.from_dict(feedback)
    label_to_keep.id = "42"
    label_to_keep.origin = "not-from-api"

    # Patch the Document Store so it returns the 2 labels above
    def get_all_labels(*args, **kwargs):
        return [label_to_delete, label_to_keep]

    monkeypatch.setattr(MockDocumentStore, "get_all_labels", get_all_labels)

    # Call the API and ensure `delete_labels` was called only on the label with id=123
    response = client.delete(url="/feedback")
    assert 200 == response.status_code
    MockDocumentStore.mocker.delete_labels.assert_called_with(ids=["123"])


def test_export_feedback(client, monkeypatch, feedback):
    def get_all_labels(*args, **kwargs):
        return [Label.from_dict(feedback)]

    monkeypatch.setattr(MockDocumentStore, "get_all_labels", get_all_labels)

    feedback_urls = [
        "/export-feedback?full_document_context=true",
        "/export-feedback?full_document_context=false&context_size=50",
        "/export-feedback?full_document_context=false&context_size=50000",
    ]
    for url in feedback_urls:
        response = client.get(url)
        response_json = response.json()
        context = response_json["data"][0]["paragraphs"][0]["context"]
        answer_start = response_json["data"][0]["paragraphs"][0]["qas"][0]["answers"][0]["answer_start"]
        answer = response_json["data"][0]["paragraphs"][0]["qas"][0]["answers"][0]["text"]
        assert context[answer_start : answer_start + len(answer)] == answer


def test_get_feedback_malformed_query(client, feedback):
    feedback["unexpected_field"] = "misplaced-value"
    response = client.post(url="/feedback", json=feedback)
    assert response.status_code == 422
