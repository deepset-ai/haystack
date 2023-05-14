import logging
import json
from pathlib import Path
from uuid import uuid4

import pytest
import responses
import numpy as np

from responses import matchers

from haystack.document_stores import DeepsetCloudDocumentStore
from haystack.utils import DeepsetCloudError
from haystack.schema import Document, Label, Answer


DC_API_ENDPOINT = "https://dc.example.com/v1"
DC_TEST_INDEX = "document_retrieval_1"
DC_API_KEY = "NO_KEY"


@pytest.fixture
def dc_api_mock(request):
    """
    This fixture contains responses activation, so either this one or ds() below must be
    passed to tests that require mocking.

    If `--mock-dc` was False, responses are never activated and it doesn't matter if the
    fixture is passed or not.
    """
    if request.config.getoption("--mock-dc"):
        responses.add(
            method=responses.GET,
            url=f"{DC_API_ENDPOINT}/workspaces/default/indexes/{DC_TEST_INDEX}",
            match=[responses.matchers.header_matcher({"authorization": f"Bearer {DC_API_KEY}"})],
            json={"indexing": {"status": "INDEXED", "pending_file_count": 0, "total_file_count": 31}},
            status=200,
        )

        responses.add(
            method=responses.GET,
            url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines",
            match=[responses.matchers.header_matcher({"authorization": f"Bearer {DC_API_KEY}"})],
            json={
                "data": [
                    {
                        "name": DC_TEST_INDEX,
                        "status": "DEPLOYED",
                        "indexing": {"status": "INDEXED", "pending_file_count": 0, "total_file_count": 31},
                    }
                ],
                "has_more": False,
                "total": 1,
            },
        )

        # activate the default mock, same as using the @responses.activate everywhere
        with responses.mock as m:
            yield m


@pytest.mark.document_store
@pytest.mark.integration
@pytest.mark.usefixtures("dc_api_mock")
class TestDeepsetCloudDocumentStore:
    # Fixtures

    @pytest.fixture
    def ds(self):
        """
        We make this fixture depend on `dc_api_mock` so that passing the document store will
        activate the mocking and we spare one function parameter.
        """
        return DeepsetCloudDocumentStore(api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY, index=DC_TEST_INDEX)

    # Integration tests

    def test_init_with_dot_product(self, ds):
        assert ds.return_embedding == False
        assert ds.similarity == "dot_product"

    def test_init_with_cosine(self):
        document_store = DeepsetCloudDocumentStore(
            api_endpoint=DC_API_ENDPOINT,
            api_key=DC_API_KEY,
            index=DC_TEST_INDEX,
            similarity="cosine",
            return_embedding=True,
        )
        assert document_store.return_embedding == True
        assert document_store.similarity == "cosine"

    def test_invalid_token(self):
        responses.add(
            method=responses.GET,
            url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines",
            match=[matchers.header_matcher({"authorization": "Bearer invalid_token"})],
            body="Internal Server Error",
            status=500,
        )

        with pytest.raises(
            DeepsetCloudError,
            match=f"Could not connect to deepset Cloud:\nGET {DC_API_ENDPOINT}/workspaces/default/pipelines failed: HTTP 500 - Internal Server Error",
        ):
            DeepsetCloudDocumentStore(api_endpoint=DC_API_ENDPOINT, api_key="invalid_token", index=DC_TEST_INDEX)

    def test_invalid_api_endpoint(self):
        responses.add(
            method=responses.GET, url=f"{DC_API_ENDPOINT}00/workspaces/default/pipelines", body="Not Found", status=404
        )

        with pytest.raises(
            DeepsetCloudError,
            match=f"Could not connect to deepset Cloud:\nGET {DC_API_ENDPOINT}00/workspaces/default/pipelines failed: "
            f"HTTP 404 - Not Found\nNot Found",
        ):
            DeepsetCloudDocumentStore(api_endpoint=f"{DC_API_ENDPOINT}00", api_key=DC_API_KEY, index=DC_TEST_INDEX)

    def test_invalid_index(self, caplog):
        responses.add(
            method=responses.GET,
            url=f"{DC_API_ENDPOINT}/workspaces/default/indexes/invalid_index",
            body="Not Found",
            status=404,
        )

        with caplog.at_level(logging.INFO):
            DeepsetCloudDocumentStore(api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY, index="invalid_index")
            assert (
                "You are using a DeepsetCloudDocumentStore with an index that does not exist on deepset Cloud."
                in caplog.text
            )

    def test_documents(self, ds, samples_path):
        with open(samples_path / "dc" / "documents-stream.response", "r") as f:
            documents_stream_response = f.read()
            docs = [json.loads(l) for l in documents_stream_response.splitlines()]
            filtered_docs = [doc for doc in docs if doc["meta"]["file_id"] == docs[0]["meta"]["file_id"]]
            documents_stream_filtered_response = "\n".join([json.dumps(d) for d in filtered_docs])

            responses.add(
                method=responses.POST,
                url=f"{DC_API_ENDPOINT}/workspaces/default/indexes/{DC_TEST_INDEX}/documents-stream",
                body=documents_stream_response,
                status=200,
            )

            responses.add(
                method=responses.POST,
                url=f"{DC_API_ENDPOINT}/workspaces/default/indexes/{DC_TEST_INDEX}/documents-stream",
                match=[
                    matchers.json_params_matcher(
                        {"filters": {"file_id": [docs[0]["meta"]["file_id"]]}, "return_embedding": False}
                    )
                ],
                body=documents_stream_filtered_response,
                status=200,
            )

            for doc in filtered_docs:
                responses.add(
                    method=responses.GET,
                    url=f"{DC_API_ENDPOINT}/workspaces/default/indexes/{DC_TEST_INDEX}/documents/{doc['id']}",
                    json=doc,
                    status=200,
                )

        docs = ds.get_all_documents()
        assert len(docs) > 1
        assert isinstance(docs[0], Document)

        first_doc = next(ds.get_all_documents_generator())
        assert isinstance(first_doc, Document)
        assert first_doc.meta["file_id"] is not None

        filtered_docs = ds.get_all_documents(filters={"file_id": [first_doc.meta["file_id"]]})
        assert len(filtered_docs) > 0
        assert len(filtered_docs) < len(docs)

        ids = [doc.id for doc in filtered_docs]
        single_doc_by_id = ds.get_document_by_id(ids[0])
        assert single_doc_by_id is not None
        assert single_doc_by_id.meta["file_id"] == first_doc.meta["file_id"]

        docs_by_id = ds.get_documents_by_id(ids)
        assert len(docs_by_id) == len(filtered_docs)
        for doc in docs_by_id:
            assert doc.meta["file_id"] == first_doc.meta["file_id"]

    def test_query(self, ds, samples_path):
        with open(samples_path / "dc" / "query_winterfell.response", "r") as f:
            query_winterfell_response = f.read()
            query_winterfell_docs = json.loads(query_winterfell_response)
            query_winterfell_filtered_docs = [
                doc
                for doc in query_winterfell_docs
                if doc["meta"]["file_id"] == query_winterfell_docs[0]["meta"]["file_id"]
            ]
            query_winterfell_filtered_response = json.dumps(query_winterfell_filtered_docs)

        responses.add(
            method=responses.POST,
            url=f"{DC_API_ENDPOINT}/workspaces/default/indexes/{DC_TEST_INDEX}/documents-query",
            match=[
                matchers.json_params_matcher(
                    {"query": "winterfell", "top_k": 50, "all_terms_must_match": False, "scale_score": True}
                )
            ],
            status=200,
            body=query_winterfell_response,
        )

        responses.add(
            method=responses.POST,
            url=f"{DC_API_ENDPOINT}/workspaces/default/indexes/{DC_TEST_INDEX}/documents-query",
            match=[
                matchers.json_params_matcher(
                    {
                        "query": "winterfell",
                        "top_k": 50,
                        "filters": {"file_id": [query_winterfell_docs[0]["meta"]["file_id"]]},
                        "all_terms_must_match": False,
                        "scale_score": True,
                    }
                )
            ],
            status=200,
            body=query_winterfell_filtered_response,
        )

        docs = ds.query("winterfell", top_k=50)
        assert docs is not None
        assert len(docs) > 0

        first_doc = docs[0]
        filtered_docs = ds.query("winterfell", top_k=50, filters={"file_id": [first_doc.meta["file_id"]]})
        assert len(filtered_docs) > 0
        assert len(filtered_docs) < len(docs)

    @pytest.mark.parametrize(
        "body, expected_count",
        [
            (
                {
                    "data": [
                        {
                            "evaluation_set_id": str(uuid4()),
                            "name": DC_TEST_INDEX,
                            "created_at": "2022-03-22T13:40:27.535Z",
                            "matched_labels": 2,
                            "total_labels": 10,
                        }
                    ],
                    "has_more": False,
                    "total": 1,
                },
                10,
            ),
            (
                {
                    "data": [
                        {
                            "evaluation_set_id": str(uuid4()),
                            "name": DC_TEST_INDEX,
                            "created_at": "2022-03-22T13:40:27.535Z",
                            "matched_labels": 0,
                            "total_labels": 0,
                        }
                    ],
                    "has_more": False,
                    "total": 1,
                },
                0,
            ),
        ],
    )
    def test_count_of_labels_for_evaluation_set(self, ds, body: dict, expected_count: int):
        responses.add(
            method=responses.GET,
            url=f"{DC_API_ENDPOINT}/workspaces/default/evaluation_sets",
            status=200,
            body=json.dumps(body),
        )

        count = ds.get_label_count(index=DC_TEST_INDEX)
        assert count == expected_count

    def test_count_of_labels_for_evaluation_set_raises_DC_error_when_nothing_found(self, ds):
        responses.add(
            method=responses.GET,
            url=f"{DC_API_ENDPOINT}/workspaces/default/evaluation_sets",
            status=200,
            body=json.dumps({"data": [], "has_more": False, "total": 0}),
        )

        with pytest.raises(DeepsetCloudError, match=f"No evaluation set found with the name {DC_TEST_INDEX}"):
            ds.get_label_count(index=DC_TEST_INDEX)

    def test_lists_evaluation_sets(self, ds):
        response_evaluation_set = {
            "evaluation_set_id": str(uuid4()),
            "name": DC_TEST_INDEX,
            "created_at": "2022-03-22T13:40:27.535Z",
            "matched_labels": 2,
            "total_labels": 10,
        }

        responses.add(
            method=responses.GET,
            url=f"{DC_API_ENDPOINT}/workspaces/default/evaluation_sets",
            status=200,
            body=json.dumps({"data": [response_evaluation_set], "has_more": False, "total": 1}),
        )

        evaluation_sets = ds.get_evaluation_sets()
        assert evaluation_sets == [response_evaluation_set]

    def test_fetches_labels_for_evaluation_set(self, ds):
        responses.add(
            method=responses.GET,
            url=f"{DC_API_ENDPOINT}/workspaces/default/evaluation_sets/{DC_TEST_INDEX}",
            status=200,
            body=json.dumps(
                [
                    {
                        "label_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                        "query": "What is berlin?",
                        "answer": "biggest city in germany",
                        "answer_start": 0,
                        "answer_end": 0,
                        "meta": {},
                        "context": "Berlin is the biggest city in germany.",
                        "external_file_name": "string",
                        "file_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                        "state": "Label matching status",
                        "candidates": "Candidates that were found in the label <-> file matching",
                    }
                ]
            ),
        )

        labels = ds.get_all_labels(index=DC_TEST_INDEX)
        assert labels == [
            Label(
                query="What is berlin?",
                document=Document(content="Berlin is the biggest city in germany."),
                is_correct_answer=True,
                is_correct_document=True,
                origin="user-feedback",
                answer=Answer("biggest city in germany"),
                id="3fa85f64-5717-4562-b3fc-2c963f66afa6",
                pipeline_id=None,
                created_at=None,
                updated_at=None,
                meta={},
                filters={},
            )
        ]

    def test_fetches_labels_for_evaluation_set_raises_deepsetclouderror_when_nothing_found(self, ds):
        responses.add(
            method=responses.GET,
            url=f"{DC_API_ENDPOINT}/workspaces/default/evaluation_sets/{DC_TEST_INDEX}",
            status=404,
        )

        with pytest.raises(DeepsetCloudError, match=f"No evaluation set found with the name {DC_TEST_INDEX}"):
            ds.get_all_labels(index=DC_TEST_INDEX)

    def test_query_by_embedding(self, ds):
        query_emb = np.random.randn(768)

        responses.add(
            method=responses.POST,
            url=f"{DC_API_ENDPOINT}/workspaces/default/indexes/{DC_TEST_INDEX}/documents-query",
            match=[
                matchers.json_params_matcher(
                    {
                        "query_emb": query_emb.tolist(),
                        "top_k": 10,
                        "return_embedding": False,
                        "scale_score": True,
                        "use_prefiltering": False,
                    }
                )
            ],
            json=[],
            status=200,
        )

        emb_docs = ds.query_by_embedding(query_emb)
        assert len(emb_docs) == 0

    def test_get_all_docs_without_index(self):
        document_store = DeepsetCloudDocumentStore(api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY, index=None)
        assert document_store.get_all_documents() == []

    def test_get_all_docs_generator_without_index(self):
        document_store = DeepsetCloudDocumentStore(api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY, index=None)
        assert list(document_store.get_all_documents_generator()) == []

    def test_get_doc_by_id_without_index(self):
        document_store = DeepsetCloudDocumentStore(api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY, index=None)
        assert document_store.get_document_by_id(id="some id") == None

    def test_get_docs_by_id_without_index(self):
        document_store = DeepsetCloudDocumentStore(api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY, index=None)
        assert document_store.get_documents_by_id(ids=["some id"]) == []

    def test_get_doc_count_without_index(self):
        document_store = DeepsetCloudDocumentStore(api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY, index=None)
        assert document_store.get_document_count() == 0

    def test_query_by_emb_without_index(self):
        query_emb = np.random.randn(768)
        document_store = DeepsetCloudDocumentStore(api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY, index=None)
        assert document_store.query_by_embedding(query_emb=query_emb) == []

    def test_query_without_index(self):
        document_store = DeepsetCloudDocumentStore(api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY, index=None)
        assert document_store.query(query="some query") == []
