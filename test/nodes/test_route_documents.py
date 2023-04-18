import pytest
import pandas as pd

from haystack.schema import Document
from haystack.nodes import RouteDocuments


@pytest.fixture
def docs_diff_types():
    return [
        Document(content="text document", content_type="text"),
        Document(
            content=pd.DataFrame(columns=["col 1", "col 2"], data=[["row 1", "row 1"], ["row 2", "row 2"]]),
            content_type="table",
        ),
        Document(content="image/path", content_type="image"),
    ]


@pytest.fixture
def docs_with_meta():
    return [
        Document(content="text document 1", content_type="text", meta={"meta_field": "test1"}),
        Document(content="text document 2", content_type="text", meta={"meta_field": "test2"}),
        Document(content="text document 3", content_type="text", meta={"meta_field": "test3"}),
    ]


@pytest.mark.unit
def test_routedocuments_by_content_type(docs_diff_types):
    route_documents = RouteDocuments()
    result, _ = route_documents.run(documents=docs_diff_types)
    assert route_documents.outgoing_edges == 2
    assert len(result["output_1"]) == 1
    assert len(result["output_2"]) == 1
    assert "output_3" not in result
    assert result["output_1"][0].content_type == "text"
    assert result["output_2"][0].content_type == "table"


@pytest.mark.unit
def test_routedocuments_by_content_type_return_remaining(docs_diff_types):
    route_documents = RouteDocuments(return_remaining=True)
    result, _ = route_documents.run(documents=docs_diff_types)
    assert route_documents.outgoing_edges == 3
    assert len(result["output_1"]) == 1
    assert len(result["output_2"]) == 1
    assert len(result["output_3"]) == 1
    assert result["output_1"][0].content_type == "text"
    assert result["output_2"][0].content_type == "table"
    assert result["output_3"][0].content_type == "image"


@pytest.mark.unit
def test_routedocuments_by_metafield(docs):
    route_documents = RouteDocuments(split_by="meta_field", metadata_values=["test1", "test3", "test5"])
    assert route_documents.outgoing_edges == 3
    result, _ = route_documents.run(docs)
    assert len(result["output_1"]) == 1
    assert len(result["output_2"]) == 1
    assert len(result["output_3"]) == 1
    assert "output_4" not in result
    assert result["output_1"][0].meta["meta_field"] == "test1"
    assert result["output_2"][0].meta["meta_field"] == "test3"
    assert result["output_3"][0].meta["meta_field"] == "test5"


@pytest.mark.unit
def test_routedocuments_by_metafield_return_remaning(docs):
    route_documents = RouteDocuments(
        split_by="meta_field", metadata_values=["test1", "test3", "test5"], return_remaining=True
    )
    assert route_documents.outgoing_edges == 4
    result, _ = route_documents.run(docs)
    assert len(result["output_1"]) == 1
    assert len(result["output_2"]) == 1
    assert len(result["output_3"]) == 1
    assert len(result["output_4"]) == 2
    assert result["output_1"][0].meta["meta_field"] == "test1"
    assert result["output_2"][0].meta["meta_field"] == "test3"
    assert result["output_3"][0].meta["meta_field"] == "test5"
    assert result["output_4"][0].meta["meta_field"] == "test2"


@pytest.mark.unit
def test_routedocuments_by_metafield_list_of_lists(docs):
    route_documents = RouteDocuments(
        split_by="meta_field", metadata_values=[["test1", "test3"], "test5"], return_remaining=True
    )
    assert route_documents.outgoing_edges == 3
    result, _ = route_documents.run(docs)
    assert len(result["output_1"]) == 2
    assert len(result["output_2"]) == 1
    assert len(result["output_3"]) == 2
    assert result["output_1"][0].meta["meta_field"] == "test1"
    assert result["output_1"][1].meta["meta_field"] == "test3"
    assert result["output_2"][0].meta["meta_field"] == "test5"
    assert result["output_3"][0].meta["meta_field"] == "test2"
