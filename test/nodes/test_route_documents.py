import pytest
import pandas as pd

from haystack.schema import Document
from haystack.nodes import RouteDocuments


@pytest.mark.unit
def test_routedocuments_by_content_type():
    docs = [
        Document(content="text document", content_type="text"),
        Document(
            content=pd.DataFrame(columns=["col 1", "col 2"], data=[["row 1", "row 1"], ["row 2", "row 2"]]),
            content_type="table",
        ),
    ]
    route_documents = RouteDocuments()
    result, _ = route_documents.run(documents=docs)
    assert len(result["output_1"]) == 1
    assert len(result["output_2"]) == 1
    assert result["output_1"][0].content_type == "text"
    assert result["output_2"][0].content_type == "table"


@pytest.mark.unit
def test_routedocuments_by_metafield(docs):
    route_documents = RouteDocuments(split_by="meta_field", metadata_values=["test1", "test3", "test5"])
    assert route_documents.outgoing_edges == 3
    result, _ = route_documents.run(docs)
    assert len(result["output_1"]) == 1
    assert len(result["output_2"]) == 1
    assert len(result["output_3"]) == 1
    assert result["output_1"][0].meta["meta_field"] == "test1"
    assert result["output_2"][0].meta["meta_field"] == "test3"
    assert result["output_3"][0].meta["meta_field"] == "test5"
