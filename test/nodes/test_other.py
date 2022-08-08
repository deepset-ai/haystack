import pytest
import pandas as pd

from haystack import Document, Answer
from haystack.nodes.other.route_documents import RouteDocuments
from haystack.nodes.other.join_answers import JoinAnswers
from haystack.nodes.other.join_docs import JoinDocuments


#
# RouteDocuments tests
#


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


#
# JoinAnswers tests
#


@pytest.mark.parametrize("join_mode", ["concatenate", "merge"])
def test_joinanswers(join_mode):
    inputs = [{"answers": [Answer(answer="answer 1", score=0.7)]}, {"answers": [Answer(answer="answer 2", score=0.8)]}]

    join_answers = JoinAnswers(join_mode=join_mode)
    result, _ = join_answers.run(inputs)
    assert len(result["answers"]) == 2
    assert result["answers"] == sorted(result["answers"], reverse=True)

    result, _ = join_answers.run(inputs, top_k_join=1)
    assert len(result["answers"]) == 1
    assert result["answers"][0].answer == "answer 2"


#
# JoinDocuments tests
#


@pytest.mark.parametrize("join_mode", ["concatenate", "merge", "reciprocal_rank_fusion"])
def test_joindocuments(join_mode):
    inputs = [
        {"documents": [Document(content="text document 1", content_type="text", score=0.2)]},
        {"documents": [Document(content="text document 2", content_type="text", score=0.7)]},
    ]

    join_docs = JoinDocuments(join_mode=join_mode)
    result, _ = join_docs.run(inputs)
    assert len(result["documents"]) == 2
    assert result["documents"] == sorted(result["documents"], reverse=True)

    result, _ = join_docs.run(inputs, top_k_join=1)
    assert len(result["documents"]) == 1
    if join_mode == "reciprocal_rank_fusion":
        assert result["documents"][0].content == "text document 1"
    else:
        assert result["documents"][0].content == "text document 2"


@pytest.mark.parametrize("join_mode", ["concatenate", "merge", "reciprocal_rank_fusion"])
@pytest.mark.parametrize("sort_by_score", [True, False])
def test_joindocuments_score_none(join_mode, sort_by_score):
    """Testing JoinDocuments() node when some of the documents have `score=None`"""
    inputs = [
        {"documents": [Document(content="text document 1", content_type="text", score=0.2)]},
        {"documents": [Document(content="text document 2", content_type="text", score=None)]},
    ]

    join_docs = JoinDocuments(join_mode=join_mode, sort_by_score=sort_by_score)
    result, _ = join_docs.run(inputs)
    assert len(result["documents"]) == 2

    result, _ = join_docs.run(inputs, top_k_join=1)
    assert len(result["documents"]) == 1
