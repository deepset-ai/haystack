import pytest


from haystack import Document
from haystack.nodes.other.join_docs import JoinDocuments


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
def test_joindocuments_preserves_root_node():
    # https://github.com/deepset-ai/haystack-private/issues/51
    inputs = [
        {"documents": [Document(content="text document 1", content_type="text", score=0.2)], "root_node": "File"},
        {"documents": [Document(content="text document 2", content_type="text", score=None)], "root_node": "File"},
    ]
    join_docs = JoinDocuments()
    result, _ = join_docs.run(inputs)
    assert result["root_node"] == "File"
