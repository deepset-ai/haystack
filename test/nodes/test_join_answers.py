import pytest

from haystack.schema import Answer
from haystack.nodes import JoinAnswers


@pytest.mark.unit
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


@pytest.mark.unit
def test_joinanswers_preserves_root_node():
    # https://github.com/deepset-ai/haystack-private/issues/51
    inputs = [
        {"answers": [Answer(answer="answer 1", score=0.7)], "root_node": "Query"},
        {"answers": [Answer(answer="answer 2", score=0.8)], "root_node": "Query"},
    ]
    join_docs = JoinAnswers()
    result, _ = join_docs.run(inputs)
    assert result["root_node"] == "Query"
