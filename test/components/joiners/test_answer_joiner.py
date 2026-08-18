# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack import Document
from haystack.components.joiners.answer_joiner import AnswerJoiner, AnswerType, JoinMode
from haystack.dataclasses.answer import ExtractedAnswer, GeneratedAnswer


class TestAnswerJoiner:
    def test_init(self):
        joiner = AnswerJoiner()
        assert joiner.join_mode == JoinMode.CONCATENATE
        assert joiner.top_k is None
        assert joiner.sort_by_score is False

    def test_init_with_custom_parameters(self):
        joiner = AnswerJoiner(join_mode="concatenate", top_k=5, sort_by_score=True)
        assert joiner.join_mode == JoinMode.CONCATENATE
        assert joiner.top_k == 5
        assert joiner.sort_by_score is True

    def test_init_with_top_k_none_is_valid(self):
        joiner = AnswerJoiner(top_k=None)
        assert joiner.top_k is None

    @pytest.mark.parametrize("top_k", [1, 5])
    def test_init_with_positive_top_k_is_valid(self, top_k):
        joiner = AnswerJoiner(top_k=top_k)
        assert joiner.top_k == top_k

    @pytest.mark.parametrize("top_k", [0, -1])
    def test_init_with_non_positive_top_k_raises(self, top_k):
        with pytest.raises(ValueError, match="top_k must be greater than 0"):
            AnswerJoiner(top_k=top_k)

    def test_to_dict(self):
        joiner = AnswerJoiner()
        data = joiner.to_dict()
        assert data == {
            "type": "haystack.components.joiners.answer_joiner.AnswerJoiner",
            "init_parameters": {"join_mode": "concatenate", "top_k": None, "sort_by_score": False},
        }

    def test_to_from_dict_custom_parameters(self):
        joiner = AnswerJoiner("concatenate", top_k=5, sort_by_score=True)
        data = joiner.to_dict()
        assert data == {
            "type": "haystack.components.joiners.answer_joiner.AnswerJoiner",
            "init_parameters": {"join_mode": "concatenate", "top_k": 5, "sort_by_score": True},
        }

        deserialized_joiner = AnswerJoiner.from_dict(data)
        assert deserialized_joiner.join_mode == JoinMode.CONCATENATE
        assert deserialized_joiner.top_k == 5
        assert deserialized_joiner.sort_by_score is True

    def test_from_dict(self):
        data = {"type": "haystack.components.joiners.answer_joiner.AnswerJoiner", "init_parameters": {}}
        answer_joiner = AnswerJoiner.from_dict(data)
        assert answer_joiner.join_mode == JoinMode.CONCATENATE
        assert answer_joiner.top_k is None
        assert answer_joiner.sort_by_score is False

    def test_from_dict_customs_parameters(self):
        data = {
            "type": "haystack.components.joiners.answer_joiner.AnswerJoiner",
            "init_parameters": {"join_mode": "concatenate", "top_k": 5, "sort_by_score": True},
        }
        answer_joiner = AnswerJoiner.from_dict(data)
        assert answer_joiner.join_mode == JoinMode.CONCATENATE
        assert answer_joiner.top_k == 5
        assert answer_joiner.sort_by_score is True

    def test_empty_list(self):
        joiner = AnswerJoiner()
        result = joiner.run([])
        assert result == {"answers": []}

    def test_list_of_empty_lists(self):
        joiner = AnswerJoiner()
        result = joiner.run([[], []])
        assert result == {"answers": []}

    def test_list_of_single_answer(self):
        joiner = AnswerJoiner()
        answers: list[AnswerType] = [
            GeneratedAnswer(query="a", data="a", meta={}, documents=[Document(content="a")]),
            GeneratedAnswer(query="b", data="b", meta={}, documents=[Document(content="b")]),
            GeneratedAnswer(query="c", data="c", meta={}, documents=[Document(content="c")]),
        ]
        result = joiner.run([answers])
        assert result == {"answers": answers}

    def test_two_lists_of_generated_answers(self):
        joiner = AnswerJoiner()
        answers1: list[AnswerType] = [GeneratedAnswer(query="a", data="a", meta={}, documents=[Document(content="a")])]
        answers2: list[AnswerType] = [GeneratedAnswer(query="d", data="d", meta={}, documents=[Document(content="d")])]
        result = joiner.run([answers1, answers2])
        assert result == {"answers": answers1 + answers2}

    def test_multiple_lists_of_mixed_answers(self):
        joiner = AnswerJoiner()
        answers1: list[AnswerType] = [GeneratedAnswer(query="a", data="a", meta={}, documents=[Document(content="a")])]
        answers2: list[AnswerType] = [ExtractedAnswer(query="d", score=0.9, meta={}, document=Document(content="d"))]
        answers3: list[AnswerType] = [GeneratedAnswer(query="f", data="f", meta={}, documents=[Document(content="f")])]
        all_answers = answers1 + answers2 + answers3
        result = joiner.run([answers1, answers2, answers3])
        assert result == {"answers": all_answers}

    def test_unsupported_join_mode(self):
        unsupported_mode = "unsupported_mode"
        with pytest.raises(ValueError):
            AnswerJoiner(join_mode=unsupported_mode)

    def test_run_with_top_k_in_run_method_overrides_init_top_k(self):
        joiner = AnswerJoiner(top_k=5)
        answers1: list[AnswerType] = [GeneratedAnswer(query="a", data="a", meta={}, documents=[Document(content="a")])]
        answers2: list[AnswerType] = [GeneratedAnswer(query="b", data="b", meta={}, documents=[Document(content="b")])]
        answers3: list[AnswerType] = [GeneratedAnswer(query="c", data="c", meta={}, documents=[Document(content="c")])]
        result = joiner.run([answers1, answers2, answers3], top_k=2)
        assert len(result["answers"]) == 2

    def test_run_with_top_k_zero_in_run_method_overrides_init_top_k(self):
        # A run-time top_k=0 must be honored (return no answers), not treated as "unset"
        # and fall back to the instance's top_k.
        joiner = AnswerJoiner(top_k=5)
        answers1: list[AnswerType] = [GeneratedAnswer(query="a", data="a", meta={}, documents=[Document(content="a")])]
        answers2: list[AnswerType] = [GeneratedAnswer(query="b", data="b", meta={}, documents=[Document(content="b")])]
        result = joiner.run([answers1, answers2], top_k=0)
        assert len(result["answers"]) == 0

    def test_run_with_negative_top_k_in_run_method_raises(self):
        joiner = AnswerJoiner(top_k=5)
        answers1: list[AnswerType] = [GeneratedAnswer(query="a", data="a", meta={}, documents=[Document(content="a")])]
        answers2: list[AnswerType] = [GeneratedAnswer(query="b", data="b", meta={}, documents=[Document(content="b")])]
        with pytest.raises(ValueError, match="top_k must not be negative"):
            joiner.run([answers1, answers2], top_k=-1)

    def test_sort_by_score(self):
        joiner = AnswerJoiner(sort_by_score=True)
        answers1: list[AnswerType] = [ExtractedAnswer(query="a", score=0.3, meta={}, document=Document(content="a"))]
        answers2: list[AnswerType] = [ExtractedAnswer(query="b", score=0.9, meta={}, document=Document(content="b"))]
        result = joiner.run([answers1, answers2])
        scores = [answer.score for answer in result["answers"]]
        assert scores == [0.9, 0.3]

    def test_sort_by_score_with_none_score(self):
        # The docstring promises that an answer with no score is handled as if its score is -infinity.
        # ExtractedAnswer with score=None must not raise a TypeError during sorting and must be sorted last.
        joiner = AnswerJoiner(sort_by_score=True)
        answers1: list[AnswerType] = [ExtractedAnswer(query="a", score=0.5, meta={}, document=Document(content="a"))]
        answers2: list[AnswerType] = [
            ExtractedAnswer(query="b", score=None, meta={}, document=Document(content="b"))  # type: ignore[arg-type]
        ]
        result = joiner.run([answers1, answers2])
        assert [answer.data for answer in result["answers"]] == [None, None]
        assert [answer.score for answer in result["answers"]] == [0.5, None]

    def test_sort_by_score_with_answers_missing_score_attribute(self):
        # GeneratedAnswer has no score attribute at all; it must be handled as -infinity and sorted last.
        joiner = AnswerJoiner(sort_by_score=True)
        answers1: list[AnswerType] = [GeneratedAnswer(query="a", data="a", meta={}, documents=[Document(content="a")])]
        answers2: list[AnswerType] = [ExtractedAnswer(query="b", score=0.9, meta={}, document=Document(content="b"))]
        result = joiner.run([answers1, answers2])
        # The ExtractedAnswer (score 0.9) comes first, the GeneratedAnswer (no score) comes last.
        assert isinstance(result["answers"][0], ExtractedAnswer)
        assert isinstance(result["answers"][1], GeneratedAnswer)
