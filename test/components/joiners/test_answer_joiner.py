# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest

from haystack import Document, GeneratedAnswer, Answer
from haystack.components.joiners.answer_joiner import AnswerJoiner, JoinMode


# custom join function that returns a single Answer, just for serde testing
def single_answer_function(answers: List[List[Answer]]) -> List[Answer]:
    return [GeneratedAnswer(data="custom", query="custom", documents=[], meta={})]


class TestAnswerJoiner:
    def test_init(self):
        joiner = AnswerJoiner()
        assert joiner.join_mode == JoinMode.CONCATENATE
        assert joiner.top_k == 10

    def test_init_with_custom_parameters(self):
        joiner = AnswerJoiner(join_mode="concatenate", top_k=5)
        assert joiner.join_mode == JoinMode.CONCATENATE
        assert joiner.top_k == 5

    def test_to_dict(self):
        joiner = AnswerJoiner()
        data = joiner.to_dict()
        assert data == {
            "type": "haystack.components.joiners.answer_joiner.AnswerJoiner",
            "init_parameters": {"join_mode": "concatenate", "custom_join_function": None, "top_k": 10},
        }

    def test_to_from_dict_with_custom_function(self):
        joiner = AnswerJoiner("concatenate", custom_join_function=single_answer_function, top_k=5)
        data = joiner.to_dict()
        assert data == {
            "type": "haystack.components.joiners.answer_joiner.AnswerJoiner",
            "init_parameters": {
                "join_mode": "concatenate",
                "custom_join_function": "joiners.test_answer_joiner.single_answer_function",
                "top_k": 5,
            },
        }

        deserialized_joiner = AnswerJoiner.from_dict(data)
        assert deserialized_joiner.join_mode == JoinMode.CONCATENATE
        assert deserialized_joiner.top_k == 5
        assert deserialized_joiner.custom_join_function == single_answer_function

    def test_from_dict(self):
        data = {"type": "haystack.components.joiners.answer_joiner.AnswerJoiner", "init_parameters": {}}
        document_joiner = AnswerJoiner.from_dict(data)
        assert document_joiner.join_mode == JoinMode.CONCATENATE
        assert document_joiner.top_k == 10

    def test_from_dict_customs_parameters(self):
        data = {
            "type": "haystack.components.joiners.answer_joiner.AnswerJoiner",
            "init_parameters": {"join_mode": "concatenate", "top_k": 5},
        }
        document_joiner = AnswerJoiner.from_dict(data)
        assert document_joiner.join_mode == JoinMode.CONCATENATE
        assert document_joiner.top_k == 5

    def test_empty_list(self):
        joiner = AnswerJoiner()
        result = joiner.run([])
        assert result == {"answers": []}

    def test_list_of_empty_lists(self):
        joiner = AnswerJoiner()
        result = joiner.run([[], []])
        assert result == {"answers": []}

    def test_list_with_one_empty_list(self):
        joiner = AnswerJoiner()
        answers = [
            GeneratedAnswer(query="a", data="a", meta={}, documents=[Document(content="a")]),
            GeneratedAnswer(query="b", data="b", meta={}, documents=[Document(content="b")]),
            GeneratedAnswer(query="c", data="c", meta={}, documents=[Document(content="c")]),
        ]
        result = joiner.run([answers])
        assert result == {"answers": answers}

    def test_unsupported_join_mode(self):
        unsupported_mode = "unsupported_mode"
        with pytest.raises(ValueError):
            AnswerJoiner(join_mode=unsupported_mode)
