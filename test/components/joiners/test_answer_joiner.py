# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os

import pytest

from haystack.components.builders import AnswerBuilder

from haystack import Document, Pipeline
from haystack.dataclasses.answer import ExtractedAnswer, GeneratedAnswer, ExtractedTableAnswer
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.joiners.answer_joiner import AnswerJoiner, JoinMode
from haystack.dataclasses import ChatMessage


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
        answers = [
            GeneratedAnswer(query="a", data="a", meta={}, documents=[Document(content="a")]),
            GeneratedAnswer(query="b", data="b", meta={}, documents=[Document(content="b")]),
            GeneratedAnswer(query="c", data="c", meta={}, documents=[Document(content="c")]),
        ]
        result = joiner.run([answers])
        assert result == {"answers": answers}

    def test_two_lists_of_generated_answers(self):
        joiner = AnswerJoiner()
        answers1 = [GeneratedAnswer(query="a", data="a", meta={}, documents=[Document(content="a")])]
        answers2 = [GeneratedAnswer(query="d", data="d", meta={}, documents=[Document(content="d")])]
        result = joiner.run([answers1, answers2])
        assert result == {"answers": answers1 + answers2}

    def test_multiple_lists_of_mixed_answers(self):
        joiner = AnswerJoiner()
        answers1 = [GeneratedAnswer(query="a", data="a", meta={}, documents=[Document(content="a")])]
        answers2 = [ExtractedAnswer(query="d", score=0.9, meta={}, document=Document(content="d"))]
        answers3 = [ExtractedTableAnswer(query="e", score=0.7, meta={}, document=Document(content="e"))]
        answers4 = [GeneratedAnswer(query="f", data="f", meta={}, documents=[Document(content="f")])]
        all_answers = answers1 + answers2 + answers3 + answers4  # type: ignore
        result = joiner.run([answers1, answers2, answers3, answers4])
        assert result == {"answers": all_answers}

    def test_unsupported_join_mode(self):
        unsupported_mode = "unsupported_mode"
        with pytest.raises(ValueError):
            AnswerJoiner(join_mode=unsupported_mode)

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY", ""), reason="Needs OPENAI_API_KEY to run this test.")
    @pytest.mark.integration
    def test_with_pipeline(self):
        query = "What's Natural Language Processing?"
        messages = [
            ChatMessage.from_system("You are a helpful, respectful and honest assistant. Be super concise."),
            ChatMessage.from_user(query),
        ]

        pipe = Pipeline()
        pipe.add_component("gpt-4o", OpenAIChatGenerator(model="gpt-4o"))
        pipe.add_component("llama", OpenAIChatGenerator(model="gpt-3.5-turbo"))
        pipe.add_component("aba", AnswerBuilder())
        pipe.add_component("abb", AnswerBuilder())
        pipe.add_component("joiner", AnswerJoiner())

        pipe.connect("gpt-4o.replies", "aba")
        pipe.connect("llama.replies", "abb")
        pipe.connect("aba.answers", "joiner")
        pipe.connect("abb.answers", "joiner")

        results = pipe.run(
            data={
                "gpt-4o": {"messages": messages},
                "llama": {"messages": messages},
                "aba": {"query": query},
                "abb": {"query": query},
            }
        )

        assert "joiner" in results
        assert len(results["joiner"]["answers"]) == 2
