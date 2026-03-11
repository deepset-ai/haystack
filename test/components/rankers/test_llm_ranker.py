# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
from jinja2 import TemplateSyntaxError

from haystack import Document
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.components.rankers.llm_ranker import DEFAULT_PROMPT_TEMPLATE, LLMRanker
from haystack.dataclasses import ChatMessage


@pytest.fixture
def mock_chat_generator():
    return Mock(spec=OpenAIChatGenerator)


def test_init_invalid_top_k():
    with pytest.raises(ValueError, match="top_k must be > 0"):
        LLMRanker(top_k=0)


def test_init_default_generator(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    ranker = LLMRanker()

    assert ranker.top_k == 10
    assert ranker.raise_on_failure is False
    assert ranker.prompt_template == DEFAULT_PROMPT_TEMPLATE
    assert isinstance(ranker._chat_generator, OpenAIChatGenerator)
    assert ranker._chat_generator.model == "gpt-4.1-mini"
    assert ranker._prompt_builder is not None


def test_init_custom_generator(mock_chat_generator):
    ranker = LLMRanker(chat_generator=mock_chat_generator, top_k=5, raise_on_failure=True)

    assert ranker._chat_generator is mock_chat_generator
    assert ranker.top_k == 5
    assert ranker.raise_on_failure is True


def test_to_dict(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    chat_generator = OpenAIChatGenerator(generation_kwargs={"temperature": 0.5})
    ranker = LLMRanker(
        chat_generator=chat_generator,
        prompt_template="Rank {{ documents|length }} docs for {{ query }} with {{ top_k }}",
        top_k=3,
        raise_on_failure=True,
    )

    assert ranker.to_dict() == {
        "type": "haystack.components.rankers.llm_ranker.LLMRanker",
        "init_parameters": {
            "chat_generator": chat_generator.to_dict(),
            "prompt_template": "Rank {{ documents|length }} docs for {{ query }} with {{ top_k }}",
            "top_k": 3,
            "raise_on_failure": True,
        },
    }


def test_from_dict(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    chat_generator = OpenAIChatGenerator(generation_kwargs={"temperature": 0.5})
    data = {
        "type": "haystack.components.rankers.llm_ranker.LLMRanker",
        "init_parameters": {
            "chat_generator": chat_generator.to_dict(),
            "prompt_template": "Rank {{ documents|length }} docs for {{ query }} with {{ top_k }}",
            "top_k": 3,
            "raise_on_failure": True,
        },
    }

    ranker = LLMRanker.from_dict(data)

    assert ranker.top_k == 3
    assert ranker.raise_on_failure is True
    assert ranker.prompt_template == "Rank {{ documents|length }} docs for {{ query }} with {{ top_k }}"
    assert ranker._chat_generator.to_dict() == chat_generator.to_dict()


def test_run_invalid_runtime_top_k(mock_chat_generator):
    ranker = LLMRanker(chat_generator=mock_chat_generator)

    with pytest.raises(ValueError, match="top_k must be > 0"):
        ranker.run(query="test", documents=[Document(content="doc")], top_k=0)


def test_run_empty_documents(mock_chat_generator):
    ranker = LLMRanker(chat_generator=mock_chat_generator)

    assert ranker.run(query="test", documents=[]) == {"documents": []}


def test_run_whitespace_query_returns_fallback(mock_chat_generator):
    documents = [Document(id="1", content="first"), Document(id="2", content="second")]
    ranker = LLMRanker(chat_generator=mock_chat_generator, top_k=1)

    result = ranker.run(query="   ", documents=documents)

    assert result == {"documents": [documents[0]]}
    mock_chat_generator.run.assert_not_called()


def test_run_successful_ranking(mock_chat_generator):
    documents = [
        Document(id="1", content="first"),
        Document(id="2", content="second"),
        Document(id="3", content="third"),
    ]
    mock_chat_generator.run.return_value = {
        "replies": [ChatMessage.from_assistant('{"documents": [{"id": "2"}, {"id": "1", "score": 0.75}, {"id": "3"}]}')]
    }
    ranker = LLMRanker(chat_generator=mock_chat_generator, top_k=2)

    result = ranker.run(query="test query", documents=documents)

    assert [document.id for document in result["documents"]] == ["2", "1"]
    assert result["documents"][0].score is None
    assert result["documents"][1].score == pytest.approx(0.75)
    assert documents[1].score is None


def test_run_dynamic_cutoff_returns_fewer_documents(mock_chat_generator):
    documents = [Document(id="1", content="first"), Document(id="2", content="second")]
    mock_chat_generator.run.return_value = {"replies": [ChatMessage.from_assistant('{"documents": [{"id": "2"}]}')]}
    ranker = LLMRanker(chat_generator=mock_chat_generator, top_k=2)

    result = ranker.run(query="test query", documents=documents)

    assert [document.id for document in result["documents"]] == ["2"]


def test_run_ignores_unknown_and_duplicate_ids(mock_chat_generator):
    documents = [Document(id="1", content="first"), Document(id="2", content="second")]
    mock_chat_generator.run.return_value = {
        "replies": [
            ChatMessage.from_assistant('{"documents": [{"id": "missing"}, {"id": "2"}, {"id": "2"}, {"id": "1"}]}')
        ]
    }
    ranker = LLMRanker(chat_generator=mock_chat_generator)

    result = ranker.run(query="test query", documents=documents)

    assert [document.id for document in result["documents"]] == ["2", "1"]


def test_run_empty_ranking_result_returns_empty_documents(mock_chat_generator):
    documents = [Document(id="1", content="first"), Document(id="2", content="second")]
    mock_chat_generator.run.return_value = {"replies": [ChatMessage.from_assistant('{"documents": []}')]}
    ranker = LLMRanker(chat_generator=mock_chat_generator)

    result = ranker.run(query="test query", documents=documents)

    assert result == {"documents": []}


def test_run_invalid_json_falls_back(mock_chat_generator):
    documents = [Document(id="1", content="first"), Document(id="2", content="second")]
    mock_chat_generator.run.return_value = {"replies": [ChatMessage.from_assistant("not-json")]}
    ranker = LLMRanker(chat_generator=mock_chat_generator, top_k=1, raise_on_failure=False)

    result = ranker.run(query="test query", documents=documents)

    assert result == {"documents": [documents[0]]}


def test_run_invalid_json_raises(mock_chat_generator):
    documents = [Document(id="1", content="first")]
    mock_chat_generator.run.return_value = {"replies": [ChatMessage.from_assistant("not-json")]}
    ranker = LLMRanker(chat_generator=mock_chat_generator, raise_on_failure=True)

    with pytest.raises(ValueError):
        ranker.run(query="test query", documents=documents)


def test_run_generator_exception_falls_back(mock_chat_generator):
    documents = [Document(id="1", content="first"), Document(id="2", content="second")]
    mock_chat_generator.run.side_effect = RuntimeError("generator failed")
    ranker = LLMRanker(chat_generator=mock_chat_generator, top_k=1)

    result = ranker.run(query="test query", documents=documents)

    assert result == {"documents": [documents[0]]}


def test_run_generator_exception_raises(mock_chat_generator):
    documents = [Document(id="1", content="first")]
    mock_chat_generator.run.side_effect = RuntimeError("generator failed")
    ranker = LLMRanker(chat_generator=mock_chat_generator, raise_on_failure=True)

    with pytest.raises(RuntimeError, match="generator failed"):
        ranker.run(query="test query", documents=documents)


def test_run_no_replies_falls_back(mock_chat_generator):
    documents = [Document(id="1", content="first"), Document(id="2", content="second")]
    mock_chat_generator.run.return_value = {"replies": []}
    ranker = LLMRanker(chat_generator=mock_chat_generator, top_k=1)

    result = ranker.run(query="test query", documents=documents)

    assert result == {"documents": [documents[0]]}


def test_run_reply_without_text_falls_back(mock_chat_generator):
    documents = [Document(id="1", content="first"), Document(id="2", content="second")]
    mock_chat_generator.run.return_value = {"replies": [ChatMessage.from_assistant(tool_calls=[])]}
    ranker = LLMRanker(chat_generator=mock_chat_generator, top_k=1)

    result = ranker.run(query="test query", documents=documents)

    assert result == {"documents": [documents[0]]}


def test_run_no_valid_document_ids_falls_back(mock_chat_generator):
    documents = [Document(id="1", content="first"), Document(id="2", content="second")]
    mock_chat_generator.run.return_value = {
        "replies": [ChatMessage.from_assistant('{"documents": [{"id": "missing"}]}')]
    }
    ranker = LLMRanker(chat_generator=mock_chat_generator, top_k=1)

    result = ranker.run(query="test query", documents=documents)

    assert result == {"documents": [documents[0]]}


def test_run_deduplicates_documents_before_ranking(mock_chat_generator):
    documents = [
        Document(id="duplicate", content="keep me", score=0.9),
        Document(id="duplicate", content="drop me", score=0.1),
        Document(id="unique", content="unique", score=0.2),
    ]
    mock_chat_generator.run.return_value = {
        "replies": [ChatMessage.from_assistant('{"documents": [{"id": "unique"}, {"id": "duplicate"}]}')]
    }
    ranker = LLMRanker(chat_generator=mock_chat_generator)

    result = ranker.run(query="test query", documents=documents)

    assert [document.content for document in result["documents"]] == ["unique", "keep me"]


def test_init_invalid_custom_prompt_raises(mock_chat_generator):
    with pytest.raises(TemplateSyntaxError):
        LLMRanker(chat_generator=mock_chat_generator, prompt_template="Rank {{ query }")
