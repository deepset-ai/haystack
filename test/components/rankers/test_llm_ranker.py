# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from unittest.mock import AsyncMock, Mock

import pytest
from jinja2 import TemplateSyntaxError

from haystack import Document
from haystack.components.generators.chat import MockChatGenerator
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.components.generators.chat.openai_responses import OpenAIResponsesChatGenerator
from haystack.components.rankers.llm_ranker import DEFAULT_PROMPT_TEMPLATE, LLMRanker
from haystack.dataclasses import ChatMessage


@pytest.fixture
def mock_chat_generator():
    return Mock(spec=OpenAIChatGenerator)


class FakeSyncOnlyChatGenerator:
    """A chat generator exposing only a synchronous `run` (no `run_async`) for the fallback path."""

    def __init__(self):
        self.run = Mock()


class TestInit:
    def test_init_default_generator(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        ranker = LLMRanker()

        assert ranker.top_k == 10
        assert ranker.raise_on_failure is False
        assert ranker.prompt == DEFAULT_PROMPT_TEMPLATE
        assert isinstance(ranker._chat_generator, OpenAIResponsesChatGenerator)
        assert ranker._chat_generator.model == "gpt-4.1-mini"
        assert ranker._chat_generator.generation_kwargs["store"] is False
        assert ranker._prompt_builder is not None

    def test_init_custom_generator(self, mock_chat_generator):
        ranker = LLMRanker(chat_generator=mock_chat_generator, top_k=5, raise_on_failure=True)

        assert ranker._chat_generator is mock_chat_generator
        assert ranker.top_k == 5
        assert ranker.raise_on_failure is True

    @pytest.mark.parametrize("top_k", [0, -1])
    def test_init_invalid_top_k(self, top_k):
        with pytest.raises(ValueError, match=rf"top_k must be > 0, but got {top_k}"):
            LLMRanker(top_k=top_k)

    def test_init_invalid_custom_prompt_raises(self, mock_chat_generator):
        with pytest.raises(TemplateSyntaxError):
            LLMRanker(chat_generator=mock_chat_generator, prompt="Rank {{ query }")

    @pytest.mark.parametrize(
        "prompt", ["Rank {{ query }}", "Rank {{ query }} using {{ documents|length }} docs with top_k={{ top_k }}"]
    )
    def test_init_prompt_requires_query_and_documents(self, mock_chat_generator, prompt):
        with pytest.raises(ValueError, match="prompt must include exactly the variables 'documents' and 'query'"):
            LLMRanker(chat_generator=mock_chat_generator, prompt=prompt)


class TestSerialization:
    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        chat_generator = OpenAIChatGenerator(generation_kwargs={"temperature": 0.5})
        ranker = LLMRanker(
            chat_generator=chat_generator,
            prompt="Rank {{ documents|length }} docs for {{ query }}",
            top_k=3,
            raise_on_failure=True,
        )

        assert ranker.to_dict() == {
            "type": "haystack.components.rankers.llm_ranker.LLMRanker",
            "init_parameters": {
                "chat_generator": chat_generator.to_dict(),
                "prompt": "Rank {{ documents|length }} docs for {{ query }}",
                "top_k": 3,
                "raise_on_failure": True,
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        chat_generator = OpenAIChatGenerator(generation_kwargs={"temperature": 0.5})
        data = {
            "type": "haystack.components.rankers.llm_ranker.LLMRanker",
            "init_parameters": {
                "chat_generator": chat_generator.to_dict(),
                "prompt": "Rank {{ documents|length }} docs for {{ query }}",
                "top_k": 3,
                "raise_on_failure": True,
            },
        }

        ranker = LLMRanker.from_dict(data)

        assert ranker.top_k == 3
        assert ranker.raise_on_failure is True
        assert ranker.prompt == "Rank {{ documents|length }} docs for {{ query }}"
        assert isinstance(ranker._chat_generator, OpenAIChatGenerator)
        assert ranker._chat_generator.to_dict() == chat_generator.to_dict()


class TestComponentLifecycle:
    def test_warm_up_delegates_to_chat_generator(self, mock_chat_generator):
        ranker = LLMRanker(chat_generator=mock_chat_generator)
        ranker.warm_up()
        mock_chat_generator.warm_up.assert_called_once()

    async def test_warm_up_async_delegates_to_chat_generator(self, mock_chat_generator):
        mock_chat_generator.warm_up_async = AsyncMock()
        ranker = LLMRanker(chat_generator=mock_chat_generator)
        await ranker.warm_up_async()
        mock_chat_generator.warm_up_async.assert_awaited_once()

    async def test_warm_up_async_falls_back_to_sync_warm_up(self):
        chat_generator = Mock(spec=["run", "warm_up"])
        ranker = LLMRanker(chat_generator=chat_generator)
        await ranker.warm_up_async()
        chat_generator.warm_up.assert_called_once()

    def test_close_delegates_to_chat_generator(self, mock_chat_generator):
        ranker = LLMRanker(chat_generator=mock_chat_generator)
        ranker.close()
        mock_chat_generator.close.assert_called_once()

    async def test_close_async_delegates_to_chat_generator(self, mock_chat_generator):
        mock_chat_generator.close_async = AsyncMock()
        ranker = LLMRanker(chat_generator=mock_chat_generator)
        await ranker.close_async()
        mock_chat_generator.close_async.assert_awaited_once()

    async def test_close_async_falls_back_to_sync_close(self):
        chat_generator = Mock(spec=["run", "close"])
        ranker = LLMRanker(chat_generator=chat_generator)
        await ranker.close_async()
        chat_generator.close.assert_called_once()

    def test_lifecycle_is_safe_when_chat_generator_lacks_methods(self):
        chat_generator = Mock(spec=["run"])
        ranker = LLMRanker(chat_generator=chat_generator)
        ranker.warm_up()
        ranker.close()


class TestRun:
    @pytest.mark.parametrize("run_top_k", [0, -1])
    def test_run_invalid_runtime_top_k(self, mock_chat_generator, run_top_k):
        ranker = LLMRanker(chat_generator=mock_chat_generator)

        with pytest.raises(ValueError, match=rf"top_k must be > 0, but got {run_top_k}"):
            ranker.run(query="test", documents=[Document(content="doc")], top_k=run_top_k)

    def test_run_empty_documents(self, mock_chat_generator):
        ranker = LLMRanker(chat_generator=mock_chat_generator)

        assert ranker.run(query="test", documents=[]) == {"documents": []}

    @pytest.mark.parametrize("bad_query", ["", "   ", None, 123])
    def test_run_invalid_query_returns_fallback(self, mock_chat_generator, bad_query):
        documents = [Document(id="1", content="first"), Document(id="2", content="second")]
        ranker = LLMRanker(chat_generator=mock_chat_generator, top_k=1)

        result = ranker.run(query=bad_query, documents=documents)

        assert result == {"documents": documents}
        mock_chat_generator.run.assert_not_called()

    def test_run_successful_ranking(self):
        documents = [
            Document(id="1", content="first"),
            Document(id="2", content="second"),
            Document(id="3", content="third"),
        ]
        chat_generator = MockChatGenerator('{"documents": [{"index": 2}, {"index": 1}, {"index": 3}]}')
        ranker = LLMRanker(chat_generator=chat_generator, top_k=2)

        result = ranker.run(query="test query", documents=documents)

        assert [document.id for document in result["documents"]] == ["2", "1"]

    def test_run_runtime_top_k_overrides_instance_top_k(self):
        documents = [
            Document(id="doc_1", content="first"),
            Document(id="doc_2", content="second"),
            Document(id="doc_3", content="third"),
        ]
        chat_generator = MockChatGenerator('{"documents": [{"index": 3}, {"index": 2}, {"index": 1}]}')
        ranker = LLMRanker(chat_generator=chat_generator, top_k=3)

        result = ranker.run(query="test query", documents=documents, top_k=1)

        assert [document.id for document in result["documents"]] == ["doc_3"]

    def test_run_deduplicates_documents_before_ranking(self, mock_chat_generator):
        documents = [
            Document(id="duplicate", content="keep me", score=0.9),
            Document(id="duplicate", content="drop me", score=0.1),
            Document(id="unique", content="unique", score=0.2),
        ]
        mock_chat_generator.run.return_value = {
            "replies": [ChatMessage.from_assistant('{"documents": [{"index": 2}, {"index": 1}]}')]
        }
        ranker = LLMRanker(chat_generator=mock_chat_generator)

        result = ranker.run(query="test query", documents=documents)

        assert [document.content for document in result["documents"]] == ["unique", "keep me"]

        prompt = mock_chat_generator.run.call_args.kwargs["messages"][0].text
        assert "Document 1:\ncontent: keep me" in prompt
        assert "Document 2:\ncontent: unique" in prompt
        assert "drop me" not in prompt
        assert "Document 3:" not in prompt

    def test_run_returns_only_documents_listed_by_the_llm(self):
        documents = [Document(id="1", content="first"), Document(id="2", content="second")]
        chat_generator = MockChatGenerator('{"documents": [{"index": 2}]}')
        ranker = LLMRanker(chat_generator=chat_generator, top_k=2)

        result = ranker.run(query="test query", documents=documents)

        assert [document.id for document in result["documents"]] == ["2"]

    def test_run_empty_ranking_result_returns_empty_documents(self):
        documents = [Document(id="1", content="first"), Document(id="2", content="second")]
        chat_generator = MockChatGenerator('{"documents": []}')
        ranker = LLMRanker(chat_generator=chat_generator)

        result = ranker.run(query="test query", documents=documents)

        assert result == {"documents": []}

    @pytest.mark.parametrize("index_json", ['"2"', "2.0"])
    def test_run_numeric_index_is_accepted(self, index_json):
        documents = [Document(id="1", content="first"), Document(id="2", content="second")]
        chat_generator = MockChatGenerator('{"documents": [{"index": ' + index_json + "}]}")
        ranker = LLMRanker(chat_generator=chat_generator)

        result = ranker.run(query="test query", documents=documents)

        assert result == {"documents": [documents[1]]}

    def test_run_preserves_duplicate_indices(self):
        documents = [Document(id="1", content="first"), Document(id="2", content="second")]
        chat_generator = MockChatGenerator('{"documents": [{"index": 2}, {"index": 2}, {"index": 1}]}')
        ranker = LLMRanker(chat_generator=chat_generator)

        result = ranker.run(query="test query", documents=documents)

        assert [document.id for document in result["documents"]] == ["2", "2", "1"]

    @pytest.mark.parametrize("index_json", ["99", "true", "1.9", "1e400"])
    def test_run_ignores_invalid_indices(self, index_json):
        documents = [Document(id="1", content="first"), Document(id="2", content="second")]
        chat_generator = MockChatGenerator('{"documents": [{"index": ' + index_json + '}, {"index": 2}, {"index": 1}]}')
        ranker = LLMRanker(chat_generator=chat_generator)

        result = ranker.run(query="test query", documents=documents)

        assert [document.id for document in result["documents"]] == ["2", "1"]

    def test_run_generator_exception_falls_back(self, mock_chat_generator, caplog):
        documents = [Document(id="1", content="first"), Document(id="2", content="second")]
        mock_chat_generator.run.side_effect = RuntimeError("generator failed")
        ranker = LLMRanker(chat_generator=mock_chat_generator, top_k=1)

        with caplog.at_level(logging.WARNING):
            result = ranker.run(query="test query", documents=documents)

        assert result == {"documents": documents}
        assert "Returning the deduplicated input documents unranked" in caplog.text
        assert "generator failed" in caplog.text

    def test_run_generator_exception_raises(self, mock_chat_generator):
        documents = [Document(id="1", content="first")]
        mock_chat_generator.run.side_effect = RuntimeError("generator failed")
        ranker = LLMRanker(chat_generator=mock_chat_generator, raise_on_failure=True)

        with pytest.raises(RuntimeError, match="generator failed"):
            ranker.run(query="test query", documents=documents)

    def test_run_no_replies_falls_back(self, mock_chat_generator):
        documents = [Document(id="1", content="first"), Document(id="2", content="second")]
        mock_chat_generator.run.return_value = {"replies": []}
        ranker = LLMRanker(chat_generator=mock_chat_generator, top_k=1)

        result = ranker.run(query="test query", documents=documents)

        assert result == {"documents": documents}

    def test_run_reply_without_text_falls_back(self):
        documents = [Document(id="1", content="first"), Document(id="2", content="second")]
        chat_generator = MockChatGenerator(ChatMessage.from_assistant(tool_calls=[]))
        ranker = LLMRanker(chat_generator=chat_generator, top_k=1)

        result = ranker.run(query="test query", documents=documents)

        assert result == {"documents": documents}

    def test_run_invalid_json_falls_back(self):
        documents = [
            Document(id="1", content="first"),
            Document(id="1", content="duplicate"),
            Document(id="2", content="second"),
        ]
        chat_generator = MockChatGenerator("not-json")
        ranker = LLMRanker(chat_generator=chat_generator, top_k=1, raise_on_failure=False)

        result = ranker.run(query="test query", documents=documents)

        assert result == {"documents": [documents[0], documents[2]]}

    def test_run_invalid_json_raises(self):
        documents = [Document(id="1", content="first")]
        chat_generator = MockChatGenerator("not-json")
        ranker = LLMRanker(chat_generator=chat_generator, raise_on_failure=True)

        with pytest.raises(ValueError):
            ranker.run(query="test query", documents=documents)

    @pytest.mark.parametrize(
        "reply", ['{"documents": [{"index": 0}, {"index": 3}]}', '{"documents": [{"index": "invalid"}]}']
    )
    def test_run_no_valid_document_indices_falls_back(self, reply):
        documents = [Document(id="1", content="first"), Document(id="2", content="second")]
        chat_generator = MockChatGenerator(reply)
        ranker = LLMRanker(chat_generator=chat_generator, top_k=1)

        result = ranker.run(query="test query", documents=documents)

        assert result == {"documents": documents}


class TestRunAsync:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("run_top_k", [0, -1])
    async def test_run_async_invalid_runtime_top_k(self, mock_chat_generator, run_top_k):
        ranker = LLMRanker(chat_generator=mock_chat_generator)

        with pytest.raises(ValueError, match=rf"top_k must be > 0, but got {run_top_k}"):
            await ranker.run_async(query="test", documents=[Document(content="doc")], top_k=run_top_k)

    @pytest.mark.asyncio
    async def test_run_async_none_query_returns_fallback(self):
        documents = [Document(id="1", content="first"), Document(id="2", content="second")]
        mock_chat_generator = Mock(spec=OpenAIChatGenerator)
        mock_chat_generator.run_async = AsyncMock()
        ranker = LLMRanker(chat_generator=mock_chat_generator)

        assert await ranker.run_async(query=None, documents=documents) == {"documents": documents}  # type: ignore[arg-type]
        mock_chat_generator.run_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_async(self):
        documents = [
            Document(id="1", content="first"),
            Document(id="2", content="second"),
            Document(id="3", content="third"),
        ]
        mock_chat_generator = Mock(spec=OpenAIChatGenerator)
        mock_chat_generator.run_async = AsyncMock(
            return_value={
                "replies": [ChatMessage.from_assistant('{"documents": [{"index": 2}, {"index": 1}, {"index": 3}]}')]
            }
        )
        ranker = LLMRanker(chat_generator=mock_chat_generator, top_k=2)

        result = await ranker.run_async(query="test query", documents=documents)

        assert [document.id for document in result["documents"]] == ["2", "1"]
        mock_chat_generator.run_async.assert_awaited_once()
        mock_chat_generator.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_async_fallback_to_sync_run(self):
        documents = [
            Document(id="1", content="first"),
            Document(id="2", content="second"),
            Document(id="3", content="third"),
        ]
        fake_chat_generator = FakeSyncOnlyChatGenerator()
        fake_chat_generator.run.return_value = {
            "replies": [ChatMessage.from_assistant('{"documents": [{"index": 2}, {"index": 1}, {"index": 3}]}')]
        }
        assert not hasattr(fake_chat_generator, "run_async")
        ranker = LLMRanker(chat_generator=fake_chat_generator, top_k=2)

        result = await ranker.run_async(query="test query", documents=documents)

        assert [document.id for document in result["documents"]] == ["2", "1"]
        fake_chat_generator.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_async_generator_exception_falls_back(self, caplog):
        documents = [Document(id="1", content="first"), Document(id="2", content="second")]
        mock_chat_generator = Mock(spec=OpenAIChatGenerator)
        mock_chat_generator.run_async = AsyncMock(side_effect=RuntimeError("generator failed"))
        ranker = LLMRanker(chat_generator=mock_chat_generator, top_k=1, raise_on_failure=False)

        with caplog.at_level(logging.WARNING):
            result = await ranker.run_async(query="test query", documents=documents)

        assert result == {"documents": documents}
        assert "Returning the deduplicated input documents unranked" in caplog.text
        assert "generator failed" in caplog.text

    @pytest.mark.asyncio
    async def test_run_async_generator_exception_raises(self):
        documents = [Document(id="1", content="first")]
        mock_chat_generator = Mock(spec=OpenAIChatGenerator)
        mock_chat_generator.run_async = AsyncMock(side_effect=RuntimeError("generator failed"))
        ranker = LLMRanker(chat_generator=mock_chat_generator, raise_on_failure=True)

        with pytest.raises(RuntimeError, match="generator failed"):
            await ranker.run_async(query="test query", documents=documents)

    @pytest.mark.parametrize("raise_on_failure", [False, True])
    async def test_run_async_parsing_failure_policy(self, raise_on_failure):
        documents = [Document(id="1", content="first"), Document(id="2", content="second")]
        ranker = LLMRanker(
            chat_generator=MockChatGenerator('{"documents": [{"index": 0}]}'),
            top_k=1,
            raise_on_failure=raise_on_failure,
        )
        if raise_on_failure:
            with pytest.raises(ValueError, match="^Ranking response did not contain any valid document indices\\.$"):
                await ranker.run_async(query="test query", documents=documents)
        else:
            assert await ranker.run_async(query="test query", documents=documents) == {"documents": documents}


class TestTracing:
    def test_run_traces_chat_generator_token_usage(self, spying_tracer):
        ranker = LLMRanker(chat_generator=MockChatGenerator('{"documents": [{"index": 1}]}'))

        ranker.run(query="capital of Germany", documents=[Document(content="Berlin")])

        gen_spans = [s for s in spying_tracer.spans if s.operation_name == "haystack.chat_generator.run"]
        assert len(gen_spans) == 1
        output = gen_spans[0].tags["haystack.component.output"]
        assert output["replies"][0].meta["usage"]["total_tokens"] > 0

    @pytest.mark.asyncio
    async def test_run_async_traces_chat_generator_token_usage(self, spying_tracer):
        ranker = LLMRanker(chat_generator=MockChatGenerator('{"documents": [{"index": 1}]}'))

        await ranker.run_async(query="capital of Germany", documents=[Document(content="Berlin")])

        gen_spans = [s for s in spying_tracer.spans if s.operation_name == "haystack.chat_generator.run"]
        assert len(gen_spans) == 1
        output = gen_spans[0].tags["haystack.component.output"]
        assert output["replies"][0].meta["usage"]["total_tokens"] > 0


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="Export an env var called OPENAI_API_KEY containing the API key to run this test.",
)
class TestIntegration:
    def test_live_run_ranks_rust_for_programming_language_query(self):
        documents = [
            Document(id="doc-berlin", content="Berlin is the capital of Germany."),
            Document(id="doc-paris", content="Paris is the capital of France."),
            Document(id="doc-rust", content="Rust is a systems programming language focused on safety."),
        ]
        ranker = LLMRanker(top_k=1)

        result = ranker.run(query="Which document is about a programming language?", documents=documents)

        assert [document.id for document in result["documents"]] == ["doc-rust"]

    async def test_live_run_async_ranks_rust_for_programming_language_query(self):
        documents = [
            Document(id="doc-berlin", content="Berlin is the capital of Germany."),
            Document(id="doc-paris", content="Paris is the capital of France."),
            Document(id="doc-rust", content="Rust is a systems programming language focused on safety."),
        ]
        ranker = LLMRanker(top_k=1)

        result = await ranker.run_async(query="Which document is about a programming language?", documents=documents)

        assert [document.id for document in result["documents"]] == ["doc-rust"]
