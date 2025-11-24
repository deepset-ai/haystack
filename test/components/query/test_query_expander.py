# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from unittest.mock import Mock

import pytest

from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.components.query.query_expander import DEFAULT_PROMPT_TEMPLATE, QueryExpander
from haystack.dataclasses.chat_message import ChatMessage


@pytest.fixture
def mock_chat_generator():
    mock_generator = Mock(spec=OpenAIChatGenerator)
    return mock_generator


@pytest.fixture
def mock_chat_generator_with_warm_up():
    mock_generator = Mock(spec=OpenAIChatGenerator)
    mock_generator.warm_up = lambda: None
    return mock_generator


class TestQueryExpander:
    def test_init_default_generator(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")
        expander = QueryExpander()

        assert expander.n_expansions == 4
        assert expander.include_original_query is True
        assert isinstance(expander.chat_generator, OpenAIChatGenerator)
        assert expander.chat_generator.model == "gpt-4.1-mini"
        assert expander._prompt_builder is not None

    def test_init_custom_generator(self, mock_chat_generator):
        expander = QueryExpander(chat_generator=mock_chat_generator, n_expansions=3)

        assert expander.n_expansions == 3
        assert expander.chat_generator is mock_chat_generator

    def test_run_warm_up(self, mock_chat_generator_with_warm_up):
        expander = QueryExpander(chat_generator=mock_chat_generator_with_warm_up)
        mock_chat_generator_with_warm_up.run.return_value = {"queries": ["test query"]}

        expander.warm_up()
        expander.run("test query")

        assert expander._is_warmed_up is True
        assert expander.run("test query") == {"queries": ["test query"]}

    def test_warm_up(self, mock_chat_generator):
        expander = QueryExpander(chat_generator=mock_chat_generator)
        expander.warm_up()
        assert expander._is_warmed_up is True

    def test_init_negative_expansions_raises_error(self):
        with pytest.raises(ValueError, match="n_expansions must be positive"):
            QueryExpander(n_expansions=-1)

    def test_init_zero_expansions_raises_error(self):
        with pytest.raises(ValueError, match="n_expansions must be positive"):
            QueryExpander(n_expansions=0)

    def test_init_custom_prompt_template(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")
        custom_template = "Custom template: {{ query }} with {{ n_expansions }} expansions"
        expander = QueryExpander(prompt_template=custom_template)

        assert expander.prompt_template == custom_template

    def test_run_negative_expansions_raises_error(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")
        expander = QueryExpander()
        expander.warm_up()
        with pytest.raises(ValueError, match="n_expansions must be positive"):
            expander.run("test query", n_expansions=-1)

    def test_run_zero_expansions_raises_error(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")
        expander = QueryExpander(n_expansions=4)
        expander.warm_up()
        with pytest.raises(ValueError, match="n_expansions must be positive"):
            expander.run("test query", n_expansions=0)

    def test_run_with_runtime_n_expansions_override(self, mock_chat_generator):
        mock_chat_generator.run.return_value = {
            "replies": [ChatMessage.from_assistant('{"queries": ["alt1", "alt2"]}')]
        }

        expander = QueryExpander(chat_generator=mock_chat_generator, n_expansions=4, include_original_query=False)
        expander.warm_up()
        result = expander.run("test query", n_expansions=2)

        # should request 2 expansions
        call_args = mock_chat_generator.run.call_args[1]["messages"][0].text
        assert "2" in call_args
        assert len(result["queries"]) == 2
        assert result["queries"] == ["alt1", "alt2"]

    def test_run_successful_expansion(self, mock_chat_generator):
        mock_chat_generator.run.return_value = {
            "replies": [
                ChatMessage.from_assistant(
                    '{"queries": ["alternative query 1", "alternative query 2", "alternative query 3"]}'
                )
            ]
        }

        expander = QueryExpander(chat_generator=mock_chat_generator, n_expansions=3)
        expander.warm_up()
        result = expander.run("original query")

        assert result["queries"] == [
            "alternative query 1",
            "alternative query 2",
            "alternative query 3",
            "original query",
        ]
        mock_chat_generator.run.assert_called_once()

    def test_run_without_including_original(self, mock_chat_generator):
        mock_chat_generator.run.return_value = {
            "replies": [ChatMessage.from_assistant('{"queries": ["alt1", "alt2"]}')]
        }

        expander = QueryExpander(chat_generator=mock_chat_generator, include_original_query=False)
        expander.warm_up()
        result = expander.run("original")

        assert result["queries"] == ["alt1", "alt2"]

    def test_run_empty_query(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")
        expander = QueryExpander()
        expander.warm_up()
        result = expander.run("")

        assert result["queries"] == [""]

    def test_run_empty_query_no_original(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")
        expander = QueryExpander(include_original_query=False)
        expander.warm_up()
        result = expander.run("   ")

        assert result["queries"] == []

    def test_run_whitespace_only_query(self, monkeypatch, caplog):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")
        expander = QueryExpander()
        expander.warm_up()
        with caplog.at_level(logging.WARNING):
            result = expander.run("\t\n  \r")
        assert result["queries"] == ["\t\n  \r"]
        assert "Empty query provided" in caplog.text

    def test_run_generator_no_replies(self, mock_chat_generator):
        mock_chat_generator.run.return_value = {"replies": []}
        expander = QueryExpander(chat_generator=mock_chat_generator)
        expander.warm_up()
        result = expander.run("test query")

        assert result["queries"] == ["test query"]

    def test_run_generator_exception(self, mock_chat_generator):
        mock_chat_generator.run.side_effect = Exception("Generator error")
        expander = QueryExpander(chat_generator=mock_chat_generator)
        expander.warm_up()
        result = expander.run("test query")
        assert result["queries"] == ["test query"]

    def test_run_invalid_json_response(self, mock_chat_generator):
        mock_chat_generator.run.return_value = {"replies": [ChatMessage.from_assistant("invalid json response")]}
        expander = QueryExpander(chat_generator=mock_chat_generator)
        expander.warm_up()
        result = expander.run("test query")
        assert result["queries"] == ["test query"]

    def test_parse_expanded_queries_valid_json(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")
        expander = QueryExpander()
        expander.warm_up()
        queries = expander._parse_expanded_queries('{"queries": ["query1", "query2", "query3"]}')
        assert queries == ["query1", "query2", "query3"]

    def test_parse_expanded_queries_invalid_json(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")
        expander = QueryExpander()
        expander.warm_up()
        queries = expander._parse_expanded_queries("not json")
        assert queries == []

    def test_parse_expanded_queries_empty_string(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")
        expander = QueryExpander()
        expander.warm_up()
        queries = expander._parse_expanded_queries("")
        assert queries == []

    def test_parse_expanded_queries_non_list_json(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")
        expander = QueryExpander()
        expander.warm_up()
        queries = expander._parse_expanded_queries('{"not": "a list"}')
        assert queries == []

    def test_parse_expanded_queries_mixed_types(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")
        expander = QueryExpander()
        expander.warm_up()
        queries = expander._parse_expanded_queries('{"queries": ["valid query", 123, "", "another valid"]}')
        assert queries == ["valid query", "another valid"]

    def test_run_query_deduplication(self, mock_chat_generator):
        mock_chat_generator.run.return_value = {
            "replies": [ChatMessage.from_assistant('{"queries": ["original query", "alt1", "alt2"]}')]
        }
        expander = QueryExpander(chat_generator=mock_chat_generator, include_original_query=True)
        expander.warm_up()
        result = expander.run("original query")
        assert result["queries"] == ["original query", "alt1", "alt2"]
        assert len(result["queries"]) == 3

    def test_run_truncates_excess_queries(self, mock_chat_generator, caplog):
        mock_chat_generator.run.return_value = {
            "replies": [ChatMessage.from_assistant('{"queries": ["q1", "q2", "q3", "q4", "q5"]}')]
        }
        expander = QueryExpander(chat_generator=mock_chat_generator, n_expansions=3, include_original_query=False)
        expander.warm_up()

        with caplog.at_level(logging.WARNING):
            result = expander.run("test query")

        assert len(result["queries"]) == 3
        assert result["queries"] == ["q1", "q2", "q3"]
        assert "Generated 5 queries but only 3 were requested" in caplog.text
        assert "Truncating" in caplog.text

    def test_run_with_custom_template(self, mock_chat_generator):
        custom_template = """
        Create {{ n_expansions }} alternative search queries for: {{ query }}
        Return as JSON: {"queries": ["query1", "query2"]}
        """

        mock_chat_generator.run.return_value = {
            "replies": [ChatMessage.from_assistant('{"queries": ["custom alt 1", "custom alt 2"]}')]
        }

        expander = QueryExpander(
            chat_generator=mock_chat_generator,
            prompt_template=custom_template,
            n_expansions=2,
            include_original_query=False,
        )
        expander.warm_up()
        result = expander.run("test query")

        assert result["queries"] == ["custom alt 1", "custom alt 2"]

        mock_chat_generator.run.assert_called_once()
        call_args = mock_chat_generator.run.call_args[1]["messages"][0].text
        assert "Create 2 alternative search queries for: test query" in call_args
        assert "Return as JSON" in call_args

    def test_component_output_types(self, mock_chat_generator, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")
        expander = QueryExpander()
        expander.warm_up()

        mock_chat_generator.run.return_value = {
            "replies": [ChatMessage.from_assistant('{"queries": ["test1", "test2"]}')]
        }
        expander.chat_generator = mock_chat_generator

        result = expander.run("test")
        assert "queries" in result
        assert isinstance(result["queries"], list)
        assert all(isinstance(q, str) for q in result["queries"])

    @pytest.mark.parametrize("variable", ["query", "n_expansions"])
    def test_prompt_template_missing_variable(self, caplog, variable, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")
        if variable == "query":
            template_missing_variable = "Generate {{ n_expansions }} expansions"
        else:
            template_missing_variable = "Generate expansions for {{ query }}"

        with caplog.at_level(logging.WARNING):
            QueryExpander(prompt_template=template_missing_variable)

        assert f"The prompt template does not contain the '{variable}' variable" in caplog.text
        assert "This may cause issues during execution" in caplog.text

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")
        generator = OpenAIChatGenerator(model="gpt-4.1-mini")
        expander = QueryExpander(chat_generator=generator, n_expansions=2, include_original_query=False)

        serialized_query_expander = expander.to_dict()

        assert serialized_query_expander == {
            "type": "haystack.components.query.query_expander.QueryExpander",
            "init_parameters": {
                "chat_generator": {
                    "type": "haystack.components.generators.chat.openai.OpenAIChatGenerator",
                    "init_parameters": {
                        "model": "gpt-4.1-mini",
                        "streaming_callback": None,
                        "api_base_url": None,
                        "organization": None,
                        "generation_kwargs": {},
                        "api_key": {"type": "env_var", "env_vars": ["OPENAI_API_KEY"], "strict": True},
                        "timeout": None,
                        "max_retries": None,
                        "tools": None,
                        "tools_strict": False,
                        "http_client_kwargs": None,
                    },
                },
                "prompt_template": DEFAULT_PROMPT_TEMPLATE,
                "n_expansions": 2,
                "include_original_query": False,
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")

        data = {
            "type": "haystack.components.query.query_expander.QueryExpander",
            "init_parameters": {
                "chat_generator": {
                    "type": "haystack.components.generators.chat.openai.OpenAIChatGenerator",
                    "init_parameters": {
                        "model": "gpt-4.1-mini",
                        "streaming_callback": None,
                        "api_base_url": None,
                        "organization": None,
                        "generation_kwargs": {},
                        "api_key": {"type": "env_var", "env_vars": ["OPENAI_API_KEY"], "strict": True},
                        "timeout": None,
                        "max_retries": None,
                        "tools": None,
                        "tools_strict": False,
                        "http_client_kwargs": None,
                    },
                },
                "prompt_template": DEFAULT_PROMPT_TEMPLATE,
                "n_expansions": 2,
                "include_original_query": False,
            },
        }

        expander = QueryExpander.from_dict(data)

        assert expander.n_expansions == 2
        assert expander.include_original_query == False
        assert expander.prompt_template == DEFAULT_PROMPT_TEMPLATE
        assert isinstance(expander.chat_generator, OpenAIChatGenerator)
        assert expander.chat_generator.model == "gpt-4.1-mini"


@pytest.mark.integration
class TestQueryExpanderIntegration:
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    def test_query_expansion(self):
        expander = QueryExpander(n_expansions=3)
        expander.warm_up()
        result = expander.run("renewable energy sources")

        assert len(result["queries"]) == 4
        assert all(len(q.strip()) > 0 for q in result["queries"])
        assert "renewable energy sources" in result["queries"]

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    def test_different_domains(self):
        test_queries = ["machine learning algorithms", "climate change effects", "quantum computing applications"]

        expander = QueryExpander(n_expansions=2, include_original_query=False)
        expander.warm_up()

        for query in test_queries:
            result = expander.run(query)

            # Should return exactly 2 expansions (no original)
            assert len(result["queries"]) == 2

            # Should be different from original
            assert query not in result["queries"]
