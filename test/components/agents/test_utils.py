# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.components.agents.state import State, replace_values
from haystack.components.agents.utils import (
    _accumulate_usage,
    _context_tokens_from_usage,
    _record_context_tokens,
    _render_prompt_messages,
    _select_tools_by_name,
    _template_for_role,
    _validate_prompt_message_blocks,
)
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.chat_message import ChatRole
from haystack.tools import Tool
from haystack.tools.toolset import Toolset


def _user_msg(text: str) -> str:
    return f'{{% message role="user" %}}{text}{{% endmessage %}}'


def _sys_msg(text: str) -> str:
    return f'{{% message role="system" %}}{text}{{% endmessage %}}'


def _tool_function(value: str) -> str:
    return value


@pytest.fixture
def first_tool() -> Tool:
    return Tool(
        name="first_tool", description="First test tool.", parameters={"type": "object"}, function=_tool_function
    )


@pytest.fixture
def second_tool() -> Tool:
    return Tool(
        name="second_tool", description="Second test tool.", parameters={"type": "object"}, function=_tool_function
    )


class TestAccumulateUsage:
    def test_sums_flat_numeric_keys(self):
        current = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        new = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        assert _accumulate_usage(current, new) == {"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20}

    def test_merges_nested_detail_dicts_recursively(self):
        current = {"prompt_tokens": 10, "completion_tokens_details": {"reasoning_tokens": 2, "audio_tokens": 0}}
        new = {
            "prompt_tokens": 4,
            "completion_tokens_details": {"reasoning_tokens": 3, "audio_tokens": 1},
            "prompt_tokens_details": {"cached_tokens": 6},
        }
        assert _accumulate_usage(current, new) == {
            "prompt_tokens": 14,
            "completion_tokens_details": {"reasoning_tokens": 5, "audio_tokens": 1},
            "prompt_tokens_details": {"cached_tokens": 6},
        }

    def test_adds_keys_missing_in_current(self):
        assert _accumulate_usage({"prompt_tokens": 5}, {"completion_tokens": 7}) == {
            "prompt_tokens": 5,
            "completion_tokens": 7,
        }

    def test_copies_new_nested_values(self):
        new = {"details": {"cached_tokens": 1}}
        result = _accumulate_usage({}, new)
        new["details"]["cached_tokens"] = 999
        assert result == {"details": {"cached_tokens": 1}}

    def test_falls_back_to_new_for_non_numeric_values(self):
        assert _accumulate_usage("old-model", "new-model") == "new-model"
        assert _accumulate_usage(5, "stringified") == "stringified"
        assert _accumulate_usage({"model": "old"}, {"model": "new"}) == {"model": "new"}

    def test_sums_floats(self):
        assert _accumulate_usage(1.5, 2.25) == 3.75


class TestSelectToolsByName:
    def test_selects_standalone_tools_by_name(self, first_tool: Tool, second_tool: Tool):
        assert _select_tools_by_name([first_tool, second_tool], [first_tool.name]) == [first_tool]

    def test_raises_for_invalid_name(self, first_tool: Tool):
        with pytest.raises(ValueError, match="The following tool names are not valid"):
            _select_tools_by_name([first_tool], ["unknown"])

    @pytest.mark.parametrize("configured_tools", [[], Toolset([])], ids=["empty_list", "empty_toolset"])
    def test_raises_when_no_tools_configured(self, configured_tools, first_tool: Tool):
        with pytest.raises(ValueError, match="No tools were configured for the Agent at initialization."):
            _select_tools_by_name(configured_tools, [first_tool.name])

    def test_reduces_plain_toolsets_to_matching_tools(self, first_tool: Tool, second_tool: Tool):
        toolset = Toolset([first_tool, second_tool])
        selected = _select_tools_by_name([toolset], [first_tool.name])
        assert selected == [first_tool]
        # The configured toolset is untouched.
        assert list(toolset) == [first_tool, second_tool]

    def test_selects_standalone_tools_and_toolsets(self, first_tool: Tool, second_tool: Tool):
        toolset = Toolset([first_tool])
        selected = _select_tools_by_name([second_tool, toolset], [first_tool.name, second_tool.name])
        assert selected == [second_tool, first_tool]

    def test_warms_up_lazy_toolsets_to_resolve_names(self, first_tool: Tool, second_tool: Tool):
        class LazyToolset(Toolset):
            """A Toolset that loads its tools on warm_up(), like toolsets backed by external services."""

            def __init__(self, tools_to_load):
                self._tools_to_load = tools_to_load
                super().__init__([])  # no tools until warm_up

            def warm_up(self):
                if not self.tools:
                    self.tools = list(self._tools_to_load)

        toolset = LazyToolset([first_tool, second_tool])
        assert list(toolset) == []  # not loaded yet

        # Name resolution warms the toolset first, so lazily loaded tools are selectable.
        selected = _select_tools_by_name([toolset], [first_tool.name])
        assert selected == [first_tool]

    def test_spawns_toolsets_without_mutating_them(self, first_tool: Tool, second_tool: Tool):
        class RunScopedToolset(Toolset):
            """A Toolset overriding spawn(), signaling run-scoped state."""

            def __init__(self, tools):
                super().__init__(tools)
                self.selected: set[str] | None = None

            def spawn(self, selected_tool_names: set[str] | None = None) -> "RunScopedToolset":
                new = RunScopedToolset(list(self.tools))
                new.selected = set(selected_tool_names) if selected_tool_names is not None else None
                return new

        toolset = RunScopedToolset([first_tool, second_tool])
        selected = _select_tools_by_name([toolset], [first_tool.name])

        run_copy = selected[0]
        assert isinstance(run_copy, RunScopedToolset)
        assert run_copy is not toolset
        assert run_copy.selected == {first_tool.name}
        # The configured toolset is untouched.
        assert toolset.selected is None

    def test_selects_tools_not_surfaced_by_iteration(self, first_tool: Tool, second_tool: Tool):
        class DiscoveryToolset(Toolset):
            """A dynamic Toolset without a spawn() override: iteration yields less than get_selectable_tools()."""

            def __init__(self, catalog):
                self._catalog = catalog
                super().__init__([])

            def __iter__(self):
                yield from []  # nothing discovered yet

            def get_selectable_tools(self) -> list[Tool]:
                return list(self._catalog)

        toolset = DiscoveryToolset([first_tool, second_tool])
        # The requested tool must be selected even though iteration does not surface it.
        selected = _select_tools_by_name([toolset], [first_tool.name])
        assert selected == [first_tool]


class TestContextTokensFromUsage:
    """`_context_tokens_from_usage` normalizes real provider `meta["usage"]` shapes to input + output tokens."""

    # OpenAI Chat Completions (core repo): reasoning_tokens is a subset of completion_tokens (64 of 74), so
    # context_tokens includes reasoning.
    OPENAI_CHAT_USAGE = {
        "completion_tokens": 74,
        "prompt_tokens": 19,
        "total_tokens": 93,
        "completion_tokens_details": {
            "accepted_prediction_tokens": 0,
            "audio_tokens": 0,
            "reasoning_tokens": 64,
            "rejected_prediction_tokens": 0,
        },
        "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
    }
    # OpenAI Responses API (core repo).
    OPENAI_RESPONSES_USAGE = {
        "input_tokens": 19,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens": 58,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 77,
    }
    # Anthropic chat generator (integration): no total_tokens, and thinking_tokens is a subset of completion_tokens
    # (57 of 63).
    ANTHROPIC_USAGE = {
        "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0},
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "inference_geo": "not_available",
        "output_tokens_details": {"thinking_tokens": 57},
        "server_tool_use": None,
        "service_tier": "standard",
        "prompt_tokens": 49,
        "completion_tokens": 63,
    }
    # Amazon Bedrock chat generator (integration).
    BEDROCK_USAGE = {
        "prompt_tokens": 20,
        "completion_tokens": 5,
        "total_tokens": 25,
        "cache_read_input_tokens": 0,
        "cache_write_input_tokens": 0,
        "cache_details": {},
    }
    # Cohere chat generator (integration).
    COHERE_USAGE = {"prompt_tokens": 15.0, "completion_tokens": 3.0}
    # Mistral chat generator (integration).
    MISTRAL_USAGE = {
        "prompt_tokens": 30,
        "total_tokens": 34,
        "completion_tokens": 4,
        "prompt_tokens_details": {"cached_tokens": 0},
    }
    # Nvidia chat generator (integration).
    NVIDIA_USAGE = {
        "completion_tokens": 2,
        "prompt_tokens": 48,
        "total_tokens": 50,
        "completion_tokens_details": None,
        "prompt_tokens_details": {"audio_tokens": None, "cached_tokens": 16},
    }
    # Google GenAI chat generator (integration): thoughts_token_count (281) is NOT part of completion_tokens;
    # total_tokens (300) includes it, so context_tokens (19) excludes thoughts.
    GOOGLE_GENAI_USAGE = {
        "prompt_tokens": 16,
        "completion_tokens": 3,
        "total_tokens": 300,
        "thoughts_token_count": 281,
        "prompt_token_count": 16,
        "candidates_token_count": 3,
        "total_token_count": 300,
        "prompt_tokens_details": [{"modality": "TEXT", "token_count": 16}],
    }

    @pytest.mark.parametrize(
        "usage, expected",
        [
            (OPENAI_CHAT_USAGE, 93),
            (OPENAI_RESPONSES_USAGE, 77),
            (ANTHROPIC_USAGE, 112),
            (BEDROCK_USAGE, 25),
            (COHERE_USAGE, 18),
            (MISTRAL_USAGE, 34),
            (NVIDIA_USAGE, 50),
            (GOOGLE_GENAI_USAGE, 19),  # 16 + 3, deliberately not the 300 total (which includes 281 thoughts tokens)
            ({"prompt_tokens": 10}, 10),  # only one side reported -> partial count
            ({"completion_tokens": 7}, 7),
            ({}, 0),  # no usage reported
            ({"foo": 5, "bar": 9}, 0),  # no recognized keys
        ],
    )
    def test_normalizes_provider_shapes(self, usage, expected):
        assert _context_tokens_from_usage(usage) == expected

    def test_bool_values_are_not_counted_as_tokens(self):
        # bool is an int subclass; True/False under a token key must be skipped, not summed.
        assert _context_tokens_from_usage({"prompt_tokens": True, "completion_tokens": 5}) == 5


class TestRecordContextTokens:
    """`_record_context_tokens` replaces the value with the latest reply's input+output, only when usage is reported."""

    def _state(self) -> State:
        state = State(schema={"context_tokens": {"type": int, "handler": replace_values}})
        state.set("context_tokens", 0)
        return state

    def test_records_latest_reply_usage_replacing_previous_value(self):
        state = self._state()
        state.set("context_tokens", 999)
        _record_context_tokens(
            state=state,
            llm_messages=[
                ChatMessage.from_assistant("Hi", meta={"usage": {"prompt_tokens": 12, "completion_tokens": 3}})
            ],
        )
        assert state.get("context_tokens") == 15

    def test_no_messages_leaves_value_untouched(self):
        state = self._state()
        state.set("context_tokens", 42)
        _record_context_tokens(state=state, llm_messages=[])
        assert state.get("context_tokens") == 42

    def test_missing_or_empty_usage_leaves_value_untouched(self):
        state = self._state()
        _record_context_tokens(state=state, llm_messages=[ChatMessage.from_assistant("no usage here")])
        _record_context_tokens(state=state, llm_messages=[ChatMessage.from_assistant("empty", meta={"usage": {}})])
        assert state.get("context_tokens") == 0


class TestPrompts:
    def test_system_prompt_plain_string(self):
        prompt_builder = ChatPromptBuilder(template=_template_for_role("You are a helpful assistant.", "system"))
        messages = _render_prompt_messages(
            prompt_builder=prompt_builder, expected_role=ChatRole.SYSTEM, prompt_label="system_prompt", kwargs={}
        )
        assert messages[0].is_from(ChatRole.SYSTEM)
        assert messages[0].text == "You are a helpful assistant."

    def test_system_prompt_with_template_variables(self):
        prompt_builder = ChatPromptBuilder(
            template=_template_for_role(
                _sys_msg("You are an assistant for {{company}}. Your role is {{role}}."), "system"
            )
        )
        messages = _render_prompt_messages(
            prompt_builder=prompt_builder,
            expected_role=ChatRole.SYSTEM,
            prompt_label="system_prompt",
            kwargs={"company": "Acme", "role": "support agent"},
        )
        sys_msg = messages[0]
        assert sys_msg.is_from(ChatRole.SYSTEM)
        assert sys_msg.text == "You are an assistant for Acme. Your role is support agent."

    def test_system_prompt_with_meta(self):
        prompt_builder = ChatPromptBuilder(
            template="{% message role='system' meta={'key': 'value'} %}System message with meta{% endmessage %}"
        )
        messages = _render_prompt_messages(
            prompt_builder=prompt_builder, expected_role=ChatRole.SYSTEM, prompt_label="system_prompt", kwargs={}
        )
        assert messages[0].is_from(ChatRole.SYSTEM)
        assert messages[0].text == "System message with meta"
        assert messages[0].meta == {"key": "value"}

    def test_user_prompt_only_variables_forwarded_to_builder(self):
        prompt_builder = ChatPromptBuilder(template=_user_msg("Question: {{question}}"))
        # 'irrelevant_kwarg' is not a template variable — must not raise
        messages = _render_prompt_messages(
            prompt_builder=prompt_builder,
            expected_role=ChatRole.USER,
            prompt_label="user_prompt",
            kwargs={"question": "Will it snow?", "irrelevant_kwarg": "unused"},
        )
        assert messages[0].text == "Question: Will it snow?"

    def test_user_prompt_with_template_variables(self):
        prompt_builder = ChatPromptBuilder(
            template=_user_msg(
                "Hello {{name|upper}}, check weather for: "
                + "{% for c in cities %}{{c}}{% if not loop.last %}, {% endif %}{% endfor %}"
                + " on {{date}}?"
            )
        )
        messages = _render_prompt_messages(
            prompt_builder=prompt_builder,
            expected_role=ChatRole.USER,
            prompt_label="user_prompt",
            kwargs={"name": "Alice", "cities": ["Berlin", "Paris", "Rome"], "date": "2024-01-15"},
        )
        assert messages[0].text == "Hello ALICE, check weather for: Berlin, Paris, Rome on 2024-01-15?"

    def test_prompt_wrong_role_raises(self):
        with pytest.raises(ValueError, match="system_prompt message block must have role 'system'"):
            _validate_prompt_message_blocks(
                user_prompt=None, system_prompt=_user_msg("This is a user message, not system.")
            )

        with pytest.raises(ValueError, match="user_prompt message block must have role 'user'"):
            _validate_prompt_message_blocks(
                user_prompt=_sys_msg("This is a system message, not user."), system_prompt=None
            )

    def test_dynamic_prompt_role_raises(self):
        prompt_builder = ChatPromptBuilder(
            template="{% message role=role_name %}Question: {{question}}{% endmessage %}"
        )
        with pytest.raises(ValueError, match="user_prompt must render to a user message"):
            _render_prompt_messages(
                prompt_builder=prompt_builder,
                expected_role=ChatRole.USER,
                prompt_label="user_prompt",
                kwargs={"role_name": "assistant", "question": "Will it snow?"},
            )

    def test_prompt_multiple_message_blocks_raises(self):
        multi_message_prompt = """{% message role='system' %}You are a helpful assistant.{% endmessage %}
        {% message role='user' %}How are you?{% endmessage %}"""

        with pytest.raises(ValueError, match="system_prompt must define exactly one message block"):
            _validate_prompt_message_blocks(user_prompt=None, system_prompt=multi_message_prompt)

        with pytest.raises(ValueError, match="user_prompt must define exactly one message block"):
            _validate_prompt_message_blocks(user_prompt=multi_message_prompt, system_prompt=None)
