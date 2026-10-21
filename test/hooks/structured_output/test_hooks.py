# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest
from pydantic import BaseModel

from haystack.components.agents import Agent
from haystack.components.agents.state import State
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.hooks import hook
from haystack.hooks.structured_output import StructuredOutputHook
from haystack.hooks.structured_output.hooks import _CORRECTION_META_KEY


class CityInfo(BaseModel):
    city: str
    population: int


CITY_JSON_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "population": {"type": "integer"}},
    "required": ["city", "population"],
}

VALID_REPLY = '{"city": "Paris", "population": 2100000}'


@hook
def _ask_for_sources_once(state: State) -> None:
    if len(state.data["messages"]) <= 2:
        state.set("messages", [ChatMessage.from_user("Add your sources.")])
        state.set("continue_run", True)


def _agent(responses: list[str | ChatMessage], hook: StructuredOutputHook, max_agent_steps: int = 100) -> Agent:
    agent = Agent(
        chat_generator=MockChatGenerator(responses=responses),
        state_schema={hook.state_key: {"type": dict}},
        hooks={"on_exit": [hook]},
        max_agent_steps=max_agent_steps,
    )
    agent.warm_up()
    return agent


class TestStructuredOutputHook:
    @pytest.mark.parametrize(
        "reply",
        [
            VALID_REPLY,
            f"```json\n{VALID_REPLY}\n```",
            f"```\n{VALID_REPLY}\n```",
            f"\n  ```JSON\n{VALID_REPLY}\n```  \n",
        ],
        ids=["plain", "json-fence", "bare-fence", "padded-fence"],
    )
    @pytest.mark.parametrize("schema", [CityInfo, CITY_JSON_SCHEMA], ids=["pydantic", "json-schema"])
    def test_stores_the_parsed_reply(self, schema, reply):
        agent = _agent(responses=[reply], hook=StructuredOutputHook(schema=schema, state_key="city_info"))
        result = agent.run(messages=[ChatMessage.from_user("Tell me about Paris")])
        assert result["city_info"] == {"city": "Paris", "population": 2100000}
        assert result["step_count"] == 1

    @pytest.mark.parametrize(
        ("bad_reply", "expected_error"),
        [
            ("Paris has about 2.1 million people.", "The reply is not valid JSON"),
            ('{"city": "Paris"}', "The reply does not match the schema"),
            (f"Here it is:\n```json\n{VALID_REPLY}\n```", "The reply is not valid JSON"),
        ],
        ids=["invalid-json", "schema-mismatch", "text-before-fence"],
    )
    @pytest.mark.parametrize("schema", [CityInfo, CITY_JSON_SCHEMA], ids=["pydantic", "json-schema"])
    def test_retries_with_a_correction_message(self, schema, bad_reply, expected_error):
        agent = _agent(responses=[bad_reply, VALID_REPLY], hook=StructuredOutputHook(schema=schema, max_retries=1))
        result = agent.run(messages=[ChatMessage.from_user("Tell me about Paris")])

        assert result["structured_output"] == {"city": "Paris", "population": 2100000}
        assert result["step_count"] == 2
        correction = result["messages"][-2]
        assert correction.meta[_CORRECTION_META_KEY] is True
        assert expected_error in correction.text
        assert '"population"' in correction.text

    @pytest.mark.parametrize(
        ("hook_kwargs", "max_retries"), [({"max_retries": 0}, 0), ({}, 2)], ids=["no-retries", "default"]
    )
    def test_stores_none_when_retries_are_exhausted(self, hook_kwargs, max_retries, caplog):
        agent = _agent(responses=["not json"], hook=StructuredOutputHook(schema=CityInfo, **hook_kwargs))
        with caplog.at_level(logging.WARNING):
            result = agent.run(messages=[ChatMessage.from_user("Tell me about Paris")])

        assert result["structured_output"] is None
        assert result["exit_reason"] == "text"
        assert result["step_count"] == max_retries + 1
        assert f"after {max_retries} retries" in caplog.text

    def test_retries_stop_at_max_agent_steps(self):
        agent = _agent(
            responses=["not json"], hook=StructuredOutputHook(schema=CityInfo, max_retries=5), max_agent_steps=2
        )
        result = agent.run(messages=[ChatMessage.from_user("Tell me about Paris")])

        assert result["structured_output"] is None
        assert result["exit_reason"] == "max_agent_steps"
        assert result["step_count"] == 2

    def test_does_not_parse_a_truncated_reply(self, caplog):
        truncated = ChatMessage.from_assistant('{"city": "Par', meta={"finish_reason": "length"})
        agent = _agent(responses=[truncated], hook=StructuredOutputHook(schema=CityInfo))
        with caplog.at_level(logging.WARNING):
            result = agent.run(messages=[ChatMessage.from_user("Tell me about Paris")])

        assert result["structured_output"] is None
        assert result["exit_reason"] == "length"
        assert result["step_count"] == 1
        assert "exit reason 'length'" in caplog.text

    def test_corrections_from_earlier_turns_do_not_use_up_retries(self):
        earlier_turn = [
            ChatMessage.from_user("Tell me about Paris"),
            ChatMessage.from_assistant("not json"),
            ChatMessage.from_user("Your previous reply could not be used", meta={_CORRECTION_META_KEY: True}),
            ChatMessage.from_assistant(VALID_REPLY),
        ]
        agent = _agent(responses=["not json", VALID_REPLY], hook=StructuredOutputHook(schema=CityInfo, max_retries=1))
        result = agent.run(messages=[*earlier_turn, ChatMessage.from_user("Now tell me about Paris again")])

        assert result["structured_output"] == {"city": "Paris", "population": 2100000}

    def test_skips_parsing_while_another_hook_keeps_the_agent_running(self, caplog):
        hook = StructuredOutputHook(schema=CityInfo, max_retries=0)
        agent = Agent(
            chat_generator=MockChatGenerator(responses=[VALID_REPLY]),
            state_schema={"structured_output": {"type": dict}},
            hooks={"on_exit": [_ask_for_sources_once, hook]},
        )
        agent.warm_up()
        with caplog.at_level(logging.WARNING):
            result = agent.run(messages=[ChatMessage.from_user("Tell me about Paris")])

        assert result["structured_output"] == {"city": "Paris", "population": 2100000}
        assert result["step_count"] == 2
        assert "could not be parsed" not in caplog.text

    def test_missing_state_key_raises(self):
        agent = Agent(
            chat_generator=MockChatGenerator(responses=[VALID_REPLY]),
            hooks={"on_exit": [StructuredOutputHook(schema=CityInfo)]},
        )
        agent.warm_up()
        with pytest.raises(ValueError, match="'structured_output', which is not in the Agent's `state_schema`"):
            agent.run(messages=[ChatMessage.from_user("Tell me about Paris")])

    @pytest.mark.parametrize(
        ("kwargs", "error", "match"),
        [
            ({"schema": {"type": "not-a-type"}}, ValueError, "`schema` is not a valid JSON schema"),
            ({"schema": "CityInfo"}, TypeError, "`schema` must be a Pydantic model or a JSON schema dict, got str."),
            ({"schema": CityInfo, "max_retries": -1}, ValueError, "`max_retries` must be 0 or greater, got -1."),
        ],
        ids=["invalid-json-schema", "wrong-schema-type", "negative-retries"],
    )
    def test_invalid_init_parameters_raise(self, kwargs, error, match):
        with pytest.raises(error, match=match):
            StructuredOutputHook(**kwargs)

    @pytest.mark.parametrize("schema", [CityInfo, CITY_JSON_SCHEMA], ids=["pydantic", "json-schema"])
    def test_to_dict_from_dict_roundtrip(self, schema):
        hook = StructuredOutputHook(schema=schema, state_key="city_info", max_retries=3)
        restored = StructuredOutputHook.from_dict(hook.to_dict())
        assert restored.schema == schema
        assert restored.state_key == "city_info"
        assert restored.max_retries == 3


class TestStructuredOutputHookAsync:
    @pytest.mark.asyncio
    async def test_retries_with_a_correction_message(self):
        agent = _agent(responses=["not json", VALID_REPLY], hook=StructuredOutputHook(schema=CityInfo, max_retries=1))
        result = await agent.run_async(messages=[ChatMessage.from_user("Tell me about Paris")])

        assert result["structured_output"] == {"city": "Paris", "population": 2100000}
        assert result["step_count"] == 2
