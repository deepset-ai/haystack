# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest
from pydantic import BaseModel

from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.hooks.structured_output import StructuredOutputHook
from haystack.hooks.structured_output.hooks import _CORRECTION_META_KEY

pytestmark = pytest.mark.filterwarnings("ignore::haystack.utils.experimental.ExperimentalWarning")


class CityInfo(BaseModel):
    city: str
    population: int


CITY_JSON_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "population": {"type": "integer"}},
    "required": ["city", "population"],
}

VALID_REPLY = '{"city": "Paris", "population": 2100000}'


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
    @pytest.mark.parametrize("schema", [CityInfo, CITY_JSON_SCHEMA], ids=["pydantic", "json-schema"])
    def test_stores_the_parsed_reply(self, schema):
        agent = _agent(responses=[VALID_REPLY], hook=StructuredOutputHook(schema=schema, state_key="city_info"))
        result = agent.run(messages=[ChatMessage.from_user("Tell me about Paris")])
        assert result["city_info"] == {"city": "Paris", "population": 2100000}
        assert result["step_count"] == 1

    @pytest.mark.parametrize(
        ("bad_reply", "expected_error"),
        [
            ("Paris has about 2.1 million people.", "The reply is not valid JSON"),
            ('{"city": "Paris"}', "The reply does not match the schema"),
        ],
        ids=["invalid-json", "schema-mismatch"],
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

    @pytest.mark.parametrize("max_retries", [0, 2])
    def test_stores_none_when_retries_are_exhausted(self, max_retries, caplog):
        agent = _agent(responses=["not json"], hook=StructuredOutputHook(schema=CityInfo, max_retries=max_retries))
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
        agent = _agent(responses=[truncated], hook=StructuredOutputHook(schema=CityInfo, max_retries=2))
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
        hook = StructuredOutputHook(schema=schema, state_key="city_info", max_retries=2)
        restored = StructuredOutputHook.from_dict(hook.to_dict())
        assert restored.schema == schema
        assert restored.state_key == "city_info"
        assert restored.max_retries == 2


class TestStructuredOutputHookAsync:
    @pytest.mark.asyncio
    async def test_retries_with_a_correction_message(self):
        agent = _agent(responses=["not json", VALID_REPLY], hook=StructuredOutputHook(schema=CityInfo, max_retries=1))
        result = await agent.run_async(messages=[ChatMessage.from_user("Tell me about Paris")])

        assert result["structured_output"] == {"city": "Paris", "population": 2100000}
        assert result["step_count"] == 2
