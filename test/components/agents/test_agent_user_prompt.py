# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from unittest.mock import MagicMock

import pytest

from haystack import Pipeline, component
from haystack.components.agents import Agent
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import ChatMessage, Document
from haystack.dataclasses.chat_message import ChatRole
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.tools import Tool
from haystack.tools.toolset import Toolset


def _user_msg(text: str) -> str:
    return f'{{% message role="user" %}}{text}{{% endmessage %}}'


def weather_function(location: str) -> dict:
    return {"weather": "sunny", "temperature": 20, "unit": "celsius"}


@pytest.fixture
def weather_tool():
    return Tool(
        name="weather_tool",
        description="Provides weather information for a given location.",
        parameters={"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]},
        function=weather_function,
    )


@component
class MockChatGenerator:
    def to_dict(self) -> dict[str, Any]:
        return {"type": "MockChatGenerator", "data": {}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MockChatGenerator":
        return cls()

    @component.output_types(replies=list[ChatMessage])
    def run(
        self, messages: list[ChatMessage], tools: list[Tool] | Toolset | None = None, **kwargs
    ) -> dict[str, list[ChatMessage]]:
        return self._run()

    @component.output_types(replies=list[ChatMessage])
    async def run_async(
        self, messages: list[ChatMessage], tools: list[Tool] | Toolset | None = None, **kwargs
    ) -> dict[str, list[ChatMessage]]:
        return self._run()

    def _run(self) -> dict[str, list[ChatMessage]]:
        return {"replies": [ChatMessage.from_assistant("Hello!")]}


def _make_agent_with_user_prompt(
    user_prompt: str, *, chat_generator: MockChatGenerator | None = None, **agent_kwargs
) -> Agent:
    return Agent(chat_generator=chat_generator or MockChatGenerator(), user_prompt=user_prompt, **agent_kwargs)


class TestAgentInitialization:
    def test_user_prompt_raises_when_no_messages_and_no_prompt(self, weather_tool):
        agent = Agent(chat_generator=MockChatGenerator(), tools=[weather_tool])
        with pytest.raises(ValueError, match="No messages provided to the Agent and user_prompt is not set"):
            agent.run()

    def test_user_prompt_conflict_with_state_schema_raises(self, weather_tool):
        with pytest.raises(ValueError, match="already defined in the state schema"):
            _make_agent_with_user_prompt(
                _user_msg("Query: {{custom_field}}"), tools=[weather_tool], state_schema={"custom_field": {"type": str}}
            )

    def test_user_prompt_conflict_with_run_param_raises(self, weather_tool):
        with pytest.raises(ValueError, match="conflicts with input names in the run method"):
            _make_agent_with_user_prompt(_user_msg("{{system_prompt}} is the system prompt."), tools=[weather_tool])

    def test_user_prompt_only_variables_forwarded_to_builder(self, weather_tool):
        agent = _make_agent_with_user_prompt(_user_msg("Question: {{question}}"), tools=[weather_tool])
        # 'irrelevant_kwarg' is not a template variable — must not raise
        result = agent.run(question="Will it snow?", irrelevant_kwarg="unused")
        assert "messages" in result


class TestUserPromptOnly:
    def test_simple_literal_user_prompt(self, weather_tool):
        agent = _make_agent_with_user_prompt(_user_msg("Tell me the weather."), tools=[weather_tool])
        result = agent.run()
        messages = result["messages"]
        # The rendered user_prompt should be the first (and only) non-system message
        user_messages = [m for m in messages if m.is_from(ChatRole.USER)]
        assert len(user_messages) == 1
        assert user_messages[0].text == "Tell me the weather."

    def test_user_prompt_with_template_variables(self, weather_tool):
        agent = _make_agent_with_user_prompt(
            _user_msg(
                "Hello {{name|upper}}, check weather for: "
                + "{% for c in cities %}{{c}}{% if not loop.last %}, {% endif %}{% endfor %}"
                + " on {{date}}?"
            ),
            tools=[weather_tool],
        )
        result = agent.run(name="Alice", cities=["Berlin", "Paris", "Rome"], date="2024-01-15")
        user_messages = [m for m in result["messages"] if m.is_from(ChatRole.USER)]
        assert user_messages[0].text == "Hello ALICE, check weather for: Berlin, Paris, Rome on 2024-01-15?"

        input_names = set(agent.__haystack_input__._sockets_dict.keys())
        assert "name" in input_names
        assert "cities" in input_names
        assert "date" in input_names

    def test_user_prompt_with_system_prompt(self, weather_tool):
        agent = _make_agent_with_user_prompt(
            _user_msg("What is the weather in {{city}}?"),
            tools=[weather_tool],
            system_prompt="You are a helpful weather assistant.",
        )
        result = agent.run(city="Berlin")
        messages = result["messages"]
        assert messages[0].is_from(ChatRole.SYSTEM)
        assert messages[0].text == "You are a helpful weather assistant."
        user_messages = [m for m in messages if m.is_from(ChatRole.USER)]
        assert user_messages[0].text == "What is the weather in Berlin?"

    def test_user_prompt_with_documents_variable(self, weather_tool):
        agent = _make_agent_with_user_prompt(
            _user_msg(
                "Answer based on these documents:\n"
                "{% for doc in documents %}{{doc.content}}\n{% endfor %}"
                "Question: {{question}}"
            ),
            tools=[weather_tool],
        )
        docs = [Document(content="Doc A"), Document(content="Doc B")]
        result = agent.run(documents=docs, question="What is in the docs?")
        user_messages = [m for m in result["messages"] if m.is_from(ChatRole.USER)]
        assert "Doc A" in user_messages[0].text
        assert "Doc B" in user_messages[0].text
        assert "What is in the docs?" in user_messages[0].text

    def test_runtime_user_prompt_overrides_init_prompt(self, weather_tool):
        agent = _make_agent_with_user_prompt(_user_msg("Default prompt for {{city}}."), tools=[weather_tool])
        result = agent.run(user_prompt=_user_msg("Runtime prompt for {{city}}."), city="Berlin")
        user_messages = [m for m in result["messages"] if m.is_from(ChatRole.USER)]
        assert user_messages[0].text == "Runtime prompt for Berlin."


class TestUserPromptWithMessages:
    def test_user_prompt_appended_after_initial_messages(self, weather_tool):
        agent = _make_agent_with_user_prompt(_user_msg("And now: {{query}}"), tools=[weather_tool])
        initial_messages = [ChatMessage.from_user("First message")]
        result = agent.run(messages=initial_messages, query="What is the weather?")
        user_messages = [m for m in result["messages"] if m.is_from(ChatRole.USER)]
        assert user_messages[0].text == "First message"
        assert user_messages[1].text == "And now: What is the weather?"

    def test_runtime_user_prompt_appended_after_initial_messages(self, weather_tool):
        agent = _make_agent_with_user_prompt(_user_msg("Init prompt: {{question}}"), tools=[weather_tool])
        initial_messages = [ChatMessage.from_user("Context message")]
        result = agent.run(
            messages=initial_messages, user_prompt=_user_msg("Follow-up: {{question}}"), question="Is it raining?"
        )
        user_messages = [m for m in result["messages"] if m.is_from(ChatRole.USER)]
        assert len(user_messages) == 2
        assert user_messages[0].text == "Context message"
        assert user_messages[1].text == "Follow-up: Is it raining?"

    def test_messages_plus_user_prompt_with_multiple_kwargs(self, weather_tool):
        agent = _make_agent_with_user_prompt(
            _user_msg("Documents:\n{% for d in documents %}{{d.content}}\n{% endfor %}Q: {{question}}"),
            tools=[weather_tool],
            system_prompt="You are very smart.",
        )
        history = [ChatMessage.from_user("Previous question?"), ChatMessage.from_assistant("Previous answer.")]
        docs = [Document(content="Fact A"), Document(content="Fact B")]
        result = agent.run(messages=history, documents=docs, question="Summarise the facts.")
        messages = result["messages"]
        assert len(messages) == 5

        assert messages[0].role.value == ChatRole.SYSTEM
        assert messages[0].text == "You are very smart."

        assert messages[1].role.value == ChatRole.USER
        assert messages[1].text == "Previous question?"

        assert messages[2].role.value == ChatRole.ASSISTANT
        assert messages[2].text == "Previous answer."

        assert messages[3].role.value == ChatRole.USER
        rendered = messages[3].text
        assert "Fact A" in rendered
        assert "Fact B" in rendered
        assert "Summarise the facts." in rendered

        assert messages[4].role.value == ChatRole.ASSISTANT
        assert messages[4].text == "Hello!"


def _make_rag_pipeline(
    document_store_with_docs: InMemoryDocumentStore, weather_tool: Tool, *, user_prompt: str | None = None
):
    agent = _make_agent_with_user_prompt(
        user_prompt=user_prompt
        or _user_msg(
            "Use the following documents to answer the question.\n"
            "Documents:\n{% for doc in documents %}{{doc.content}}\n{% endfor %}"
            "Question: {{query}}"
        ),
        tools=[weather_tool],
        system_prompt="You are a knowledgeable assistant.",
    )

    pp = Pipeline()
    pp.add_component("retriever", InMemoryBM25Retriever(document_store=document_store_with_docs))
    pp.add_component("agent", agent)
    pp.connect("retriever.documents", "agent.documents")

    return pp


class TestAgentUserPromptInPipeline:
    @pytest.fixture
    def document_store_with_docs(self):
        store = InMemoryDocumentStore()
        store.write_documents(
            [
                Document(content="The Eiffel Tower is located in Paris."),
                Document(content="The Brandenburg Gate is in Berlin."),
                Document(content="The Colosseum is in Rome."),
            ]
        )
        return store

    def test_rag_pipeline_user_prompt_init_only(self, document_store_with_docs, weather_tool):
        pipeline = _make_rag_pipeline(document_store_with_docs, weather_tool)
        query = "Where is the Colosseum?"
        result = pipeline.run(data={"retriever": {"query": query}, "agent": {"query": query}})
        assert "agent" in result
        agent_output = result["agent"]
        assert "messages" in agent_output
        assert "last_message" in agent_output

        messages = agent_output["messages"]
        assert messages[0].is_from(ChatRole.SYSTEM)
        assert messages[0].text == "You are a knowledgeable assistant."

        user_messages = [m for m in messages if m.is_from(ChatRole.USER)]
        assert len(user_messages) == 1
        rendered = user_messages[0].text
        assert "Question: Where is the Colosseum?" in rendered
        assert "Documents:" in rendered

    def test_rag_pipeline_user_prompt_runtime_override(self, document_store_with_docs, weather_tool):
        user_prompt = _user_msg(
            "Documents:\n{% for doc in documents %}{{doc.content}}\n{% endfor %}Question: {{query}}"
        )
        pipeline = _make_rag_pipeline(document_store_with_docs, weather_tool, user_prompt=user_prompt)

        query = "Where is the Eiffel Tower?"
        result = pipeline.run(
            data={
                "retriever": {"query": query},
                "agent": {
                    "user_prompt": _user_msg(
                        "OVERRIDE: Using docs:\n"
                        "{% for doc in documents %}{{doc.content}}\n{% endfor %}"
                        "Answer: {{query}}"
                    ),
                    "query": query,
                },
            }
        )
        messages = result["agent"]["messages"]
        user_messages = [m for m in messages if m.is_from(ChatRole.USER)]
        rendered = user_messages[0].text
        assert "OVERRIDE:" in rendered
        assert "Where is the Eiffel Tower?" in rendered

    def test_rag_pipeline_messages_plus_user_prompt(self, document_store_with_docs, weather_tool):
        from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder

        chat_generator = MockChatGenerator()

        agent = Agent(
            chat_generator=chat_generator,
            tools=[weather_tool],
            user_prompt=_user_msg("Relevant docs:\n{% for doc in documents %}{{doc.content}}\n{% endfor %}"),
        )
        chat_generator.run = MagicMock(return_value={"replies": [ChatMessage.from_assistant("Berlin")]})

        pipeline = Pipeline()
        pipeline.add_component(
            "prompt_builder", ChatPromptBuilder(template=[ChatMessage.from_user("History: {{history_note}}")])
        )
        pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=document_store_with_docs))
        pipeline.add_component("agent", agent)

        pipeline.connect("prompt_builder.prompt", "agent.messages")
        pipeline.connect("retriever.documents", "agent.documents")

        result = pipeline.run(
            data={
                "prompt_builder": {"history_note": "User previously asked about European cities."},
                "retriever": {"query": "Brandenburg Gate"},
            }
        )
        messages = result["agent"]["messages"]
        user_messages = [m for m in messages if m.is_from(ChatRole.USER)]
        assert "History:" in user_messages[0].text
        rendered = user_messages[1].text
        assert "Relevant docs:" in rendered
