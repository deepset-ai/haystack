# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from haystack import Document, Pipeline, component
from haystack.components.agents.agent import Agent
from haystack.components.generators.chat import LLM
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.core.component.types import OutputSocket
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.chat_message import ChatRole
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.tools import Tool
from haystack.tools.toolset import Toolset


@component
class MockChatGeneratorWithTools:
    """A mock chat generator that accepts a tools parameter."""

    def to_dict(self) -> dict[str, Any]:
        return {"type": "test_llm.MockChatGeneratorWithTools", "data": {}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MockChatGeneratorWithTools":
        return cls()

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage], tools: list[Tool] | Toolset | None = None, **kwargs) -> dict[str, Any]:
        return {"replies": [ChatMessage.from_assistant("Reply with tools support")]}

    @component.output_types(replies=list[ChatMessage])
    async def run_async(
        self, messages: list[ChatMessage], tools: list[Tool] | Toolset | None = None, **kwargs
    ) -> dict[str, Any]:
        return {"replies": [ChatMessage.from_assistant("Async reply with tools support")]}


@component
class MockChatGenerator:
    """A mock chat generator that does NOT accept a tools parameter."""

    def to_dict(self) -> dict[str, Any]:
        return {"type": "test_llm.MockChatGenerator", "data": {}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MockChatGenerator":
        return cls()

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage], **kwargs) -> dict[str, Any]:
        return {"replies": [ChatMessage.from_assistant("Sync reply")]}

    @component.output_types(replies=list[ChatMessage])
    async def run_async(self, messages: list[ChatMessage], **kwargs) -> dict[str, Any]:
        return {"replies": [ChatMessage.from_assistant("Async reply")]}


class TestLLM:
    class TestInit:
        def test_is_subclass_of_agent(self):
            assert issubclass(LLM, Agent)

        def test_defaults(self):
            llm = LLM(chat_generator=MockChatGenerator())
            assert llm.chat_generator is not None
            assert llm.tools == []
            assert llm.system_prompt is None
            assert llm.user_prompt is None
            assert llm.streaming_callback is None
            assert llm._tool_invoker is None

        def test_output_sockets(self):
            llm = LLM(chat_generator=MockChatGenerator())
            assert llm.__haystack_output__._sockets_dict == {
                "messages": OutputSocket(name="messages", type=list[ChatMessage], receivers=[]),
                "last_message": OutputSocket(name="last_message", type=ChatMessage, receivers=[]),
            }

        def test_detects_no_tools_support(self):
            llm = LLM(chat_generator=MockChatGenerator())
            assert llm._chat_generator_supports_tools is False

        def test_detects_tools_support(self):
            llm = LLM(chat_generator=MockChatGeneratorWithTools())
            assert llm._chat_generator_supports_tools is True

    class TestSerialization:
        def test_to_dict_excludes_agent_only_params(self, monkeypatch):
            monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
            llm = LLM(chat_generator=OpenAIChatGenerator(), system_prompt="You are helpful.")

            serialized = llm.to_dict()

            assert serialized["type"] == "haystack.components.generators.chat.llm.LLM"
            assert "chat_generator" in serialized["init_parameters"]
            assert serialized["init_parameters"]["system_prompt"] == "You are helpful."

            agent_only_params = [
                "tools",
                "exit_conditions",
                "max_agent_steps",
                "raise_on_tool_invocation_failure",
                "tool_invoker_kwargs",
                "confirmation_strategies",
                "state_schema",
            ]
            for param in agent_only_params:
                assert param not in serialized["init_parameters"], (
                    f"Agent-only param '{param}' should not be serialized"
                )

        def test_to_dict_includes_llm_params(self, monkeypatch):
            monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
            llm = LLM(
                chat_generator=OpenAIChatGenerator(),
                system_prompt="Be concise.",
                user_prompt='{% message role="user" %}{{ query }}{% endmessage %}',
                required_variables=["query"],
            )

            serialized = llm.to_dict()

            assert serialized["init_parameters"]["system_prompt"] == "Be concise."
            assert "{{ query }}" in serialized["init_parameters"]["user_prompt"]
            assert serialized["init_parameters"]["required_variables"] == ["query"]
            assert serialized["init_parameters"]["streaming_callback"] is None

        def test_from_dict(self, monkeypatch):
            monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
            data = {
                "type": "haystack.components.generators.chat.llm.LLM",
                "init_parameters": {
                    "chat_generator": {
                        "type": "haystack.components.generators.chat.openai.OpenAIChatGenerator",
                        "init_parameters": {
                            "model": "gpt-4o-mini",
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
                    "system_prompt": "You are helpful.",
                    "user_prompt": None,
                    "required_variables": None,
                    "streaming_callback": None,
                },
            }

            llm = LLM.from_dict(data)

            assert isinstance(llm, LLM)
            assert isinstance(llm.chat_generator, OpenAIChatGenerator)
            assert llm.system_prompt == "You are helpful."
            assert llm.tools == []

        def test_roundtrip(self, monkeypatch):
            monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
            original = LLM(chat_generator=OpenAIChatGenerator(), system_prompt="You are a poet.")

            restored = LLM.from_dict(original.to_dict())

            assert isinstance(restored, LLM)
            assert isinstance(restored.chat_generator, OpenAIChatGenerator)
            assert restored.system_prompt == original.system_prompt
            assert restored.tools == []

    class TestPipelineIntegration:
        @pytest.fixture()
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

        def test_rag_pipeline(self, document_store_with_docs):
            user_prompt = (
                '{% message role="user" %}'
                "Use the following documents to answer the question.\n"
                "Documents:\n{% for doc in documents %}{{ doc.content }}\n{% endfor %}"
                "Question: {{ query }}"
                "{% endmessage %}"
            )
            llm = LLM(
                chat_generator=MockChatGenerator(),
                system_prompt="You are a knowledgeable assistant.",
                user_prompt=user_prompt,
                required_variables=["query", "documents"],
            )

            pipe = Pipeline()
            pipe.add_component("retriever", InMemoryBM25Retriever(document_store=document_store_with_docs))
            pipe.add_component("llm", llm)
            pipe.connect("retriever.documents", "llm.documents")

            query = "Where is the Colosseum?"
            result = pipe.run(data={"retriever": {"query": query}, "llm": {"query": query}})

            assert "llm" in result
            llm_output = result["llm"]
            assert "messages" in llm_output
            assert "last_message" in llm_output

            messages = llm_output["messages"]

            assert messages[0].is_from(ChatRole.SYSTEM)
            assert messages[0].text == "You are a knowledgeable assistant."

            user_messages = [m for m in messages if m.is_from(ChatRole.USER)]
            assert len(user_messages) == 1
            rendered = user_messages[0].text
            assert "Question: Where is the Colosseum?" in rendered
            assert "Documents:" in rendered
            assert "Colosseum" in rendered

            assert llm_output["last_message"].is_from(ChatRole.ASSISTANT)
            assert llm_output["last_message"].text == "Sync reply"
