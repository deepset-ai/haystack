# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from haystack import Document, Pipeline, component
from haystack.components.agents.agent import Agent
from haystack.components.generators.chat import LLM
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.core.component.types import OutputSocket
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.chat_message import ChatRole
from haystack.dataclasses.streaming_chunk import StreamingChunk
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
            assert llm.exit_conditions == ["text"]
            assert llm.max_agent_steps == 1
            assert llm.streaming_callback is None
            assert llm._tool_invoker is None

        def test_with_system_prompt(self):
            llm = LLM(chat_generator=MockChatGenerator(), system_prompt="You are helpful.")
            assert llm.system_prompt == "You are helpful."

        def test_with_user_prompt(self):
            llm = LLM(chat_generator=MockChatGenerator(), user_prompt="Tell me about {{ topic }}")
            assert llm.user_prompt == "Tell me about {{ topic }}"

        def test_does_not_accept_state_schema(self):
            with pytest.raises(TypeError, match="unexpected keyword argument"):
                LLM(chat_generator=MockChatGenerator(), state_schema={"custom_key": {"type": str}})

        def test_does_not_accept_tools(self):
            with pytest.raises(TypeError, match="unexpected keyword argument"):
                LLM(chat_generator=MockChatGeneratorWithTools(), tools=[])

        def test_does_not_accept_exit_conditions(self):
            with pytest.raises(TypeError, match="unexpected keyword argument"):
                LLM(chat_generator=MockChatGenerator(), exit_conditions=["text"])

        def test_does_not_accept_max_agent_steps(self):
            with pytest.raises(TypeError, match="unexpected keyword argument"):
                LLM(chat_generator=MockChatGenerator(), max_agent_steps=5)

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

    class TestRunSync:
        def test_basic(self):
            llm = LLM(chat_generator=MockChatGenerator())
            response = llm.run(messages=[ChatMessage.from_user("What is the capital of France?")])

            assert isinstance(response, dict)
            assert "messages" in response
            assert "last_message" in response
            assert len(response["messages"]) == 2
            assert response["messages"][0].text == "What is the capital of France?"
            assert response["messages"][1].text == "Sync reply"
            assert response["last_message"] is response["messages"][-1]

        def test_with_system_prompt(self):
            llm = LLM(chat_generator=MockChatGenerator(), system_prompt="You are a helpful assistant.")
            response = llm.run(messages=[ChatMessage.from_user("Hi there")])

            assert len(response["messages"]) == 3
            assert response["messages"][0].is_from(ChatRole.SYSTEM)
            assert response["messages"][0].text == "You are a helpful assistant."
            assert response["messages"][1].is_from(ChatRole.USER)
            assert response["messages"][1].text == "Hi there"
            assert response["messages"][2].is_from(ChatRole.ASSISTANT)
            assert response["messages"][2].text == "Sync reply"

        def test_runtime_system_prompt_overrides_init(self):
            llm = LLM(chat_generator=MockChatGenerator(), system_prompt="Default system prompt")
            response = llm.run(messages=[ChatMessage.from_user("Hi")], system_prompt="Runtime system prompt")

            assert response["messages"][0].is_from(ChatRole.SYSTEM)
            assert response["messages"][0].text == "Runtime system prompt"

        def test_with_user_prompt_template(self):
            user_prompt = '{% message role="user" %}Tell me about {{ topic }}{% endmessage %}'
            llm = LLM(chat_generator=MockChatGenerator(), user_prompt=user_prompt)

            response = llm.run(topic="Python")

            user_messages = [m for m in response["messages"] if m.is_from(ChatRole.USER)]
            assert len(user_messages) == 1
            assert "Python" in user_messages[0].text

        def test_runtime_user_prompt_overrides_init(self):
            init_prompt = '{% message role="user" %}Init prompt about {{ topic }}{% endmessage %}'
            runtime_prompt = '{% message role="user" %}Runtime prompt about {{ topic }}{% endmessage %}'
            llm = LLM(chat_generator=MockChatGenerator(), user_prompt=init_prompt)

            response = llm.run(messages=[], user_prompt=runtime_prompt, topic="haystack")

            user_messages = [m for m in response["messages"] if m.is_from(ChatRole.USER)]
            assert len(user_messages) == 1
            assert "Runtime prompt" in user_messages[0].text
            assert "haystack" in user_messages[0].text

        def test_required_variables_missing_raises(self):
            user_prompt = '{% message role="user" %}Tell me about {{ topic }}{% endmessage %}'
            llm = LLM(chat_generator=MockChatGenerator(), user_prompt=user_prompt, required_variables=["topic"])

            with pytest.raises(ValueError):
                llm.run(messages=[])

        def test_required_variables_provided(self):
            user_prompt = '{% message role="user" %}Tell me about {{ topic }}{% endmessage %}'
            llm = LLM(chat_generator=MockChatGenerator(), user_prompt=user_prompt, required_variables=["topic"])

            response = llm.run(messages=[], topic="Python")

            user_messages = [m for m in response["messages"] if m.is_from(ChatRole.USER)]
            assert len(user_messages) == 1
            assert "Python" in user_messages[0].text

        def test_required_variables_wildcard(self):
            user_prompt = '{% message role="user" %}Tell me about {{ topic }}{% endmessage %}'
            llm = LLM(chat_generator=MockChatGenerator(), user_prompt=user_prompt, required_variables="*")

            response = llm.run(messages=[], topic="Python")

            user_messages = [m for m in response["messages"] if m.is_from(ChatRole.USER)]
            assert len(user_messages) == 1
            assert "Python" in user_messages[0].text

        def test_required_variables_wildcard_missing_raises(self):
            user_prompt = '{% message role="user" %}Tell me about {{ topic }}{% endmessage %}'
            llm = LLM(chat_generator=MockChatGenerator(), user_prompt=user_prompt, required_variables="*")

            with pytest.raises(ValueError):
                llm.run(messages=[])

        def test_with_chat_generator_that_supports_tools(self):
            llm = LLM(chat_generator=MockChatGeneratorWithTools())
            response = llm.run(messages=[ChatMessage.from_user("Hi")])

            assert len(response["messages"]) == 2
            assert response["messages"][1].text == "Reply with tools support"

        def test_multi_turn_conversation(self):
            llm = LLM(chat_generator=MockChatGenerator())
            history = [
                ChatMessage.from_user("Hi"),
                ChatMessage.from_assistant("Hello! How can I help?"),
                ChatMessage.from_user("Tell me a joke"),
            ]
            response = llm.run(messages=history)

            assert len(response["messages"]) == 4
            assert response["messages"][:3] == history
            assert response["last_message"].is_from(ChatRole.ASSISTANT)
            assert response["last_message"].text == "Sync reply"

        def test_no_messages_and_no_prompts_raises(self):
            llm = LLM(chat_generator=MockChatGenerator())

            with pytest.raises(ValueError, match="No messages provided"):
                llm.run()

    class TestRunAsync:
        async def test_basic_async(self):
            llm = LLM(chat_generator=MockChatGenerator())
            response = await llm.run_async(messages=[ChatMessage.from_user("What is the capital of France?")])

            assert isinstance(response, dict)
            assert "messages" in response
            assert "last_message" in response
            assert len(response["messages"]) == 2
            assert response["messages"][0].text == "What is the capital of France?"
            assert response["messages"][1].text == "Async reply"
            assert response["last_message"] is response["messages"][-1]

        async def test_with_system_prompt_async(self):
            llm = LLM(chat_generator=MockChatGenerator(), system_prompt="You are a helpful assistant.")
            response = await llm.run_async(messages=[ChatMessage.from_user("Hi there")])

            assert len(response["messages"]) == 3
            assert response["messages"][0].is_from(ChatRole.SYSTEM)
            assert response["messages"][0].text == "You are a helpful assistant."
            assert response["messages"][1].is_from(ChatRole.USER)
            assert response["messages"][1].text == "Hi there"
            assert response["messages"][2].is_from(ChatRole.ASSISTANT)
            assert response["messages"][2].text == "Async reply"

        async def test_runtime_system_prompt_overrides_init_async(self):
            llm = LLM(chat_generator=MockChatGenerator(), system_prompt="Default system prompt")
            response = await llm.run_async(
                messages=[ChatMessage.from_user("Hi")], system_prompt="Runtime system prompt"
            )

            assert response["messages"][0].is_from(ChatRole.SYSTEM)
            assert response["messages"][0].text == "Runtime system prompt"

        async def test_with_user_prompt_template_async(self):
            user_prompt = '{% message role="user" %}Tell me about {{ topic }}{% endmessage %}'
            llm = LLM(chat_generator=MockChatGenerator(), user_prompt=user_prompt)

            response = await llm.run_async(topic="Python")

            user_messages = [m for m in response["messages"] if m.is_from(ChatRole.USER)]
            assert len(user_messages) == 1
            assert "Python" in user_messages[0].text

        async def test_generation_kwargs_forwarded_async(self):
            generator = MockChatGenerator()
            llm = LLM(chat_generator=generator)

            generator.run_async = AsyncMock(return_value={"replies": [ChatMessage.from_assistant("OK")]})
            await llm.run_async(messages=[ChatMessage.from_user("Hi")], generation_kwargs={"temperature": 0.5})

            call_kwargs = generator.run_async.call_args.kwargs
            assert call_kwargs["generation_kwargs"] == {"temperature": 0.5}

        async def test_multi_turn_conversation_async(self):
            llm = LLM(chat_generator=MockChatGenerator())
            history = [
                ChatMessage.from_user("Hi"),
                ChatMessage.from_assistant("Hello! How can I help?"),
                ChatMessage.from_user("Tell me a joke"),
            ]
            response = await llm.run_async(messages=history)

            assert len(response["messages"]) == 4
            assert response["messages"][:3] == history
            assert response["last_message"].is_from(ChatRole.ASSISTANT)
            assert response["last_message"].text == "Async reply"

        async def test_no_messages_and_no_prompts_raises_async(self):
            llm = LLM(chat_generator=MockChatGenerator())

            with pytest.raises(ValueError, match="No messages provided"):
                await llm.run_async()

        async def test_async_with_tools_supporting_generator(self):
            llm = LLM(chat_generator=MockChatGeneratorWithTools())
            response = await llm.run_async(messages=[ChatMessage.from_user("Hi")])

            assert len(response["messages"]) == 2
            assert response["messages"][1].text == "Async reply with tools support"

    class TestStreamingCallback:
        def test_streaming_callback_forwarded_to_generator(self):
            generator = MockChatGenerator()
            generator.run = MagicMock(return_value={"replies": [ChatMessage.from_assistant("OK")]})

            def my_callback(chunk: StreamingChunk) -> None:
                pass

            llm = LLM(chat_generator=generator)
            llm.run(messages=[ChatMessage.from_user("Hi")], streaming_callback=my_callback)

            call_kwargs = generator.run.call_args.kwargs
            assert "streaming_callback" in call_kwargs
            assert call_kwargs["streaming_callback"] is my_callback

        def test_init_streaming_callback_forwarded_to_generator(self):
            generator = MockChatGenerator()
            generator.run = MagicMock(return_value={"replies": [ChatMessage.from_assistant("OK")]})

            def my_callback(chunk: StreamingChunk) -> None:
                pass

            llm = LLM(chat_generator=generator, streaming_callback=my_callback)
            llm.run(messages=[ChatMessage.from_user("Hi")])

            call_kwargs = generator.run.call_args.kwargs
            assert "streaming_callback" in call_kwargs
            assert call_kwargs["streaming_callback"] is my_callback

        def test_streaming_callback_invoked(self):
            callback_called = False

            def my_callback(chunk: StreamingChunk) -> None:
                nonlocal callback_called
                callback_called = True

            @component
            class StreamingMockChatGenerator:
                @component.output_types(replies=list[ChatMessage])
                def run(self, messages, streaming_callback=None, **kwargs):
                    if streaming_callback:
                        streaming_callback(StreamingChunk(content="chunk"))
                    return {"replies": [ChatMessage.from_assistant("Streamed reply")]}

            llm = LLM(chat_generator=StreamingMockChatGenerator())
            response = llm.run(messages=[ChatMessage.from_user("Hi")], streaming_callback=my_callback)

            assert callback_called
            assert response["last_message"].text == "Streamed reply"

        async def test_async_streaming_callback_forwarded(self):
            generator = MockChatGenerator()
            generator.run_async = AsyncMock(return_value={"replies": [ChatMessage.from_assistant("OK")]})

            async def my_async_callback(chunk: StreamingChunk) -> None:
                pass

            llm = LLM(chat_generator=generator, streaming_callback=my_async_callback)
            await llm.run_async(messages=[ChatMessage.from_user("Hi")])

            call_kwargs = generator.run_async.call_args.kwargs
            assert "streaming_callback" in call_kwargs
            assert call_kwargs["streaming_callback"] is my_async_callback

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
            assert llm.exit_conditions == ["text"]
            assert llm.max_agent_steps == 1

        def test_roundtrip(self, monkeypatch):
            monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
            original = LLM(chat_generator=OpenAIChatGenerator(), system_prompt="You are a poet.")

            restored = LLM.from_dict(original.to_dict())

            assert isinstance(restored, LLM)
            assert isinstance(restored.chat_generator, OpenAIChatGenerator)
            assert restored.system_prompt == original.system_prompt
            assert restored.tools == []
            assert restored.exit_conditions == ["text"]
            assert restored.max_agent_steps == 1

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
