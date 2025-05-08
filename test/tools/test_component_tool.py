# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import json
import os
from dataclasses import dataclass
from typing import Dict, List

import pytest

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from haystack import Pipeline, component
from haystack.components.builders import PromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.tools import ToolInvoker
from haystack.components.websearch.serper_dev import SerperDevWebSearch
from haystack.core.pipeline.utils import deepcopy_with_fallback
from haystack.dataclasses import ChatMessage, ChatRole, Document
from haystack.tools import ComponentTool
from haystack.utils.auth import Secret

### Component and Model Definitions


@component
class SimpleComponent:
    """A simple component that generates text."""

    @component.output_types(reply=str)
    def run(self, text: str) -> Dict[str, str]:
        """
        A simple component that generates text.

        :param text: user's name
        :return: A dictionary with the generated text.
        """
        return {"reply": f"Hello, {text}!"}


@dataclass
class User:
    """A simple user dataclass."""

    name: str = "Anonymous"
    age: int = 0


@component
class UserGreeter:
    """A simple component that processes a User."""

    @component.output_types(message=str)
    def run(self, user: User) -> Dict[str, str]:
        """
        A simple component that processes a User.

        :param user: The User object to process.
        :return: A dictionary with a message about the user.
        """
        return {"message": f"User {user.name} is {user.age} years old"}


@component
class ListProcessor:
    """A component that processes a list of strings."""

    @component.output_types(concatenated=str)
    def run(self, texts: List[str]) -> Dict[str, str]:
        """
        Concatenates a list of strings into a single string.

        :param texts: The list of strings to concatenate.
        :return: A dictionary with the concatenated string.
        """
        return {"concatenated": " ".join(texts)}


@dataclass
class Address:
    """A dataclass representing a physical address."""

    street: str
    city: str


@dataclass
class Person:
    """A person with an address."""

    name: str
    address: Address


@component
class PersonProcessor:
    """A component that processes a Person with nested Address."""

    @component.output_types(info=str)
    def run(self, person: Person) -> Dict[str, str]:
        """
        Creates information about the person.

        :param person: The Person to process.
        :return: A dictionary with the person's information.
        """
        return {"info": f"{person.name} lives at {person.address.street}, {person.address.city}."}


@component
class DocumentProcessor:
    """A component that processes a list of Documents."""

    @component.output_types(concatenated=str)
    def run(self, documents: List[Document], top_k: int = 5) -> Dict[str, str]:
        """
        Concatenates the content of multiple documents with newlines.

        :param documents: List of Documents whose content will be concatenated
        :param top_k: The number of top documents to concatenate
        :returns: Dictionary containing the concatenated document contents
        """
        return {"concatenated": "\n".join(doc.content for doc in documents[:top_k])}


def output_handler(old, new):
    """
    Output handler to test serialization.
    """
    return old + new


## Unit tests
class TestToolComponent:
    def test_from_component_basic(self):
        component = SimpleComponent()

        tool = ComponentTool(component=component)

        assert tool.name == "simple_component"
        assert tool.description == "A simple component that generates text."
        assert tool.parameters == {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "user's name"}},
            "required": ["text"],
        }

        # Test tool invocation
        result = tool.invoke(text="world")
        assert isinstance(result, dict)
        assert "reply" in result
        assert result["reply"] == "Hello, world!"

    def test_from_component_long_description(self):
        component = SimpleComponent()
        tool = ComponentTool(component=component, description="".join(["A"] * 1024))

        assert len(tool.description) == 1024

    def test_from_component_with_inputs(self):
        component = SimpleComponent()

        tool = ComponentTool(component=component, inputs_from_state={"text": "text"})

        assert tool.inputs_from_state == {"text": "text"}
        # Inputs should be excluded from schema generation
        assert tool.parameters == {"type": "object", "properties": {}}

    def test_from_component_with_outputs(self):
        component = SimpleComponent()

        tool = ComponentTool(component=component, outputs_to_state={"replies": {"source": "reply"}})

        assert tool.outputs_to_state == {"replies": {"source": "reply"}}

    def test_from_component_with_dataclass(self):
        component = UserGreeter()

        tool = ComponentTool(component=component)
        assert tool.parameters == {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "description": "The User object to process.",
                    "properties": {
                        "name": {"type": "string", "description": "Field 'name' of 'User'."},
                        "age": {"type": "integer", "description": "Field 'age' of 'User'."},
                    },
                }
            },
            "required": ["user"],
        }

        assert tool.name == "user_greeter"
        assert tool.description == "A simple component that processes a User."

        # Test tool invocation
        result = tool.invoke(user={"name": "Alice", "age": 30})
        assert isinstance(result, dict)
        assert "message" in result
        assert result["message"] == "User Alice is 30 years old"

    def test_from_component_with_list_input(self):
        component = ListProcessor()

        tool = ComponentTool(
            component=component, name="list_processing_tool", description="A tool that concatenates strings"
        )

        assert tool.parameters == {
            "type": "object",
            "properties": {
                "texts": {
                    "type": "array",
                    "description": "The list of strings to concatenate.",
                    "items": {"type": "string"},
                }
            },
            "required": ["texts"],
        }

        # Test tool invocation
        result = tool.invoke(texts=["hello", "world"])
        assert isinstance(result, dict)
        assert "concatenated" in result
        assert result["concatenated"] == "hello world"

    def test_from_component_with_nested_dataclass(self):
        component = PersonProcessor()

        tool = ComponentTool(component=component, name="person_tool", description="A tool that processes people")

        assert tool.parameters == {
            "type": "object",
            "properties": {
                "person": {
                    "type": "object",
                    "description": "The Person to process.",
                    "properties": {
                        "name": {"type": "string", "description": "Field 'name' of 'Person'."},
                        "address": {
                            "type": "object",
                            "description": "Field 'address' of 'Person'.",
                            "properties": {
                                "street": {"type": "string", "description": "Field 'street' of 'Address'."},
                                "city": {"type": "string", "description": "Field 'city' of 'Address'."},
                            },
                        },
                    },
                }
            },
            "required": ["person"],
        }

        # Test tool invocation
        result = tool.invoke(person={"name": "Diana", "address": {"street": "123 Elm Street", "city": "Metropolis"}})
        assert isinstance(result, dict)
        assert "info" in result
        assert result["info"] == "Diana lives at 123 Elm Street, Metropolis."

    def test_from_component_with_document_list(self):
        component = DocumentProcessor()

        tool = ComponentTool(
            component=component, name="document_processor", description="A tool that concatenates document contents"
        )

        assert tool.parameters == {
            "type": "object",
            "properties": {
                "documents": {
                    "type": "array",
                    "description": "List of Documents whose content will be concatenated",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "Field 'id' of 'Document'."},
                            "content": {"type": "string", "description": "Field 'content' of 'Document'."},
                            "blob": {
                                "type": "object",
                                "description": "Field 'blob' of 'Document'.",
                                "properties": {
                                    "data": {"type": "string", "description": "Field 'data' of 'ByteStream'."},
                                    "meta": {"type": "string", "description": "Field 'meta' of 'ByteStream'."},
                                    "mime_type": {
                                        "type": "string",
                                        "description": "Field 'mime_type' of 'ByteStream'.",
                                    },
                                },
                            },
                            "meta": {"type": "string", "description": "Field 'meta' of 'Document'."},
                            "score": {"type": "number", "description": "Field 'score' of 'Document'."},
                            "embedding": {
                                "type": "array",
                                "description": "Field 'embedding' of 'Document'.",
                                "items": {"type": "number"},
                            },
                            "sparse_embedding": {
                                "type": "object",
                                "description": "Field 'sparse_embedding' of 'Document'.",
                                "properties": {
                                    "indices": {
                                        "type": "array",
                                        "description": "Field 'indices' of 'SparseEmbedding'.",
                                        "items": {"type": "integer"},
                                    },
                                    "values": {
                                        "type": "array",
                                        "description": "Field 'values' of 'SparseEmbedding'.",
                                        "items": {"type": "number"},
                                    },
                                },
                            },
                        },
                    },
                },
                "top_k": {"description": "The number of top documents to concatenate", "type": "integer"},
            },
            "required": ["documents"],
        }

        # Test tool invocation
        result = tool.invoke(documents=[{"content": "First document"}, {"content": "Second document"}])
        assert isinstance(result, dict)
        assert "concatenated" in result
        assert result["concatenated"] == "First document\nSecond document"

    def test_from_component_with_non_component(self):
        class NotAComponent:
            def foo(self, text: str):
                return {"reply": f"Hello, {text}!"}

        not_a_component = NotAComponent()

        with pytest.raises(ValueError):
            ComponentTool(component=not_a_component, name="invalid_tool", description="This should fail")


## Integration tests
class TestToolComponentInPipelineWithOpenAI:
    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.integration
    def test_component_tool_in_pipeline(self):
        # Create component and convert it to tool
        component = SimpleComponent()
        tool = ComponentTool(
            component=component, name="hello_tool", description="A tool that generates a greeting message for the user"
        )

        # Create pipeline with OpenAIChatGenerator and ToolInvoker
        pipeline = Pipeline()
        pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini", tools=[tool]))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=[tool]))

        # Connect components
        pipeline.connect("llm.replies", "tool_invoker.messages")

        message = ChatMessage.from_user(text="Vladimir")

        # Run pipeline
        result = pipeline.run({"llm": {"messages": [message]}})

        # Check results
        tool_messages = result["tool_invoker"]["tool_messages"]
        assert len(tool_messages) == 1

        tool_message = tool_messages[0]
        assert tool_message.is_from(ChatRole.TOOL)
        assert "Vladimir" in tool_message.tool_call_result.result
        assert not tool_message.tool_call_result.error

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.integration
    def test_component_tool_in_pipeline_openai_tools_strict(self):
        # Create component and convert it to tool
        component = SimpleComponent()
        tool = ComponentTool(
            component=component, name="hello_tool", description="A tool that generates a greeting message for the user"
        )

        # Create pipeline with OpenAIChatGenerator and ToolInvoker
        pipeline = Pipeline()
        pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini", tools=[tool], tools_strict=True))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=[tool]))

        # Connect components
        pipeline.connect("llm.replies", "tool_invoker.messages")

        message = ChatMessage.from_user(text="Vladimir")

        # Run pipeline
        result = pipeline.run({"llm": {"messages": [message]}})

        # Check results
        tool_messages = result["tool_invoker"]["tool_messages"]
        assert len(tool_messages) == 1

        tool_message = tool_messages[0]
        assert tool_message.is_from(ChatRole.TOOL)
        assert "Vladimir" in tool_message.tool_call_result.result
        assert not tool_message.tool_call_result.error

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.integration
    def test_user_greeter_in_pipeline(self):
        component = UserGreeter()
        tool = ComponentTool(
            component=component, name="user_greeter", description="A tool that greets users with their name and age"
        )

        pipeline = Pipeline()
        pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini", tools=[tool]))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=[tool]))
        pipeline.connect("llm.replies", "tool_invoker.messages")

        message = ChatMessage.from_user(text="I am Alice and I'm 30 years old")

        result = pipeline.run({"llm": {"messages": [message]}})
        tool_messages = result["tool_invoker"]["tool_messages"]
        assert len(tool_messages) == 1

        tool_message = tool_messages[0]
        assert tool_message.is_from(ChatRole.TOOL)
        assert tool_message.tool_call_result.result == str({"message": "User Alice is 30 years old"})
        assert not tool_message.tool_call_result.error

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.integration
    def test_list_processor_in_pipeline(self):
        component = ListProcessor()
        tool = ComponentTool(
            component=component, name="list_processor", description="A tool that concatenates a list of strings"
        )

        pipeline = Pipeline()
        pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini", tools=[tool]))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=[tool]))
        pipeline.connect("llm.replies", "tool_invoker.messages")

        message = ChatMessage.from_user(text="Can you join these words: hello, beautiful, world")

        result = pipeline.run({"llm": {"messages": [message]}})
        tool_messages = result["tool_invoker"]["tool_messages"]
        assert len(tool_messages) == 1

        tool_message = tool_messages[0]
        assert tool_message.is_from(ChatRole.TOOL)
        assert tool_message.tool_call_result.result == str({"concatenated": "hello beautiful world"})
        assert not tool_message.tool_call_result.error

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.integration
    def test_person_processor_in_pipeline(self):
        component = PersonProcessor()
        tool = ComponentTool(
            component=component,
            name="person_processor",
            description="A tool that processes information about a person and their address",
        )

        pipeline = Pipeline()
        pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini", tools=[tool]))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=[tool]))
        pipeline.connect("llm.replies", "tool_invoker.messages")

        message = ChatMessage.from_user(text="Diana lives at 123 Elm Street in Metropolis")

        result = pipeline.run({"llm": {"messages": [message]}})
        tool_messages = result["tool_invoker"]["tool_messages"]
        assert len(tool_messages) == 1

        tool_message = tool_messages[0]
        assert tool_message.is_from(ChatRole.TOOL)
        assert "Diana" in tool_message.tool_call_result.result and "Metropolis" in tool_message.tool_call_result.result
        assert not tool_message.tool_call_result.error

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.integration
    def test_document_processor_in_pipeline(self):
        component = DocumentProcessor()
        tool = ComponentTool(
            component=component,
            name="document_processor",
            description="A tool that concatenates the content of multiple documents",
        )

        pipeline = Pipeline()
        pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini", tools=[tool]))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=[tool], convert_result_to_json_string=True))
        pipeline.connect("llm.replies", "tool_invoker.messages")

        message = ChatMessage.from_user(
            text="Concatenate these documents: First one says 'Hello world' and second one says 'Goodbye world' and third one says 'Hello again', but use top_k=2. Set only content field of the document only. Do not set id, meta, score, embedding, sparse_embedding, dataframe, blob fields."
        )

        result = pipeline.run({"llm": {"messages": [message]}})

        tool_messages = result["tool_invoker"]["tool_messages"]
        assert len(tool_messages) == 1

        tool_message = tool_messages[0]
        assert tool_message.is_from(ChatRole.TOOL)
        result = json.loads(tool_message.tool_call_result.result)
        assert "concatenated" in result
        assert "Hello world" in result["concatenated"]
        assert "Goodbye world" in result["concatenated"]
        assert not tool_message.tool_call_result.error

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.integration
    def test_lost_in_middle_ranker_in_pipeline(self):
        from haystack.components.rankers import LostInTheMiddleRanker

        component = LostInTheMiddleRanker()
        tool = ComponentTool(
            component=component,
            name="lost_in_middle_ranker",
            description="A tool that ranks documents using the Lost in the Middle algorithm and returns top k results",
        )

        pipeline = Pipeline()
        pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini", tools=[tool]))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=[tool]))
        pipeline.connect("llm.replies", "tool_invoker.messages")

        message = ChatMessage.from_user(
            text="I have three documents with content: 'First doc', 'Middle doc', and 'Last doc'. Rank them top_k=2. Set only content field of the document only. Do not set id, meta, score, embedding, sparse_embedding, dataframe, blob fields."
        )

        result = pipeline.run({"llm": {"messages": [message]}})

        tool_messages = result["tool_invoker"]["tool_messages"]
        assert len(tool_messages) == 1
        tool_message = tool_messages[0]
        assert tool_message.is_from(ChatRole.TOOL)

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.skipif(not os.environ.get("SERPERDEV_API_KEY"), reason="SERPERDEV_API_KEY not set")
    @pytest.mark.integration
    def test_serper_dev_web_search_in_pipeline(self):
        component = SerperDevWebSearch(api_key=Secret.from_env_var("SERPERDEV_API_KEY"), top_k=3)
        tool = ComponentTool(
            component=component, name="web_search", description="Search the web for current information on any topic"
        )

        pipeline = Pipeline()
        pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini", tools=[tool]))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=[tool]))
        pipeline.connect("llm.replies", "tool_invoker.messages")

        result = pipeline.run(
            {
                "llm": {
                    "messages": [
                        ChatMessage.from_user(text="Use the web search tool to find information about Nikola Tesla")
                    ]
                }
            }
        )

        assert len(result["tool_invoker"]["tool_messages"]) == 1
        tool_message = result["tool_invoker"]["tool_messages"][0]
        assert tool_message.is_from(ChatRole.TOOL)
        assert "Nikola Tesla" in tool_message.tool_call_result.result
        assert not tool_message.tool_call_result.error

    def test_serde_in_pipeline(self, monkeypatch):
        monkeypatch.setenv("SERPERDEV_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Create the search component and tool
        search = SerperDevWebSearch(top_k=3)
        tool = ComponentTool(component=search, name="web_search", description="Search the web for current information")

        # Create and configure the pipeline
        pipeline = Pipeline()
        pipeline.add_component("tool_invoker", ToolInvoker(tools=[tool]))
        pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini", tools=[tool]))
        pipeline.connect("tool_invoker.tool_messages", "llm.messages")

        # Serialize to dict and verify structure
        pipeline_dict = pipeline.to_dict()
        assert (
            pipeline_dict["components"]["tool_invoker"]["type"] == "haystack.components.tools.tool_invoker.ToolInvoker"
        )
        assert len(pipeline_dict["components"]["tool_invoker"]["init_parameters"]["tools"]) == 1

        tool_dict = pipeline_dict["components"]["tool_invoker"]["init_parameters"]["tools"][0]
        assert tool_dict["type"] == "haystack.tools.component_tool.ComponentTool"
        assert tool_dict["data"]["name"] == "web_search"
        assert tool_dict["data"]["component"]["type"] == "haystack.components.websearch.serper_dev.SerperDevWebSearch"
        assert tool_dict["data"]["component"]["init_parameters"]["top_k"] == 3
        assert tool_dict["data"]["component"]["init_parameters"]["api_key"]["type"] == "env_var"

        # Test round-trip serialization
        pipeline_yaml = pipeline.dumps()
        new_pipeline = Pipeline.loads(pipeline_yaml)
        assert new_pipeline == pipeline

    def test_component_tool_serde(self):
        component = SimpleComponent()

        tool = ComponentTool(
            component=component,
            name="simple_tool",
            description="A simple tool",
            inputs_from_state={"test": "input"},
            outputs_to_state={"output": {"source": "out", "handler": output_handler}},
        )

        # Test serialization
        tool_dict = tool.to_dict()
        assert tool_dict["type"] == "haystack.tools.component_tool.ComponentTool"
        assert tool_dict["data"]["name"] == "simple_tool"
        assert tool_dict["data"]["description"] == "A simple tool"
        assert "component" in tool_dict["data"]
        assert tool_dict["data"]["inputs_from_state"] == {"test": "input"}
        assert tool_dict["data"]["outputs_to_state"]["output"]["handler"] == "test_component_tool.output_handler"

        # Test deserialization
        new_tool = ComponentTool.from_dict(tool_dict)
        assert new_tool.name == tool.name
        assert new_tool.description == tool.description
        assert new_tool.parameters == tool.parameters
        assert new_tool.inputs_from_state == tool.inputs_from_state
        assert new_tool.outputs_to_state == tool.outputs_to_state
        assert isinstance(new_tool._component, SimpleComponent)

    def test_pipeline_component_fails(self):
        component = SimpleComponent()

        # Create a pipeline and add the component to it
        pipeline = Pipeline()
        pipeline.add_component("simple", component)

        # Try to create a tool from the component and it should fail because the component has been added to a pipeline and
        # thus can't be used as tool
        with pytest.raises(ValueError, match="Component has been added to a pipeline"):
            ComponentTool(component=component)

    def test_deepcopy_with_jinja_based_component(self):
        builder = PromptBuilder("{{query}}")
        tool = ComponentTool(component=builder)
        result = tool.function(query="Hello")
        tool_copy = deepcopy_with_fallback(tool)
        result_from_copy = tool_copy.function(query="Hello")
        assert "prompt" in result_from_copy
        assert result_from_copy["prompt"] == result["prompt"]

    def test_jinja_based_component_tool_in_pipeline(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        with patch("openai.resources.chat.completions.Completions.create") as mock_create:
            mock_create.return_value = ChatCompletion(
                id="test",
                model="gpt-4o-mini",
                object="chat.completion",
                choices=[
                    Choice(
                        finish_reason="length",
                        index=0,
                        message=ChatCompletionMessage(role="assistant", content="A response from the model"),
                    )
                ],
                created=1234567890,
            )

            builder = PromptBuilder("{{query}}")
            tool = ComponentTool(component=builder)
            pipeline = Pipeline()
            pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini"))
            result = pipeline.run({"llm": {"messages": [ChatMessage.from_user(text="Hello")], "tools": [tool]}})

        assert result["llm"]["replies"][0].text == "A response from the model"
