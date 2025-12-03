# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from dataclasses import dataclass
from unittest.mock import patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from haystack import Pipeline, SuperComponent, component
from haystack.components.builders import PromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.tools import ToolInvoker
from haystack.components.websearch.serper_dev import SerperDevWebSearch
from haystack.core.pipeline.utils import _deepcopy_with_exceptions
from haystack.dataclasses import ChatMessage, ChatRole, Document
from haystack.tools import ComponentTool
from haystack.utils.auth import Secret
from test.tools.test_parameters_schema_utils import BYTE_STREAM_SCHEMA, DOCUMENT_SCHEMA, SPARSE_EMBEDDING_SCHEMA

# Component and Model Definitions


@component
class SimpleComponentUsingChatMessages:
    """A simple component that generates text."""

    @component.output_types(reply=str)
    def run(self, messages: list[ChatMessage]) -> dict[str, str]:
        """
        A simple component that generates text.

        :param messages: Users messages
        :return: A dictionary with the generated text.
        """
        return {"reply": f"Hello, {messages[0].text}!"}


@component
class SimpleComponent:
    """A simple component that generates text."""

    @component.output_types(reply=str)
    def run(self, text: str) -> dict[str, str]:
        """
        A simple component that generates text.

        :param text: user's name
        :return: A dictionary with the generated text.
        """
        return {"reply": f"Hello, {text}!"}


def reply_formatter(input_text: str) -> str:
    return f"Formatted reply: {input_text}"


@dataclass
class User:
    """A simple user dataclass."""

    name: str = "Anonymous"
    age: int = 0


@component
class UserGreeter:
    """A simple component that processes a User."""

    @component.output_types(message=str)
    def run(self, user: User) -> dict[str, str]:
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
    def run(self, texts: list[str]) -> dict[str, str]:
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
    def run(self, person: Person) -> dict[str, str]:
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
    def run(self, documents: list[Document], top_k: int = 5) -> dict[str, str]:
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


class TestComponentTool:
    def test_from_component_basic(self):
        tool = ComponentTool(component=SimpleComponent())

        assert tool.name == "simple_component"
        assert tool.description == "A simple component that generates text."
        assert tool.parameters == {
            "type": "object",
            "description": "A simple component that generates text.",
            "properties": {"text": {"type": "string", "description": "user's name"}},
            "required": ["text"],
        }

        # Test tool invocation
        result = tool.invoke(text="world")
        assert isinstance(result, dict)
        assert "reply" in result
        assert result["reply"] == "Hello, world!"

    def test_from_component_long_description(self):
        tool = ComponentTool(component=SimpleComponent(), description="".join(["A"] * 1024))
        assert len(tool.description) == 1024

    def test_from_component_with_inputs_from_state(self):
        tool = ComponentTool(component=SimpleComponent(), inputs_from_state={"text": "text"})
        assert tool.inputs_from_state == {"text": "text"}
        # Inputs should be excluded from schema generation
        assert tool.parameters == {
            "type": "object",
            "properties": {},
            "description": "A simple component that generates text.",
        }

    def test_from_component_with_inputs_from_state_different_names(self):
        tool = ComponentTool(component=SimpleComponent(), inputs_from_state={"state_text": "text"})
        assert tool.inputs_from_state == {"state_text": "text"}
        # Inputs should be excluded from schema generation
        assert tool.parameters == {
            "type": "object",
            "properties": {},
            "description": "A simple component that generates text.",
        }

    def test_from_component_with_outputs_to_state(self):
        tool = ComponentTool(component=SimpleComponent(), outputs_to_state={"replies": {"source": "reply"}})
        assert tool.outputs_to_state == {"replies": {"source": "reply"}}

    def test_from_component_with_dataclass(self):
        tool = ComponentTool(component=UserGreeter())
        assert tool.parameters == {
            "$defs": {
                "User": {
                    "properties": {
                        "name": {"description": "Field 'name' of 'User'.", "type": "string", "default": "Anonymous"},
                        "age": {"description": "Field 'age' of 'User'.", "type": "integer", "default": 0},
                    },
                    "type": "object",
                }
            },
            "description": "A simple component that processes a User.",
            "properties": {"user": {"$ref": "#/$defs/User", "description": "The User object to process."}},
            "required": ["user"],
            "type": "object",
        }

        assert tool.name == "user_greeter"
        assert tool.description == "A simple component that processes a User."

        # Test tool invocation
        result = tool.invoke(user={"name": "Alice", "age": 30})
        assert isinstance(result, dict)
        assert "message" in result
        assert result["message"] == "User Alice is 30 years old"

    def test_from_component_with_list_input(self):
        tool = ComponentTool(
            component=ListProcessor(), name="list_processing_tool", description="A tool that concatenates strings"
        )

        assert tool.parameters == {
            "type": "object",
            "description": "Concatenates a list of strings into a single string.",
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
        tool = ComponentTool(
            component=PersonProcessor(), name="person_tool", description="A tool that processes people"
        )

        assert tool.parameters == {
            "$defs": {
                "Address": {
                    "properties": {
                        "street": {"description": "Field 'street' of 'Address'.", "type": "string"},
                        "city": {"description": "Field 'city' of 'Address'.", "type": "string"},
                    },
                    "required": ["street", "city"],
                    "type": "object",
                },
                "Person": {
                    "properties": {
                        "name": {"description": "Field 'name' of 'Person'.", "type": "string"},
                        "address": {"$ref": "#/$defs/Address", "description": "Field 'address' of 'Person'."},
                    },
                    "required": ["name", "address"],
                    "type": "object",
                },
            },
            "description": "Creates information about the person.",
            "properties": {"person": {"$ref": "#/$defs/Person", "description": "The Person to process."}},
            "required": ["person"],
            "type": "object",
        }

        # Test tool invocation
        result = tool.invoke(person={"name": "Diana", "address": {"street": "123 Elm Street", "city": "Metropolis"}})
        assert isinstance(result, dict)
        assert "info" in result
        assert result["info"] == "Diana lives at 123 Elm Street, Metropolis."

    def test_from_component_with_list_of_documents(self):
        tool = ComponentTool(
            component=DocumentProcessor(),
            name="document_processor",
            description="A tool that concatenates document contents",
        )

        assert tool.parameters == {
            "$defs": {
                "ByteStream": BYTE_STREAM_SCHEMA,
                "Document": DOCUMENT_SCHEMA,
                "SparseEmbedding": SPARSE_EMBEDDING_SCHEMA,
            },
            "description": "Concatenates the content of multiple documents with newlines.",
            "properties": {
                "documents": {
                    "description": "List of Documents whose content will be concatenated",
                    "items": {"$ref": "#/$defs/Document"},
                    "type": "array",
                },
                "top_k": {"description": "The number of top documents to concatenate", "type": "integer", "default": 5},
            },
            "required": ["documents"],
            "type": "object",
        }

        # Test tool invocation
        result = tool.invoke(documents=[{"content": "First document"}, {"content": "Second document"}])
        assert isinstance(result, dict)
        assert "concatenated" in result
        assert result["concatenated"] == "First document\nSecond document"

    def test_from_component_with_dynamic_input_types(self):
        builder = PromptBuilder(template="Hello, {{name}}!")
        tool = ComponentTool(component=builder, name="prompt_builder_tool")
        assert tool.parameters == {
            "description": "Renders the prompt template with the provided variables.",
            "properties": {
                "name": {"default": "", "description": "Input 'name' for the component."},
                "template": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                    "description": "An optional string template to overwrite PromptBuilder's default template. If "
                    "None, the default template\nprovided at initialization is used.",
                },
                "template_variables": {
                    "anyOf": [{"additionalProperties": True, "type": "object"}, {"type": "null"}],
                    "default": None,
                    "description": "An optional dictionary of template variables to overwrite the pipeline variables.",
                },
            },
            "type": "object",
        }

    def test_from_component_with_invalid_component(self):
        class NotAComponent:
            def foo(self, text: str):
                return {"reply": f"Hello, {text}!"}

        not_a_component = NotAComponent()

        with pytest.raises(ValueError):
            ComponentTool(component=not_a_component, name="invalid_tool", description="This should fail")

    def test_component_invoker_with_chat_message_input(self):
        tool = ComponentTool(
            component=SimpleComponentUsingChatMessages(), name="simple_tool", description="A simple tool"
        )
        result = tool.invoke(messages=[ChatMessage.from_user(text="world")])
        assert result == {"reply": "Hello, world!"}

    def test_component_tool_with_super_component_docstrings(self, monkeypatch):
        """Test that ComponentTool preserves docstrings from underlying pipeline components in SuperComponents."""

        @component
        class AnnotatedComponent:
            """An annotated component with descriptive parameter docstrings."""

            @component.output_types(result=str)
            def run(self, text: str, number: int = 42):
                """
                Process inputs and return result.

                :param text: A detailed description of the text parameter that should be preserved
                :param number: A detailed description of the number parameter that should be preserved
                """
                return {"result": f"Processed: {text} and {number}"}

        # Create a pipeline with the annotated component
        pipeline = Pipeline()
        pipeline.add_component("processor", AnnotatedComponent())
        # Create SuperComponent with mapping
        super_comp = SuperComponent(
            pipeline=pipeline,
            input_mapping={"input_text": ["processor.text"], "input_number": ["processor.number"]},
            output_mapping={"processor.result": "processed_result"},
        )

        # Create ComponentTool from SuperComponent
        tool = ComponentTool(component=super_comp, name="text_processor")

        # Verify that schema includes the docstrings from the original component
        assert tool.parameters == {
            "type": "object",
            "description": "A component that combines: 'processor': Process inputs and return result.",
            "properties": {
                "input_text": {
                    "type": "string",
                    "description": "Provided to the 'processor' component as: 'A detailed description of the text "
                    "parameter that should be preserved'.",
                },
                "input_number": {
                    "type": "integer",
                    "description": "Provided to the 'processor' component as: 'A detailed description of the number "
                    "parameter that should be preserved'.",
                },
            },
            "required": ["input_text"],
        }

        # Test the tool functionality works
        result = tool.invoke(input_text="Hello", input_number=42)
        assert result["processed_result"] == "Processed: Hello and 42"

    def test_component_tool_with_multiple_mapped_docstrings(self):
        """
        Test ComponentTool combines docstrings from multiple components when a single input maps to multiple components.
        """

        @component
        class ComponentA:
            """Component A with descriptive docstrings."""

            @component.output_types(output_a=str)
            def run(self, query: str):
                """
                Process query in component A.

                :param query: The query string for component A
                """
                return {"output_a": f"A processed: {query}"}

        @component
        class ComponentB:
            """Component B with descriptive docstrings."""

            @component.output_types(output_b=str)
            def run(self, text: str):
                """
                Process text in component B.

                :param text: Text to process in component B
                """
                return {"output_b": f"B processed: {text}"}

        # Create a pipeline with both components
        pipeline = Pipeline()
        pipeline.add_component("comp_a", ComponentA())
        pipeline.add_component("comp_b", ComponentB())

        # Create SuperComponent with a single input mapped to both components
        super_comp = SuperComponent(
            pipeline=pipeline, input_mapping={"combined_input": ["comp_a.query", "comp_b.text"]}
        )

        # Create ComponentTool from SuperComponent
        tool = ComponentTool(component=super_comp, name="combined_processor")

        # Verify that schema includes combined docstrings from both components
        assert tool.parameters == {
            "type": "object",
            "description": "A component that combines: 'comp_a': Process query in component A., 'comp_b': Process "
            "text in component B.",
            "properties": {
                "combined_input": {
                    "type": "string",
                    "description": "Provided to the 'comp_a' component as: 'The query string for component A', and "
                    "Provided to the 'comp_b' component as: 'Text to process in component B'.",
                }
            },
            "required": ["combined_input"],
        }

        # Test the tool functionality works
        result = tool.invoke(combined_input="test input")
        assert result["output_a"] == "A processed: test input"
        assert result["output_b"] == "B processed: test input"

    def test_warm_up_is_idempotent(self):
        """Test that calling warm_up multiple times only warms up the component once."""
        from unittest.mock import MagicMock

        component = SimpleComponent()
        component.warm_up = MagicMock()

        tool = ComponentTool(component=component)

        # Call warm_up multiple times
        tool.warm_up()
        tool.warm_up()
        tool.warm_up()

        # Component's warm_up should only be called once
        component.warm_up.assert_called_once()


class TestComponentToolInPipeline:
    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.integration
    def test_component_tool_in_pipeline(self):
        # Create component and convert it to tool
        tool = ComponentTool(
            component=SimpleComponent(),
            name="hello_tool",
            description="A tool that generates a greeting message for the user",
        )

        # Create pipeline with OpenAIChatGenerator and ToolInvoker
        pipeline = Pipeline()
        pipeline.add_component("llm", OpenAIChatGenerator(tools=[tool]))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=[tool]))

        # Connect components
        pipeline.connect("llm.replies", "tool_invoker.messages")

        message = ChatMessage.from_user(text="Using tools, greet Vladimir")

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
    @pytest.mark.flaky(reruns=3, reruns_delay=10)
    def test_component_tool_in_pipeline_openai_tools_strict(self):
        # Create component and convert it to tool
        tool = ComponentTool(
            component=SimpleComponent(),
            name="hello_tool",
            description="A tool that generates a greeting message for the user",
        )

        # Create pipeline with OpenAIChatGenerator and ToolInvoker
        pipeline = Pipeline()
        pipeline.add_component("llm", OpenAIChatGenerator(tools=[tool], tools_strict=True))
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
        tool = ComponentTool(
            component=UserGreeter(), name="user_greeter", description="A tool that greets users with their name and age"
        )

        pipeline = Pipeline()
        pipeline.add_component("llm", OpenAIChatGenerator(tools=[tool]))
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
        tool = ComponentTool(
            component=ListProcessor(), name="list_processor", description="A tool that concatenates a list of strings"
        )

        pipeline = Pipeline()
        pipeline.add_component("llm", OpenAIChatGenerator(tools=[tool]))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=[tool]))
        pipeline.connect("llm.replies", "tool_invoker.messages")

        # Be explicit about using tools, otherwise model will ignore the tool call and return the result directly.
        message = ChatMessage.from_user(text="Using tools, join these words: hello, beautiful, world")

        result = pipeline.run({"llm": {"messages": [message]}})
        tool_messages = result["tool_invoker"]["tool_messages"]
        assert len(tool_messages) == 1

        tool_message = tool_messages[0]
        assert tool_message.is_from(ChatRole.TOOL)
        # Check that the result contains the expected words (handle whitespace variations)
        result_str = tool_message.tool_call_result.result
        assert "concatenated" in result_str
        # Normalize whitespace in the result string and check it contains the expected words
        normalized_result = " ".join(result_str.split())
        assert "hello" in normalized_result
        assert "beautiful" in normalized_result
        assert "world" in normalized_result
        assert not tool_message.tool_call_result.error

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.integration
    def test_person_processor_in_pipeline(self):
        tool = ComponentTool(
            component=PersonProcessor(),
            name="person_processor",
            description="A tool that processes information about a person and their address",
        )

        pipeline = Pipeline()
        pipeline.add_component("llm", OpenAIChatGenerator(tools=[tool]))
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
        tool = ComponentTool(
            component=DocumentProcessor(),
            name="document_processor",
            description="A tool that concatenates the content of multiple documents",
        )

        pipeline = Pipeline()
        pipeline.add_component("llm", OpenAIChatGenerator(tools=[tool]))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=[tool], convert_result_to_json_string=True))
        pipeline.connect("llm.replies", "tool_invoker.messages")

        message = ChatMessage.from_user(
            text="Concatenate these documents: First one says 'Hello world' and second one says 'Goodbye world' and "
            "third one says 'Hello again', but use top_k=2. Set only content field of the document only. Do "
            "not set id, meta, score, embedding, sparse_embedding, dataframe, blob fields."
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

        tool = ComponentTool(
            component=LostInTheMiddleRanker(),
            name="lost_in_middle_ranker",
            description="A tool that ranks documents using the Lost in the Middle algorithm and returns top k results",
        )

        pipeline = Pipeline()
        pipeline.add_component("llm", OpenAIChatGenerator(tools=[tool]))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=[tool]))
        pipeline.connect("llm.replies", "tool_invoker.messages")

        message = ChatMessage.from_user(
            text="I have three documents with content: 'First doc', 'Middle doc', and 'Last doc'. Rank them top_k=2. "
            "Set only content field of the document only. Do not set id, meta, score, embedding, "
            "sparse_embedding, dataframe, blob fields."
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
        tool = ComponentTool(
            component=SerperDevWebSearch(api_key=Secret.from_env_var("SERPERDEV_API_KEY"), top_k=3),
            name="web_search",
            description="Search the web for current information on any topic",
        )

        pipeline = Pipeline()
        pipeline.add_component("llm", OpenAIChatGenerator(tools=[tool]))
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
        pipeline.add_component("llm", OpenAIChatGenerator(tools=[tool]))
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
        tool = ComponentTool(
            component=SimpleComponent(),
            name="simple_tool",
            description="A simple tool",
            outputs_to_string={"source": "reply", "handler": reply_formatter},
            inputs_from_state={"test": "input"},
            outputs_to_state={"output": {"source": "out", "handler": output_handler}},
        )

        # Test serialization
        expected_tool_dict = {
            "type": "haystack.tools.component_tool.ComponentTool",
            "data": {
                "component": {"type": "test_component_tool.SimpleComponent", "init_parameters": {}},
                "name": "simple_tool",
                "description": "A simple tool",
                "parameters": None,
                "outputs_to_string": {"source": "reply", "handler": "test_component_tool.reply_formatter"},
                "inputs_from_state": {"test": "input"},
                "outputs_to_state": {"output": {"source": "out", "handler": "test_component_tool.output_handler"}},
            },
        }
        tool_dict = tool.to_dict()
        assert tool_dict == expected_tool_dict

        # Test deserialization
        new_tool = ComponentTool.from_dict(expected_tool_dict)
        assert new_tool.name == tool.name
        assert new_tool.description == tool.description
        assert new_tool.parameters == tool.parameters
        assert new_tool.outputs_to_string == tool.outputs_to_string
        assert new_tool.inputs_from_state == tool.inputs_from_state
        assert new_tool.outputs_to_state == tool.outputs_to_state
        assert isinstance(new_tool._component, SimpleComponent)

    def test_pipeline_component_fails(self):
        comp = SimpleComponent()

        # Create a pipeline and add the component to it
        pipeline = Pipeline()
        pipeline.add_component("simple", comp)

        # Try to create a tool from the component and it should fail because the component has been added to a pipeline
        # and thus can't be used as tool
        with pytest.raises(ValueError, match="Component has been added to a pipeline"):
            ComponentTool(component=comp)

    def test_deepcopy_with_jinja_based_component(self):
        builder = PromptBuilder("{{query}}")
        tool = ComponentTool(component=builder)
        result = tool.function(query="Hello")
        tool_copy = _deepcopy_with_exceptions(tool)
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

            tool = ComponentTool(component=PromptBuilder("{{query}}"))
            pipeline = Pipeline()
            pipeline.add_component("llm", OpenAIChatGenerator())
            result = pipeline.run({"llm": {"messages": [ChatMessage.from_user(text="Hello")], "tools": [tool]}})

        assert result["llm"]["replies"][0].text == "A response from the model"
