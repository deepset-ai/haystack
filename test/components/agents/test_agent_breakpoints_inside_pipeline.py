# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from haystack import component
from haystack.components.agents import Agent
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.core.errors import BreakpointException
from haystack.core.pipeline import Pipeline
from haystack.core.pipeline.breakpoint import load_state
from haystack.dataclasses import ByteStream, ChatMessage, Document, ToolCall
from haystack.dataclasses.breakpoints import AgentBreakpoint, Breakpoint, ToolBreakpoint
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.tools import tool

document_store = InMemoryDocumentStore()


@component
class MockLinkContentFetcher:
    @component.output_types(streams=List[ByteStream])
    def run(self, urls: List[str]) -> Dict[str, List[ByteStream]]:
        mock_html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Deepset - About Our Team</title>
        </head>
        <body>
            <h1>About Deepset</h1>
            <p>Deepset is a company focused on natural language processing and AI.</p>
            <h2>Our Leadership Team</h2>
            <div class="team-member">
                <h3>Malte Pietsch</h3>
                <p>Malte Pietsch is the CEO and co-founder of Deepset. He has extensive experience in machine learning
                and natural language processing.</p>
                <p>Job Title: Chief Executive Officer</p>
            </div>
            <div class="team-member">
                <h3>Milos Rusic</h3>
                <p>Milos Rusic is the CTO and co-founder of Deepset. He specializes in building scalable AI systems
                and has worked on various NLP projects.</p>
                <p>Job Title: Chief Technology Officer</p>
            </div>
            <h2>Our Mission</h2>
            <p>Deepset aims to make natural language processing accessible to developers and businesses worldwide
            through open-source tools and enterprise solutions.</p>
        </body>
        </html>
        """

        bytestream = ByteStream(
            data=mock_html_content.encode("utf-8"),
            mime_type="text/html",
            meta={"url": urls[0] if urls else "https://en.wikipedia.org/wiki/Deepset"},
        )

        return {"streams": [bytestream]}


@component
class MockHTMLToDocument:
    @component.output_types(documents=List[Document])
    def run(self, sources: List[ByteStream]) -> Dict[str, List[Document]]:
        """Mock HTML to Document converter that extracts text content from HTML ByteStreams."""

        documents = []
        for source in sources:
            # Extract the HTML content from the ByteStream
            html_content = source.data.decode("utf-8")

            # Simple text extraction - remove HTML tags and extract meaningful content
            # This is a simplified version that extracts the main content
            # Remove HTML tags
            text_content = re.sub(r"<[^>]+>", " ", html_content)
            # Remove extra whitespace
            text_content = re.sub(r"\s+", " ", text_content).strip()

            # Create a Document with the extracted text
            document = Document(
                content=text_content,
                meta={"url": source.meta.get("url", "unknown"), "mime_type": source.mime_type, "source_type": "html"},
            )
            documents.append(document)

        return {"documents": documents}


@tool
def add_database_tool(name: str, surname: str, job_title: Optional[str], other: Optional[str]):
    document_store.write_documents(
        [Document(content=name + " " + surname + " " + (job_title or ""), meta={"other": other})]
    )


@pytest.fixture
def pipeline_with_agent(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test_key")
    generator = OpenAIChatGenerator()
    call_count = 0

    def mock_run(messages, tools=None, **kwargs):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            return {
                "replies": [
                    ChatMessage.from_assistant(
                        "I'll extract the information about the people mentioned in the context.",
                        tool_calls=[
                            ToolCall(
                                tool_name="add_database_tool",
                                arguments={
                                    "name": "Malte",
                                    "surname": "Pietsch",
                                    "job_title": "Chief Executive Officer",
                                    "other": "CEO and co-founder of Deepset with extensive experience in machine "
                                    "learning and natural language processing",
                                },
                            ),
                            ToolCall(
                                tool_name="add_database_tool",
                                arguments={
                                    "name": "Milos",
                                    "surname": "Rusic",
                                    "job_title": "Chief Technology Officer",
                                    "other": "CTO and co-founder of Deepset specializing in building scalable "
                                    "AI systems and NLP projects",
                                },
                            ),
                        ],
                    )
                ]
            }
        else:
            return {
                "replies": [
                    ChatMessage.from_assistant(
                        "I have successfully extracted and stored information about the following people:\n\n"
                        "1. **Malte Pietsch** - Chief Executive Officer\n"
                        "   - CEO and co-founder of Deepset\n"
                        "   - Extensive experience in machine learning and natural language processing\n\n"
                        "2. **Milos Rusic** - Chief Technology Officer\n"
                        "   - CTO and co-founder of Deepset\n"
                        "   - Specializes in building scalable AI systems and NLP projects\n\n"
                        "Both individuals have been added to the knowledge base with their respective information."
                    )
                ]
            }

    generator.run = mock_run

    database_assistant = Agent(
        chat_generator=generator,
        tools=[add_database_tool],
        system_prompt="""
        You are a database assistant.
        Your task is to extract the names of people mentioned in the given context and add them to a knowledge base,
        along with additional relevant information about them that can be extracted from the context.
        Do not use you own knowledge, stay grounded to the given context.
        Do not ask the user for confirmation. Instead, automatically update the knowledge base and return a brief
        summary of the people added, including the information stored for each.
        """,
        exit_conditions=["text"],
        max_agent_steps=100,
        raise_on_tool_invocation_failure=False,
    )

    extraction_agent = Pipeline()
    extraction_agent.add_component("fetcher", MockLinkContentFetcher())
    extraction_agent.add_component("converter", MockHTMLToDocument())
    extraction_agent.add_component(
        "builder",
        ChatPromptBuilder(
            template=[
                ChatMessage.from_user("""
        {% for doc in docs %}
        {{ doc.content|default|truncate(25000) }}
        {% endfor %}
        """)
            ],
            required_variables=["docs"],
        ),
    )
    extraction_agent.add_component("database_agent", database_assistant)

    extraction_agent.connect("fetcher.streams", "converter.sources")
    extraction_agent.connect("converter.documents", "builder.docs")
    extraction_agent.connect("builder", "database_agent")

    return extraction_agent


def run_pipeline_without_any_breakpoints(pipeline_with_agent):
    agent_output = pipeline_with_agent.run(data={"fetcher": {"urls": ["https://en.wikipedia.org/wiki/Deepset"]}})

    # pipeline completed
    assert "database_agent" in agent_output
    assert "messages" in agent_output["database_agent"]
    assert len(agent_output["database_agent"]["messages"]) > 0

    # final message contains the expected summary
    final_message = agent_output["database_agent"]["messages"][-1].text
    assert "Malte Pietsch" in final_message
    assert "Milos Rusic" in final_message
    assert "Chief Executive Officer" in final_message
    assert "Chief Technology Officer" in final_message


def test_chat_generator_breakpoint_in_pipeline_agent(pipeline_with_agent):
    with tempfile.TemporaryDirectory() as debug_path:
        agent_generator_breakpoint = Breakpoint("chat_generator", 0, debug_path=debug_path)
        agent_breakpoint = AgentBreakpoint(break_point=agent_generator_breakpoint, agent_name="database_agent")
        try:
            pipeline_with_agent.run(
                data={"fetcher": {"urls": ["https://en.wikipedia.org/wiki/Deepset"]}}, break_point=agent_breakpoint
            )
            assert False, "Expected exception was not raised"

        except BreakpointException as e:  # this is the exception from the Agent
            assert e.component == "chat_generator"
            assert e.state is not None
            assert "messages" in e.state
            assert e.results is not None

        # verify that debug/state file was created
        chat_generator_state_files = list(Path(debug_path).glob("database_agent_chat_generator_*.json"))
        assert len(chat_generator_state_files) > 0, f"No chat_generator state files found in {debug_path}"


def test_tool_breakpoint_in_pipeline_agent(pipeline_with_agent):
    with tempfile.TemporaryDirectory() as debug_path:
        agent_tool_breakpoint = ToolBreakpoint("tool_invoker", 0, tool_name="add_database_tool", debug_path=debug_path)
        agent_breakpoints = AgentBreakpoint(break_point=agent_tool_breakpoint, agent_name="database_agent")
        try:
            pipeline_with_agent.run(
                data={"fetcher": {"urls": ["https://en.wikipedia.org/wiki/Deepset"]}}, break_point=agent_breakpoints
            )
            assert False, "Expected exception was not raised"
        except BreakpointException as e:  # this is the exception from the Agent
            assert e.component == "tool_invoker"
            assert e.state is not None
            assert "messages" in e.state
            assert e.results is not None

        # verify that debug/state file was created
        tool_invoker_state_files = list(Path(debug_path).glob("database_agent_tool_invoker_*.json"))
        assert len(tool_invoker_state_files) > 0, f"No tool_invoker state files found in {debug_path}"


def test_agent_breakpoint_chat_generator_and_resume_pipeline(pipeline_with_agent):
    with tempfile.TemporaryDirectory() as debug_path:
        agent_generator_breakpoint = Breakpoint("chat_generator", 0, debug_path=debug_path)
        agent_breakpoints = AgentBreakpoint(break_point=agent_generator_breakpoint, agent_name="database_agent")
        try:
            pipeline_with_agent.run(
                data={"fetcher": {"urls": ["https://en.wikipedia.org/wiki/Deepset"]}}, break_point=agent_breakpoints
            )
            assert False, "Expected PipelineBreakpointException was not raised"

        except BreakpointException as e:
            assert e.component == "chat_generator"
            assert e.state is not None
            assert "messages" in e.state
            assert e.results is not None

        # verify that the state file was created
        chat_generator_state_files = list(Path(debug_path).glob("database_agent_chat_generator_*.json"))
        assert len(chat_generator_state_files) > 0, f"No chat_generator state files found in {debug_path}"

        # resume the pipeline from the saved state
        latest_state_file = max(chat_generator_state_files, key=os.path.getctime)
        result = pipeline_with_agent.run(data={}, pipeline_snapshot=load_state(latest_state_file))

        # pipeline completed successfully after resuming
        assert "database_agent" in result
        assert "messages" in result["database_agent"]
        assert len(result["database_agent"]["messages"]) > 0

        # final message contains the expected summary
        final_message = result["database_agent"]["messages"][-1].text
        assert "Malte Pietsch" in final_message
        assert "Milos Rusic" in final_message
        assert "Chief Executive Officer" in final_message
        assert "Chief Technology Officer" in final_message

        # tool should have been called during the resumed execution
        documents = document_store.filter_documents()
        assert len(documents) >= 2, "Expected at least 2 documents to be added to the database"

        # both people were added
        person_names = [doc.content for doc in documents]
        assert any("Malte Pietsch" in name for name in person_names)
        assert any("Milos Rusic" in name for name in person_names)


def test_agent_breakpoint_tool_and_resume_pipeline(pipeline_with_agent):
    with tempfile.TemporaryDirectory() as debug_path:
        agent_tool_breakpoint = ToolBreakpoint("tool_invoker", 0, tool_name="add_database_tool", debug_path=debug_path)
        agent_breakpoints = AgentBreakpoint(break_point=agent_tool_breakpoint, agent_name="database_agent")
        try:
            pipeline_with_agent.run(
                data={"fetcher": {"urls": ["https://en.wikipedia.org/wiki/Deepset"]}}, break_point=agent_breakpoints
            )
            assert False, "Expected PipelineBreakpointException was not raised"

        except BreakpointException as e:
            assert e.component == "tool_invoker"
            assert e.state is not None
            assert "messages" in e.state
            assert e.results is not None

        # verify that the state file was created
        tool_invoker_state_files = list(Path(debug_path).glob("database_agent_tool_invoker_*.json"))
        assert len(tool_invoker_state_files) > 0, f"No tool_invoker state files found in {debug_path}"

        # resume the pipeline from the saved state
        latest_state_file = max(tool_invoker_state_files, key=os.path.getctime)
        result = pipeline_with_agent.run(data={}, pipeline_snapshot=load_state(latest_state_file))

        # pipeline completed successfully after resuming
        assert "database_agent" in result
        assert "messages" in result["database_agent"]
        assert len(result["database_agent"]["messages"]) > 0

        # final message contains the expected summary
        final_message = result["database_agent"]["messages"][-1].text
        assert "Malte Pietsch" in final_message
        assert "Milos Rusic" in final_message
        assert "Chief Executive Officer" in final_message
        assert "Chief Technology Officer" in final_message

        # tool should have been called during the resumed execution
        documents = document_store.filter_documents()
        assert len(documents) >= 2, "Expected at least 2 documents to be added to the database"

        # both people were added
        person_names = [doc.content for doc in documents]
        assert any("Malte Pietsch" in name for name in person_names)
        assert any("Milos Rusic" in name for name in person_names)
