from pathlib import Path
from typing import Optional

from haystack.components.agents import Agent
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.components.converters.html import HTMLToDocument
from haystack.components.fetchers.link_content import LinkContentFetcher
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.core.pipeline import Pipeline
from haystack.core.pipeline.pipeline import PersistenceSaving
from haystack.dataclasses import ChatMessage, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.tools import tool

document_store = InMemoryDocumentStore()  # create a document store or an SQL database


def create_agent_pipeline():
    @tool
    def add_database_tool(name: str, surname: str, job_title: Optional[str], other: Optional[str]):
        """Use this tool to add names to the database with information about them"""
        document_store.write_documents(
            [Document(content=name + " " + surname + " " + (job_title or ""), meta={"other": other})]
        )

    database_assistant = Agent(
        chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
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

    chat_prompt_builder = ChatPromptBuilder(
        template=[
            ChatMessage.from_user("""
        {% for doc in docs %}
        {{ doc.content|default|truncate(25000) }}
        {% endfor %}
        """)
        ],
        required_variables=["docs"],
    )

    extraction_agent = Pipeline()
    extraction_agent.add_component("fetcher", LinkContentFetcher())
    extraction_agent.add_component("converter", HTMLToDocument())
    extraction_agent.add_component("builder", chat_prompt_builder)

    extraction_agent.add_component("database_agent", database_assistant)
    extraction_agent.connect("fetcher.streams", "converter.sources")
    extraction_agent.connect("converter.documents", "builder.docs")
    extraction_agent.connect("builder", "database_agent")
    return extraction_agent


def main():
    snapshots_dir = "snapshots_agent_single_tool"
    agent = create_agent_pipeline()
    agent_output = agent.run(
        data={"fetcher": {"urls": ["https://en.wikipedia.org/wiki/Deepset"]}},
        state_persistence=PersistenceSaving.FULL,
        state_persistence_path=snapshots_dir,
    )
    print(agent_output["database_agent"]["messages"][-1].text)

    snapshot_files = list(Path(snapshots_dir).glob("*.json"))
    print(f"\nSnapshot files created: {len(snapshot_files)}")

    # resume from each snapshot and print details
    for snapshot_file in snapshot_files:
        print(f"  - {snapshot_file.name}")
        from haystack.core.pipeline.breakpoint import load_pipeline_snapshot

        try:
            snapshot = load_pipeline_snapshot(snapshot_file)
            print(f"    Component: {snapshot.break_point.component_name}")
            print(f"    Visit count: {snapshot.break_point.visit_count}")
            resumed_results = agent.run(data={}, pipeline_snapshot=snapshot)
            print(f"    Resumed results: {resumed_results}")
        except Exception as e:
            print(f"    Error loading snapshot: {e}")


if __name__ == "__main__":
    main()
