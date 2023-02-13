import logging
import os
from typing import Tuple

import pytest

from haystack import BaseComponent
from haystack.agents import Agent
from haystack.agents.base import Tool
from haystack.errors import AgentError
from haystack.nodes import PromptModel, PromptNode, PromptTemplate
from haystack.pipelines import ExtractiveQAPipeline, DocumentSearchPipeline, BaseStandardPipeline
from test.conftest import MockRetriever, MockPromptNode


def test_add_and_overwrite_tool():
    # Add a node as a tool to an agent
    agent = Agent(prompt_node=MockPromptNode())
    retriever = MockRetriever()
    agent.add_tool(
        Tool(
            name="Retriever",
            pipeline_or_node=retriever,
            description="useful for when you need to " "retrieve documents from your index",
        )
    )
    assert len(agent.tools) == 1
    assert "Retriever" in agent.tools
    assert isinstance(agent.tools["Retriever"].pipeline_or_node, BaseComponent)

    agent.add_tool(
        Tool(
            name="Retriever",
            pipeline_or_node=retriever,
            description="useful for when you need to " "retrieve documents from your index",
        )
    )

    # Add a pipeline as a tool to an agent and overwrite the previously added tool
    retriever_pipeline = DocumentSearchPipeline(MockRetriever())
    agent.add_tool(
        Tool(
            name="Retriever",
            pipeline_or_node=retriever_pipeline,
            description="useful for when you need " "to retrieve documents " "from your index",
        )
    )
    assert len(agent.tools) == 1
    assert "Retriever" in agent.tools
    assert isinstance(agent.tools["Retriever"].pipeline_or_node, BaseStandardPipeline)


def test_agent_chooses_no_action():
    agent = Agent(prompt_node=MockPromptNode())
    retriever = MockRetriever()
    agent.add_tool(
        Tool(
            name="Retriever",
            pipeline_or_node=retriever,
            description="useful for when you need to " "retrieve documents from your index",
        )
    )
    with pytest.raises(AgentError, match=r"Wrong output format.*"):
        agent.run("How many letters does the name of the town where Christelle lives have?")


def test_max_iterations(caplog, monkeypatch):
    # Run agent and stop because max_iterations is reached
    agent = Agent(prompt_node=MockPromptNode(), max_iterations=1)
    retriever = MockRetriever()
    agent.add_tool(
        Tool(
            name="Retriever",
            pipeline_or_node=retriever,
            description="useful for when you need to " "retrieve documents from your index",
        )
    )

    def mock_parse_tool_and_tool_input(self, pred: str) -> Tuple[str, str]:
        return "Retriever", ""

    monkeypatch.setattr(Agent, "parse_tool_name_and_tool_input", mock_parse_tool_and_tool_input)
    with caplog.at_level(logging.WARN, logger="haystack.agents"):
        agent.run("Where does Christelle live?")
    assert "Maximum number of agent iterations (1) reached" in caplog.text


@pytest.mark.integration
@pytest.mark.parametrize("reader", ["farm"], indirect=True)
@pytest.mark.parametrize("retriever_with_docs, document_store_with_docs", [("bm25", "memory")], indirect=True)
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="Please export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
def test_agent_run(reader, retriever_with_docs, document_store_with_docs):
    search = ExtractiveQAPipeline(reader, retriever_with_docs)
    prompt_model = PromptModel(model_name_or_path="text-davinci-003", api_key=os.environ.get("OPENAI_API_KEY"))
    prompt_node = PromptNode(model_name_or_path=prompt_model, stop_words=["Observation:"])
    calculator = PromptNode(
        model_name_or_path=prompt_model,
        default_prompt_template=PromptTemplate(
            name="calculator_template",
            prompt_text="You can do calculations and answer questions about math. "
            "When you do calculations, you return the question and the "
            "final result. $query?",
            prompt_params=["query"],
        ),
    )

    agent = Agent(prompt_node=prompt_node)
    agent.add_tool(
        Tool(
            name="Search",
            pipeline_or_node=search,
            description="useful for when you need to answer "
            "questions about where people live. You "
            "should ask targeted questions",
        )
    )
    agent.add_tool(
        Tool(
            name="Calculator",
            pipeline_or_node=calculator,
            description="useful for when you need to " "answer questions about math",
        )
    )

    result = agent.run("What is 2 to the power of 3?")
    assert "8" in result["answers"][0].answer or "eight" in result["answers"][0].answer

    result = agent.run("How many letters does the name of the town where Christelle lives have?")
    assert "5" in result["answers"][0].answer or "five" in result["answers"][0].answer


@pytest.mark.integration
@pytest.mark.parametrize("reader", ["farm"], indirect=True)
@pytest.mark.parametrize("retriever_with_docs, document_store_with_docs", [("bm25", "memory")], indirect=True)
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="Please export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
def test_agent_run_batch(reader, retriever_with_docs, document_store_with_docs):
    search = ExtractiveQAPipeline(reader, retriever_with_docs)
    prompt_model = PromptModel(model_name_or_path="text-davinci-003", api_key=os.environ.get("OPENAI_API_KEY"))
    prompt_node = PromptNode(model_name_or_path=prompt_model, stop_words=["Observation:"])
    calculator = PromptNode(
        model_name_or_path=prompt_model,
        default_prompt_template=PromptTemplate(
            name="calculator_template",
            prompt_text="You can do calculations and answer questions about math. "
            "When you do calculations, you return the question and the "
            "final result. $query?",
            prompt_params=["query"],
        ),
    )

    agent = Agent(prompt_node=prompt_node)
    agent.add_tool(
        Tool(
            name="Search",
            pipeline_or_node=search,
            description="useful for when you need to answer "
            "questions about where people live. You "
            "should ask targeted questions",
        )
    )
    agent.add_tool(
        Tool(
            name="Calculator",
            pipeline_or_node=calculator,
            description="useful for when you need to " "answer questions about math",
        )
    )

    results = agent.run_batch(
        queries=[
            "What is 2 to the power of 3?",
            "How many letters does the name of the town where Christelle lives have?",
        ]
    )
    assert "8" in results["answers"][0][0].answer or "eight" in results["answers"][0][0].answer
    assert "5" in results["answers"][1][0].answer or "five" in results["answers"][1][0].answer
