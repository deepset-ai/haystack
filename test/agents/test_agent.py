import logging
import os
from typing import Tuple

import pytest

from haystack import BaseComponent, Answer, Document
from haystack.agents import Agent
from haystack.agents.base import Tool
from haystack.errors import AgentError
from haystack.nodes import PromptModel, PromptNode, PromptTemplate
from haystack.pipelines import ExtractiveQAPipeline, DocumentSearchPipeline, BaseStandardPipeline
from test.conftest import MockRetriever, MockPromptNode


@pytest.mark.unit
def test_add_and_overwrite_tool():
    # Add a Node as a Tool to an Agent
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
    assert agent.has_tool(tool_name="Retriever")
    assert isinstance(agent.tools["Retriever"].pipeline_or_node, BaseComponent)

    agent.add_tool(
        Tool(
            name="Retriever",
            pipeline_or_node=retriever,
            description="useful for when you need to retrieve documents from your index",
        )
    )

    # Add a Pipeline as a Tool to an Agent and overwrite the previously added Tool
    retriever_pipeline = DocumentSearchPipeline(MockRetriever())
    agent.add_tool(
        Tool(
            name="Retriever",
            pipeline_or_node=retriever_pipeline,
            description="useful for when you need to retrieve documents from your index",
        )
    )
    assert len(agent.tools) == 1
    assert "Retriever" in agent.tools
    assert agent.has_tool(tool_name="Retriever")
    assert isinstance(agent.tools["Retriever"].pipeline_or_node, BaseStandardPipeline)


@pytest.mark.unit
def test_agent_chooses_no_action():
    agent = Agent(prompt_node=MockPromptNode())
    retriever = MockRetriever()
    agent.add_tool(
        Tool(
            name="Retriever",
            pipeline_or_node=retriever,
            description="useful for when you need to retrieve documents from your index",
        )
    )
    with pytest.raises(
        AgentError, match=r"Could not identify the next tool or input for that tool from Agent's output.*"
    ):
        agent.run("How many letters does the name of the town where Christelle lives have?")


@pytest.mark.unit
def test_max_iterations(caplog, monkeypatch):
    # Run an Agent and stop because max_iterations is reached
    agent = Agent(prompt_node=MockPromptNode(), max_iterations=3)
    retriever = MockRetriever()
    agent.add_tool(
        Tool(
            name="Retriever",
            pipeline_or_node=retriever,
            description="useful for when you need to retrieve documents from your index",
            output_variable="documents",
        )
    )

    # Let the Agent always choose "Retriever" as the Tool with "" as input
    def mock_extract_tool_name_and_tool_input(self, pred: str) -> Tuple[str, str]:
        return "Retriever", ""

    monkeypatch.setattr(Agent, "_extract_tool_name_and_tool_input", mock_extract_tool_name_and_tool_input)

    # Using max_iterations as specified in the Agent's init method
    with caplog.at_level(logging.WARN, logger="haystack.agents"):
        result = agent.run("Where does Christelle live?")
    assert result["answers"] == [Answer(answer="", type="generative")]
    assert "Maximum number of iterations (3) reached" in caplog.text

    # Setting max_iterations in the Agent's run method
    with caplog.at_level(logging.WARN, logger="haystack.agents"):
        result = agent.run("Where does Christelle live?", max_iterations=2)
    assert result["answers"] == [Answer(answer="", type="generative")]
    assert "Maximum number of iterations (2) reached" in caplog.text


@pytest.mark.unit
def test_run_tool():
    agent = Agent(prompt_node=MockPromptNode())
    retriever = MockRetriever()
    agent.add_tool(
        Tool(
            name="Retriever",
            pipeline_or_node=retriever,
            description="useful for when you need to retrieve documents from your index",
            output_variable="documents",
        )
    )
    result = agent._run_tool(tool_name="Retriever", tool_input="", transcript="")
    assert result == "[]"  # empty list of documents


@pytest.mark.unit
def test_extract_tool_name_and_tool_input():
    agent = Agent(prompt_node=MockPromptNode())

    pred = "need to find out what city he was born.\nTool: Search\nTool Input: Where was Jeremy McKinnon born"
    tool_name, tool_input = agent._extract_tool_name_and_tool_input(pred)
    assert tool_name == "Search" and tool_input == "Where was Jeremy McKinnon born"


@pytest.mark.unit
def test_extract_final_answer():
    agent = Agent(prompt_node=MockPromptNode())

    pred = "have the final answer to the question.\nFinal Answer: Florida"
    final_answer = agent._extract_final_answer(pred)
    assert final_answer == "Florida"


@pytest.mark.unit
def test_format_answer():
    agent = Agent(prompt_node=MockPromptNode())
    formatted_answer = agent._format_answer(query="query", answer="answer", transcript="transcript")
    assert formatted_answer["query"] == "query"
    assert formatted_answer["answers"] == [Answer(answer="answer", type="generative")]
    assert formatted_answer["transcript"] == "transcript"


@pytest.mark.integration
@pytest.mark.parametrize("reader", ["farm"], indirect=True)
@pytest.mark.parametrize("retriever_with_docs, document_store_with_docs", [("bm25", "memory")], indirect=True)
def test_tool_result_extraction(reader, retriever_with_docs):
    # Test that the result of a Tool is correctly extracted as a string

    # Pipeline as a Tool
    search = ExtractiveQAPipeline(reader, retriever_with_docs)
    t = Tool(
        name="Search",
        pipeline_or_node=search,
        description="useful for when you need to answer "
        "questions about where people live. You "
        "should ask targeted questions",
        output_variable="answers",
    )
    result = t.run("Where does Christelle live?")
    assert isinstance(result, str)
    assert result == "Paris" or result == "Madrid"

    # PromptNode as a Tool
    pt = PromptTemplate("test", "Here is a question: $query, Answer:")
    pn = PromptNode(default_prompt_template=pt)

    t = Tool(name="Search", pipeline_or_node=pn, description="N/A", output_variable="results")
    result = t.run(tool_input="What is the capital of Germany?")
    assert isinstance(result, str)
    assert "berlin" in result.lower()

    # Retriever as a Tool
    t = Tool(
        name="Retriever",
        pipeline_or_node=retriever_with_docs,
        description="useful for when you need to retrieve documents from your index",
        output_variable="documents",
    )
    result = t.run(tool_input="Where does Christelle live?")
    assert isinstance(result, str)
    assert "Christelle" in result


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
    counter = PromptNode(
        model_name_or_path=prompt_model,
        default_prompt_template=PromptTemplate(
            name="calculator_template",
            prompt_text="When I give you a word, respond with the number of characters that this word contains.\n"
            "Word: Rome\nLength: 4\n"
            "Word: Arles\nLength: 5\n"
            "Word: Berlin\nLength: 6\n"
            "Word: $query?\nLength: ",
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
            output_variable="answers",
        )
    )
    agent.add_tool(
        Tool(
            name="Count",
            pipeline_or_node=counter,
            description="useful for when you need to count how many characters are in a word. Ask only with a single word.",
        )
    )

    # TODO Replace Count tool once more tools are implemented so that we do not need to account for off-by-one errors
    result = agent.run("How many characters are in the word Madrid?")
    assert any(digit in result["answers"][0].answer for digit in ["5", "6", "five", "six"])

    result = agent.run("How many letters does the name of the town where Christelle lives have?")
    assert any(digit in result["answers"][0].answer for digit in ["5", "6", "five", "six"])


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
    counter = PromptNode(
        model_name_or_path=prompt_model,
        default_prompt_template=PromptTemplate(
            name="calculator_template",
            prompt_text="When I give you a word, respond with the number of characters that this word contains.\n"
            "Word: Rome\nLength: 4\n"
            "Word: Arles\nLength: 5\n"
            "Word: Berlin\nLength: 6\n"
            "Word: $query?\nLength: ",
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
            output_variable="answers",
        )
    )
    agent.add_tool(
        Tool(
            name="Count",
            pipeline_or_node=counter,
            description="useful for when you need to count how many characters are in a word. Ask only with a single word.",
        )
    )

    results = agent.run_batch(
        queries=[
            "How many characters are in the word Madrid?",
            "How many letters does the name of the town where Christelle lives have?",
        ]
    )
    # TODO Replace Count tool once more tools are implemented so that we do not need to account for off-by-one errors
    assert any(digit in results["answers"][0][0].answer for digit in ["5", "6", "five", "six"])
    assert any(digit in results["answers"][1][0].answer for digit in ["5", "6", "five", "six"])
