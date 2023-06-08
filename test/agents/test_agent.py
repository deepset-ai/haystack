import logging
import os
import re
from typing import Tuple
from unittest.mock import patch

from test.conftest import MockRetriever, MockPromptNode
from unittest import mock
import pytest

from haystack import BaseComponent, Answer, Document
from haystack.agents import Agent, AgentStep
from haystack.agents.base import Tool, ToolsManager
from haystack.nodes import PromptModel, PromptNode, PromptTemplate
from haystack.pipelines import ExtractiveQAPipeline, DocumentSearchPipeline, BaseStandardPipeline


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
    assert len(agent.tm.tools) == 1
    assert agent.has_tool(tool_name="Retriever")
    assert isinstance(agent.tm.tools["Retriever"].pipeline_or_node, BaseComponent)

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
    assert len(agent.tm.tools) == 1
    assert agent.has_tool(tool_name="Retriever")
    assert isinstance(agent.tm.tools["Retriever"].pipeline_or_node, BaseStandardPipeline)


@pytest.mark.unit
def test_max_steps(caplog, monkeypatch):
    # Run an Agent and stop because max_steps is reached
    agent = Agent(prompt_node=MockPromptNode(), max_steps=3)
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

    monkeypatch.setattr(ToolsManager, "extract_tool_name_and_tool_input", mock_extract_tool_name_and_tool_input)

    # Using max_steps as specified in the Agent's init method
    with caplog.at_level(logging.WARN, logger="haystack.agents"):
        result = agent.run("Where does Christelle live?")
    assert result["answers"] == [Answer(answer="", type="generative")]
    assert "maximum number of iterations (3)" in caplog.text.lower()

    # Setting max_steps in the Agent's run method
    with caplog.at_level(logging.WARN, logger="haystack.agents"):
        result = agent.run("Where does Christelle live?", max_steps=2)
    assert result["answers"] == [Answer(answer="", type="generative")]
    assert "maximum number of iterations (2)" in caplog.text.lower()


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
    pn_response = "need to find out what city he was born.\nTool: Retriever\nTool Input: Where was Jeremy McKinnon born"

    step = AgentStep(prompt_node_response=pn_response)
    result = agent.tm.run_tool(step.prompt_node_response)
    assert result == "[]"  # empty list of documents


@pytest.mark.unit
def test_extract_final_answer():
    match_examples = [
        "have the final answer to the question.\nFinal Answer: Florida",
        "Final Answer: 42 is the answer",
        "Final Answer:  1234",
        "Final Answer:  Answer",
        "Final Answer:  This list: one and two and three",
        "Final Answer:42",
        "Final Answer:   ",
        "Final Answer:    The answer is 99    ",
    ]
    expected_answers = [
        "Florida",
        "42 is the answer",
        "1234",
        "Answer",
        "This list: one and two and three",
        "42",
        "",
        "The answer is 99",
    ]

    for example, expected_answer in zip(match_examples, expected_answers):
        agent_step = AgentStep(prompt_node_response=example, final_answer_pattern=r"Final Answer\s*:\s*(.*)")
        final_answer = agent_step.final_answer(query="irrelevant")
        assert final_answer["answers"][0].answer == expected_answer


@pytest.mark.unit
def test_final_answer_regex():
    match_examples = [
        "Final Answer: 42 is the answer",
        "Final Answer:  1234",
        "Final Answer:  Answer",
        "Final Answer:  This list: one and two and three",
        "Final Answer:42",
        "Final Answer:   ",
        "Final Answer:    The answer is 99    ",
    ]

    non_match_examples = ["Final answer: 42 is the answer", "Final Answer", "The final answer is: 100"]
    final_answer_pattern = r"Final Answer\s*:\s*(.*)"
    for example in match_examples:
        match = re.match(final_answer_pattern, example)
        assert match is not None

    for example in non_match_examples:
        match = re.match(final_answer_pattern, example)
        assert match is None


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
    pt = PromptTemplate("Here is a question: {query}, Answer:")
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


@pytest.mark.skip("FIXME")
@pytest.mark.integration
@pytest.mark.parametrize("reader", ["farm"], indirect=True)
@pytest.mark.parametrize("retriever_with_docs, document_store_with_docs", [("bm25", "memory")], indirect=True)
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="Please export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
def test_agent_run(reader, retriever_with_docs, document_store_with_docs):
    search = ExtractiveQAPipeline(reader, retriever_with_docs)
    prompt_model = PromptModel(model_name_or_path="gpt-3.5-turbo", api_key=os.environ.get("OPENAI_API_KEY"))
    prompt_node = PromptNode(model_name_or_path=prompt_model, stop_words=["Observation:"])
    country_finder = PromptNode(
        model_name_or_path=prompt_model,
        default_prompt_template=PromptTemplate(
            "When I give you a name of the city, respond with the country where the city is located.\n"
            "City: Rome\nCountry: Italy\n"
            "City: Berlin\nCountry: Germany\n"
            "City: Belgrade\nCountry: Serbia\n"
            "City: {query}?\nCountry: "
        ),
    )

    agent = Agent(prompt_node=prompt_node, max_steps=12)
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
            name="CountryFinder",
            pipeline_or_node=country_finder,
            description="useful for when you need to find the country where a city is located",
        )
    )

    result = agent.run("Where is Madrid?")
    country = result["answers"][0].answer
    assert "spain" in country.lower()

    result = agent.run("In which country is the city where Christelle lives?")
    country = result["answers"][0].answer
    assert "france" in country.lower()


@pytest.mark.unit
def test_update_hash():
    agent = Agent(prompt_node=MockPromptNode(), prompt_template=mock.Mock())
    assert agent.hash == "d41d8cd98f00b204e9800998ecf8427e"
    agent.add_tool(
        Tool(
            name="Search",
            pipeline_or_node=mock.Mock(),
            description="useful for when you need to answer "
            "questions about where people live. You "
            "should ask targeted questions",
            output_variable="answers",
        )
    )
    assert agent.hash == "d41d8cd98f00b204e9800998ecf8427e"
    agent.add_tool(
        Tool(
            name="Count",
            pipeline_or_node=mock.Mock(),
            description="useful for when you need to count how many characters are in a word. Ask only with a single word.",
        )
    )
    assert agent.hash == "d41d8cd98f00b204e9800998ecf8427e"
    agent.update_hash()
    assert agent.hash == "5ac8eca2f92c9545adcce3682b80d4c5"


@pytest.mark.unit
def test_tool_fails_processing_dict_result():
    tool = Tool(name="name", pipeline_or_node=MockPromptNode(), description="description")
    with pytest.raises(ValueError):
        tool._process_result({"answer": "answer"})


@pytest.mark.unit
def test_tool_processes_answer_result_and_document_result():
    tool = Tool(name="name", pipeline_or_node=MockPromptNode(), description="description")
    assert tool._process_result(Answer(answer="answer")) == "answer"
    assert tool._process_result(Document(content="content")) == "content"


def test_invalid_agent_template():
    pn = PromptNode()
    with pytest.raises(ValueError, match="some_non_existing_template not supported"):
        Agent(prompt_node=pn, prompt_template="some_non_existing_template")

    # if prompt_template is None, then we'll use zero-shot-react
    a = Agent(prompt_node=pn, prompt_template=None)
    assert isinstance(a.prompt_template, PromptTemplate)
    assert a.prompt_template.name == "zero-shot-react"


@pytest.mark.unit
@patch.object(PromptNode, "prompt")
@patch("haystack.nodes.prompt.prompt_node.PromptModel")
def test_default_template_order(mock_model, mock_prompt):
    pn = PromptNode("abc")
    a = Agent(prompt_node=pn)
    assert a.prompt_template.name == "zero-shot-react"

    pn.default_prompt_template = "language-detection"
    a = Agent(prompt_node=pn)
    assert a.prompt_template.name == "language-detection"

    a = Agent(prompt_node=pn, prompt_template="translation")
    assert a.prompt_template.name == "translation"
