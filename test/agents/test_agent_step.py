import pytest

from haystack.agents import AgentStep
from haystack.agents.answer_parser import AgentAnswerParser, BasicAnswerParser


class MockAnswerParser(AgentAnswerParser):
    def can_parse(self, prompt_node_response: str) -> bool:
        return "Final Answer:" in prompt_node_response

    def parse(self, prompt_node_response: str) -> str:
        return prompt_node_response.split("Final Answer:")[1].strip()


@pytest.fixture
def agent_step():
    return AgentStep()


@pytest.mark.unit
def test_agent_step_init(agent_step):
    assert agent_step.current_step == 1
    assert agent_step.max_steps == 10
    assert isinstance(agent_step.final_answer_parser, BasicAnswerParser)
    assert agent_step.prompt_node_response == ""
    assert agent_step.transcript == ""


@pytest.mark.unit
def test_create_next_step(agent_step):
    prompt_node_response = ["Test response"]
    next_step = agent_step.create_next_step(prompt_node_response)
    assert next_step.current_step == 2
    assert next_step.prompt_node_response == "Test response"
    assert next_step.transcript == ""


@pytest.mark.unit
def test_final_answer(agent_step):
    query = "Test query"
    agent_step.final_answer_parser = MockAnswerParser()
    agent_step.prompt_node_response = "Final Answer: Test final answer"
    answer = agent_step.final_answer(query)
    assert answer["query"] == query
    assert len(answer["answers"]) == 1
    assert answer["answers"][0].answer == "Test final answer"


@pytest.mark.unit
def test_is_last(agent_step):
    agent_step.final_answer_parser = MockAnswerParser()
    agent_step.prompt_node_response = "Final Answer: Test final answer"
    assert agent_step.is_last()

    agent_step.current_step = 11
    assert agent_step.is_last()

    agent_step.current_step = 1
    agent_step.prompt_node_response = "Non-final response"
    assert not agent_step.is_last()

    agent_step.current_step = 1
    assert not agent_step.is_last()


@pytest.mark.unit
def test_completed(agent_step):
    agent_step.prompt_node_response = "Test response"
    agent_step.completed("Test observation")
    assert agent_step.transcript == "Test response\nObservation: Test observation\nThought:"
