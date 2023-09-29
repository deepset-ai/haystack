import pytest

from haystack import Answer
from haystack.agents import AgentStep
from haystack.errors import AgentError


@pytest.fixture
def agent_step():
    return AgentStep(
        current_step=1, max_steps=10, final_answer_pattern=None, prompt_node_response="Hello", transcript="Hello"
    )


@pytest.mark.unit
def test_create_next_step(agent_step):
    # Test normal case
    next_step = agent_step.create_next_step(["Hello again"])
    assert next_step.current_step == 2
    assert next_step.prompt_node_response == "Hello again"
    assert next_step.transcript == "Hello"

    # Test with invalid prompt_node_response
    with pytest.raises(AgentError):
        agent_step.create_next_step({})

    # Test with empty prompt_node_response
    with pytest.raises(AgentError):
        agent_step.create_next_step([])


@pytest.mark.unit
def test_final_answer(agent_step):
    # Test normal case
    result = agent_step.final_answer("query")
    assert result["query"] == "query"
    assert isinstance(result["answers"][0], Answer)
    assert result["answers"][0].answer == "Hello"
    assert result["answers"][0].type == "generative"
    assert result["transcript"] == "Hello"

    # Test with max_steps reached
    agent_step.current_step = 11
    result = agent_step.final_answer("query")
    assert result["answers"][0].answer == ""


@pytest.mark.unit
def test_is_last():
    # Test is last, and it is last because of valid prompt_node_response and default final_answer_pattern
    agent_step = AgentStep(current_step=1, max_steps=10, prompt_node_response="Hello", transcript="Hello")
    assert agent_step.is_last()

    # Test not last
    agent_step.current_step = 1
    agent_step.prompt_node_response = "final answer not satisfying pattern"
    agent_step.final_answer_pattern = r"Final Answer\s*:\s*(.*)"
    assert not agent_step.is_last()

    # Test border cases for max_steps
    agent_step.current_step = 9
    assert not agent_step.is_last()
    agent_step.current_step = 10
    assert not agent_step.is_last()

    # Test when last due to max_steps
    agent_step.current_step = 11
    assert agent_step.is_last()


@pytest.mark.unit
def test_completed(agent_step):
    # Test without observation
    agent_step.completed(None)
    assert agent_step.transcript == "HelloHello"

    # Test with observation, adds Hello from prompt_node_response
    agent_step.completed("observation")
    assert agent_step.transcript == "HelloHelloHello\nObservation: observation\nThought:"


@pytest.mark.unit
def test_repr(agent_step):
    assert repr(agent_step) == (
        "AgentStep(current_step=1, max_steps=10, "
        "prompt_node_response=Hello, final_answer_pattern=^([\\s\\S]+)$, "
        "transcript=Hello)"
    )


@pytest.mark.unit
def test_parse_final_answer(agent_step):
    # Test when pattern matches
    assert agent_step.parse_final_answer() == "Hello"

    # Test when pattern does not match
    agent_step.final_answer_pattern = "goodbye"
    assert agent_step.parse_final_answer() is None


@pytest.mark.unit
def test_format_react_answer():
    step = AgentStep(
        final_answer_pattern=r"Final Answer\s*:\s*(.*)",
        prompt_node_response="have the final answer to the question.\nFinal Answer: Florida",
    )
    formatted_answer = step.final_answer(query="query")
    assert formatted_answer["query"] == "query"
    assert formatted_answer["answers"] == [Answer(answer="Florida", type="generative")]
