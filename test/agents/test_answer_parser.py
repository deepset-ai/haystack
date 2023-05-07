import pytest
from haystack.agents.answer_parser import AgentAnswerParser, RegexAnswerParser

final_answer_pattern = r"Final Answer\s*:\s*(.*)"


def test_agent_answer_parser_abstract_base_class():
    with pytest.raises(TypeError):
        _ = AgentAnswerParser()


@pytest.mark.parametrize(
    "input_str, expected",
    [("Hello, my name is John", True), ("This is a test", True), ("", False), (" ", True), (123, False), (None, False)],
)
def test_basic_answer_parser_can_parse(input_str, expected):
    parser = RegexAnswerParser()
    assert parser.can_parse(input_str) == expected


@pytest.mark.parametrize(
    "input_str, expected",
    [
        ("Hello, my name is John", "Hello, my name is John"),
        ("This is a test", "This is a test"),
        ("", ""),
        (123, ""),
        (None, ""),
        (" ", ""),
    ],
)
def test_answer_parser_parse_any_string(input_str, expected):
    parser = RegexAnswerParser()
    assert parser.parse(input_str) == expected


@pytest.mark.parametrize(
    "input_str, pattern, expected",
    [
        ("Final Answer: 42 is the answer", final_answer_pattern, True),
        ("Final Answer:  1234", final_answer_pattern, True),
        ("Final Answer:  Answer", final_answer_pattern, True),
        ("Final Answer:  This list: one and two and three", final_answer_pattern, True),
        ("Final Answer:42", final_answer_pattern, True),
        ("Final Answer:   ", final_answer_pattern, True),
        ("Final Answer:    The answer is 99    ", final_answer_pattern, True),
        ("Final answer: 42 is the answer", final_answer_pattern, False),
        ("Final Answer", final_answer_pattern, False),
        ("The final answer is: 100", final_answer_pattern, False),
    ],
)
def test_final_answer_regex_can_parse(input_str, pattern, expected):
    parser = RegexAnswerParser(pattern)
    assert parser.can_parse(input_str) == expected


@pytest.mark.parametrize(
    "input_str, pattern, expected",
    [
        ("Final Answer: 42 is the answer", final_answer_pattern, "42 is the answer"),
        ("Final Answer:  1234", final_answer_pattern, "1234"),
        ("Final Answer:  Answer", final_answer_pattern, "Answer"),
        ("Final Answer:  This list: one and two and three", final_answer_pattern, "This list: one and two and three"),
        ("Final Answer:42", final_answer_pattern, "42"),
        ("Final Answer:   ", final_answer_pattern, ""),
        ("Final Answer:    The answer is 99    ", final_answer_pattern, "The answer is 99"),
        ("Final answer: 42 is the answer", final_answer_pattern, ""),
        ("Final Answer", final_answer_pattern, ""),
        ("The final answer is: 100", final_answer_pattern, ""),
    ],
)
def test_final_answer_regex_parse(input_str, pattern, expected):
    parser = RegexAnswerParser(pattern)
    assert parser.parse(input_str) == expected
