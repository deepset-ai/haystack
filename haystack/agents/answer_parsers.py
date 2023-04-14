import re
from abc import abstractmethod, ABC
from typing import Any


class AgentAnswerParser(ABC):
    """
    Abstract base class for parsing agent's answer.
    """

    @abstractmethod
    def can_parse(self, parser_input: Any) -> bool:
        """
        Check if the parser can parse the input.
        :param parser_input: The input to parse.
        :return: True if the parser can parse the input, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def parse(self, parser_input: Any) -> str:
        """
        Parse the input.
        :param parser_input: The input to parse.
        :return: The parsed input.
        """
        raise NotImplementedError


class RegexAnswerParser(AgentAnswerParser):
    """
    Parser that uses a regex to parse the agent's answer.
    """

    def __init__(self, final_answer_pattern: str):
        self.pattern = final_answer_pattern

    def can_parse(self, parser_input: Any) -> bool:
        if isinstance(parser_input, str):
            return bool(re.search(self.pattern, parser_input))
        return False

    def parse(self, parser_input: Any) -> str:
        if self.can_parse(parser_input):
            final_answer_match = re.search(self.pattern, parser_input)
            if final_answer_match:
                final_answer = final_answer_match.group(1)
                return final_answer.strip('" ')  # type: ignore
        return ""


class BasicAnswerParser(AgentAnswerParser):
    """
    Parser that returns the input if it is a non-empty string.
    """

    def can_parse(self, parser_input: Any) -> bool:
        return isinstance(parser_input, str) and parser_input

    def parse(self, parser_input: Any) -> str:
        if self.can_parse(parser_input):
            return parser_input
        return ""
