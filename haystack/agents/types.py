from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any

from events import Events
from haystack.nodes.prompt.invocation_layer.handlers import TokenStreamingHandler


class Color(Enum):
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    RESET = "\x1b[0m"


class AgentTokenStreamingHandler(TokenStreamingHandler):
    def __init__(self, events: Events):
        self.events = events

    def __call__(self, token_received, **kwargs) -> str:
        self.events.on_new_token(token_received, **kwargs)
        return token_received


class PromptParametersResolver(ABC):
    """
    Abstract base class for resolving parameters of Agent's PromptTemplate. During Agent's step planning stage, Agent
    invokes implementations of this class to resolve parameters of its PromptTemplate. In addition to query parameter,
    the resolver is given access to Agent runtime via **kwargs.

    See various implementations for more details and examples.
    """

    @abstractmethod
    def resolve(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Resolve parameters of PromptTemplate.
        :param query: The query to resolve parameters for.
        :param kwargs: Additional parameters to resolve parameters for.
        :return: A dictionary containing resolved parameters.
        """
        raise NotImplementedError
