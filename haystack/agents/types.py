from enum import Enum

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
