from enum import Enum
from typing import Any, Dict, List, Optional

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


class AgentToolLogger:
    def __init__(self, agent_events: Events, tool_events: Events):
        agent_events.on_agent_start += self.on_agent_start
        tool_events.on_tool_finish += self.on_tool_finish
        self.logs: List[Dict[str, Any]] = []

    def on_agent_start(self, **kwargs: Any) -> None:
        self.logs = []

    def on_tool_finish(
        self, tool_result: str, tool_name: Optional[str] = None, tool_input: Optional[str] = None, **kwargs: Any
    ) -> None:
        self.logs.append({"tool_name": tool_name, "tool_input": tool_input, "tool_output": tool_result})

    def __repr__(self):
        return f"<DefaultToolLogger with {len(self.logs)} tool log(s)>"
