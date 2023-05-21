from haystack.agents.agent_step import AgentStep
from events import Events
from typing import Any, Optional

from haystack.agents.types import Color

class ToolResultsGatherer:
    results = []

    def __init__(self, agent_callback_manager: Events):
        agent_callback_manager.on_agent_start += self.on_agent_start
        agent_callback_manager.on_agent_finish += self.on_agent_finish

        agent_callback_manager.on_tool_finish += self.on_tool_finish

    def on_agent_start(self, **kwargs: Any) -> None:
        self.results = []

    def on_tool_finish(self,
            tool_result: str,
            color: Optional[Color] = None,
            observation_prefix: Optional[str] = None,
            llm_prefix: Optional[str] = None,
            tool_name: Optional[str] = None,
            unprocessed_tool_result: Any = None,
            **kwargs: Any) -> None:
        self.results.append({
            "name": tool_name,
            "result": tool_result,
            "unprocessed_result": unprocessed_tool_result
        })

    def on_agent_finish(self, agent_step: AgentStep) -> None:
        agent_step.tool_results = self.results