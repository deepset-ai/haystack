from typing import Optional, Dict, Any

from haystack.agents.types import Color
from haystack.agents import Agent
from haystack.agents.agent_step import AgentStep


def print_text(text: str, end="", color: Optional[Color] = None) -> None:
    """
    Print text with optional color.
    :param text: Text to print.
    :param end: End character to use (defaults to "").
    :param color: Color to print text in (defaults to None).
    """
    if color:
        print(f"{color.value}{text}{Color.RESET.value}", end=end, flush=True)
    else:
        print(text, end=end, flush=True)


def react_parameter_resolver(query: str, agent: Agent, agent_step: AgentStep, **kwargs) -> Dict[str, Any]:
    """
    A parameter resolver for ReAct based agents that returns the query, the tool names, the tool names
    with descriptions, and the transcript (internal monologue).
    """
    return {
        "query": query,
        "tool_names": agent.tm.get_tool_names(),
        "tool_names_with_descriptions": agent.tm.get_tool_names_with_descriptions(),
        "transcript": agent_step.transcript,
    }


def agent_without_tools_parameter_resolver(query: str, agent: Agent, **kwargs) -> Dict[str, Any]:
    """
    A parameter resolver for ReAct based agents without tools that returns the query, the history.
    """
    return {"query": query, "history": agent.memory.load()}
