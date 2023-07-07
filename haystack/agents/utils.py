from typing import Optional, TYPE_CHECKING, Dict, Any

from haystack.agents.types import Color
from haystack.agents.agent_step import AgentStep

if TYPE_CHECKING:
    from haystack.agents import Agent


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


def react_parameter_resolver(query: str, agent: "Agent", agent_step: AgentStep, **kwargs) -> Dict[str, Any]:
    """
    A parameter resolver for ReAct-based agents that returns the query, the tool names, the tool names
    with descriptions, and the transcript (internal monologue).
    """
    return {
        "query": query,
        "tool_names": agent.tm.get_tool_names(),
        "tool_names_with_descriptions": agent.tm.get_tool_names_with_descriptions(),
        "transcript": agent_step.transcript,
    }


def agent_without_tools_parameter_resolver(query: str, agent: "Agent", **kwargs) -> Dict[str, Any]:
    """
    A parameter resolver for simple chat agents without tools that returns the query and the history.
    """
    return {"query": query, "memory": agent.memory.load()}


def conversational_agent_parameter_resolver(
    query: str, agent: "Agent", agent_step: AgentStep, **kwargs
) -> Dict[str, Any]:
    """
    A parameter resolver for ReAct-based conversational agent that returns the query, the tool names, the tool names
    with descriptions, the history of the conversation, and the transcript (internal monologue).
    """
    return {
        "query": query,
        "tool_names": agent.tm.get_tool_names(),
        "tool_names_with_descriptions": agent.tm.get_tool_names_with_descriptions(),
        "transcript": agent_step.transcript,
        "memory": agent.memory.load(),
    }
