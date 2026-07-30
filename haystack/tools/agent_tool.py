# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Callable
from typing import Any

from haystack.components.agents import Agent
from haystack.components.agents.agent import _EXIT_REASON_MAX_STEPS
from haystack.core.serialization import component_from_dict, import_class_by_name
from haystack.tools.component_tool import ComponentTool
from haystack.tools.tool import _deserialize_outputs_to_state, _deserialize_outputs_to_string


def _uncovered_agent_inputs(agent: Agent, inputs_from_state: dict[str, str] | None) -> list[str]:
    """
    Names of the mandatory Agent inputs that the caller has to supply, such as the variables of a templated prompt.

    :param agent: The Agent wrapped by the tool.
    :param inputs_from_state: The tool's `inputs_from_state`, whose targets are already covered by the calling Agent.
    :returns: The mandatory Agent inputs other than `messages` that are not mapped from the calling Agent's state.
    """
    covered = {"messages", *(inputs_from_state or {}).values()}
    return sorted(
        name
        for name, socket in agent.__haystack_input__._sockets_dict.items()  # type: ignore[attr-defined]
        if socket.is_mandatory and name not in covered
    )


def _build_parameters(uncovered: list[str]) -> dict[str, Any]:
    """
    Build the schema the calling LLM fills in: the task to delegate, plus one string per uncovered Agent input.

    :param uncovered: The Agent inputs returned by `_uncovered_agent_inputs`.
    :returns: A JSON schema for the Tool parameters.
    """
    extra = {name: {"type": "string"} for name in uncovered}
    return {
        "type": "object",
        "properties": {
            "messages": {
                "type": "array",
                "description": "Exactly one user message.",
                "minItems": 1,
                "maxItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string", "enum": ["user"]},
                        "content": {"type": "string", "description": "The task to delegate to this tool."},
                    },
                    "required": ["role", "content"],
                },
            },
            **extra,
        },
        "required": ["messages", *uncovered],
    }


def _agent_result_to_string(result: dict[str, Any]) -> str:
    """
    Default `outputs_to_string` handler: the text of the agent's final reply.

    :param result: The output of the Agent run.
    :returns: The text of the final reply, or the whole message if it has none, with a warning appended if the agent
        ran out of steps.
    """
    last_message = result["last_message"]
    text = last_message.text or json.dumps(last_message.to_dict())
    if result["exit_reason"] == _EXIT_REASON_MAX_STEPS:
        text += "\n\n[The agent reached max_agent_steps and stopped, so this result may be incomplete.]"
    return text


class AgentTool(ComponentTool):
    """
    A Tool that wraps an Agent, so that another Agent can delegate work to it.

    ## Usage Example:

    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack.tools import AgentTool, ComponentTool
    from haystack_integrations.components.websearch.serperdev import SerperDevWebSearch

    researcher = Agent(
        chat_generator=OpenAIChatGenerator(model="gpt-5.4-mini"),
        system_prompt="You are a research specialist. Investigate the task and report your findings.",
        tools=[
            ComponentTool(
                component=SerperDevWebSearch(
                    top_k=3,
                ),
                name="web_search",
                description="Search the web for current information on any topic",
            ),
        ],
    )

    research = AgentTool(
        researcher,
        name="research",
        description="You are a research specialist. Search the web to find information.",
    )

    coordinator = Agent(
        chat_generator=OpenAIChatGenerator(model="gpt-5.4"),
        tools=[research],
        system_prompt="You coordinate specialists. Delegate research questions, then answer the user.",
    )

    result = coordinator.run([ChatMessage.from_user("Who was Nikola Tesla?")])
    print(result["last_message"].text)
    ```
    """

    def __init__(
        self,
        agent: Agent,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any] | None = None,
        outputs_to_string: dict[str, str | Callable[[Any], str]] | None = None,
        inputs_from_state: dict[str, str] | None = None,
        outputs_to_state: dict[str, dict[str, str | Callable]] | None = None,
    ) -> None:
        """
        Create a Tool instance from a Haystack Agent.
        """
        if not isinstance(agent, Agent):
            raise TypeError(f"The 'agent' parameter must be an instance of Agent. Got {type(agent)} instead.")

        uncovered = _uncovered_agent_inputs(agent, inputs_from_state)
        if parameters is None:
            parameters = _build_parameters(uncovered)
        else:
            missing = [name for name in uncovered if name not in parameters.get("properties", {})]
            if missing:
                raise ValueError(
                    f"The Agent requires the inputs {missing}, which this tool does not provide, so it could never "
                    f"be invoked. Add them to 'parameters' so that the calling LLM fills them in, or map them from "
                    f"the calling Agent's state with 'inputs_from_state'."
                )

        super().__init__(
            component=agent,
            name=name,
            description=description,
            parameters=parameters,
            outputs_to_string=outputs_to_string or {"handler": _agent_result_to_string},
            inputs_from_state=inputs_from_state,
            outputs_to_state=outputs_to_state,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the AgentTool to a dictionary.

        :returns:
            The serialized dictionary representation of AgentTool.
        """
        serialized = super().to_dict()
        serialized["data"]["agent"] = serialized["data"].pop("component")
        return serialized

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentTool":
        """
        Deserializes the AgentTool from a dictionary.

        :param data: The dictionary representation of AgentTool.
        :returns:
            The deserialized AgentTool instance.
        """
        inner_data = data["data"]
        agent_class = import_class_by_name(inner_data["agent"]["type"])
        agent = component_from_dict(cls=agent_class, data=inner_data["agent"], name=inner_data["name"])

        outputs_to_state = inner_data.get("outputs_to_state")
        if outputs_to_state:
            outputs_to_state = _deserialize_outputs_to_state(outputs_to_state)

        outputs_to_string = inner_data.get("outputs_to_string")
        if outputs_to_string is not None:
            outputs_to_string = _deserialize_outputs_to_string(outputs_to_string)

        return cls(
            agent=agent,
            name=inner_data["name"],
            description=inner_data["description"],
            parameters=inner_data.get("parameters"),
            outputs_to_string=outputs_to_string,
            inputs_from_state=inner_data.get("inputs_from_state"),
            outputs_to_state=outputs_to_state,
        )
