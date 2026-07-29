# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Any

from haystack import logging
from haystack.components.agents import Agent
from haystack.core.serialization import component_from_dict, import_class_by_name
from haystack.tools.component_tool import ComponentTool
from haystack.tools.tool import _deserialize_outputs_to_state, _deserialize_outputs_to_string

logger = logging.getLogger(__name__)

_DEFAULT_PARAMETERS = {
    "type": "object",
    "properties": {
        "messages": {
            "type": "array",
            "description": "Exactly one user message",
            "minItems": 1,
            "maxItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "enum": ["user"]},
                    "content": {"type": "string", "description": ("The task to delegate to this tool.")},
                },
                "required": ["role", "content"],
            },
        }
    },
    "required": ["messages"],
}


class AgentTool(ComponentTool):
    """
    A Tool that wraps an Agent, so that another Agent can delegate work to it.

    ## Usage Example:

    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack.tools import AgentTool
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

        super().__init__(
            component=agent,
            name=name,
            description=description,
            parameters=parameters or _DEFAULT_PARAMETERS,
            outputs_to_string=outputs_to_string or {"source": "last_message"},
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
