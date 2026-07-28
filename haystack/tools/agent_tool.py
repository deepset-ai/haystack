# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Any

from haystack import logging
from haystack.components.agents import Agent
from haystack.core.serialization import component_from_dict, import_class_by_name
from haystack.dataclasses import ChatMessage
from haystack.tools.component_tool import ComponentTool
from haystack.tools.tool import _deserialize_outputs_to_state, _deserialize_outputs_to_string

logger = logging.getLogger(__name__)

# Shared by every AgentTool built with the default schema: treat it as read-only.
_DEFAULT_PARAMETERS = {
    "type": "object",
    "properties": {
        "messages": {
            "type": "array",
            "description": "Exactly one user message whose content is the task to delegate.",
            "minItems": 1,
            "maxItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "enum": ["user"]},
                    "content": {
                        "type": "string",
                        "description": (
                            "The task to delegate. The agent does not see this conversation, so state everything "
                            "it needs to know in a single self-contained instruction."
                        ),
                    },
                },
                "required": ["role", "content"],
            },
        }
    },
    "required": ["messages"],
}


def _last_message_text(message: ChatMessage) -> str:
    """
    Default `outputs_to_string` handler: return the agent's final reply as plain text.

    :param message: The `last_message` produced by the agent.
    :returns: The text of the message, or an empty string if it has none.
    """
    return message.text or ""


class AgentTool(ComponentTool):
    """
    A Tool that wraps an Agent, so that another Agent can delegate work to it.

    It is a `ComponentTool` with two defaults changed:

    - `parameters` asks for a single user message carrying the task, instead of the schema derived from `Agent.run`.
      The derived one is over 7,000 characters, because it describes every `ChatMessage` content block, and it also
      exposes inputs such as `tools` that the calling LLM should not be able to set.
    - `outputs_to_string` returns the text of the agent's final reply, so only the conclusion of the delegated work
      enters the calling agent's context, not the whole run result.

    Both defaults can be overridden, and everything else is inherited from `ComponentTool`.

    ## Usage Example:

    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack.tools import AgentTool

    researcher = Agent(
        chat_generator=OpenAIChatGenerator(model="gpt-5.4-mini"),
        system_prompt="You are a research specialist. Investigate the task and report your findings.",
    )

    research = AgentTool(
        researcher,
        name="research",
        description="Delegate a focused research question and get back a summary of the findings.",
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
        parameters: dict[str, Any] | None = _DEFAULT_PARAMETERS,
        outputs_to_string: dict[str, str | Callable[[Any], str]] | None = None,
        inputs_from_state: dict[str, str] | None = None,
        outputs_to_state: dict[str, dict[str, str | Callable]] | None = None,
    ) -> None:
        """
        Create a Tool instance from a Haystack Agent.

        :param agent: The Agent to wrap as a tool.
        :param name: Name of the tool.
        :param description: Description of the tool. Tell the calling LLM what the agent is good at and when to
            delegate to it.
        :param parameters:
            A JSON schema defining the parameters expected by the Tool. Defaults to a single user message carrying
            the task; the default schema is shared between instances, so do not modify it in place. A custom schema
            must use the Agent's own input names, since they are resolved against its input sockets when the tool is
            invoked. Pass `None` to fall back to the schema derived from `Agent.run`.
        :param outputs_to_string:
            Optional dictionary defining how tool outputs should be converted into string(s) or results.
            Defaults to the text of the agent's final reply. See `ComponentTool` for the supported formats.
        :param inputs_from_state:
            Optional dictionary mapping state keys to tool parameter names.
            Example: `{"repository": "repo"}` maps state's "repository" to tool's "repo" parameter.
        :param outputs_to_state:
            Optional dictionary defining how tool outputs map to keys within state as well as optional handlers.
            See `ComponentTool` for the supported formats.
        :raises TypeError: If the object passed is not an Agent instance.
        """
        if not isinstance(agent, Agent):
            raise TypeError(f"The 'agent' parameter must be an instance of Agent. Got {type(agent)} instead.")

        super().__init__(
            component=agent,
            name=name,
            description=description,
            parameters=parameters,
            outputs_to_string=outputs_to_string or {"source": "last_message", "handler": _last_message_text},
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
