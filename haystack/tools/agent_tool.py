# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Callable
from typing import Any

from haystack.components.agents import Agent
from haystack.components.agents.agent import _EXIT_REASON_MAX_STEPS
from haystack.tools.component_tool import ComponentTool
from haystack.tools.tool import _deserialize_outputs_to_state, _deserialize_outputs_to_string
from haystack.utils.deserialization import deserialize_component_inplace


def _required_tool_parameters(agent: Agent, inputs_from_state: dict[str, Any] | None) -> list[str]:
    """
    Return the additional required Tool parameters for the wrapped Agent.

    These are mandatory Agent inputs that are not supplied through `inputs_from_state`.
    `messages` is excluded because AgentTool always adds it to the generated schema.

    :param agent: The wrapped Agent.
    :param inputs_from_state: Maps the calling Agent's state keys to Agent inputs.
    :returns: Additional required Tool parameters, sorted by name.
    """
    covered = {"messages", *(inputs_from_state or {}).values()}
    # prompt variables are registered as input sockets in a non-deterministic order, so sort to keep the schema stable
    return sorted(
        name
        # __haystack_input__ is attached to the instance by the @component decorator, so mypy cannot see it
        for name, socket in agent.__haystack_input__._sockets_dict.items()  # type: ignore[attr-defined]
        if socket.is_mandatory and name not in covered
    )


def agent_result_to_string(result: dict[str, Any]) -> str:
    """Default `outputs_to_string` handler"""
    last_message = result["last_message"]
    text = last_message.text or json.dumps(last_message.to_dict())
    if result["exit_reason"] == _EXIT_REASON_MAX_STEPS:
        text += "\n\n[The Agent reached max_agent_steps and stopped, so this result may be incomplete.]"
    return text


class AgentTool(ComponentTool):
    """
    A Tool that wraps a Haystack Agent, allowing it to be used as a tool by another Agent.

    AgentTool is a building block for multi-agent systems: an Agent specialized in one task becomes a tool that
    other Agents can delegate to. The calling Agent only sees the final reply, so all the steps the wrapped Agent
    takes stay out of its context. Sensible defaults make this work out of the box: the task is delegated as a
    single user message and comes back as text.

    To use AgentTool, you first need a Haystack Agent. Below is an example of creating an AgentTool from an Agent
    that searches the web with a SerperDevWebSearch component from the `serperdev-haystack` integration package
    (`pip install serperdev-haystack`).

    ## Usage Example:
    <!-- test-ignore -->
    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIResponsesChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack.tools import AgentTool, ComponentTool
    from haystack_integrations.components.websearch.serperdev import SerperDevWebSearch

    researcher = Agent(
        chat_generator=OpenAIResponsesChatGenerator(model="gpt-5.4-mini"),
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
        agent=researcher,
        name="research",
        description="Research a question on the web and report the findings",
    )

    coordinator = Agent(
        chat_generator=OpenAIResponsesChatGenerator(model="gpt-5.4"),
        tools=[research],
        system_prompt="You coordinate specialists. Delegate research questions, then answer the user.",
    )

    result = coordinator.run([ChatMessage.from_user("What are the latest developments in the Haystack framework?")])
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

        :param agent: The Haystack Agent to wrap as a tool.
        :param name: Name of the tool.
        :param description: Description of the tool. It should tell the calling LLM what the Agent is specialized in
            and when to delegate to it.
        :param parameters:
            A JSON schema defining the parameters expected by the Tool.
            Will fall back to a schema with the task to delegate as a single user message, plus one string parameter
            for every other mandatory input of the Agent, if not provided.
        :param outputs_to_string:
            Optional dictionary defining how tool outputs should be converted into string(s) or results.
            If not provided, the tool result is the text of the Agent's final reply, or the serialized message if
            the reply has no text. A warning is appended if the Agent stopped because it reached `max_agent_steps`.

            `outputs_to_string` supports two formats:

            1. Single output format - use "source", "handler", and/or "raw_result" at the root level:
                ```python
                {
                    "source": "last_message", "handler": format_reply, "raw_result": False
                }
                ```
                - `source`: If provided, only the specified output key is sent to the handler.
                - `handler`: A function that takes the tool output (or the extracted source value) and returns the
                  final result.
                - `raw_result`: If `True`, the result is returned raw without string conversion, but applying the
                   `handler` if provided. This is intended for tools that return images. In this mode, the `handler`
                   is required, since the Agent returns a dictionary, and it must return a list of
                   `TextContent`/`ImageContent` objects to ensure compatibility with Chat Generators.

            2. Multiple output format - map keys to individual configurations:
                ```python
                {
                    "reply": {"source": "last_message", "handler": format_reply},
                    "steps": {"source": "step_count", "handler": str}
                }
                ```
                Each key maps to a dictionary that can contain "source" and/or "handler".
                Note that `raw_result` is not supported in the multiple output format.
        :param inputs_from_state:
            Optional dictionary mapping the calling Agent's state keys to Agent input names.
            Example: `{"subject": "topic"}` maps state's "subject" to the Agent's "topic" input.
            Inputs mapped this way are not added to the generated `parameters` schema, since the calling Agent
            provides them.
        :param outputs_to_state:
            Optional dictionary defining how tool outputs map to keys within state as well as optional handlers.
            The keys must be declared in the `state_schema` of the calling Agent.
            Handlers merge the tool output into the state and are called as `handler(current_value, tool_output)`.
            If the source is provided only the specified output key is sent to the handler.
            Example:
            ```python
            {
                "notes": {"source": "last_message", "handler": custom_handler}
            }
            ```
            If the source is omitted the whole tool result is sent to the handler.
            Example:
            ```python
            {
                "notes": {"handler": custom_handler}
            }
            ```
        :raises TypeError: If the object passed is not a Haystack Agent instance.
        :raises ValueError: If `parameters` is provided but does not cover all the mandatory inputs of the Agent.
        """
        if not isinstance(agent, Agent):
            raise TypeError(f"The 'agent' parameter must be an instance of Agent. Got {type(agent)} instead.")

        if parameters is not None:
            required_tool_parameters = _required_tool_parameters(agent=agent, inputs_from_state=inputs_from_state)
            missing_required_parameters = [
                name for name in required_tool_parameters if name not in parameters.get("properties", {})
            ]
            if missing_required_parameters:
                raise ValueError(
                    f"The Agent wrapped by this tool requires the inputs {missing_required_parameters}, but this tool "
                    f"does not supply them, so it can never run. Add them to 'parameters', the schema that the calling "
                    f"LLM fills in, or to 'inputs_from_state', which takes them from the calling Agent's state."
                )

        super().__init__(
            component=agent,
            name=name,
            description=description,
            parameters=parameters,
            outputs_to_string=outputs_to_string or {"handler": agent_result_to_string},
            inputs_from_state=inputs_from_state,
            outputs_to_state=outputs_to_state,
        )

    def _create_tool_parameters_schema(self, component: Any, inputs_from_state: dict[str, Any]) -> dict[str, Any]:
        """
        Override ComponentTool schema generation for AgentTool defaults.

        ComponentTool calls this when users do not provide an explicit `parameters` schema. The generated schema always
        includes `messages` for the delegated task, plus one string parameter for each mandatory Agent input not
        supplied through `inputs_from_state`.
        """
        additional_required_parameters = _required_tool_parameters(agent=component, inputs_from_state=inputs_from_state)
        additional_properties = {name: {"type": "string"} for name in additional_required_parameters}
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
                **additional_properties,
            },
            "required": ["messages", *additional_required_parameters],
        }

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
        deserialize_component_inplace(data=inner_data, key="agent")

        outputs_to_state = inner_data.get("outputs_to_state")
        if outputs_to_state:
            outputs_to_state = _deserialize_outputs_to_state(outputs_to_state=outputs_to_state)

        outputs_to_string = inner_data.get("outputs_to_string")
        if outputs_to_string is not None:
            outputs_to_string = _deserialize_outputs_to_string(outputs_to_string=outputs_to_string)

        return cls(
            agent=inner_data["agent"],
            name=inner_data["name"],
            description=inner_data["description"],
            parameters=inner_data.get("parameters"),
            outputs_to_string=outputs_to_string,
            inputs_from_state=inner_data.get("inputs_from_state"),
            outputs_to_state=outputs_to_state,
        )
