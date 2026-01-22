# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from haystack.components.agents import State
from haystack.components.tools.tool_invoker import ToolInvoker
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage, StreamingCallbackT
from haystack.human_in_the_loop import ToolExecutionDecision
from haystack.human_in_the_loop.types import ConfirmationPolicy, ConfirmationStrategy, ConfirmationUI
from haystack.tools import Tool
from haystack.utils.deserialization import deserialize_component_inplace

# To prevent circular imports
if TYPE_CHECKING:
    from haystack.components.agents.agent import _ExecutionContext

REJECTION_FEEDBACK_TEMPLATE = "Tool execution for '{tool_name}' was rejected by the user."
MODIFICATION_FEEDBACK_TEMPLATE = (
    "The parameters for tool '{tool_name}' were updated by the user to:\n{final_tool_params}"
)
USER_FEEDBACK_TEMPLATE = "With user feedback: {feedback}"


class BlockingConfirmationStrategy:
    """
    Confirmation strategy that blocks execution to gather user feedback.
    """

    def __init__(
        self,
        *,
        confirmation_policy: ConfirmationPolicy,
        confirmation_ui: ConfirmationUI,
        reject_template: str = REJECTION_FEEDBACK_TEMPLATE,
        modify_template: str = MODIFICATION_FEEDBACK_TEMPLATE,
        user_feedback_template: str = USER_FEEDBACK_TEMPLATE,
    ) -> None:
        """
        Initialize the BlockingConfirmationStrategy with a confirmation policy and UI.

        :param confirmation_policy:
            The confirmation policy to determine when to ask for user confirmation.
        :param confirmation_ui:
            The user interface to interact with the user for confirmation.
        :param reject_template:
            Template for rejection feedback messages. It should include a `{tool_name}` placeholder.
        :param modify_template:
            Template for modification feedback messages. It should include `{tool_name}` and `{final_tool_params}`
            placeholders.
        :param user_feedback_template:
            Template for user feedback messages. It should include a `{feedback}` placeholder.
        """
        self.confirmation_policy = confirmation_policy
        self.confirmation_ui = confirmation_ui
        self.reject_template = reject_template
        self.modify_template = modify_template
        self.user_feedback_template = user_feedback_template

    def run(
        self,
        *,
        tool_name: str,
        tool_description: str,
        tool_params: dict[str, Any],
        tool_call_id: str | None = None,
        confirmation_strategy_context: dict[str, Any] | None = None,
    ) -> ToolExecutionDecision:
        """
        Run the human-in-the-loop strategy for a given tool and its parameters.

        :param tool_name:
            The name of the tool to be executed.
        :param tool_description:
            The description of the tool.
        :param tool_params:
            The parameters to be passed to the tool.
        :param tool_call_id:
            Optional unique identifier for the tool call. This can be used to track and correlate the decision with a
            specific tool invocation.
        :param confirmation_strategy_context:
            Optional dictionary for passing request-scoped resources. Useful in web/server environments
            to provide per-request objects (e.g., WebSocket connections, async queues, Redis pub/sub clients)
            that strategies can use for non-blocking user interaction.

        :returns:
            A ToolExecutionDecision indicating whether to execute the tool with the given parameters, or a
            feedback message if rejected.
        """
        # Check if we should ask based on policy
        if not self.confirmation_policy.should_ask(
            tool_name=tool_name, tool_description=tool_description, tool_params=tool_params
        ):
            return ToolExecutionDecision(
                tool_name=tool_name, execute=True, tool_call_id=tool_call_id, final_tool_params=tool_params
            )

        # Get user confirmation through UI
        confirmation_ui_result = self.confirmation_ui.get_user_confirmation(tool_name, tool_description, tool_params)

        # Pass back the result to the policy for any learning/updating
        self.confirmation_policy.update_after_confirmation(
            tool_name, tool_description, tool_params, confirmation_ui_result
        )

        # Process the confirmation result
        final_args = {}
        if confirmation_ui_result.action == "reject":
            explanation_text = self.reject_template.format(tool_name=tool_name)
            if confirmation_ui_result.feedback:
                explanation_text += " "
                explanation_text += self.user_feedback_template.format(feedback=confirmation_ui_result.feedback)
            return ToolExecutionDecision(
                tool_name=tool_name, execute=False, tool_call_id=tool_call_id, feedback=explanation_text
            )
        elif confirmation_ui_result.action == "modify" and confirmation_ui_result.new_tool_params:
            # Update the tool call params with the new params
            final_args.update(confirmation_ui_result.new_tool_params)
            explanation_text = self.modify_template.format(tool_name=tool_name, final_tool_params=final_args)
            if confirmation_ui_result.feedback:
                explanation_text += " "
                explanation_text += self.user_feedback_template.format(feedback=confirmation_ui_result.feedback)
            return ToolExecutionDecision(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                execute=True,
                feedback=explanation_text,
                final_tool_params=final_args,
            )
        else:  # action == "confirm"
            return ToolExecutionDecision(
                tool_name=tool_name, execute=True, tool_call_id=tool_call_id, final_tool_params=tool_params
            )

    async def run_async(
        self,
        *,
        tool_name: str,
        tool_description: str,
        tool_params: dict[str, Any],
        tool_call_id: str | None = None,
        confirmation_strategy_context: dict[str, Any] | None = None,
    ) -> ToolExecutionDecision:
        """
        Async version of run. Calls the sync run() method by default.

        :param tool_name:
            The name of the tool to be executed.
        :param tool_description:
            The description of the tool.
        :param tool_params:
            The parameters to be passed to the tool.
        :param tool_call_id:
            Optional unique identifier for the tool call.
        :param confirmation_strategy_context:
            Optional dictionary for passing request-scoped resources.

        :returns:
            A ToolExecutionDecision indicating whether to execute the tool with the given parameters.
        """
        return self.run(
            tool_name=tool_name,
            tool_description=tool_description,
            tool_params=tool_params,
            tool_call_id=tool_call_id,
            confirmation_strategy_context=confirmation_strategy_context,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the BlockingConfirmationStrategy to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            confirmation_policy=self.confirmation_policy.to_dict(),
            confirmation_ui=self.confirmation_ui.to_dict(),
            reject_template=self.reject_template,
            modify_template=self.modify_template,
            user_feedback_template=self.user_feedback_template,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BlockingConfirmationStrategy":
        """
        Deserializes the BlockingConfirmationStrategy from a dictionary.

        :param data:
            Dictionary to deserialize from.

        :returns:
            Deserialized BlockingConfirmationStrategy.
        """
        deserialize_component_inplace(data["init_parameters"], key="confirmation_policy")
        deserialize_component_inplace(data["init_parameters"], key="confirmation_ui")
        return default_from_dict(cls, data)


def _prepare_tool_args(
    *,
    tool: Tool,
    tool_call_arguments: dict[str, Any],
    state: State,
    streaming_callback: StreamingCallbackT | None = None,
    enable_streaming_passthrough: bool = False,
) -> dict[str, Any]:
    """
    Prepare the final arguments for a tool by injecting state inputs and optionally a streaming callback.

    :param tool:
        The tool instance to prepare arguments for.
    :param tool_call_arguments:
        The initial arguments provided for the tool call.
    :param state:
        The current state containing inputs to be injected into the tool arguments.
    :param streaming_callback:
        Optional streaming callback to be injected if enabled and applicable.
    :param enable_streaming_passthrough:
        Flag indicating whether to inject the streaming callback into the tool arguments.

    :returns:
        A dictionary of final arguments ready for tool invocation.
    """
    # Combine user + state inputs
    final_args = ToolInvoker._inject_state_args(tool, tool_call_arguments.copy(), state)
    # Check whether to inject streaming_callback
    if (
        enable_streaming_passthrough
        and streaming_callback is not None
        and "streaming_callback" not in final_args
        and "streaming_callback" in ToolInvoker._get_func_params(tool)
    ):
        final_args["streaming_callback"] = streaming_callback
    return final_args


def _process_confirmation_strategies(
    *,
    confirmation_strategies: dict[str, ConfirmationStrategy],
    messages_with_tool_calls: list[ChatMessage],
    execution_context: "_ExecutionContext",
) -> tuple[list[ChatMessage], list[ChatMessage]]:
    """
    Run the confirmation strategies and return modified tool call messages and updated chat history.

    :param confirmation_strategies: Mapping of tool names to their corresponding confirmation strategies
    :param messages_with_tool_calls: Chat messages containing tool calls
    :param execution_context: The current execution context of the agent
    :returns:
        Tuple of modified messages with confirmed tool calls and updated chat history
    """
    # If confirmations strategies is empty, return original messages and chat history
    if not confirmation_strategies:
        return messages_with_tool_calls, execution_context.state.get("messages")

    # Run confirmation strategies and get tool execution decisions
    teds = _run_confirmation_strategies(
        confirmation_strategies=confirmation_strategies,
        messages_with_tool_calls=messages_with_tool_calls,
        execution_context=execution_context,
    )

    # Apply tool execution decisions to messages_with_tool_calls
    rejection_messages, modified_tool_call_messages = _apply_tool_execution_decisions(
        tool_call_messages=messages_with_tool_calls, tool_execution_decisions=teds
    )

    # Update the chat history with rejection messages and new tool call messages
    new_chat_history = _update_chat_history(
        chat_history=execution_context.state.get("messages"),
        rejection_messages=rejection_messages,
        tool_call_and_explanation_messages=modified_tool_call_messages,
    )

    return modified_tool_call_messages, new_chat_history


async def _process_confirmation_strategies_async(
    *,
    confirmation_strategies: dict[str, ConfirmationStrategy],
    messages_with_tool_calls: list[ChatMessage],
    execution_context: "_ExecutionContext",
) -> tuple[list[ChatMessage], list[ChatMessage]]:
    """
    Async version of _process_confirmation_strategies.

    Run the confirmation strategies and return modified tool call messages and updated chat history.

    :param confirmation_strategies: Mapping of tool names to their corresponding confirmation strategies
    :param messages_with_tool_calls: Chat messages containing tool calls
    :param execution_context: The current execution context of the agent
    :returns:
        Tuple of modified messages with confirmed tool calls and updated chat history
    """
    # If confirmations strategies is empty, return original messages and chat history
    if not confirmation_strategies:
        return messages_with_tool_calls, execution_context.state.get("messages")

    # Run confirmation strategies and get tool execution decisions (async version)
    teds = await _run_confirmation_strategies_async(
        confirmation_strategies=confirmation_strategies,
        messages_with_tool_calls=messages_with_tool_calls,
        execution_context=execution_context,
    )

    # Apply tool execution decisions to messages_with_tool_calls
    rejection_messages, modified_tool_call_messages = _apply_tool_execution_decisions(
        tool_call_messages=messages_with_tool_calls, tool_execution_decisions=teds
    )

    # Update the chat history with rejection messages and new tool call messages
    new_chat_history = _update_chat_history(
        chat_history=execution_context.state.get("messages"),
        rejection_messages=rejection_messages,
        tool_call_and_explanation_messages=modified_tool_call_messages,
    )

    return modified_tool_call_messages, new_chat_history


def _run_confirmation_strategies(
    confirmation_strategies: dict[str, ConfirmationStrategy],
    messages_with_tool_calls: list[ChatMessage],
    execution_context: "_ExecutionContext",
) -> list[ToolExecutionDecision]:
    """
    Run confirmation strategies for tool calls in the provided chat messages.

    :param confirmation_strategies: Mapping of tool names to their corresponding confirmation strategies
    :param messages_with_tool_calls: Messages containing tool calls to process
    :param execution_context: The current execution context containing state and inputs
    :returns:
        A list of ToolExecutionDecision objects representing the decisions made for each tool call.
    """
    state = execution_context.state
    tools_with_names = {tool.name: tool for tool in execution_context.tool_invoker_inputs["tools"]}
    existing_teds = execution_context.tool_execution_decisions if execution_context.tool_execution_decisions else []
    existing_teds_by_name = {ted.tool_name: ted for ted in existing_teds if ted.tool_name}
    existing_teds_by_id = {ted.tool_call_id: ted for ted in existing_teds if ted.tool_call_id}

    teds = []
    for message in messages_with_tool_calls:
        if not message.tool_calls:
            continue

        for tool_call in message.tool_calls:
            tool_name = tool_call.tool_name
            tool_to_invoke = tools_with_names[tool_name]

            # Prepare final tool args
            final_args = _prepare_tool_args(
                tool=tool_to_invoke,
                tool_call_arguments=tool_call.arguments,
                state=state,
                streaming_callback=execution_context.tool_invoker_inputs.get("streaming_callback"),
                enable_streaming_passthrough=execution_context.tool_invoker_inputs.get(
                    "enable_streaming_passthrough", False
                ),
            )

            # Get tool execution decisions from confirmation strategies
            # If no confirmation strategy is defined for this tool, proceed with execution
            if tool_name not in confirmation_strategies:
                teds.append(
                    ToolExecutionDecision(
                        tool_call_id=tool_call.id, tool_name=tool_name, execute=True, final_tool_params=final_args
                    )
                )
                continue

            # Check if there's already a decision for this tool call in the execution context
            ted = existing_teds_by_id.get(tool_call.id or "") or existing_teds_by_name.get(tool_name)

            # If not, run the confirmation strategy
            if not ted:
                ted = confirmation_strategies[tool_name].run(
                    tool_name=tool_name,
                    tool_description=tool_to_invoke.description,
                    tool_params=final_args,
                    tool_call_id=tool_call.id,
                    confirmation_strategy_context=execution_context.confirmation_strategy_context,
                )
            teds.append(ted)

    return teds


async def _run_confirmation_strategies_async(
    confirmation_strategies: dict[str, ConfirmationStrategy],
    messages_with_tool_calls: list[ChatMessage],
    execution_context: "_ExecutionContext",
) -> list[ToolExecutionDecision]:
    """
    Async version of _run_confirmation_strategies.

    Run confirmation strategies for tool calls in the provided chat messages.

    :param confirmation_strategies: Mapping of tool names to their corresponding confirmation strategies
    :param messages_with_tool_calls: Messages containing tool calls to process
    :param execution_context: The current execution context containing state and inputs
    :returns:
        A list of ToolExecutionDecision objects representing the decisions made for each tool call.
    """
    state = execution_context.state
    tools_with_names = {tool.name: tool for tool in execution_context.tool_invoker_inputs["tools"]}
    existing_teds = execution_context.tool_execution_decisions if execution_context.tool_execution_decisions else []
    existing_teds_by_name = {ted.tool_name: ted for ted in existing_teds if ted.tool_name}
    existing_teds_by_id = {ted.tool_call_id: ted for ted in existing_teds if ted.tool_call_id}

    teds = []
    for message in messages_with_tool_calls:
        if not message.tool_calls:
            continue

        for tool_call in message.tool_calls:
            tool_name = tool_call.tool_name
            tool_to_invoke = tools_with_names[tool_name]

            # Prepare final tool args
            final_args = _prepare_tool_args(
                tool=tool_to_invoke,
                tool_call_arguments=tool_call.arguments,
                state=state,
                streaming_callback=execution_context.tool_invoker_inputs.get("streaming_callback"),
                enable_streaming_passthrough=execution_context.tool_invoker_inputs.get(
                    "enable_streaming_passthrough", False
                ),
            )

            # Get tool execution decisions from confirmation strategies
            # If no confirmation strategy is defined for this tool, proceed with execution
            if tool_name not in confirmation_strategies:
                teds.append(
                    ToolExecutionDecision(
                        tool_call_id=tool_call.id, tool_name=tool_name, execute=True, final_tool_params=final_args
                    )
                )
                continue

            # Check if there's already a decision for this tool call in the execution context
            ted = existing_teds_by_id.get(tool_call.id or "") or existing_teds_by_name.get(tool_name)

            # If not, run the confirmation strategy (async version)
            if not ted:
                strategy = confirmation_strategies[tool_name]
                # Use run_async if available, otherwise fall back to sync run
                if hasattr(strategy, "run_async"):
                    ted = await strategy.run_async(
                        tool_name=tool_name,
                        tool_description=tool_to_invoke.description,
                        tool_params=final_args,
                        tool_call_id=tool_call.id,
                        confirmation_strategy_context=execution_context.confirmation_strategy_context,
                    )
                else:
                    ted = strategy.run(
                        tool_name=tool_name,
                        tool_description=tool_to_invoke.description,
                        tool_params=final_args,
                        tool_call_id=tool_call.id,
                        confirmation_strategy_context=execution_context.confirmation_strategy_context,
                    )
            teds.append(ted)

    return teds


def _apply_tool_execution_decisions(
    tool_call_messages: list[ChatMessage], tool_execution_decisions: list[ToolExecutionDecision]
) -> tuple[list[ChatMessage], list[ChatMessage]]:
    """
    Apply the tool execution decisions to the tool call messages.

    :param tool_call_messages: The tool call messages to apply the decisions to.
    :param tool_execution_decisions: The tool execution decisions to apply.
    :returns:
        A tuple containing:
        - A list of rejection messages for rejected tool calls. These are pairs of tool call and tool call result
          messages.
        - A list of tool call messages for confirmed or modified tool calls. If tool parameters were modified,
          a user message explaining the modification is included before the tool call message.
    """
    decision_by_id = {d.tool_call_id: d for d in tool_execution_decisions if d.tool_call_id}
    decision_by_name = {d.tool_name: d for d in tool_execution_decisions if d.tool_name}

    # Known limitation: If tool calls are missing IDs, we rely on tool names to match decisions to tool calls.
    # This can lead to incorrect matches if there are multiple tool calls in the provided messages with duplicate names.
    if not decision_by_id and len(decision_by_name) < len(tool_execution_decisions):
        raise ValueError(
            "ToolExecutionDecisions are missing tool_call_id fields and there are multiple tool calls with the same "
            "name. When multiple tool calls with the same name are present, tool_call_id is required to correctly "
            "match decisions to tool calls."
        )

    def make_assistant_message(chat_message, tool_calls):
        return ChatMessage.from_assistant(
            text=chat_message.text,
            meta=chat_message.meta,
            name=chat_message.name,
            tool_calls=tool_calls,
            reasoning=chat_message.reasoning,
        )

    new_tool_call_messages = []
    rejection_messages = []

    for chat_msg in tool_call_messages:
        new_tool_calls = []
        for tc in chat_msg.tool_calls or []:
            ted = decision_by_id.get(tc.id or "") or decision_by_name.get(tc.tool_name)
            if not ted:
                # This shouldn't happen, if so something went wrong in _run_confirmation_strategies
                continue

            if not ted.execute:
                # rejected tool call
                tool_result_text = ted.feedback or REJECTION_FEEDBACK_TEMPLATE.format(tool_name=tc.tool_name)
                rejection_messages.extend(
                    [
                        make_assistant_message(chat_msg, [tc]),
                        ChatMessage.from_tool(tool_result=tool_result_text, origin=tc, error=True),
                    ]
                )
                continue

            # Covers confirm and modify cases
            final_args = ted.final_tool_params or {}
            if tc.arguments != final_args:
                # In the modify case we add a user message explaining the modification otherwise the LLM won't know
                # why the tool parameters changed and will likely just try and call the tool again with the
                # original parameters.
                user_text = ted.feedback or MODIFICATION_FEEDBACK_TEMPLATE.format(
                    tool_name=tc.tool_name, final_tool_params=final_args
                )
                new_tool_call_messages.append(ChatMessage.from_user(text=user_text))
            new_tool_calls.append(replace(tc, arguments=final_args))

        # Only add the tool call message if there are any tool calls left (i.e. not all were rejected)
        if new_tool_calls:
            new_tool_call_messages.append(make_assistant_message(chat_msg, new_tool_calls))

    return rejection_messages, new_tool_call_messages


def _update_chat_history(
    chat_history: list[ChatMessage],
    rejection_messages: list[ChatMessage],
    tool_call_and_explanation_messages: list[ChatMessage],
) -> list[ChatMessage]:
    """
    Update the chat history to include rejection messages and tool call messages at the appropriate positions.

    Steps:
    1. Identify the last user message and the last tool message in the current chat history.
    2. Determine the insertion point as the maximum index of these two messages.
    3. Create a new chat history that includes:
       - All messages up to the insertion point.
       - Any rejection messages (pairs of tool call and tool call result messages).
       - Any tool call messages for confirmed or modified tool calls, including user messages explaining modifications.

    :param chat_history: The current chat history.
    :param rejection_messages: Chat messages to add for rejected tool calls (pairs of tool call and tool call result
        messages).
    :param tool_call_and_explanation_messages: Tool call messages for confirmed or modified tool calls, which may
        include user messages explaining modifications.
    :returns:
        The updated chat history.
    """
    user_indices = [i for i, message in enumerate(chat_history) if message.is_from("user")]
    tool_indices = [i for i, message in enumerate(chat_history) if message.is_from("tool")]

    last_user_idx = max(user_indices) if user_indices else -1
    last_tool_idx = max(tool_indices) if tool_indices else -1

    insertion_point = max(last_user_idx, last_tool_idx)

    new_chat_history = chat_history[: insertion_point + 1] + rejection_messages + tool_call_and_explanation_messages
    return new_chat_history
