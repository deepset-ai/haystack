# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import List

from haystack import component
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.tool import Tool

_FUNCTION_NAME_FAILURE = (
    "I'm sorry, I tried to run a function that did not exist. Would you like me to correct it and try again?"
)
_FUNCTION_RUN_FAILURE = "Seems there was an error while running the function: {error}"


# JUST A SLIGHTLY MODIFIED VERSION OF OpenAIFunctionCaller
@component
class ToolInvoker:
    """
    OpenAIFunctionCaller processes a list of chat messages and call tools if encounters a tool call.
    """

    def __init__(self, tools: List[Tool]):
        self.tools = tools
        self._tool_names = {tool.name for tool in tools}

    @component.output_types(tool_messages=List[ChatMessage])
    def run(self, messages: List[ChatMessage]):
        """
        Process a list of chat messages and call tools if encounters a tool call.
        """
        replies = []

        tool_calls = messages[-1].tool_calls
        if tool_calls:
            for tool_call in tool_calls:
                tool_name = tool_call.tool_name
                tool_arguments = tool_call.arguments
                if tool_name in self._tool_names:
                    tool_to_call = next(tool for tool in self.tools if tool.name == tool_name)
                    try:
                        tool_response = tool_to_call.invoke(**tool_arguments)
                        replies.append(
                            ChatMessage.from_tool(content=json.dumps(tool_response), tool_call_id=tool_call.id)
                        )
                    # pylint: disable=broad-exception-caught
                    except Exception as e:
                        replies.append(
                            ChatMessage.from_tool(
                                content=_FUNCTION_RUN_FAILURE.format(error=e), tool_call_id=tool_call.id
                            )
                        )
                else:
                    replies.append(ChatMessage.from_tool(_FUNCTION_NAME_FAILURE, tool_call_id=tool_call.id))
        return {"tool_messages": replies}
