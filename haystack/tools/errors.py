# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


class SchemaGenerationError(Exception):
    """
    Exception raised when automatic schema generation fails.
    """

    pass


class ToolInvocationError(Exception):
    """
    Exception raised when a Tool invocation fails.
    """

    def __init__(self, message: str, tool_name: str):
        super().__init__(message)
        self.tool_name = tool_name
