# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Test file demonstrating the type variance solution for ToolsType.

This file demonstrates the fix for the issue reported in GitHub issue #9949:
https://github.com/deepset-ai/haystack/issues/9949

The problem was: In v2.19, changing from `list[Tool]` to `list[Tool | Toolset]` broke
type compatibility because Python's `list` is invariant. This meant `list[Tool]` was
NOT a subtype of `list[Tool | Toolset]`, even though `Tool` is compatible with
`Tool | Toolset`.

Pyright error with the old implementation:
    error: Argument of type "list[Tool]" cannot be assigned to parameter "tools"
           of type "ToolsType" in function "__init__"
        Type "list[Tool]" is not assignable to type "ToolsType"
          "list[Tool]" is not assignable to "list[Tool | Toolset]"
            Type parameter "_T@list" is invariant, but "Tool" is not the same as "Tool | Toolset"
            Consider switching from "list" to "Sequence" which is covariant

The solution: Changed ToolsType to use `Sequence[Union[Tool, Toolset]]` instead of
`list[Union[Tool, Toolset]]`. Since Sequence is covariant, `Sequence[Tool]` IS
a subtype of `Sequence[Tool | Toolset]`, and the type errors are resolved.

These tests verify that the fix works correctly and document the type variance behavior.
"""

from typing import Sequence, Union

from haystack.components.tools import ToolInvoker
from haystack.tools import Tool, Toolset


def test_list_of_tools_variance_issue():
    """
    Demonstrates the type variance issue with list[Tool] and ToolsType.

    This test shows that when you create a list of Tool objects and try to pass it
    to ToolInvoker, Pyright will report a type error because list is invariant.
    """

    # Create some tools
    def add_func(a: int, b: int) -> int:
        return a + b

    def multiply_func(a: int, b: int) -> int:
        return a * b

    tool1 = Tool(
        name="add",
        description="Add two numbers",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        function=add_func,
    )

    tool2 = Tool(
        name="multiply",
        description="Multiply two numbers",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        function=multiply_func,
    )

    # This is the pattern that users naturally write
    tools = [tool1, tool2]  # Type: list[Tool]

    # FIXED: With Sequence[Tool | Toolset], this now works correctly because Sequence is covariant
    # list[Tool] is a subtype of Sequence[Tool | Toolset], so this passes type checking
    invoker = ToolInvoker(tools=tools)

    # Verify the invoker was created successfully
    assert invoker is not None


def test_mixed_tools_and_toolsets():
    """
    Test that demonstrates mixing Tool and Toolset objects.

    This should work after the fix because Sequence is covariant.
    """

    def add_func(a: int, b: int) -> int:
        return a + b

    def multiply_func(a: int, b: int) -> int:
        return a * b

    tool1 = Tool(
        name="add",
        description="Add two numbers",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        function=add_func,
    )

    tool2 = Tool(
        name="multiply",
        description="Multiply two numbers",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        function=multiply_func,
    )

    toolset = Toolset([tool1])

    # Mix Tool and Toolset - this should be supported
    mixed_tools: list[Union[Tool, Toolset]] = [toolset, tool2]
    invoker = ToolInvoker(tools=mixed_tools)

    # Verify the invoker was created successfully
    assert invoker is not None


def test_sequence_compatibility():
    """
    Demonstrates that using Sequence (covariant) solves the variance issue.

    After the fix, this pattern should work without type errors.
    """

    def add_func(a: int, b: int) -> int:
        return a + b

    tool1 = Tool(
        name="add",
        description="Add two numbers",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        function=add_func,
    )

    # Users can explicitly type as Sequence[Tool] if they want
    # Note: Both list[Tool] and Sequence[Tool] work because list is a subtype of Sequence
    tools_seq: Sequence[Tool] = [tool1]

    # With the fix (using Sequence in ToolsType), both list[Tool] and Sequence[Tool] work
    # because Sequence is covariant: Sequence[Tool] is a subtype of Sequence[Tool | Toolset]
    invoker = ToolInvoker(tools=tools_seq)

    # Verify the invoker was created successfully
    assert invoker is not None
