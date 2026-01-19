# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.dataclasses import ConfirmationUIResult, ToolExecutionDecision


class TestConfirmationUIResult:
    def test_init(self):
        original = ConfirmationUIResult(action="reject", feedback="Changed my mind")
        assert original.action == "reject"
        assert original.feedback == "Changed my mind"
        assert original.new_tool_params is None


class TestToolExecutionDecision:
    def test_init(self):
        decision = ToolExecutionDecision(
            execute=True,
            tool_name="test_tool",
            tool_call_id="test_tool_call_id",
            final_tool_params={"param1": "new_value"},
        )
        assert decision.execute is True
        assert decision.final_tool_params == {"param1": "new_value"}
        assert decision.tool_call_id == "test_tool_call_id"
        assert decision.tool_name == "test_tool"

    def test_to_dict(self):
        original = ToolExecutionDecision(
            execute=True,
            tool_name="test_tool",
            tool_call_id="test_tool_call_id",
            final_tool_params={"param1": "new_value"},
        )
        as_dict = original.to_dict()
        assert as_dict == {
            "execute": True,
            "tool_name": "test_tool",
            "tool_call_id": "test_tool_call_id",
            "feedback": None,
            "final_tool_params": {"param1": "new_value"},
        }

    def test_from_dict(self):
        data = {
            "execute": False,
            "tool_name": "another_tool",
            "tool_call_id": "another_tool_call_id",
            "feedback": "Not needed",
            "final_tool_params": {"paramA": 123},
        }
        decision = ToolExecutionDecision.from_dict(data)
        assert decision.execute is False
        assert decision.tool_name == "another_tool"
        assert decision.tool_call_id == "another_tool_call_id"
        assert decision.feedback == "Not needed"
        assert decision.final_tool_params == {"paramA": 123}
