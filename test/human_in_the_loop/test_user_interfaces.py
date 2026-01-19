# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from haystack.components.agents.human_in_the_loop import RichConsoleUI, SimpleConsoleUI
from haystack.dataclasses import ConfirmationUIResult
from haystack.tools import create_tool_from_function


def multiply_tool(x: int) -> int:
    return x * 2


@pytest.fixture
def tool():
    return create_tool_from_function(
        function=multiply_tool, name="test_tool", description="A test tool that multiplies input by 2."
    )


class TestRichConsoleUI:
    @pytest.mark.parametrize("choice", ["y"])
    def test_process_choice_confirm(self, tool, choice):
        ui = RichConsoleUI(console=MagicMock())

        with patch(
            "haystack.components.agents.human_in_the_loop.user_interfaces.Prompt.ask", side_effect=[choice, "feedback"]
        ):
            result = ui.get_user_confirmation(tool.name, tool.description, {"x": 1})

        assert isinstance(result, ConfirmationUIResult)
        assert result.action == "confirm"
        assert result.new_tool_params is None
        assert result.feedback is None

    @pytest.mark.parametrize("choice", ["m"])
    def test_process_choice_modify(self, tool, choice):
        ui = RichConsoleUI(console=MagicMock())

        with patch("haystack.components.agents.human_in_the_loop.user_interfaces.Prompt.ask", side_effect=["m", "2"]):
            result = ui.get_user_confirmation(tool.name, tool.description, {"x": 1})

        assert isinstance(result, ConfirmationUIResult)
        assert result.action == "modify"
        assert result.new_tool_params == {"x": 2}

    def test_process_choice_modify_dict_param(self, tool):
        ui = RichConsoleUI(console=MagicMock())

        with patch(
            "haystack.components.agents.human_in_the_loop.user_interfaces.Prompt.ask",
            side_effect=["m", '{"key": "value"}'],
        ):
            result = ui.get_user_confirmation(tool.name, tool.description, {"param1": {"old_key": "old_value"}})

        assert isinstance(result, ConfirmationUIResult)
        assert result.action == "modify"
        assert result.new_tool_params == {"param1": {"key": "value"}}

    def test_process_choice_modify_dict_param_invalid_json(self, tool):
        ui = RichConsoleUI(console=MagicMock())

        with patch(
            "haystack.components.agents.human_in_the_loop.user_interfaces.Prompt.ask",
            side_effect=["m", "invalid_json", '{"key": "value"}'],
        ):
            result = ui.get_user_confirmation(tool.name, tool.description, {"param1": {"old_key": "old_value"}})

        assert isinstance(result, ConfirmationUIResult)
        assert result.action == "modify"
        assert result.new_tool_params == {"param1": {"key": "value"}}

    @pytest.mark.parametrize("choice", ["n"])
    def test_process_choice_reject(self, tool, choice):
        ui = RichConsoleUI(console=MagicMock())

        with patch(
            "haystack.components.agents.human_in_the_loop.user_interfaces.Prompt.ask",
            side_effect=["n", "Changed my mind"],
        ):
            result = ui.get_user_confirmation(tool.name, tool.description, {"x": 1})

        assert isinstance(result, ConfirmationUIResult)
        assert result.action == "reject"
        assert result.feedback == "Changed my mind"

    def test_to_dict(self):
        ui = RichConsoleUI()
        data = ui.to_dict()
        assert data["type"] == ("haystack.components.agents.human_in_the_loop.user_interfaces.RichConsoleUI")
        assert data["init_parameters"]["console"] is None

    def test_from_dict(self):
        ui = RichConsoleUI()
        data = ui.to_dict()
        new_ui = RichConsoleUI.from_dict(data)
        assert isinstance(new_ui, RichConsoleUI)


class TestSimpleConsoleUI:
    @pytest.mark.parametrize("choice", ["y", "yes", "Y", "YES"])
    def test_process_choice_confirm(self, tool, choice):
        ui = SimpleConsoleUI()

        with patch("builtins.input", side_effect=[choice]):
            result = ui.get_user_confirmation(tool.name, tool.description, {"y": "abc"})

        assert isinstance(result, ConfirmationUIResult)
        assert result.action == "confirm"

    @pytest.mark.parametrize("choice", ["m", "modify", "M", "MODIFY"])
    def test_process_choice_modify(self, tool, choice):
        ui = SimpleConsoleUI()

        with patch("builtins.input", side_effect=[choice, "new_value"]):
            result = ui.get_user_confirmation(tool.name, tool.description, {"y": "abc"})

        assert isinstance(result, ConfirmationUIResult)
        assert result.action == "modify"
        assert result.new_tool_params == {"y": "new_value"}

    def test_process_choice_modify_dict_param(self, tool):
        ui = SimpleConsoleUI()

        with patch("builtins.input", side_effect=["m", '{"key": "value"}']):
            result = ui.get_user_confirmation(tool.name, tool.description, {"param1": {"old_key": "old_value"}})

        assert isinstance(result, ConfirmationUIResult)
        assert result.action == "modify"
        assert result.new_tool_params == {"param1": {"key": "value"}}

    def test_process_choice_modify_dict_param_invalid_json(self, tool):
        ui = SimpleConsoleUI()

        with patch("builtins.input", side_effect=["m", "invalid_json", '{"key": "value"}']):
            result = ui.get_user_confirmation(tool.name, tool.description, {"param1": {"old_key": "old_value"}})

        assert isinstance(result, ConfirmationUIResult)
        assert result.action == "modify"
        assert result.new_tool_params == {"param1": {"key": "value"}}

    @pytest.mark.parametrize("choice", ["n", "no", "N", "NO"])
    def test_process_choice_reject(self, tool, choice):
        ui = SimpleConsoleUI()

        with patch("builtins.input", side_effect=[choice, "Changed my mind"]):
            result = ui.get_user_confirmation(tool.name, tool.description, {"param1": "value1"})

        assert isinstance(result, ConfirmationUIResult)
        assert result.action == "reject"
        assert result.feedback == "Changed my mind"

    def test_process_choice_no_tool_params_confirm(self, tool):
        ui = SimpleConsoleUI()

        with patch("builtins.input", side_effect=["y"]):
            result = ui.get_user_confirmation(tool.name, tool.description, {})

        assert isinstance(result, ConfirmationUIResult)
        assert result.action == "confirm"
        assert result.new_tool_params is None
        assert result.feedback is None

    def test_process_choice_no_tool_params_modify(self, tool):
        ui = SimpleConsoleUI()

        with patch("builtins.input", side_effect=["m"]):
            result = ui.get_user_confirmation(tool.name, tool.description, {})

        assert isinstance(result, ConfirmationUIResult)
        assert result.action == "modify"
        assert result.new_tool_params == {}
        assert result.feedback is None

    def test_process_choice_no_tool_params_reject(self, tool):
        ui = SimpleConsoleUI()

        with patch("builtins.input", side_effect=["n", "Changed my mind"]):
            result = ui.get_user_confirmation(tool.name, tool.description, {})

        assert isinstance(result, ConfirmationUIResult)
        assert result.action == "reject"
        assert result.new_tool_params is None
        assert result.feedback == "Changed my mind"

    def test_to_dict(self):
        ui = SimpleConsoleUI()
        data = ui.to_dict()
        assert data["type"] == ("haystack.components.agents.human_in_the_loop.user_interfaces.SimpleConsoleUI")
        assert data["init_parameters"] == {}

    def test_from_dict(self):
        ui = SimpleConsoleUI()
        data = ui.to_dict()
        new_ui = SimpleConsoleUI.from_dict(data)
        assert isinstance(new_ui, SimpleConsoleUI)

    def test_get_user_confirmation_invalid_input_then_valid(self, tool):
        ui = SimpleConsoleUI()

        with patch("builtins.input", side_effect=["invalid", "y"]):
            result = ui.get_user_confirmation(tool.name, tool.description, {"x": 1})

        assert isinstance(result, ConfirmationUIResult)
        assert result.action == "confirm"
        assert result.new_tool_params is None
        assert result.feedback is None
