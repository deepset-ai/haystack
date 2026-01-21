# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from threading import Lock
from typing import Any

from haystack.core.serialization import default_to_dict
from haystack.human_in_the_loop import ConfirmationUIResult
from haystack.human_in_the_loop.types import ConfirmationUI
from haystack.lazy_imports import LazyImport

with LazyImport(message="Run 'pip install rich'") as rich_import:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt

_ui_lock = Lock()


class RichConsoleUI(ConfirmationUI):
    """Rich console interface for user interaction."""

    def __init__(self, console: "Console | None" = None) -> None:
        rich_import.check()
        self.console = console or Console()

    def get_user_confirmation(
        self, tool_name: str, tool_description: str, tool_params: dict[str, Any]
    ) -> ConfirmationUIResult:
        """
        Get user confirmation for tool execution via rich console prompts.

        :param tool_name: The name of the tool to be executed.
        :param tool_description: The description of the tool.
        :param tool_params: The parameters to be passed to the tool.
        :returns: ConfirmationUIResult based on user input.
        """
        with _ui_lock:
            self._display_tool_info(tool_name, tool_description, tool_params)
            # If wrong input is provided, Prompt.ask will re-prompt
            choice = Prompt.ask("\nYour choice", choices=["y", "n", "m"], default="y", console=self.console)
            return self._process_choice(choice, tool_params)

    def _display_tool_info(self, tool_name: str, tool_description: str, tool_params: dict[str, Any]) -> None:
        """
        Display tool information and parameters in a rich panel.

        :param tool_name: The name of the tool to be executed.
        :param tool_description: The description of the tool.
        :param tool_params: The parameters to be passed to the tool.
        """
        lines = [
            f"[bold yellow]Tool:[/bold yellow] {tool_name}",
            f"[bold yellow]Description:[/bold yellow] {tool_description}",
            "\n[bold yellow]Arguments:[/bold yellow]",
        ]

        if tool_params:
            for k, v in tool_params.items():
                lines.append(f"[cyan]{k}:[/cyan] {v}")
        else:
            lines.append("  (No arguments)")

        self.console.print(Panel("\n".join(lines), title="🔧 Tool Execution Request", title_align="left"))

    def _process_choice(self, choice: str, tool_params: dict[str, Any]) -> ConfirmationUIResult:
        """
        Process the user's choice and return the corresponding ConfirmationUIResult.

        :param choice: The user's choice ('y', 'n', or 'm').
        :param tool_params: The original tool parameters.
        :returns:
            ConfirmationUIResult based on user input.
        """
        if choice == "y":
            return ConfirmationUIResult(action="confirm")
        elif choice == "m":
            return self._modify_params(tool_params)
        else:  # reject
            feedback = Prompt.ask("Feedback message (optional)", default="", console=self.console)
            return ConfirmationUIResult(action="reject", feedback=feedback or None)

    def _modify_params(self, tool_params: dict[str, Any]) -> ConfirmationUIResult:
        """
        Prompt the user to modify tool parameters.

        :param tool_params: The original tool parameters.
        :returns:
            ConfirmationUIResult with modified parameters.
        """
        new_params: dict[str, Any] = {}
        for k, v in tool_params.items():
            # We don't JSON dump strings to avoid users needing to input extra quotes
            default_val = json.dumps(v) if not isinstance(v, str) else v
            while True:
                new_val = Prompt.ask(f"Modify '{k}'", default=default_val, console=self.console)
                try:
                    if isinstance(v, str):
                        # Always treat input as string
                        new_params[k] = new_val
                    else:
                        # Parse JSON for all non-string types
                        new_params[k] = json.loads(new_val)
                    break
                except json.JSONDecodeError:
                    self.console.print("[red]❌ Invalid JSON, please try again.[/red]")

        return ConfirmationUIResult(action="modify", new_tool_params=new_params)

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the RichConsoleConfirmationUI to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        # Note: Console object is not serializable; we store None
        return default_to_dict(self, console=None)


class SimpleConsoleUI(ConfirmationUI):
    """Simple console interface using standard input/output."""

    def get_user_confirmation(
        self, tool_name: str, tool_description: str, tool_params: dict[str, Any]
    ) -> ConfirmationUIResult:
        """
        Get user confirmation for tool execution via simple console prompts.

        :param tool_name: The name of the tool to be executed.
        :param tool_description: The description of the tool.
        :param tool_params: The parameters to be passed to the tool.
        """
        with _ui_lock:
            self._display_tool_info(tool_name, tool_description, tool_params)
            valid_choices = {"y", "yes", "n", "no", "m", "modify"}
            while True:
                choice = input("Confirm execution? (y=confirm / n=reject / m=modify): ").strip().lower()
                if choice in valid_choices:
                    break
                print("Invalid input. Please enter 'y', 'n', or 'm'.")
            return self._process_choice(choice, tool_params)

    def _display_tool_info(self, tool_name: str, tool_description: str, tool_params: dict[str, Any]) -> None:
        """
        Display tool information and parameters in the console.

        :param tool_name: The name of the tool to be executed.
        :param tool_description: The description of the tool.
        :param tool_params: The parameters to be passed to the tool.
        """
        print("\n--- Tool Execution Request ---")
        print(f"Tool: {tool_name}")
        print(f"Description: {tool_description}")
        print("Arguments:")
        if tool_params:
            for k, v in tool_params.items():
                print(f"  {k}: {v}")
        else:
            print("  (No arguments)")
        print("-" * 30)

    def _process_choice(self, choice: str, tool_params: dict[str, Any]) -> ConfirmationUIResult:
        """
        Process the user's choice and return the corresponding ConfirmationUIResult.

        :param choice: The user's choice ('y', 'n', or 'm').
        :param tool_params: The original tool parameters.
        :returns:
            ConfirmationUIResult based on user input.
        """
        if choice in ("y", "yes"):
            return ConfirmationUIResult(action="confirm")
        elif choice in ("m", "modify"):
            return self._modify_params(tool_params)
        else:  # reject
            feedback = input("Feedback message (optional): ").strip()
            return ConfirmationUIResult(action="reject", feedback=feedback or None)

    def _modify_params(self, tool_params: dict[str, Any]) -> ConfirmationUIResult:
        """
        Prompt the user to modify tool parameters.

        :param tool_params: The original tool parameters.
        :returns:
            ConfirmationUIResult with modified parameters.
        """
        new_params: dict[str, Any] = {}

        if not tool_params:
            print("No parameters to modify, skipping modification.")
            return ConfirmationUIResult(action="modify", new_tool_params=new_params)

        for k, v in tool_params.items():
            # We don't JSON dump strings to avoid users needing to input extra quotes
            default_val = json.dumps(v) if not isinstance(v, str) else v
            while True:
                new_val = input(f"Modify '{k}' (current: {default_val}): ").strip() or default_val
                try:
                    if isinstance(v, str):
                        # Always treat input as string
                        new_params[k] = new_val
                    else:
                        # Parse JSON for all non-string types
                        new_params[k] = json.loads(new_val)
                    break
                except json.JSONDecodeError:
                    print("❌ Invalid JSON, please try again.")

        return ConfirmationUIResult(action="modify", new_tool_params=new_params)
