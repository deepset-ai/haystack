# SPDX-FileCopyrightText: 2023-present deepset <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
#
# https://pylint.pycqa.org/en/latest/development_guide/how_tos/custom_checkers.html

from typing import TYPE_CHECKING, Optional, List, Any

from astroid import nodes

from pylint.checkers import BaseChecker

if TYPE_CHECKING:
    from pylint.lint import PyLinter


class DirectLoggingChecker(BaseChecker):
    name = "no-direct-logging"
    msgs = {
        "W9001": (
            "Use a logger object instead of a direct logging function like 'logging.%s()'",
            "no-direct-logging",
            "Do not use direct calls to logging functions like logging.info(), "
            "rather create a logger object with getLogger and use it instead. "
            "See https://github.com/deepset-ai/haystack/issues/4202.",
        )
    }

    def __init__(self, linter: Optional["PyLinter"] = None) -> None:
        super().__init__(linter)
        self._function_stack: List[Any] = []

    def visit_functiondef(self, node: nodes.FunctionDef) -> None:
        self._function_stack.append([])

    def leave_functiondef(self, node: nodes.FunctionDef) -> None:
        self._function_stack.pop()

    def visit_call(self, node: nodes.Call) -> None:
        if isinstance(node.func, nodes.Attribute) and isinstance(node.func.expr, nodes.Name):
            if node.func.expr.name == "logging" and node.func.attrname in [
                "debug",
                "info",
                "warning",
                "error",
                "critical",
                "exception",
            ]:
                self.add_message("no-direct-logging", args=node.func.attrname, node=node)


class NoLoggingConfigurationChecker(BaseChecker):
    name = "no-logging-basicconfig"
    msgs = {
        "W9002": (
            "Do not use 'logging.basicConfig' in Haystack code: Haystack should not configure any loggers.",
            "no-logging-basicconfig",
            "Do not configure the logger explicitly, because this would be problematic for users. "
            "Always configure the loggers only in scripts that use Haystack, like tutorials, rather than Haystack itself.",
        )
    }

    def __init__(self, linter: Optional["PyLinter"] = None) -> None:
        super().__init__(linter)
        self._function_stack: List[Any] = []

    def visit_functiondef(self, node: nodes.FunctionDef) -> None:
        self._function_stack.append([])

    def leave_functiondef(self, node: nodes.FunctionDef) -> None:
        self._function_stack.pop()

    def visit_call(self, node: nodes.Call) -> None:
        if isinstance(node.func, nodes.Attribute) and isinstance(node.func.expr, nodes.Name):
            if node.func.expr.name == "logging" and node.func.attrname in ["basicConfig"]:
                self.add_message("no-logging-basicconfig", node=node)


def register(linter: "PyLinter") -> None:
    """This required method auto registers the checker during initialization.
    :param linter: The linter to register the checker to.
    """
    linter.register_checker(DirectLoggingChecker(linter))
    linter.register_checker(NoLoggingConfigurationChecker(linter))
