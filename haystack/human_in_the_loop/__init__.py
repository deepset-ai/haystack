# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "dataclasses": ["ConfirmationUIResult", "ToolExecutionDecision"],
    "policies": ["AlwaysAskPolicy", "AskOncePolicy", "NeverAskPolicy"],
    "strategies": ["BlockingConfirmationStrategy"],
    "user_interfaces": ["RichConsoleUI", "SimpleConsoleUI"],
}

if TYPE_CHECKING:
    from .dataclasses import ConfirmationUIResult as ConfirmationUIResult
    from .dataclasses import ToolExecutionDecision as ToolExecutionDecision
    from .policies import AlwaysAskPolicy as AlwaysAskPolicy
    from .policies import AskOncePolicy as AskOncePolicy
    from .policies import NeverAskPolicy as NeverAskPolicy
    from .strategies import BlockingConfirmationStrategy as BlockingConfirmationStrategy
    from .user_interfaces import RichConsoleUI as RichConsoleUI
    from .user_interfaces import SimpleConsoleUI as SimpleConsoleUI

else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
