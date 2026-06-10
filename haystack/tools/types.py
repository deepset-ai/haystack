# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

from haystack.tools.tool import Tool
from haystack.tools.toolset import Toolset

# Type alias for tools parameter - allows mixing Tools and Toolsets in a sequence
# Accepts either:
# - Sequence[Tool | Toolset]: Any sequence (list, tuple, etc.) containing Tools, Toolsets, or a mix of both
# - Toolset: A single Toolset (not in a sequence)
ToolsType = Sequence[Tool | Toolset] | Toolset

__all__ = ["ToolsType"]
