# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

# The `tool` decorator (from `from_function`) shares its name with the `tool` submodule (`tool.py`).
# `LazyImporter` registers every key in `import_structure` as a submodule symbol, so adding `"tool"` as a key collides
# with the `tool` decorator and raises a duplicate-symbol error at construction. We therefore eagerly import the
# decorator and pass it via `extra_objects`, keeping `"tool"` out of `import_structure`.
#
# Guidance for adding future exports:
#   - Symbols defined in `tool.py` (e.g. `Tool`) CANNOT be lazy: doing so would require `"tool"` as a key, which
#     re-introduces the collision with the `tool` decorator. Import them eagerly here and add them to `extra_objects`.
#   - Symbols from `from_function.py` (e.g. `create_tool_from_function`) could be lazy, but since we already eagerly
#     import this module for the `tool` decorator, we keep its exports eager in `extra_objects` too rather than
#     splitting them.
#   - Symbols from any other submodule have no name collision and should be added lazily to `import_structure` (plus
#     the `TYPE_CHECKING` block below) like the existing entries.
from haystack.tools.from_function import create_tool_from_function, tool
from haystack.tools.tool import Tool, _check_duplicate_tool_names

_import_structure = {
    "toolset": ["Toolset"],
    "searchable_toolset": ["SearchableToolset"],
    "component_tool": ["ComponentTool"],
    "pipeline_tool": ["PipelineTool"],
    "serde_utils": ["deserialize_tools_or_toolset_inplace", "serialize_tools_or_toolset"],
    "utils": ["flatten_tools_or_toolsets", "warm_up_tools"],
    "tool_types": ["ToolsType"],
}

if TYPE_CHECKING:
    from haystack.tools.component_tool import ComponentTool as ComponentTool
    from haystack.tools.pipeline_tool import PipelineTool as PipelineTool
    from haystack.tools.searchable_toolset import SearchableToolset as SearchableToolset
    from haystack.tools.serde_utils import deserialize_tools_or_toolset_inplace as deserialize_tools_or_toolset_inplace
    from haystack.tools.serde_utils import serialize_tools_or_toolset as serialize_tools_or_toolset
    from haystack.tools.tool_types import ToolsType as ToolsType
    from haystack.tools.toolset import Toolset as Toolset
    from haystack.tools.utils import flatten_tools_or_toolsets as flatten_tools_or_toolsets
    from haystack.tools.utils import warm_up_tools as warm_up_tools
else:
    sys.modules[__name__] = LazyImporter(
        name=__name__,
        module_file=__file__,
        import_structure=_import_structure,
        extra_objects={
            "create_tool_from_function": create_tool_from_function,
            "tool": tool,
            "Tool": Tool,
            "_check_duplicate_tool_names": _check_duplicate_tool_names,
        },
    )
