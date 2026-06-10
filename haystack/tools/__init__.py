# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyModule
from lazy_imports.util import as_package

# The `tool` decorator (exported from `from_function`) shares its name with the `tool` submodule (`tool.py`).
# This is why we cannot lazily import everything: importing the `tool` submodule makes Python bind it as the `tool`
# attribute of this package, which would shadow the decorator.
# We get around this by importing `tool` and `from_function` eagerly and passing the decorator as a plain attribute,
# while the heavier modules are imported lazily.
from haystack.tools.from_function import create_tool_from_function, tool
from haystack.tools.tool import Tool, _check_duplicate_tool_names

if TYPE_CHECKING:
    from haystack.tools.component_tool import ComponentTool as ComponentTool
    from haystack.tools.pipeline_tool import PipelineTool as PipelineTool
    from haystack.tools.searchable_toolset import SearchableToolset as SearchableToolset
    from haystack.tools.serde_utils import deserialize_tools_or_toolset_inplace as deserialize_tools_or_toolset_inplace
    from haystack.tools.serde_utils import serialize_tools_or_toolset as serialize_tools_or_toolset
    from haystack.tools.toolset import Toolset as Toolset
    from haystack.tools.types import ToolsType as ToolsType
    from haystack.tools.utils import flatten_tools_or_toolsets as flatten_tools_or_toolsets
    from haystack.tools.utils import warm_up_tools as warm_up_tools
else:
    sys.modules[__name__] = LazyModule(
        "from haystack.tools.component_tool import ComponentTool",
        "from haystack.tools.pipeline_tool import PipelineTool",
        "from haystack.tools.searchable_toolset import SearchableToolset",
        "from haystack.tools.serde_utils import deserialize_tools_or_toolset_inplace, serialize_tools_or_toolset",
        "from haystack.tools.toolset import Toolset",
        "from haystack.tools.types import ToolsType",
        "from haystack.tools.utils import flatten_tools_or_toolsets, warm_up_tools",
        # Eagerly imported above; passed as plain attributes so they survive the module replacement and keep the `tool`
        # decorator from being shadowed by the `tool` submodule.
        ("create_tool_from_function", create_tool_from_function),
        ("tool", tool),
        ("Tool", Tool),
        ("_check_duplicate_tool_names", _check_duplicate_tool_names),
        *as_package(__file__),
        name=__name__,
        doc=__doc__,
    )
