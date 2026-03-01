# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from typing import TYPE_CHECKING, Annotated, Any

from haystack.core.serialization import generate_qualified_class_name, import_class_by_name
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.tools.from_function import create_tool_from_function
from haystack.tools.tool import Tool
from haystack.tools.toolset import Toolset
from haystack.tools.utils import flatten_tools_or_toolsets, warm_up_tools

if TYPE_CHECKING:
    from haystack.tools import ToolsType


class SearchableToolset(Toolset):
    """
    Dynamic tool discovery from large catalogs using BM25 search.

    This Toolset enables LLMs to discover and use tools from large catalogs through
    BM25-based search. Instead of exposing all tools at once (which can overwhelm the
    LLM context), it provides a `search_tools` bootstrap tool that allows the LLM to
    find and load specific tools as needed.

    For very small catalogs (below `search_threshold`), acts as a simple passthrough
    exposing all tools directly without any discovery mechanism.

    ### Usage Example

    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack.tools import Tool, SearchableToolset

    # Create a catalog of tools
    catalog = [
        Tool(name="get_weather", description="Get weather for a city", ...),
        Tool(name="search_web", description="Search the web", ...),
        # ... 100s more tools
    ]
    toolset = SearchableToolset(catalog=catalog)

    agent = Agent(chat_generator=OpenAIChatGenerator(), tools=toolset)

    # The agent is initially provided only with the search_tools tool and will use it to find relevant tools.
    result = agent.run(messages=[ChatMessage.from_user("What's the weather in Milan?")])
    ```
    """

    _VALID_SEARCH_TOOL_PARAMS = {"tool_keywords", "k"}

    def __init__(
        self,
        catalog: "ToolsType",
        *,
        top_k: int = 3,
        search_threshold: int = 8,
        search_tool_name: str = "search_tools",
        search_tool_description: str | None = None,
        search_tool_parameters_description: dict[str, str] | None = None,
    ):
        """
        Initialize the SearchableToolset.

        :param catalog: Source of tools - a list of Tools, list of Toolsets, or a single Toolset.
        :param top_k: Default number of results for search_tools.
        :param search_threshold: Minimum catalog size to activate search.
            If catalog has fewer tools, acts as passthrough (all tools visible).
            Default is 8.
        :param search_tool_name: Custom name for the bootstrap search tool. Default is "search_tools".
        :param search_tool_description: Custom description for the bootstrap search tool.
            If not provided, uses a default description.
        :param search_tool_parameters_description: Custom descriptions for the bootstrap search tool's parameters.
            Keys must be a subset of {"tool_keywords", "k"}.
            Example: {"tool_keywords": "Keywords to find tools, e.g. 'email send'"}
        """
        valid_catalog = isinstance(catalog, Toolset) or (
            isinstance(catalog, list) and all(isinstance(item, (Tool, Toolset)) for item in catalog)
        )
        if not valid_catalog:
            raise TypeError(
                f"Invalid catalog type: {type(catalog)}. Expected Tool, Toolset, or list of Tools and/or Toolsets."
            )

        if search_tool_parameters_description is not None:
            invalid_keys = set(search_tool_parameters_description.keys()) - self._VALID_SEARCH_TOOL_PARAMS
            if invalid_keys:
                raise ValueError(
                    f"Invalid search_tool_parameters_description keys: {invalid_keys}. "
                    f"Valid keys are: {self._VALID_SEARCH_TOOL_PARAMS}"
                )

        # Store raw catalog; flattening is deferred to warm_up() so that lazy
        # toolsets (e.g. MCPToolset with eager_connect=False) can connect first.
        self._raw_catalog: "ToolsType" = catalog
        self._catalog: list[Tool] = []

        self._top_k = top_k
        self._search_threshold = search_threshold
        self._search_tool_name = search_tool_name
        self._search_tool_description = search_tool_description
        self._search_tool_parameters_description = search_tool_parameters_description

        # Runtime state (initialized in warm_up)
        self._discovered_tools: dict[str, Tool] = {}
        self._bootstrap_tool: Tool | None = None
        self._document_store: InMemoryDocumentStore | None = None
        self._warmed_up = False

        # Initialize parent with empty tools list - we manage tools dynamically
        super().__init__(tools=[])

    def __add__(self, other: Tool | Toolset | list[Tool]) -> "Toolset":
        """Concatenation is not supported for SearchableToolset."""
        raise NotImplementedError("SearchableToolset does not support concatenation.")

    def add(self, tool: Tool | Toolset) -> None:
        """Adding new tools after initialization is not supported for SearchableToolset."""
        raise NotImplementedError("SearchableToolset does not support adding new tools after initialization.")

    def _is_passthrough(self) -> bool:
        """
        Internal method to check if operating in passthrough mode (small catalog). Must be called after warm_up().
        """
        return len(self._catalog) < self._search_threshold

    def warm_up(self) -> None:
        """
        Prepare the toolset for use.

        Warms up child toolsets first (so lazy toolsets like MCPToolset can connect),
        then flattens the catalog, indexes it, and creates the search_tools bootstrap tool.
        In passthrough mode, it warms up all catalog tools directly.
        Must be called before using the toolset with an Agent.
        """
        if self._warmed_up:
            return

        # Warm up child toolsets first (triggers lazy connections like MCPToolset)
        warm_up_tools(self._raw_catalog)
        # Now flatten — lazy toolsets will have their real tools available
        self._catalog = flatten_tools_or_toolsets(self._raw_catalog)

        if self._is_passthrough():
            for tool in self._catalog:
                tool.warm_up()
        else:
            self._document_store = InMemoryDocumentStore()
            self._tool_by_name = {tool.name: tool for tool in self._catalog}
            documents = [
                Document(content=f"{tool.name} {tool.description}", meta={"tool_name": tool.name})
                for tool in self._catalog
            ]
            self._document_store.write_documents(documents, policy=DuplicatePolicy.OVERWRITE)
            self._bootstrap_tool = self._create_search_tool()

        self._warmed_up = True

    def clear(self) -> None:
        """
        Clear all discovered tools.

        This method allows resetting the toolset's discovered tools between agent runs
        when the same toolset instance is reused. This can be useful for long-running
        applications to control memory usage or to start fresh searches.
        """
        self._discovered_tools.clear()

    def _create_search_tool(self) -> Tool:
        """Create the search_tools bootstrap tool."""

        tool_by_name = {tool.name: tool for tool in self._catalog}

        def search_tools(
            tool_keywords: Annotated[
                str,
                "Space-separated words from tool names/descriptions (e.g. 'route weather search')."
                " NOT the user's question or task—use vocabulary from the tools you need.",
            ],
            k: Annotated[int | None, f"Number of results to return (default: {self._top_k})"] = None,
        ) -> str:
            """
            ALWAYS use this tool FIRST when you need to invoke some tools but don't have the right one loaded yet.

            Provide space separated tool keywords likely to appear in tool names/descriptions
            (e.g. 'route distance weather', 'search email'). Do NOT pass the user's request or task (e.g.
            'things to do in X', 'user question'); matching is keyword-based. Returns loaded
            tool names; they become available immediately.
            """
            num_results = k if k is not None else self._top_k

            if not tool_keywords.strip():
                return (
                    "No tool keywords provided. Please provide space-separated words likely to appear in tool "
                    "names/descriptions (e.g. 'route weather search')."
                )

            # at this point, the toolset has been warmed up, so self._document_store is not None
            results = self._document_store.bm25_retrieval(query=tool_keywords, top_k=num_results)  # type: ignore[union-attr]

            if not results:
                return "No tools found matching these keywords. Try different keywords."

            # Add found tools to _discovered_tools. These become available to the LLM
            # on the next agent iteration when __iter__ is called again - the Agent
            # re-iterates over the toolset each loop, picking up newly discovered tools.
            # The return message here just confirms what was found; actual tool availability
            # comes through the dynamic iteration mechanism. This way we also save tokens
            # by not returning the full tool definitions.
            tool_names = []
            for doc in results:
                tool = tool_by_name[doc.meta["tool_name"]]
                tool.warm_up()
                self._discovered_tools[tool.name] = tool
                tool_names.append(tool.name)

            return f"Found and loaded {len(tool_names)} tool(s): {', '.join(tool_names)}. Use them directly as tools."

        bootstrap_tool = create_tool_from_function(
            function=search_tools, name=self._search_tool_name, description=self._search_tool_description
        )

        # Override parameter descriptions if custom ones were provided
        if self._search_tool_parameters_description:
            for param_name, desc in self._search_tool_parameters_description.items():
                if param_name in bootstrap_tool.parameters.get("properties", {}):
                    bootstrap_tool.parameters["properties"][param_name]["description"] = desc

        return bootstrap_tool

    def __iter__(self) -> Iterator[Tool]:
        """
        Iterate over available tools.

        In passthrough mode, yields all catalog tools.
        Otherwise, yields bootstrap tool + discovered tools.
        Automatically calls warm_up() if needed to ensure bootstrap tool is available.
        """
        if not self._warmed_up:
            self.warm_up()
        if self._is_passthrough():
            yield from self._catalog
        else:
            if self._bootstrap_tool is not None:
                yield self._bootstrap_tool
            yield from self._discovered_tools.values()

    def __len__(self) -> int:
        """Return the number of currently available tools."""
        # the number of tools is computed by invoking __iter__ on the toolset
        return sum(1 for _ in self)

    def __contains__(self, item: str | Tool) -> bool:
        """
        Check if a tool is available by Tool instance or tool name string.

        :param item: Tool instance or tool name string.
        :returns: True if the tool is available, False otherwise.
        """
        if isinstance(item, str):
            return any(tool.name == item for tool in self)
        if isinstance(item, Tool):
            return any(tool == item for tool in self)
        raise TypeError(f"Invalid item type: {type(item)}. Must be Tool or str.")

    def __getitem__(self, index: int) -> Tool:
        """
        Get a tool by index.

        :param index: Index of the tool to retrieve.
        :returns: The tool at the given index.
        :raises IndexError: If the index is out of range.
        """
        return list(self)[index]

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the toolset to a dictionary.

        :returns: Dictionary representation of the toolset.
        """
        catalog_items: list[Tool | Toolset] = (
            [self._raw_catalog] if isinstance(self._raw_catalog, Toolset) else list(self._raw_catalog)
        )

        data: dict[str, Any] = {
            "catalog": [item.to_dict() for item in catalog_items],
            "top_k": self._top_k,
            "search_threshold": self._search_threshold,
            "search_tool_name": self._search_tool_name,
        }
        if self._search_tool_description is not None:
            data["search_tool_description"] = self._search_tool_description
        if self._search_tool_parameters_description is not None:
            data["search_tool_parameters_description"] = self._search_tool_parameters_description

        return {"type": generate_qualified_class_name(type(self)), "data": data}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchableToolset":
        """
        Deserialize a toolset from a dictionary.

        :param data: Dictionary representation of the toolset.
        :returns: New SearchableToolset instance.
        """
        inner_data = data["data"]

        # Deserialize catalog items (may be Tool or Toolset instances)
        catalog_data = inner_data.get("catalog", [])
        catalog: list[Tool | Toolset] = []
        for item_data in catalog_data:
            item_class = import_class_by_name(item_data["type"])
            if not issubclass(item_class, (Tool, Toolset)):
                raise TypeError(f"Class '{item_class}' is not a subclass of Tool or Toolset")
            catalog.append(item_class.from_dict(item_data))

        optional_keys = (
            "top_k",
            "search_threshold",
            "search_tool_name",
            "search_tool_description",
            "search_tool_parameters_description",
        )
        return cls(catalog=catalog, **{k: inner_data[k] for k in optional_keys if k in inner_data})
