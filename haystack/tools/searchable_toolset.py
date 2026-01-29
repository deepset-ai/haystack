# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, Iterator

from haystack.core.serialization import generate_qualified_class_name, import_class_by_name
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.tools.tool import Tool
from haystack.tools.toolset import Toolset
from haystack.tools.utils import flatten_tools_or_toolsets

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
    from haystack.tools import Tool, SearchableToolset

    # Create a catalog of tools
    catalog = [
        Tool(name="get_weather", description="Get weather for a city", ...),
        Tool(name="search_web", description="Search the web", ...),
        # ... 100s more tools
    ]

    toolset = SearchableToolset(catalog=catalog)
    toolset.warm_up()

    agent = Agent(chat_generator=generator, tools=toolset)
    # LLM will use search_tools("weather") to find relevant tools
    ```
    """

    # Don't use dataclass fields - we manage our own state
    tools: list[Tool]

    def __init__(self, catalog: "ToolsType", *, top_k: int = 3, search_threshold: int = 8):
        """
        Initialize the SearchableToolset.

        :param catalog: Source of tools - a list of Tools, list of Toolsets, or a single Toolset.
        :param top_k: Default number of results for search_tools.
        :param search_threshold: Minimum catalog size to activate search.
            If catalog has fewer tools, acts as passthrough (all tools visible).
            Default is 8.
        """
        # Flatten catalog using standard utility
        self._catalog: list[Tool] = flatten_tools_or_toolsets(catalog)

        self._top_k = top_k
        self._search_threshold = search_threshold

        # Runtime state (initialized in warm_up)
        self._discovered_tools: dict[str, Tool] = {}
        self._bootstrap_tool: Tool | None = None
        self._document_store: InMemoryDocumentStore | None = None
        self._tool_by_name: dict[str, Tool] = {}
        self._warmed_up = False

        # Initialize parent with empty tools list - we manage tools dynamically
        super().__init__(tools=[])

    @property
    def is_passthrough(self) -> bool:
        """Check if operating in passthrough mode (small catalog)."""
        return len(self._catalog) < self._search_threshold

    def warm_up(self) -> None:
        """
        Prepare the toolset for use.

        This method indexes the catalog and creates the search_tools bootstrap tool.
        In passthrough mode, it warms up all catalog tools directly.
        Must be called before using the toolset with an Agent.
        """
        if self._warmed_up:
            return

        if self.is_passthrough:
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

        def search_tools(tool_keywords: str, k: int | None = None) -> str:
            """
            Search for tools matching tool_keywords and load them.

            :param tool_keywords: Keywords likely to appear in tool names/descriptions (not the user's request).
            :param k: Number of results to return (optional, defaults to top_k).
            :returns: Confirmation of loaded tools.
            """
            if self._document_store is None:
                return "Error: Document store not initialized. Call warm_up() first."

            num_results = k if k is not None else self._top_k

            if not tool_keywords.strip():
                return "No tools found matching these keywords. Try different keywords."

            results = self._document_store.bm25_retrieval(query=tool_keywords, top_k=num_results)

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
                tool = self._tool_by_name[doc.meta["tool_name"]]
                tool.warm_up()
                self._discovered_tools[tool.name] = tool
                tool_names.append(tool.name)

            return f"Found and loaded {len(tool_names)} tool(s): {', '.join(tool_names)}. Use them directly as tools."

        return Tool(
            name="search_tools",
            description="ALWAYS use this tool FIRST when you need to invoke some tools but don't have the right one "
            "loaded yet. Provide space separated tool keywords likely to appear in tool names/descriptions "
            "(e.g. 'route distance weather', 'search email'). Do NOT pass the user's request or task (e.g. "
            "'things to do in X', 'user question'); matching is keyword-based. Returns loaded "
            "tool names; they become available immediately.",
            parameters={
                "type": "object",
                "properties": {
                    "tool_keywords": {
                        "type": "string",
                        "description": "Space-separated words from tool names/descriptions (e.g. 'route weather "
                        "search'). NOT the user's question or task—use vocabulary from the tools you need.",
                    },
                    "k": {"type": "integer", "description": f"Number of results to return (default: {self._top_k})"},
                },
                "required": ["tool_keywords"],
            },
            function=search_tools,
        )

    def __iter__(self) -> Iterator[Tool]:
        """
        Iterate over available tools.

        In passthrough mode, yields all catalog tools.
        Otherwise, yields bootstrap tool + discovered tools.
        Automatically calls warm_up() if needed to ensure bootstrap tool is available.
        """
        if self.is_passthrough:
            yield from self._catalog
        else:
            # Auto warm-up to ensure bootstrap tool is available
            if not self._warmed_up:
                self.warm_up()
            if self._bootstrap_tool is not None:
                yield self._bootstrap_tool
            yield from self._discovered_tools.values()

    def __len__(self) -> int:
        """
        Return the number of currently available tools.

        In passthrough mode, returns catalog size.
        Otherwise, returns 1 (bootstrap) + discovered count.
        Automatically calls warm_up() if needed to ensure accurate count.
        """
        if self.is_passthrough:
            return len(self._catalog)
        # Auto warm-up to ensure bootstrap tool is counted
        if not self._warmed_up:
            self.warm_up()
        bootstrap_count = 1 if self._bootstrap_tool is not None else 0
        return bootstrap_count + len(self._discovered_tools)

    def __contains__(self, item: Any) -> bool:
        """
        Check if a tool is available.

        Supports checking by Tool instance or tool name string.
        Automatically calls warm_up() if needed.

        :param item: Tool instance or tool name string.
        :returns: True if the tool is available, False otherwise.
        """
        if self.is_passthrough:
            if isinstance(item, str):
                return any(tool.name == item for tool in self._catalog)
            return item in self._catalog if isinstance(item, Tool) else False

        # Auto warm-up to ensure bootstrap tool is available for checking
        if not self._warmed_up:
            self.warm_up()

        if isinstance(item, str):
            is_bootstrap = self._bootstrap_tool is not None and item == self._bootstrap_tool.name
            return is_bootstrap or item in self._discovered_tools
        if isinstance(item, Tool):
            is_bootstrap = self._bootstrap_tool is not None and item == self._bootstrap_tool
            return is_bootstrap or item in self._discovered_tools.values()
        return False

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
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {
                "catalog": [tool.to_dict() for tool in self._catalog],
                "top_k": self._top_k,
                "search_threshold": self._search_threshold,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchableToolset":
        """
        Deserialize a toolset from a dictionary.

        :param data: Dictionary representation of the toolset.
        :returns: New SearchableToolset instance.
        """
        inner_data = data["data"]

        # Deserialize catalog tools
        catalog_data = inner_data.get("catalog", [])
        catalog: list[Tool] = []
        for tool_data in catalog_data:
            tool_class = import_class_by_name(tool_data["type"])
            if not issubclass(tool_class, Tool):
                raise TypeError(f"Class '{tool_class}' is not a subclass of Tool")
            catalog.append(tool_class.from_dict(tool_data))

        return cls(
            catalog=catalog, top_k=inner_data.get("top_k", 3), search_threshold=inner_data.get("search_threshold", 8)
        )
