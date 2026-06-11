# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from typing import TYPE_CHECKING, Annotated, Any

from haystack.core.serialization import generate_qualified_class_name
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.tools.from_function import create_tool_from_function
from haystack.tools.serde_utils import deserialize_tools_or_toolset_inplace, serialize_tools_or_toolset
from haystack.tools.tool import Tool, _check_duplicate_tool_names
from haystack.tools.toolset import Toolset
from haystack.tools.utils import flatten_tools_or_toolsets, warm_up_tools

if TYPE_CHECKING:
    from haystack.tools import ToolsType


class SearchableToolset(Toolset):
    """
    Dynamic tool discovery from large catalogs using BM25 search.

    This Toolset enables LLMs to discover and use tools from large catalogs through BM25-based search.
    Instead of exposing all tools at once (which can overwhelm the LLM context), it provides a `search_tools` bootstrap
    tool that allows the LLM to find and load specific tools as needed.

    For very small catalogs (below `search_threshold`), acts as a simple passthrough exposing all tools directly
    without any discovery mechanism.

    ### Usage Example

    ```python
    from typing import Annotated

    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack.tools import SearchableToolset, tool

    @tool
    def get_weather(city: Annotated[str, "The city to get the weather for"]) -> str:
        '''Get the current weather for a city.'''
        return f"The weather in {city} is 22°C and sunny."

    @tool
    def search_web(query: Annotated[str, "The query to search the web for"]) -> str:
        '''Search the web for a query.'''
        return f"Top result for '{query}': ..."

    @tool
    def convert_currency(
        amount: Annotated[float, "The amount to convert"],
        to_currency: Annotated[str, "The currency to convert to, e.g. 'EUR'"],
    ) -> str:
        '''Convert an amount in USD to another currency.'''
        return f"{amount} USD is {amount * 0.9} {to_currency}"

    # search_threshold=2 means a catalog of 2+ tools activates discovery: the agent only sees the
    # `search_tools` tool and must search to load the others (set it higher for larger catalogs).
    toolset = SearchableToolset(catalog=[get_weather, search_web, convert_currency], search_threshold=2)

    agent = Agent(chat_generator=OpenAIChatGenerator(), tools=toolset)

    # The agent is initially provided only with the search_tools tool and will use it to find relevant tools.
    result = agent.run(messages=[ChatMessage.from_user("What's the weather in Milan?")])
    print(result["last_message"].text)
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
    ) -> None:
        """
        Initialize the SearchableToolset.

        :param catalog: Source of tools - a list of Tools, list of Toolsets, or a single Toolset.
        :param top_k: Default number of results for search_tools.
        :param search_threshold: Minimum catalog size to activate search. If catalog has fewer tools, acts as
            passthrough (all tools visible). Default is 8.
        :param search_tool_name: Custom name for the bootstrap search tool. Default is "search_tools".
        :param search_tool_description: Custom description for the bootstrap search tool. If not provided, uses a
            default description.
        :param search_tool_parameters_description: Custom descriptions for the bootstrap search tool's parameters.
            Keys must be a subset of `{"tool_keywords", "k"}`.
            Example: `{"tool_keywords": "Keywords to find tools, e.g. 'email send'"}`
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

        # Store raw catalog; flattening is deferred to warm_up() so that lazy toolsets
        # (e.g. MCPToolset with eager_connect=False) can connect first.
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
        self._passthrough: bool | None = None
        self._is_warmed_up = False

        # Initialize parent with empty tools list - we manage tools dynamically
        super().__init__(tools=[])

    def __add__(self, other: Tool | Toolset | list[Tool]) -> "Toolset":
        """Concatenation is not supported for SearchableToolset."""
        raise NotImplementedError("SearchableToolset does not support concatenation.")

    def add(self, tool: Tool | Toolset) -> None:
        """Adding new tools after initialization is not supported for SearchableToolset."""
        raise NotImplementedError("SearchableToolset does not support adding new tools after initialization.")

    def warm_up(self) -> None:
        """
        Prepare the toolset for use.

        Warms up the catalog (so lazy toolsets like MCPToolset can connect) and flattens it. Above the passthrough
        threshold, it also indexes the catalog and creates the search_tools bootstrap tool.

        This method is idempotent: it only warms up the toolset the first time it is called.

        :raises ValueError: If the flattened catalog contains tools with duplicate names.
        """
        if self._is_warmed_up:
            return

        # Warm up the catalog first (triggers lazy connections like MCPToolset), then flatten — lazy toolsets will
        # have their real tools available.
        warm_up_tools(self._raw_catalog)
        self._catalog = flatten_tools_or_toolsets(self._raw_catalog)
        _check_duplicate_tool_names(self._catalog)
        self._passthrough = len(self._catalog) < self._search_threshold

        # Build the BM25 search index only when the catalog is large enough to need discovery.
        if not self._passthrough:
            self._document_store = InMemoryDocumentStore()
            documents = [
                Document(content=f"{tool.name} {tool.description}", meta={"tool_name": tool.name})
                for tool in self._catalog
            ]
            self._document_store.write_documents(documents, policy=DuplicatePolicy.OVERWRITE)
            self._bootstrap_tool = self._create_search_tool()

        self._is_warmed_up = True

    def clear(self) -> None:
        """
        Clear all discovered tools.

        This method allows resetting the toolset's discovered tools between agent runs when the same toolset instance
        is reused. This can be useful for long-running applications to control memory usage or to start fresh searches.
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
            (e.g. 'route distance weather', 'search email').
            Do NOT pass the user's request or task (e.g. 'things to do in X', 'user question'); matching is
            keyword-based.
            Returns loaded tool names; they become available immediately.
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

            # Add found tools to _discovered_tools. These become available to the LLM on the next agent iteration
            # when __iter__ is called again - the Agent re-iterates over the toolset each loop, picking up newly
            # discovered tools.
            # The return message here just confirms what was found; actual tool availability comes through the dynamic
            # iteration mechanism. This way we also save tokens by not returning the full tool definitions.
            #
            # NOTE: The Agent can run tool calls in a step concurrently (ThreadPoolExecutor), so multiple search_tools
            # calls can mutate self._discovered_tools from different threads at once. This is currently safe only
            # because CPython's GIL makes individual dict assignments atomic; on a free-threaded (no-GIL) build these
            # unguarded writes could corrupt the dict.
            tool_names = []
            for doc in results:
                tool = tool_by_name[doc.meta["tool_name"]]
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
        # Unlike base Toolset/MCPToolset, which expose a placeholder tool before warm_up, this toolset materializes
        # everything (flattened catalog, bootstrap tool, passthrough decision) in warm_up.
        # Without warming here, iterating before warm_up would yield nothing, so we warm up to make the toolset usable
        # at all.
        if not self._is_warmed_up:
            self.warm_up()
        if self._passthrough:
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
        data: dict[str, Any] = {
            "catalog": serialize_tools_or_toolset(self._raw_catalog),
            "top_k": self._top_k,
            "search_threshold": self._search_threshold,
            "search_tool_name": self._search_tool_name,
            "search_tool_description": self._search_tool_description,
            "search_tool_parameters_description": self._search_tool_parameters_description,
        }

        return {"type": generate_qualified_class_name(type(self)), "data": data}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchableToolset":
        """
        Deserialize a toolset from a dictionary.

        :param data: Dictionary representation of the toolset.
        :returns: New SearchableToolset instance.
        :raises TypeError: If a serialized catalog entry is not a subclass of Tool or Toolset.
        """
        inner_data = data["data"]
        deserialize_tools_or_toolset_inplace(inner_data, key="catalog")
        optional_keys = (
            "top_k",
            "search_threshold",
            "search_tool_name",
            "search_tool_description",
            "search_tool_parameters_description",
        )
        return cls(catalog=inner_data["catalog"], **{k: inner_data[k] for k in optional_keys if k in inner_data})
