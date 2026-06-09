# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


import os
from collections.abc import Callable
from typing import Annotated, Any

import pytest

from haystack import component
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import SearchableToolset, Tool, Toolset, flatten_tools_or_toolsets
from haystack.tools.from_function import create_tool_from_function


# Test helper functions
def get_weather(city: Annotated[str, "The city to get the weather for"]) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: 22°C, sunny"


def add_numbers(a: Annotated[int, "The first number"], b: Annotated[int, "The second number"]) -> int:
    """Add two numbers together."""
    return a + b


def multiply_numbers(a: Annotated[int, "The first number"], b: Annotated[int, "The second number"]) -> int:
    """Multiply two numbers."""
    return a * b


def get_stock_price(symbol: Annotated[str, "The ticker symbol, e.g. 'AAPL'"]) -> str:
    """Get stock price by ticker symbol."""
    return f"{symbol}: $150.00"


def search_database(query: Annotated[str, "The query to search records for"]) -> str:
    """Search the database for records."""
    return f"Found 5 records matching '{query}'"


def send_email(
    to: Annotated[str, "The recipient email address"],
    subject: Annotated[str, "The email subject"],
    body: Annotated[str, "The email body"],
) -> str:
    """Send an email to a recipient."""
    return f"Email sent to {to}"


def calculate_tax(
    amount: Annotated[float, "The amount to tax"], rate: Annotated[float, "The tax rate as a fraction, e.g. 0.2"]
) -> float:
    """Calculate tax on an amount."""
    return amount * rate


def convert_currency(
    amount: Annotated[float, "The amount to convert"],
    from_currency: Annotated[str, "The source currency, e.g. 'USD'"],
    to_currency: Annotated[str, "The target currency, e.g. 'EUR'"],
) -> float:
    """Convert currency from one to another."""
    return amount * 1.1  # Simplified conversion


# Test fixtures
@pytest.fixture
def weather_tool():
    return create_tool_from_function(get_weather)


@pytest.fixture
def add_tool():
    return create_tool_from_function(add_numbers)


@pytest.fixture
def multiply_tool():
    return create_tool_from_function(multiply_numbers)


@pytest.fixture
def stock_tool():
    return create_tool_from_function(get_stock_price)


@pytest.fixture
def small_catalog(weather_tool, add_tool, multiply_tool):
    """Small catalog that triggers passthrough mode (< 8 tools)."""
    return [weather_tool, add_tool, multiply_tool]


@pytest.fixture
def large_catalog():
    """Larger catalog that requires discovery (>= 8 tools)."""
    functions: list[Callable[..., Any]] = [
        get_weather,
        add_numbers,
        multiply_numbers,
        get_stock_price,
        search_database,
        send_email,
        calculate_tax,
        convert_currency,
    ]
    return [create_tool_from_function(fn) for fn in functions]


class TestSearchableToolset:
    def test_init_with_invalid_catalog(self):
        with pytest.raises(TypeError):
            SearchableToolset(catalog=123)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            SearchableToolset(catalog=[123])  # type: ignore[list-item]
        with pytest.raises(TypeError):
            SearchableToolset(
                catalog=Tool(  # type: ignore[arg-type]
                    name="test",
                    description="test",
                    parameters={"type": "object", "properties": {}},
                    function=lambda: None,
                )
            )

    def test_not_implemented_methods(self):
        toolset = SearchableToolset(catalog=[])
        with pytest.raises(NotImplementedError):
            toolset + Tool(
                name="test", description="test", parameters={"type": "object", "properties": {}}, function=lambda: None
            )
        with pytest.raises(NotImplementedError):
            toolset.add(
                Tool(
                    name="test",
                    description="test",
                    parameters={"type": "object", "properties": {}},
                    function=lambda: None,
                )
            )

    def test_clear(self, large_catalog):
        toolset = SearchableToolset(catalog=large_catalog)
        toolset.warm_up()
        assert toolset._bootstrap_tool is not None
        toolset._bootstrap_tool.invoke(tool_keywords="weather temperature city")
        assert len(toolset._discovered_tools) > 0
        toolset.clear()
        assert len(toolset._discovered_tools) == 0


class TestSearchableToolsetPassthrough:
    """Tests for passthrough mode (small catalogs)."""

    def test_passthrough_mode_detected(self, small_catalog):
        """Test that small catalogs trigger passthrough mode."""
        toolset = SearchableToolset(catalog=small_catalog)
        toolset.warm_up()
        assert toolset._passthrough is True

    def test_passthrough_exposes_all_tools(self, small_catalog):
        """Test that passthrough mode exposes all catalog tools."""
        toolset = SearchableToolset(catalog=small_catalog)
        toolset.warm_up()
        assert len(toolset) == len(small_catalog)
        tool_names = [tool.name for tool in toolset]
        assert "get_weather" in tool_names
        assert "add_numbers" in tool_names
        assert "multiply_numbers" in tool_names

    def test_passthrough_no_bootstrap_tool(self, small_catalog):
        """Test that passthrough mode neither creates nor exposes the search bootstrap tool."""
        toolset = SearchableToolset(catalog=small_catalog)
        toolset.warm_up()
        assert toolset._bootstrap_tool is None
        # The search tool must never be present in passthrough mode.
        assert "search_tools" not in toolset
        assert all(tool.name != "search_tools" for tool in toolset)

    def test_passthrough_contains_by_name(self, small_catalog):
        """Test __contains__ by name in passthrough mode."""
        toolset = SearchableToolset(catalog=small_catalog)
        toolset.warm_up()
        assert "get_weather" in toolset
        assert "nonexistent" not in toolset

    def test_passthrough_contains_by_tool(self, small_catalog, weather_tool):
        """Test __contains__ by tool instance in passthrough mode."""
        toolset = SearchableToolset(catalog=small_catalog)
        toolset.warm_up()
        assert weather_tool in toolset

    def test_passthrough_contains_by_tool_invalid_type(self, small_catalog):
        toolset = SearchableToolset(catalog=small_catalog)
        toolset.warm_up()

        with pytest.raises(TypeError):
            123 in toolset  # type: ignore[operator] # noqa: B015

    def test_custom_search_threshold(self, large_catalog):
        """Test that custom search_threshold changes passthrough behavior."""
        # With threshold of 10, 8 tools should be passthrough
        toolset = SearchableToolset(catalog=large_catalog, search_threshold=10)
        toolset.warm_up()
        assert toolset._passthrough is True
        assert len(list(toolset)) == 8


class TestSearchableToolsetBM25Mode:
    """Tests for BM25 discovery mode."""

    def test_bm25_mode_creates_search_tools(self, large_catalog):
        """Test that BM25 mode creates search_tools bootstrap tool."""
        toolset = SearchableToolset(catalog=large_catalog)
        toolset.warm_up()

        assert toolset._bootstrap_tool is not None
        assert toolset._bootstrap_tool.name == "search_tools"

    def test_bm25_mode_initializes_document_store(self, large_catalog):
        """Test that BM25 mode initializes document store."""
        toolset = SearchableToolset(catalog=large_catalog)
        toolset.warm_up()

        assert toolset._document_store is not None

    def test_search_tools_finds_relevant_tools(self, large_catalog):
        """Test that search_tools finds relevant tools."""
        toolset = SearchableToolset(catalog=large_catalog, top_k=3)
        toolset.warm_up()
        assert toolset._bootstrap_tool is not None

        result = toolset._bootstrap_tool.invoke(tool_keywords="weather temperature city")

        assert "get_weather" in result
        assert "get_weather" in toolset._discovered_tools

    def test_search_tools_auto_loads(self, large_catalog):
        """Test that search_tools auto-loads found tools."""
        toolset = SearchableToolset(catalog=large_catalog, top_k=3)
        toolset.warm_up()
        assert toolset._bootstrap_tool is not None

        # Initially only bootstrap tool
        assert len(toolset) == 1

        toolset._bootstrap_tool.invoke(tool_keywords="add numbers multiply")

        # Should have bootstrap + discovered tools
        assert len(toolset) > 1

    def test_search_tools_respects_k(self, large_catalog):
        """Test that search_tools respects k parameter."""
        toolset = SearchableToolset(catalog=large_catalog, top_k=2)
        toolset.warm_up()
        assert toolset._bootstrap_tool is not None

        # Search with explicit k=1
        result = toolset._bootstrap_tool.invoke(tool_keywords="add numbers together", k=1)

        # Should find exactly 1 tool
        assert "Found and loaded 1 tool(s):" in result

    def test_search_tools_no_results(self, large_catalog):
        """Test search_tools with no matching results."""
        toolset = SearchableToolset(catalog=large_catalog)
        toolset.warm_up()
        assert toolset._bootstrap_tool is not None

        result = toolset._bootstrap_tool.invoke(tool_keywords="xyznonexistent123")

        assert "No tools found" in result
        assert len(toolset._discovered_tools) == 0

    def test_search_tools_no_keywords(self, large_catalog):
        """Test search_tools with no keywords."""
        toolset = SearchableToolset(catalog=large_catalog)
        toolset.warm_up()
        assert toolset._bootstrap_tool is not None

        result = toolset._bootstrap_tool.invoke(tool_keywords="")
        assert "No tool keywords provided" in result
        assert len(toolset._discovered_tools) == 0


class TestSearchableToolsetIteration:
    """Tests for iteration and collection behavior."""

    def test_iter_passthrough(self, small_catalog):
        """Test iteration in passthrough mode."""
        toolset = SearchableToolset(catalog=small_catalog)
        toolset.warm_up()

        tools = list(toolset)

        assert len(tools) == len(small_catalog)

    def test_iter_with_discovered_tools(self, large_catalog):
        """Test iteration with discovered tools."""
        toolset = SearchableToolset(catalog=large_catalog)
        toolset.warm_up()
        assert toolset._bootstrap_tool is not None

        # Search for tools
        toolset._bootstrap_tool.invoke(tool_keywords="weather")
        toolset._bootstrap_tool.invoke(tool_keywords="math addition")

        tools = list(toolset)

        assert len(tools) >= 2  # bootstrap + discovered tools
        tool_names = [t.name for t in tools]
        assert "search_tools" in tool_names
        assert "get_weather" in tool_names

    def test_iter_automatically_warms_up(self, large_catalog):
        toolset = SearchableToolset(catalog=large_catalog)
        assert not toolset._is_warmed_up

        list(toolset)
        assert toolset._is_warmed_up

    def test_contains_bootstrap_tool(self, large_catalog):
        """Test __contains__ for bootstrap tool."""
        toolset = SearchableToolset(catalog=large_catalog)
        toolset.warm_up()

        assert "search_tools" in toolset
        assert toolset._bootstrap_tool is not None
        assert toolset._bootstrap_tool in toolset

    def test_contains_discovered_tool(self, large_catalog):
        """Test __contains__ for discovered tools."""
        toolset = SearchableToolset(catalog=large_catalog, top_k=1)
        toolset.warm_up()
        assert toolset._bootstrap_tool is not None

        toolset._bootstrap_tool.invoke(tool_keywords="weather")

        assert "get_weather" in toolset
        assert "add_numbers" not in toolset  # Not discovered yet

    def test_getitem(self, large_catalog):
        toolset = SearchableToolset(catalog=large_catalog)
        toolset.warm_up()

        tool = toolset[0]
        assert tool.name == "search_tools"


class TestSearchableToolsetSerialization:
    """Tests for serialization and deserialization."""

    def test_to_dict(self, large_catalog):
        """Test serialization to dict."""
        toolset = SearchableToolset(catalog=large_catalog, top_k=3, search_threshold=5)

        data = toolset.to_dict()

        assert "type" in data
        assert "haystack.tools.searchable_toolset.SearchableToolset" in data["type"]
        assert "data" in data
        assert data["data"]["top_k"] == 3
        assert data["data"]["search_threshold"] == 5
        assert len(data["data"]["catalog"]) == len(large_catalog)

    def test_to_dict_with_toolset(self, large_catalog):
        toolset = Toolset(tools=large_catalog)

        searchable_toolset = SearchableToolset(catalog=toolset)
        data = searchable_toolset.to_dict()
        assert "type" in data
        assert "haystack.tools.searchable_toolset.SearchableToolset" in data["type"]
        assert "data" in data
        assert data["data"]["top_k"] == 3
        assert data["data"]["search_threshold"] == 8
        # A single Toolset catalog serializes as that toolset, not wrapped in a list.
        assert isinstance(data["data"]["catalog"], dict)
        assert data["data"]["catalog"]["type"] == "haystack.tools.toolset.Toolset"

    def test_serde_roundtrip_with_toolset_catalog(self, large_catalog):
        """A single Toolset catalog round-trips back to a Toolset, not a list."""
        searchable_toolset = SearchableToolset(catalog=Toolset(tools=large_catalog))
        restored = SearchableToolset.from_dict(searchable_toolset.to_dict())
        assert isinstance(restored._raw_catalog, Toolset)
        restored.warm_up()
        assert len(restored._catalog) == len(large_catalog)

    def test_from_dict(self, large_catalog):
        """Test deserialization from dict."""
        toolset = SearchableToolset(catalog=large_catalog, top_k=3)

        data = toolset.to_dict()
        restored = SearchableToolset.from_dict(data)
        restored.warm_up()

        assert restored._top_k == 3
        assert len(restored._catalog) == len(large_catalog)

    def test_from_dict_with_invalid_item_type(self):
        data = {
            "type": "haystack.tools.searchable_toolset.SearchableToolset",
            "data": {"catalog": [{"type": "haystack.dataclasses.Document", "data": "irrelevant"}]},
        }
        with pytest.raises(TypeError):
            SearchableToolset.from_dict(data)

    def test_serde_roundtrip(self, large_catalog):
        """Test full serialization roundtrip."""
        toolset = SearchableToolset(catalog=large_catalog, top_k=5, search_threshold=6)
        toolset.warm_up()

        # Serialize
        data = toolset.to_dict()

        # Deserialize
        restored = SearchableToolset.from_dict(data)
        restored.warm_up()
        assert restored._bootstrap_tool is not None

        # Verify behavior matches
        assert restored._passthrough == toolset._passthrough

        # Verify bootstrap tool works
        result = restored._bootstrap_tool.invoke(tool_keywords="weather")
        assert "get_weather" in result
        assert "get_weather" in restored._discovered_tools

    def test_serde_preserves_catalog_tools(self, large_catalog):
        """Test that serialization preserves catalog tool functionality."""
        toolset = SearchableToolset(catalog=large_catalog)
        toolset.warm_up()

        data = toolset.to_dict()
        restored = SearchableToolset.from_dict(data)
        restored.warm_up()
        assert restored._bootstrap_tool is not None

        # Search and invoke a tool
        result_text = restored._bootstrap_tool.invoke(tool_keywords="add numbers")
        assert "add_numbers" in result_text
        add_tool = restored._discovered_tools["add_numbers"]
        result = add_tool.invoke(a=10, b=5)

        assert result == 15


class TestSearchableToolsetWithToolset:
    """Tests for using a Toolset as catalog input."""

    def test_accepts_toolset_as_catalog(self, small_catalog):
        """Test that a Toolset can be used as catalog."""
        base_toolset = Toolset(tools=small_catalog)
        search_toolset = SearchableToolset(catalog=base_toolset, search_threshold=10)
        search_toolset.warm_up()

        assert len(search_toolset._catalog) == len(small_catalog)

    def test_toolset_catalog_passthrough(self, small_catalog):
        """Test passthrough mode with Toolset catalog."""
        base_toolset = Toolset(tools=small_catalog)
        search_toolset = SearchableToolset(catalog=base_toolset)
        search_toolset.warm_up()

        assert search_toolset._passthrough is True
        assert len(list(search_toolset)) == len(small_catalog)

    def test_accepts_list_of_toolsets(self, weather_tool, add_tool, multiply_tool, stock_tool):
        """Test that a list of Toolsets can be used as catalog."""
        toolset1 = Toolset(tools=[weather_tool, add_tool])
        toolset2 = Toolset(tools=[multiply_tool, stock_tool])

        search_toolset = SearchableToolset(catalog=[toolset1, toolset2], search_threshold=10)
        search_toolset.warm_up()

        assert len(search_toolset._catalog) == 4
        assert any(t.name == "get_weather" for t in search_toolset._catalog)
        assert any(t.name == "add_numbers" for t in search_toolset._catalog)
        assert any(t.name == "multiply_numbers" for t in search_toolset._catalog)
        assert any(t.name == "get_stock_price" for t in search_toolset._catalog)

    def test_accepts_mixed_list(self, weather_tool, add_tool, multiply_tool):
        """Test that a mixed list of Tools and Toolsets can be used as catalog."""
        toolset = Toolset(tools=[add_tool, multiply_tool])

        search_toolset = SearchableToolset(catalog=[weather_tool, toolset], search_threshold=10)
        search_toolset.warm_up()

        assert len(search_toolset._catalog) == 3
        assert any(t.name == "get_weather" for t in search_toolset._catalog)
        assert any(t.name == "add_numbers" for t in search_toolset._catalog)
        assert any(t.name == "multiply_numbers" for t in search_toolset._catalog)


class TestSearchableToolsetWarmUp:
    """Tests for warm_up behavior."""

    def test_not_warmed_up_after_agent_init(self, large_catalog, monkeypatch):
        """Initializing an Agent with a SearchableToolset must not warm it up (no premature flatten/connect)."""
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        toolset = SearchableToolset(catalog=large_catalog)
        assert toolset._is_warmed_up is False

        Agent(chat_generator=OpenAIChatGenerator(), tools=toolset)

        assert toolset._is_warmed_up is False

    def test_warm_up_idempotent(self, large_catalog):
        """Test that warm_up can be called multiple times safely."""
        toolset = SearchableToolset(catalog=large_catalog)

        toolset.warm_up()
        first_bootstrap = toolset._bootstrap_tool

        toolset.warm_up()
        second_bootstrap = toolset._bootstrap_tool

        # Should be the same instance
        assert first_bootstrap is second_bootstrap

    def test_catalog_empty_before_warm_up(self, small_catalog):
        """Test that catalog is empty before warm_up (deferred flattening)."""
        toolset = SearchableToolset(catalog=small_catalog)

        # Before warm_up, _catalog is empty (flattening is deferred)
        assert len(toolset._catalog) == 0

        # After warm_up, catalog is populated
        toolset.warm_up()
        assert len(list(toolset)) == len(small_catalog)

    def test_bootstrap_tool_before_warm_up(self, large_catalog):
        """Test that bootstrap tool is None before warm_up."""
        toolset = SearchableToolset(catalog=large_catalog)

        assert toolset._bootstrap_tool is None

    def test_warm_up_raises_on_duplicate_tool_names(self):
        """Test that warm_up raises when the flattened catalog has duplicate tool names."""
        params = {"type": "object", "properties": {}}
        tool1 = Tool(name="dup", description="first", parameters=params, function=lambda: 1)
        tool2 = Tool(name="dup", description="second", parameters=params, function=lambda: 2)
        toolset = SearchableToolset(catalog=[tool1, tool2])

        with pytest.raises(ValueError, match="Duplicate tool names found"):
            toolset.warm_up()


class TestSearchableToolsetEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_catalog(self):
        """Test with empty catalog."""
        toolset = SearchableToolset(catalog=[])
        toolset.warm_up()
        assert toolset._passthrough is True
        assert len(list(toolset)) == 0

    def test_single_tool_catalog(self, weather_tool):
        """Test with single tool catalog."""
        toolset = SearchableToolset(catalog=[weather_tool])
        toolset.warm_up()
        assert toolset._passthrough is True
        assert len(list(toolset)) == 1

    def test_exactly_at_threshold(self):
        """Test catalog size exactly at threshold."""
        tools = [
            Tool(
                name=f"tool_{i}",
                description=f"Tool number {i}",
                parameters={"type": "object", "properties": {}},
                function=lambda: None,
            )
            for i in range(8)
        ]

        toolset = SearchableToolset(catalog=tools, search_threshold=8)
        toolset.warm_up()
        # Should NOT be passthrough (>= threshold triggers discovery)
        assert toolset._passthrough is False

    def test_one_below_threshold(self):
        """Test catalog size one below threshold."""
        tools = [
            Tool(
                name=f"tool_{i}",
                description=f"Tool number {i}",
                parameters={"type": "object", "properties": {}},
                function=lambda: None,
            )
            for i in range(7)
        ]

        toolset = SearchableToolset(catalog=tools, search_threshold=8)
        toolset.warm_up()
        # Should be passthrough
        assert toolset._passthrough is True

    def test_multiple_loads_same_tool(self, large_catalog):
        """Test searching for the same tool multiple times."""
        toolset = SearchableToolset(catalog=large_catalog)
        toolset.warm_up()
        assert toolset._bootstrap_tool is not None

        # Search for same tool twice
        toolset._bootstrap_tool.invoke(tool_keywords="weather")
        toolset._bootstrap_tool.invoke(tool_keywords="weather temperature")

        # Should still only have discovered tools (may be multiple if they match "weather")
        assert "get_weather" in toolset._discovered_tools
        assert len(toolset) >= 2  # bootstrap + discovered tools


class TestSearchableToolsetLazyToolset:
    """Tests for lazy toolsets (e.g. MCPToolset with eager_connect=False)."""

    def test_lazy_toolset_tools_available_after_warm_up(self):
        """Test that a lazy toolset's tools become available after warm_up."""

        class LazyToolset(Toolset):
            """A toolset that only provides real tools after warm_up (like MCPToolset)."""

            def __init__(self):
                # Before warm_up, no real tools — mimics MCPToolset with eager_connect=False
                super().__init__(tools=[])
                self._connected = False

            def warm_up(self) -> None:
                if self._connected:
                    return
                # Simulate connecting and discovering tools
                self.tools = [
                    Tool(
                        name=f"lazy_tool_{i}",
                        description=f"Lazy tool number {i} for testing",
                        parameters={"type": "object", "properties": {"x": {"type": "string"}}},
                        function=lambda x: x,
                    )
                    for i in range(10)
                ]
                self._connected = True

        lazy = LazyToolset()
        # Before warm_up, iterating yields nothing
        assert len(list(lazy)) == 0

        toolset = SearchableToolset(catalog=lazy)
        # Before warm_up, catalog is empty
        assert len(toolset._catalog) == 0

        toolset.warm_up()

        # After warm_up, all 10 lazy tools should be in the catalog
        assert len(toolset._catalog) == 10
        # 10 tools >= 8 threshold -> BM25 mode
        assert toolset._passthrough is False
        assert toolset._bootstrap_tool is not None

        # Search should find lazy tools
        result = toolset._bootstrap_tool.invoke(tool_keywords="lazy tool testing")
        assert "lazy_tool" in result

    def test_lazy_toolset_passthrough_mode(self):
        """Test lazy toolset with few tools ends up in passthrough mode."""

        class SmallLazyToolset(Toolset):
            def __init__(self):
                super().__init__(tools=[])

            def warm_up(self) -> None:
                self.tools = [
                    Tool(
                        name="lazy_single",
                        description="A single lazy tool",
                        parameters={"type": "object", "properties": {}},
                        function=lambda: "result",
                    )
                ]

        toolset = SearchableToolset(catalog=SmallLazyToolset())
        toolset.warm_up()

        assert toolset._passthrough is True
        assert len(list(toolset)) == 1
        assert "lazy_single" in toolset

    def test_mixed_lazy_and_eager_toolsets(self):
        """Test catalog with both lazy and eager toolsets."""

        class LazyToolset(Toolset):
            def __init__(self):
                super().__init__(tools=[])

            def warm_up(self) -> None:
                self.tools = [
                    Tool(
                        name=f"lazy_{i}",
                        description=f"Lazy tool {i}",
                        parameters={"type": "object", "properties": {}},
                        function=lambda: None,
                    )
                    for i in range(5)
                ]

        eager_tools = [
            Tool(
                name=f"eager_{i}",
                description=f"Eager tool {i}",
                parameters={"type": "object", "properties": {}},
                function=lambda: None,
            )
            for i in range(5)
        ]

        toolset = SearchableToolset(catalog=[LazyToolset(), *eager_tools])
        toolset.warm_up()

        # Should have 5 lazy + 5 eager = 10 tools
        assert len(toolset._catalog) == 10
        assert toolset._passthrough is False
        assert any(t.name == "lazy_0" for t in toolset._catalog)
        assert any(t.name == "eager_0" for t in toolset._catalog)


class TestSearchableToolsetCustomSearchTool:
    """Tests for customizing the bootstrap search tool."""

    def test_invalid_parameters_description_key(self):
        """Test that invalid parameter keys raise ValueError."""
        with pytest.raises(ValueError, match="Invalid search_tool_parameters_description keys"):
            SearchableToolset(catalog=[], search_tool_parameters_description={"invalid_key": "some description"})

    def test_custom_tool_behavior(self, large_catalog):
        """Test custom name, description, parameter overrides, collection access, and discovery."""
        toolset = SearchableToolset(
            catalog=large_catalog,
            search_tool_name="find_tools",
            search_tool_description="Find tools by keyword.",
            search_tool_parameters_description={"tool_keywords": "Keywords.", "k": "How many."},
        )
        toolset.warm_up()
        assert toolset._bootstrap_tool is not None

        # Custom name and description applied
        assert toolset._bootstrap_tool.name == "find_tools"
        assert toolset._bootstrap_tool.description == "Find tools by keyword."

        # Both parameter descriptions overridden
        props = toolset._bootstrap_tool.parameters["properties"]
        assert props["tool_keywords"]["description"] == "Keywords."
        assert props["k"]["description"] == "How many."

        # Collection access uses custom name
        assert "find_tools" in toolset
        assert "search_tools" not in toolset
        assert toolset[0].name == "find_tools"

        # Discovery still works
        result = toolset._bootstrap_tool.invoke(tool_keywords="weather")
        assert "get_weather" in result
        assert "get_weather" in toolset._discovered_tools

    def test_serialization(self, large_catalog):
        """Test to_dict includes all fields and serde roundtrip preserves settings."""
        toolset = SearchableToolset(
            catalog=large_catalog,
            search_tool_name="find_tools",
            search_tool_description="Custom description.",
            search_tool_parameters_description={"tool_keywords": "Custom param desc."},
        )

        # to_dict includes custom values
        data = toolset.to_dict()
        assert data["data"]["search_tool_name"] == "find_tools"
        assert data["data"]["search_tool_description"] == "Custom description."
        assert data["data"]["search_tool_parameters_description"] == {"tool_keywords": "Custom param desc."}

        # Roundtrip preserves behavior
        restored = SearchableToolset.from_dict(data)
        restored.warm_up()
        assert restored._bootstrap_tool is not None
        assert restored._bootstrap_tool.name == "find_tools"
        assert restored._bootstrap_tool.description == "Custom description."
        assert restored._bootstrap_tool.parameters["properties"]["tool_keywords"]["description"] == "Custom param desc."
        result = restored._bootstrap_tool.invoke(tool_keywords="weather")
        assert "get_weather" in result


class TestSearchableToolsetAgentToolSelection:
    """Deterministic Agent tests for runtime tool-name selection and lazy tool_call_counts."""

    def test_get_selectable_tools_exposes_full_catalog(self, large_catalog):
        """get_selectable_tools() exposes the whole catalog, unlike iteration (search tool + discovered only)."""
        toolset = SearchableToolset(catalog=large_catalog, search_threshold=3)
        toolset.warm_up()

        # Iteration only exposes the bootstrap search tool before anything is discovered.
        assert [tool.name for tool in toolset] == ["search_tools"]
        # The catalog, however, is fully available for name-based selection.
        assert {tool.name for tool in toolset.get_selectable_tools()} == {tool.name for tool in large_catalog}

    def test_runtime_tool_names_register_selection_and_preserve_search(self, large_catalog, monkeypatch):
        """Selecting catalog tool names registers a selection on the live SearchableToolset and keeps search active."""
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        toolset = SearchableToolset(catalog=large_catalog, search_threshold=3)  # 8 tools -> search mode
        agent = Agent(chat_generator=OpenAIChatGenerator(), tools=toolset)

        selected = agent._select_tools(["get_weather", "add_numbers"])

        # The live toolset is kept (not flattened into static tools), with the selection registered.
        assert selected == [toolset]
        assert toolset._selected_tool_names == {"get_weather", "add_numbers"}
        # Search is preserved (not dismantled): the bootstrap tool is still the only thing exposed up front.
        assert [tool.name for tool in toolset] == ["search_tools"]
        # And search only discovers tools within the selected subset.
        assert toolset._bootstrap_tool is not None
        toolset._bootstrap_tool.invoke(tool_keywords="weather add stock multiply")
        assert set(toolset._discovered_tools) <= {"get_weather", "add_numbers"}

    def test_runtime_tool_names_passthrough_exposes_selected(self, large_catalog, monkeypatch):
        """In passthrough mode, selecting names exposes exactly those catalog tools directly."""
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        toolset = SearchableToolset(catalog=large_catalog, search_threshold=20)  # 8 < 20 -> passthrough
        agent = Agent(chat_generator=OpenAIChatGenerator(), tools=toolset)

        selected = agent._select_tools(["get_weather", "add_numbers"])

        assert selected == [toolset]
        assert {tool.name for tool in flatten_tools_or_toolsets(selected)} == {"get_weather", "add_numbers"}

    def test_agent_run_with_runtime_tool_names(self, large_catalog):
        """An Agent with a SearchableToolset runs with specific catalog tools selected by name, then clears them."""
        toolset = SearchableToolset(catalog=large_catalog, search_threshold=20)  # passthrough exposes the selection

        @component
        class WeatherCallingGenerator:
            invoked = False

            @component.output_types(replies=list[ChatMessage])
            def run(self, messages, tools=None, **kwargs):
                # In passthrough mode the selected catalog tool is exposed directly.
                assert [tool.name for tool in tools] == ["get_weather"]
                if self.invoked:
                    return {"replies": [ChatMessage.from_assistant("done")]}
                self.invoked = True
                return {
                    "replies": [
                        ChatMessage.from_assistant(
                            tool_calls=[ToolCall(tool_name="get_weather", arguments={"city": "Berlin"})]
                        )
                    ]
                }

        agent = Agent(chat_generator=WeatherCallingGenerator(), tools=toolset, max_agent_steps=5)
        result = agent.run(messages=[ChatMessage.from_user("What's the weather in Berlin?")], tools=["get_weather"])

        assert result["tool_call_counts"]["get_weather"] == 1
        # The per-run selection is cleared after the run, so it does not leak into later runs.
        assert toolset._selected_tool_names is None

    def test_discovered_tool_call_counts_added_lazily(self, large_catalog):
        """tool_call_counts seeds only the search tool up front; a discovered+called tool is added lazily."""
        toolset = SearchableToolset(catalog=large_catalog, search_threshold=3, top_k=5)

        @component
        class SearchThenWeatherGenerator:
            step = 0

            @component.output_types(replies=list[ChatMessage])
            def run(self, messages, tools=None, **kwargs):
                self.step += 1
                if self.step == 1:
                    return {
                        "replies": [
                            ChatMessage.from_assistant(
                                tool_calls=[ToolCall(tool_name="search_tools", arguments={"tool_keywords": "weather"})]
                            )
                        ]
                    }
                if self.step == 2:
                    return {
                        "replies": [
                            ChatMessage.from_assistant(
                                tool_calls=[ToolCall(tool_name="get_weather", arguments={"city": "Berlin"})]
                            )
                        ]
                    }
                return {"replies": [ChatMessage.from_assistant("done")]}

        agent = Agent(chat_generator=SearchThenWeatherGenerator(), tools=toolset, max_agent_steps=6)
        result = agent.run(messages=[ChatMessage.from_user("What's the weather in Berlin?")])

        counts = result["tool_call_counts"]
        # search_tools is seeded at init; get_weather is only counted after being discovered and called.
        assert counts["search_tools"] == 1
        assert counts["get_weather"] == 1
        # After the run, reset() clears the toolset's discovered tools.
        assert toolset._discovered_tools == {}


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
@pytest.mark.integration
class TestSearchableToolsetAgentIntegration:
    """Integration tests with real Agent and OpenAIChatGenerator."""

    def test_agent_discovers_and_uses_tools(self, large_catalog):
        """Agent discovers tools via BM25 search and uses them."""
        toolset = SearchableToolset(catalog=large_catalog, top_k=2, search_threshold=3)
        agent = Agent(chat_generator=OpenAIChatGenerator(model="gpt-4.1-nano"), tools=toolset, max_agent_steps=5)

        assert len(agent.tools) == 1
        result = agent.run(messages=[ChatMessage.from_user("What's the weather in Milan?")])

        # Discovered tools are cleared after the run (reset), so only the search tool remains exposed.
        assert len(agent.tools) == 1
        assert "messages" in result
        messages = result["messages"]
        assert len(messages) > 1

        # Discovery happened during the run: the agent searched and then called the discovered tool.
        tool_calls = [tool_call for msg in messages if msg.tool_calls for tool_call in msg.tool_calls]
        assert len(tool_calls) > 1
        assert any(tool_call.tool_name == "search_tools" for tool_call in tool_calls)
        assert any(tool_call.tool_name == "get_weather" for tool_call in tool_calls)
        assert "22" in messages[-1].text

    def test_agent_discovers_multiple_tools_across_steps(self, large_catalog):
        """A task needing two different tools forces the agent to search for and load each one."""
        # Use two tools the model cannot answer on its own (live weather and stock price) so it is forced
        # to call both. top_k=1 means a single search returns at most one tool, so the agent must search
        # again for the second tool — exercising that discovered tools accumulate across agent steps.
        toolset = SearchableToolset(catalog=large_catalog, top_k=1, search_threshold=3)
        agent = Agent(chat_generator=OpenAIChatGenerator(model="gpt-4.1-nano"), tools=toolset, max_agent_steps=8)

        result = agent.run(
            messages=[
                ChatMessage.from_user(
                    "What is the current weather in Milan, and what is the current stock price of AAPL? "
                    "Use the available tools to look up both."
                )
            ]
        )

        tool_calls = [tool_call for msg in result["messages"] if msg.tool_calls for tool_call in msg.tool_calls]
        called = {tc.tool_name for tc in tool_calls}
        # The agent had to search (more than once given top_k=1) and use both discovered tools.
        assert sum(tc.tool_name == "search_tools" for tc in tool_calls) >= 2
        assert "get_weather" in called
        assert "get_stock_price" in called
