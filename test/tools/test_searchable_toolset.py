# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


import os

import pytest

from haystack.tools import SearchableToolset, Tool, Toolset
from haystack.tools.from_function import create_tool_from_function


# Test helper functions
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: 22°C, sunny"


def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


def get_stock_price(symbol: str) -> str:
    """Get stock price by ticker symbol."""
    return f"{symbol}: $150.00"


def search_database(query: str) -> str:
    """Search the database for records."""
    return f"Found 5 records matching '{query}'"


def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient."""
    return f"Email sent to {to}"


def calculate_tax(amount: float, rate: float) -> float:
    """Calculate tax on an amount."""
    return amount * rate


def convert_currency(amount: float, from_currency: str, to_currency: str) -> float:
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
    return [
        create_tool_from_function(fn)
        for fn in [
            get_weather,
            add_numbers,
            multiply_numbers,
            get_stock_price,
            search_database,
            send_email,
            calculate_tax,
            convert_currency,
        ]
    ]


class TestSearchableToolsetPassthrough:
    """Tests for passthrough mode (small catalogs)."""

    def test_passthrough_mode_detected(self, small_catalog):
        """Test that small catalogs trigger passthrough mode."""
        toolset = SearchableToolset(catalog=small_catalog)
        toolset.warm_up()

        assert toolset.is_passthrough is True

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
        """Test that passthrough mode doesn't create bootstrap tool."""
        toolset = SearchableToolset(catalog=small_catalog)
        toolset.warm_up()

        assert toolset._bootstrap_tool is None

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

    def test_custom_search_threshold(self, large_catalog):
        """Test that custom search_threshold changes passthrough behavior."""
        # With threshold of 10, 8 tools should be passthrough
        toolset = SearchableToolset(catalog=large_catalog, search_threshold=10)
        toolset.warm_up()

        assert toolset.is_passthrough is True
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

    def test_contains_bootstrap_tool(self, large_catalog):
        """Test __contains__ for bootstrap tool."""
        toolset = SearchableToolset(catalog=large_catalog)
        toolset.warm_up()

        assert "search_tools" in toolset
        assert toolset._bootstrap_tool in toolset

    def test_contains_discovered_tool(self, large_catalog):
        """Test __contains__ for discovered tools."""
        toolset = SearchableToolset(catalog=large_catalog, top_k=1)
        toolset.warm_up()
        assert toolset._bootstrap_tool is not None

        toolset._bootstrap_tool.invoke(tool_keywords="weather")

        assert "get_weather" in toolset
        assert "add_numbers" not in toolset  # Not discovered yet


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

    def test_from_dict(self, large_catalog):
        """Test deserialization from dict."""
        toolset = SearchableToolset(catalog=large_catalog, top_k=3)

        data = toolset.to_dict()
        restored = SearchableToolset.from_dict(data)
        restored.warm_up()

        assert restored._top_k == 3
        assert len(restored._catalog) == len(large_catalog)

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
        assert restored.is_passthrough == toolset.is_passthrough

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

        assert search_toolset.is_passthrough is True
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
        assert len(list(toolset)) == 0

        # After warm_up, catalog is populated
        toolset.warm_up()
        assert len(list(toolset)) == len(small_catalog)

    def test_bootstrap_tool_before_warm_up(self, large_catalog):
        """Test that bootstrap tool is None before warm_up."""
        toolset = SearchableToolset(catalog=large_catalog)

        assert toolset._bootstrap_tool is None


class TestSearchableToolsetEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_catalog(self):
        """Test with empty catalog."""
        toolset = SearchableToolset(catalog=[])
        toolset.warm_up()

        assert toolset.is_passthrough is True
        assert len(list(toolset)) == 0

    def test_single_tool_catalog(self, weather_tool):
        """Test with single tool catalog."""
        toolset = SearchableToolset(catalog=[weather_tool])
        toolset.warm_up()

        assert toolset.is_passthrough is True
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
        assert toolset.is_passthrough is False

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
        assert toolset.is_passthrough is True

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
        assert toolset.is_passthrough is False
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

        assert toolset.is_passthrough is True
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

        toolset = SearchableToolset(catalog=[LazyToolset()] + eager_tools)
        toolset.warm_up()

        # Should have 5 lazy + 5 eager = 10 tools
        assert len(toolset._catalog) == 10
        assert toolset.is_passthrough is False
        assert any(t.name == "lazy_0" for t in toolset._catalog)
        assert any(t.name == "eager_0" for t in toolset._catalog)


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
@pytest.mark.integration
class TestSearchableToolsetAgentIntegration:
    """Integration tests with real Agent and OpenAIChatGenerator."""

    def test_agent_discovers_and_uses_tools(self, large_catalog):
        """Agent discovers tools via BM25 search and uses them."""
        from haystack.components.agents import Agent
        from haystack.components.generators.chat import OpenAIChatGenerator
        from haystack.dataclasses import ChatMessage

        toolset = SearchableToolset(catalog=large_catalog, top_k=2, search_threshold=3)
        agent = Agent(chat_generator=OpenAIChatGenerator(model="gpt-4.1-nano"), tools=toolset, max_agent_steps=5)

        assert len(agent.tools) == 1
        result = agent.run(messages=[ChatMessage.from_user("What's the weather in Milan?")])

        assert len(agent.tools) > 1
        assert "messages" in result
        messages = result["messages"]
        assert len(messages) > 1

        tool_calls = [tool_call for msg in messages if msg.tool_calls for tool_call in msg.tool_calls]
        assert len(tool_calls) > 1
        assert any(tool_call.tool_name == "search_tools" for tool_call in tool_calls)
        assert any(tool_call.tool_name == "get_weather" for tool_call in tool_calls)
        assert "22" in messages[-1].text
