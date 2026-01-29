# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


import pytest

from haystack.tools import SearchableToolset, Tool, Toolset


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
    return Tool(
        name="get_weather",
        description="Get current weather for a city",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string", "description": "The city name"}},
            "required": ["city"],
        },
        function=get_weather,
    )


@pytest.fixture
def add_tool():
    return Tool(
        name="add_numbers",
        description="Add two numbers together",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        function=add_numbers,
    )


@pytest.fixture
def multiply_tool():
    return Tool(
        name="multiply_numbers",
        description="Multiply two numbers",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        function=multiply_numbers,
    )


@pytest.fixture
def stock_tool():
    return Tool(
        name="get_stock_price",
        description="Get stock price by ticker symbol",
        parameters={
            "type": "object",
            "properties": {"symbol": {"type": "string", "description": "Stock ticker symbol"}},
            "required": ["symbol"],
        },
        function=get_stock_price,
    )


@pytest.fixture
def small_catalog(weather_tool, add_tool, multiply_tool):
    """Small catalog that triggers passthrough mode (< 8 tools)."""
    return [weather_tool, add_tool, multiply_tool]


@pytest.fixture
def large_catalog():
    """Larger catalog that requires discovery (>= 8 tools)."""
    tools = [
        Tool(
            name="get_weather",
            description="Get current weather for a city",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            function=get_weather,
        ),
        Tool(
            name="add_numbers",
            description="Add two numbers together",
            parameters={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
            function=add_numbers,
        ),
        Tool(
            name="multiply_numbers",
            description="Multiply two numbers",
            parameters={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
            function=multiply_numbers,
        ),
        Tool(
            name="get_stock_price",
            description="Get stock price by ticker symbol",
            parameters={"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]},
            function=get_stock_price,
        ),
        Tool(
            name="search_database",
            description="Search the database for records",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            function=search_database,
        ),
        Tool(
            name="send_email",
            description="Send an email to a recipient",
            parameters={
                "type": "object",
                "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
                "required": ["to", "subject", "body"],
            },
            function=send_email,
        ),
        Tool(
            name="calculate_tax",
            description="Calculate tax on an amount",
            parameters={
                "type": "object",
                "properties": {"amount": {"type": "number"}, "rate": {"type": "number"}},
                "required": ["amount", "rate"],
            },
            function=calculate_tax,
        ),
        Tool(
            name="convert_currency",
            description="Convert currency from one to another",
            parameters={
                "type": "object",
                "properties": {
                    "amount": {"type": "number"},
                    "from_currency": {"type": "string"},
                    "to_currency": {"type": "string"},
                },
                "required": ["amount", "from_currency", "to_currency"],
            },
            function=convert_currency,
        ),
    ]
    return tools


class TestSearchableToolsetPassthrough:
    """Tests for passthrough mode (small catalogs)."""

    def test_passthrough_mode_detected(self, small_catalog):
        """Test that small catalogs trigger passthrough mode."""
        toolset = SearchableToolset(catalog=small_catalog)

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

        assert len(search_toolset._catalog) == 4
        assert any(t.name == "get_weather" for t in search_toolset._catalog)
        assert any(t.name == "add_numbers" for t in search_toolset._catalog)
        assert any(t.name == "multiply_numbers" for t in search_toolset._catalog)
        assert any(t.name == "get_stock_price" for t in search_toolset._catalog)

    def test_accepts_mixed_list(self, weather_tool, add_tool, multiply_tool):
        """Test that a mixed list of Tools and Toolsets can be used as catalog."""
        toolset = Toolset(tools=[add_tool, multiply_tool])

        search_toolset = SearchableToolset(catalog=[weather_tool, toolset], search_threshold=10)

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

    def test_warm_up_not_required_for_passthrough(self, small_catalog):
        """Test that passthrough works without warm_up."""
        toolset = SearchableToolset(catalog=small_catalog)

        # Should work without warm_up
        tools = list(toolset)
        assert len(tools) == len(small_catalog)

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


# Integration tests requiring OPENAI_API_KEY
import os


@pytest.fixture
def integration_catalog():
    """Catalog with 5 tools for integration tests (triggers search with threshold=3)."""
    return [
        Tool(
            name="get_weather",
            description="Get current weather forecast for a city including temperature and conditions",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string", "description": "The city name"}},
                "required": ["city"],
            },
            function=get_weather,
        ),
        Tool(
            name="add_numbers",
            description="Add two integer numbers together and return the sum",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "First number"},
                    "b": {"type": "integer", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
            function=add_numbers,
        ),
        Tool(
            name="multiply_numbers",
            description="Multiply two integer numbers and return the product",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "First number"},
                    "b": {"type": "integer", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
            function=multiply_numbers,
        ),
        Tool(
            name="get_stock_price",
            description="Get the current stock price for a given ticker symbol",
            parameters={
                "type": "object",
                "properties": {"symbol": {"type": "string", "description": "Stock ticker symbol like AAPL or GOOGL"}},
                "required": ["symbol"],
            },
            function=get_stock_price,
        ),
        Tool(
            name="search_database",
            description="Search a database for records matching a query string",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The search query"}},
                "required": ["query"],
            },
            function=search_database,
        ),
    ]


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
@pytest.mark.integration
class TestSearchableToolsetAgentIntegration:
    """Integration tests with real Agent and OpenAIChatGenerator."""

    def test_agent_with_bm25_mode(self, integration_catalog):
        """Real Agent integration - discover and use tools via BM25 search."""
        from haystack.components.agents import Agent
        from haystack.components.generators.chat import OpenAIChatGenerator
        from haystack.dataclasses import ChatMessage

        # Create toolset with search_threshold=3 to trigger discovery (we have 5 tools)
        toolset = SearchableToolset(catalog=integration_catalog, top_k=2, search_threshold=3)

        agent = Agent(chat_generator=OpenAIChatGenerator(), tools=toolset, max_agent_steps=5)

        # Agent should: 1) search for weather tool, 2) call it, 3) respond
        result = agent.run(messages=[ChatMessage.from_user("What's the weather in Paris?")])
        # Verify the agent completed successfully
        assert "messages" in result
        assert len(result["messages"]) > 1

        # Check that the weather tool was discovered and used
        last_message = result["last_message"]
        assert last_message is not None

        # The response should contain weather information (22°C or sunny from our mock)
        full_conversation = " ".join(msg.text for msg in result["messages"] if msg.text is not None)
        assert "22" in full_conversation and "sunny" in full_conversation and "Paris" in full_conversation

    def test_agent_with_bm25_mode_math(self, integration_catalog):
        """Real Agent integration - discover and use math tools via BM25 search."""
        from haystack.components.agents import Agent
        from haystack.components.generators.chat import OpenAIChatGenerator
        from haystack.dataclasses import ChatMessage

        toolset = SearchableToolset(catalog=integration_catalog, top_k=2, search_threshold=3)

        agent = Agent(chat_generator=OpenAIChatGenerator(), tools=toolset, max_agent_steps=5)

        # Ask to add numbers - agent should search, find add_numbers tool, use it
        result = agent.run(messages=[ChatMessage.from_user("Please add 15 and 27 together")])

        assert "messages" in result
        assert len(result["messages"]) > 1

        # The final response should contain 42 (15 + 27)
        full_conversation = " ".join(msg.text for msg in result["messages"] if msg.text is not None)
        assert "42" in full_conversation

    def test_agent_with_passthrough_mode(self, integration_catalog):
        """Real Agent integration - passthrough mode exposes all tools directly."""
        from haystack.components.agents import Agent
        from haystack.components.generators.chat import OpenAIChatGenerator
        from haystack.dataclasses import ChatMessage

        # Set high threshold so 5 tools trigger passthrough mode
        toolset = SearchableToolset(catalog=integration_catalog, search_threshold=10)

        # In passthrough mode, all tools are directly available (no search needed)
        assert toolset.is_passthrough is True
        assert len(list(toolset)) == 5

        agent = Agent(chat_generator=OpenAIChatGenerator(), tools=toolset, max_agent_steps=3)

        result = agent.run(messages=[ChatMessage.from_user("What's the weather in Berlin?")])

        assert "messages" in result
        full_conversation = " ".join(msg.text for msg in result["messages"] if msg.text is not None)
        # Should get weather info without needing to search first
        assert "22" in full_conversation or "sunny" in full_conversation or "Berlin" in full_conversation

    def test_agent_discovers_multiple_tools_sequentially(self, integration_catalog):
        """Test that agent can discover and use multiple tools in sequence."""
        from haystack.components.agents import Agent
        from haystack.components.generators.chat import OpenAIChatGenerator
        from haystack.dataclasses import ChatMessage

        toolset = SearchableToolset(catalog=integration_catalog, top_k=2, search_threshold=3)

        agent = Agent(chat_generator=OpenAIChatGenerator(), tools=toolset, max_agent_steps=8)

        # Ask a question that requires multiple tool uses
        result = agent.run(messages=[ChatMessage.from_user("First add 10 and 5, then tell me the weather in Rome")])

        assert "messages" in result
        full_conversation = " ".join(msg.text for msg in result["messages"] if msg.text is not None)

        # Should contain results from both tools
        assert "15" in full_conversation  # 10 + 5
        # Weather info for Rome (22°C, sunny from our mock)
        assert "22" in full_conversation or "sunny" in full_conversation or "Rome" in full_conversation
