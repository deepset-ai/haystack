import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from haystack.tools import MCPTool, HttpMCPServerInfo, StdioMCPServerInfo, MCPError
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.components.tools import ToolInvoker
from haystack import Pipeline
from haystack.tools.from_function import tool


@tool
def echo(name: str) -> str:
    """Echo a name"""
    return f"Hello, {name}!"


class TestMCPToolWithMockedServer:
    """Tests for MCPTool with a mocked server."""

    @patch("haystack.tools.mcp_tool.HttpMCPClient")
    def test_http_mcp_tool_invoke(self, mock_client_class):
        """Test invoking an HTTP MCPTool."""
        # Setup mocks
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.connect = AsyncMock()
        mock_client.call_tool = AsyncMock(return_value=10)

        # Create a mock tool
        mock_tool = MagicMock()
        mock_tool.name = "add"
        mock_tool.description = "Add two numbers"
        mock_tool.inputSchema = {
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
            "type": "object",
        }

        # Set up the connect method to return our mock tool
        mock_client.connect.return_value = [mock_tool]

        # Create server info and tool
        server_info = HttpMCPServerInfo(base_url="http://localhost:8000")
        tool = MCPTool(name="add", server_info=server_info)

        # Invoke the tool
        result = tool.invoke(a=7, b=3)

        # Assertions
        assert result == 10
        mock_client.call_tool.assert_called_once_with("add", {"a": 7, "b": 3})

    @patch("haystack.tools.mcp_tool.StdioMCPClient")
    def test_stdio_mcp_tool_invoke(self, mock_client_class):
        """Test invoking a stdio MCPTool."""
        # Setup mocks
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.connect = AsyncMock()
        mock_client.call_tool = AsyncMock(return_value="2023-01-01T12:00:00Z")

        # Create a mock tool
        mock_tool = MagicMock()
        mock_tool.name = "get_current_time"
        mock_tool.description = "Get current time in a specific timezone"
        mock_tool.inputSchema = {
            "properties": {"timezone": {"type": "string"}},
            "required": ["timezone"],
            "type": "object",
        }

        # Set up the connect method to return our mock tool
        mock_client.connect.return_value = [mock_tool]

        # Create server info and tool
        server_info = StdioMCPServerInfo(command="python", args=["server.py"])
        tool = MCPTool(name="get_current_time", server_info=server_info)

        # Invoke the tool
        result = tool.invoke(timezone="America/New_York")

        # Assertions
        assert result == "2023-01-01T12:00:00Z"
        mock_client.call_tool.assert_called_once_with("get_current_time", {"timezone": "America/New_York"})


@pytest.mark.integration
class TestMCPToolInPipelineWithOpenAI:
    """Integration tests for MCPTool in Haystack pipelines with OpenAI."""

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    def test_mcp_tool_with_http_server(self):
        """Test using an MCPTool with a real HTTP server."""
        import tempfile
        import os
        import subprocess
        import time
        import socket
        from urllib.error import URLError
        import logging

        # Find an available port
        def find_free_port():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                return s.getsockname()[1]

        port = find_free_port()

        # Create a temporary file for the server script
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
            temp_file.write(
                f"""
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("MCP Calculator", host="127.0.0.1", port={port})
@mcp.tool()
def add(a: int, b: int) -> int:
    \"\"\"Add two numbers\"\"\"
    return a + b

# Add a subtraction tool
@mcp.tool()
def subtract(a: int, b: int) -> int:
    \"\"\"Subtract b from a\"\"\"
    return a - b

if __name__ == "__main__":
    try:
        mcp.run(transport="sse")
    except Exception as e:
        sys.exit(1)
""".encode()
            )
            server_script_path = temp_file.name

        server_process = None
        try:
            # Start the server in a separate process
            server_process = subprocess.Popen(
                ["python", server_script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            # Give the server a moment to start
            time.sleep(5)
            # Create an MCPTool that connects to the HTTP server
            server_info = HttpMCPServerInfo(base_url=f"http://127.0.0.1:{port}")

            # Create the tool
            tool = MCPTool(name="add", server_info=server_info)

            # Invoke the tool
            result = tool.invoke(a=5, b=3)

            # Verify the result
            assert not result.isError
            assert len(result.content) == 1
            assert result.content[0].text == "8"

            # Try another tool from the same server
            subtract_tool = MCPTool(name="subtract", server_info=server_info)
            result = subtract_tool.invoke(a=10, b=4)

            # Verify the result
            assert not result.isError
            assert len(result.content) == 1
            assert result.content[0].text == "6"

        except Exception as e:
            # Check server output for clues
            if server_process and server_process.poll() is None:
                server_process.terminate()
            raise

        finally:
            # Clean up
            if server_process:
                if server_process.poll() is None:  # Process is still running
                    server_process.terminate()
                    try:
                        server_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        server_process.kill()
                        server_process.wait(timeout=5)

            # Remove the temporary file
            if os.path.exists(server_script_path):
                os.remove(server_script_path)

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY") and not os.environ.get("BRAVE_API_KEY"),
        reason="OPENAI_API_KEY or BRAVE_API_KEY not set",
    )
    def test_mcp_brave_search(self):
        """Test using an MCPTool in a pipeline with OpenAI."""

        # Create an MCPTool for the brave_web_search operation
        server_info = StdioMCPServerInfo(
            command="docker",
            args=["run", "-i", "--rm", "-e", f"BRAVE_API_KEY={os.environ.get('BRAVE_API_KEY')}", "mcp/brave-search"],
            env=None,
        )
        tool = MCPTool(name="brave_web_search", server_info=server_info)

        # Create pipeline with OpenAIChatGenerator and ToolInvoker
        pipeline = Pipeline()
        pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini", tools=[tool]))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=[tool]))

        # Connect components
        pipeline.connect("llm.replies", "tool_invoker.messages")

        # Create a message that should trigger tool use
        message = ChatMessage.from_user(text="Use brave_web_search to search for the latest German elections news")

        result = pipeline.run({"llm": {"messages": [message]}})

        tool_messages = result["tool_invoker"]["tool_messages"]
        assert len(tool_messages) == 1

        tool_message = tool_messages[0]
        assert tool_message.is_from(ChatRole.TOOL)
        assert any(term in tool_message.tool_call_result.result for term in ["Bundestag", "elections"])

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    def test_mcp_tool_in_pipeline_with_multiple_tools(self):
        """Test using multiple MCPTools in a pipeline with OpenAI."""

        # Mix mcp tool with a simple echo tool
        time_server_info = StdioMCPServerInfo(
            command="python", args=["-m", "mcp_server_time", "--local-timezone=America/New_York"]
        )
        time_tool = MCPTool(name="get_current_time", server_info=time_server_info)

        # Create pipeline with OpenAIChatGenerator and ToolInvoker
        pipeline = Pipeline()
        pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini", tools=[echo, time_tool]))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=[echo, time_tool]))

        pipeline.connect("llm.replies", "tool_invoker.messages")

        # Create a message that should trigger tool use
        message = ChatMessage.from_user(text="What is the current time in New York?")

        result = pipeline.run({"llm": {"messages": [message]}})

        tool_messages = result["tool_invoker"]["tool_messages"]
        assert len(tool_messages) == 1

        tool_message = tool_messages[0]
        assert tool_message.is_from(ChatRole.TOOL)
        assert "timezone" in tool_message.tool_call_result.result
        assert "datetime" in tool_message.tool_call_result.result
        assert "New_York" in tool_message.tool_call_result.result

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    def test_mcp_tool_error_handling(self):
        """Test error handling with MCPTool in a pipeline."""

        # Create a custom server info for a server that might return errors
        server_info = HttpMCPServerInfo(base_url="http://localhost:8000")
        with pytest.raises(MCPError):
            tool = MCPTool(name="divide", server_info=server_info)
