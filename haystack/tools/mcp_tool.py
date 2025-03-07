# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack
from dataclasses import dataclass, fields
from typing import Any, Coroutine, Dict, List, Optional, Tuple

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp import ClientSession, types
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client

from haystack import logging
from haystack.core.serialization import generate_qualified_class_name, import_class_by_name
from haystack.tools import Tool
from haystack.tools.errors import ToolInvocationError

logger = logging.getLogger(__name__)


class MCPError(Exception):
    """Base class for MCP-related errors."""

    pass


class MCPConnectionError(MCPError):
    """Error connecting to MCP server."""

    pass


class MCPToolNotFoundError(MCPError):
    """Error when a tool is not found on the server."""

    pass


class MCPInvocationError(ToolInvocationError):
    """Error during tool invocation."""

    pass


class MCPClient(ABC):
    """
    Abstract base class for MCP clients.

    This class defines the common interface and shared functionality for all MCP clients,
    regardless of the transport mechanism used.
    """

    def __init__(self) -> None:
        self.session: Optional[ClientSession] = None
        self.exit_stack: AsyncExitStack = AsyncExitStack()
        self.stdio: Optional[MemoryObjectReceiveStream[types.JSONRPCMessage | Exception]] = None
        self.write: Optional[MemoryObjectSendStream[types.JSONRPCMessage]] = None

    @abstractmethod
    async def connect(self) -> List[types.Tool]:
        """
        Connect to an MCP server.

        :returns: List of available tools on the server
        :raises MCPConnectionError: If connection to the server fails
        """
        pass

    async def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """
        Call a tool on the connected MCP server.

        :param tool_name: Name of the tool to call
        :param tool_args: Arguments to pass to the tool
        :returns: Result of the tool invocation
        :raises MCPError: If not connected to an MCP server
        :raises MCPInvocationError: If tool invocation fails
        """
        if not self.session:
            raise MCPError("Not connected to an MCP server")

        try:
            result = await self.session.call_tool(tool_name, tool_args)
            return result
        except Exception as e:
            raise MCPInvocationError(f"Failed to invoke tool '{tool_name}'") from e

    async def close(self) -> None:
        """
        Close the connection and clean up resources.
        """
        try:
            await self.exit_stack.aclose()
            self.session = None
            self.stdio = None
            self.write = None
        except Exception:
            # Best effort cleanup
            pass

    async def _initialize_session_with_transport(
        self,
        transport_tuple: Tuple[
            MemoryObjectReceiveStream[types.JSONRPCMessage | Exception], MemoryObjectSendStream[types.JSONRPCMessage]
        ],
        connection_type: str,
    ) -> List[types.Tool]:
        """
        Common session initialization logic for all transports.

        :param transport_tuple: Tuple containing (stdio, write) from the transport
        :param connection_type: String describing the connection type for error messages
        :returns: List of available tools on the server
        :raises MCPConnectionError: If connection to the server fails
        """
        try:
            self.stdio, self.write = transport_tuple
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

            await self.session.initialize()

            # List available tools
            response = await self.session.list_tools()
            return response.tools

        except Exception as e:
            await self.close()
            raise MCPConnectionError(f"Failed to connect to {connection_type}: {e}") from e


class StdioMCPClient(MCPClient):
    """
    MCP client that connects to servers using stdio transport.
    """

    def __init__(self, command: str, args: Optional[List[str]] = None, env: Optional[Dict[str, str]] = None) -> None:
        """
        Initialize a stdio MCP client.

        :param command: Command to run (e.g., "python", "node")
        :param args: Arguments to pass to the command
        :param env: Environment variables for the command
        """
        super().__init__()
        self.command: str = command
        self.args: List[str] = args or []
        self.env: Optional[Dict[str, str]] = env

    async def connect(self) -> List[types.Tool]:
        """
        Connect to an MCP server using stdio transport.

        :returns: List of available tools on the server
        :raises MCPConnectionError: If connection to the server fails
        """
        server_params = StdioServerParameters(command=self.command, args=self.args, env=self.env)
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        return await self._initialize_session_with_transport(stdio_transport, f"stdio server (command: {self.command})")


class HttpMCPClient(MCPClient):
    """
    MCP client that connects to servers using HTTP transport.
    """

    def __init__(self, base_url: str, token: Optional[str] = None, timeout: int = 5) -> None:
        """
        Initialize an HTTP MCP client.

        :param base_url: Base URL of the server
        :param token: Authentication token for the server (optional)
        :param timeout: Connection timeout in seconds
        """
        super().__init__()
        self.base_url: str = base_url.rstrip("/")  # Remove any trailing slashes
        self.token: Optional[str] = token
        self.timeout: int = timeout

    async def connect(self) -> List[types.Tool]:
        """
        Connect to an MCP server using HTTP transport.

        :returns: List of available tools on the server
        :raises MCPConnectionError: If connection to the server fails
        """
        sse_url = f"{self.base_url}/sse"
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else None
        sse_transport = await self.exit_stack.enter_async_context(
            sse_client(sse_url, headers=headers, timeout=self.timeout)
        )
        return await self._initialize_session_with_transport(sse_transport, f"HTTP server at {self.base_url}")


@dataclass
class MCPServerInfo(ABC):
    """
    Abstract base class for MCP server connection parameters.

    This class defines the common interface for all MCP server connection types.
    """

    @abstractmethod
    def create_client(self) -> MCPClient:
        """
        Create an appropriate MCP client for this server info.

        :returns: An instance of MCPClient configured with this server info
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this server info to a dictionary.

        :returns: Dictionary representation of this server info
        """
        # Store the fully qualified class name for deserialization
        result = {"type": generate_qualified_class_name(type(self))}

        # Add all fields from the dataclass
        for field in fields(self):
            result[field.name] = getattr(self, field.name)

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPServerInfo":
        """
        Deserialize server info from a dictionary.

        :param data: Dictionary containing serialized server info
        :returns: Instance of the appropriate server info class
        """
        # Remove the type field as it's not a constructor parameter
        data_copy = data.copy()
        data_copy.pop("type", None)

        # Create an instance of the class with the remaining fields
        return cls(**data_copy)


@dataclass
class HttpMCPServerInfo(MCPServerInfo):
    """
    Data class that encapsulates HTTP MCP server connection parameters.

    :param base_url: Base URL of the MCP server
    :param token: Authentication token for the server (optional)
    :param timeout: Connection timeout in seconds
    """

    base_url: str
    token: Optional[str] = None
    timeout: int = 30

    def create_client(self) -> MCPClient:
        """
        Create an HTTP MCP client.

        :returns: Configured HttpMCPClient instance
        """
        return HttpMCPClient(self.base_url, self.token, self.timeout)


@dataclass
class StdioMCPServerInfo(MCPServerInfo):
    """
    Data class that encapsulates stdio MCP server connection parameters.

    :param command: Command to run (e.g., "python", "node")
    :param args: Arguments to pass to the command
    :param env: Environment variables for the command
    """

    command: str
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None

    def create_client(self) -> MCPClient:
        """
        Create a stdio MCP client.

        :returns: Configured StdioMCPClient instance
        """
        return StdioMCPClient(self.command, self.args, self.env)


class MCPTool(Tool):
    """
    A Tool that represents a single tool from an MCP server.

    This implementation uses the official MCP SDK for protocol handling while maintaining
    compatibility with the Haystack tool ecosystem.

    Example using HTTP:
    ```python
    from haystack.tools import MCPTool, HttpMCPServerInfo

    # Create tool instance
    tool = MCPTool(
        name="add",
        server_info=HttpMCPServerInfo(base_url="http://localhost:8000")
    )

    # Use the tool
    result = tool.invoke(a=5, b=3)
    ```

    Example using stdio:
    ```python
    from haystack.tools import MCPTool, StdioMCPServerInfo

    # Create tool instance
    tool = MCPTool(
        name="get_current_time",
        server_info=StdioMCPServerInfo(command="python", args=["path/to/server.py"])
    )

    # Use the tool
    result = tool.invoke(timezone="America/New_York")
    ```
    """

    # Shared thread pool with optimal sizing
    _executor = ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) + 4))
    _loop_policy = None

    def __init__(self, name: str, server_info: MCPServerInfo, description: Optional[str] = None):
        """
        Initialize the MCP tool.

        :param name: Name of the tool to use
        :param server_info: Server connection information
        :param description: Custom description (if None, server description will be used)
        :raises MCPConnectionError: If connection to the server fails
        :raises MCPToolNotFoundError: If no tools are available or the requested tool is not found
        """

        # Store connection parameters for serialization
        self._server_info = server_info
        client = None

        # Initialize the connection
        try:
            # Create the appropriate client using the factory method
            client = server_info.create_client()

            # Connect and get available tools
            tools = self._run_sync(client.connect())

            # Handle no tools case
            if not tools:
                raise MCPToolNotFoundError("No tools available on server")

            # Find the specified tool
            tool_info = next((t for t in tools if t.name == name), None)
            if not tool_info:
                raise MCPToolNotFoundError(f"Tool '{name}' not found on server")

            # Store the client for later use
            self._client = client

            # Initialize the parent class with the final values
            logger.debug(f"Initializing MCPTool with name: {name}")

            # Hook into the Tool base class
            super().__init__(
                name=name,
                description=description or tool_info.description,
                parameters=tool_info.inputSchema,
                function=self._invoke_tool,
            )

        except Exception as e:
            if client is not None:
                self._run_sync(client.close())
            raise MCPError(f"Failed to initialize MCPTool: {e}") from e

    def _invoke_tool(self, **kwargs: Any) -> Any:
        """
        Synchronous tool invocation.

        This method is called by the Tool base class's invoke() method.

        :param kwargs: Arguments to pass to the tool
        :returns: Result of the tool invocation
        :raises: Any exception that might occur during tool invocation
        """
        try:
            return self._run_sync(self._client.call_tool(self.name, kwargs))
        except Exception:
            # Preserve the original exception
            raise

    async def ainvoke(self, **kwargs: Any) -> Any:
        """
        Asynchronous tool invocation.

        :param kwargs: Arguments to pass to the tool
        :returns: Result of the tool invocation
        :raises: Any exception that might occur during tool invocation
        """
        return await self._client.call_tool(self.name, kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the MCPTool to a dictionary.

        Note: The active connection is not serialized. A new connection will need to be
        established when deserializing.
        """
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {"name": self.name, "description": self.description, "server_info": self._server_info.to_dict()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Tool":
        """
        Deserializes the MCPTool from a dictionary.

        Note: This will establish a new connection to the MCP server.
        """
        inner_data = data["data"]
        server_info_dict = inner_data.get("server_info", {})

        # Import the server info class by name
        server_info_class = import_class_by_name(server_info_dict["type"])

        # Create an instance of the server info class
        server_info = server_info_class.from_dict(server_info_dict)

        return cls(name=inner_data["name"], description=inner_data.get("description"), server_info=server_info)

    @classmethod
    def _get_loop_policy(cls):
        """
        Get the event loop policy with caching for better performance.

        :returns: The cached event loop policy
        """
        if cls._loop_policy is None:
            cls._loop_policy = asyncio.get_event_loop_policy()
        return cls._loop_policy

    def _run_sync(self, coro: Coroutine, timeout: Optional[float] = 30) -> Any:
        """
        Run a coroutine in a synchronous context with improved handling.

        :param coro: Coroutine to run
        :param timeout: Optional timeout in seconds
        :returns: Result of the coroutine
        :raises TimeoutError: If the operation times out
        :raises Exception: Any exception that might occur during execution
        """
        try:
            try:
                # Try to get the current event loop
                running_loop = self._get_loop_policy().get_event_loop()

                if running_loop.is_running():
                    # We're in an async context, use the shared thread pool
                    future = self._executor.submit(lambda: asyncio.run(asyncio.wait_for(coro, timeout)))
                    return future.result()
                else:
                    # We have an event loop but it's not running
                    return running_loop.run_until_complete(asyncio.wait_for(coro, timeout))
            except RuntimeError:
                # No event loop exists, create one
                return asyncio.run(asyncio.wait_for(coro, timeout))
        except asyncio.TimeoutError:
            raise TimeoutError(f"Operation timed out after {timeout} seconds") from None
        except Exception:
            # Preserve the original exception type and traceback
            raise
