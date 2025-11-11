---
title: "MCP"
id: integrations-mcp
description: "MCP integration for Haystack"
slug: "/integrations-mcp"
---

<a id="haystack_integrations.tools.mcp.mcp_tool"></a>

## Module haystack\_integrations.tools.mcp.mcp\_tool

<a id="haystack_integrations.tools.mcp.mcp_tool.AsyncExecutor"></a>

### AsyncExecutor

Thread-safe event loop executor for running async code from sync contexts.

<a id="haystack_integrations.tools.mcp.mcp_tool.AsyncExecutor.get_instance"></a>

#### AsyncExecutor.get\_instance

```python
@classmethod
def get_instance(cls) -> "AsyncExecutor"
```

Get or create the global singleton executor instance.

<a id="haystack_integrations.tools.mcp.mcp_tool.AsyncExecutor.__init__"></a>

#### AsyncExecutor.\_\_init\_\_

```python
def __init__()
```

Initialize a dedicated event loop

<a id="haystack_integrations.tools.mcp.mcp_tool.AsyncExecutor.run"></a>

#### AsyncExecutor.run

```python
def run(coro: Coroutine[Any, Any, Any], timeout: float | None = None) -> Any
```

Run a coroutine in the event loop.

**Arguments**:

- `coro`: Coroutine to execute
- `timeout`: Optional timeout in seconds

**Raises**:

- `TimeoutError`: If execution exceeds timeout

**Returns**:

Result of the coroutine

<a id="haystack_integrations.tools.mcp.mcp_tool.AsyncExecutor.get_loop"></a>

#### AsyncExecutor.get\_loop

```python
def get_loop()
```

Get the event loop.

**Returns**:

The event loop

<a id="haystack_integrations.tools.mcp.mcp_tool.AsyncExecutor.run_background"></a>

#### AsyncExecutor.run\_background

```python
def run_background(
    coro_factory: Callable[[asyncio.Event], Coroutine[Any, Any, Any]],
    timeout: float | None = None
) -> tuple[concurrent.futures.Future[Any], asyncio.Event]
```

Schedule `coro_factory` to run in the executor's event loop **without** blocking the

caller thread.

The factory receives an :class:`asyncio.Event` that can be used to cooperatively shut
the coroutine down. The method returns **both** the concurrent future (to observe
completion or failure) and the created *stop_event* so that callers can signal termination.

**Arguments**:

- `coro_factory`: A callable receiving the stop_event and returning the coroutine to execute.
- `timeout`: Optional timeout while waiting for the stop_event to be created.

**Returns**:

Tuple ``(future, stop_event)``.

<a id="haystack_integrations.tools.mcp.mcp_tool.AsyncExecutor.shutdown"></a>

#### AsyncExecutor.shutdown

```python
def shutdown(timeout: float = 2) -> None
```

Shut down the background event loop and thread.

**Arguments**:

- `timeout`: Timeout in seconds for shutting down the event loop

<a id="haystack_integrations.tools.mcp.mcp_tool.MCPError"></a>

### MCPError

Base class for MCP-related errors.

<a id="haystack_integrations.tools.mcp.mcp_tool.MCPError.__init__"></a>

#### MCPError.\_\_init\_\_

```python
def __init__(message: str) -> None
```

Initialize the MCPError.

**Arguments**:

- `message`: Descriptive error message

<a id="haystack_integrations.tools.mcp.mcp_tool.MCPConnectionError"></a>

### MCPConnectionError

Error connecting to MCP server.

<a id="haystack_integrations.tools.mcp.mcp_tool.MCPConnectionError.__init__"></a>

#### MCPConnectionError.\_\_init\_\_

```python
def __init__(message: str,
             server_info: "MCPServerInfo | None" = None,
             operation: str | None = None) -> None
```

Initialize the MCPConnectionError.

**Arguments**:

- `message`: Descriptive error message
- `server_info`: Server connection information that was used
- `operation`: Name of the operation that was being attempted

<a id="haystack_integrations.tools.mcp.mcp_tool.MCPToolNotFoundError"></a>

### MCPToolNotFoundError

Error when a tool is not found on the server.

<a id="haystack_integrations.tools.mcp.mcp_tool.MCPToolNotFoundError.__init__"></a>

#### MCPToolNotFoundError.\_\_init\_\_

```python
def __init__(message: str,
             tool_name: str,
             available_tools: list[str] | None = None) -> None
```

Initialize the MCPToolNotFoundError.

**Arguments**:

- `message`: Descriptive error message
- `tool_name`: Name of the tool that was requested but not found
- `available_tools`: List of available tool names, if known

<a id="haystack_integrations.tools.mcp.mcp_tool.MCPInvocationError"></a>

### MCPInvocationError

Error during tool invocation.

<a id="haystack_integrations.tools.mcp.mcp_tool.MCPInvocationError.__init__"></a>

#### MCPInvocationError.\_\_init\_\_

```python
def __init__(message: str,
             tool_name: str,
             tool_args: dict[str, Any] | None = None) -> None
```

Initialize the MCPInvocationError.

**Arguments**:

- `message`: Descriptive error message
- `tool_name`: Name of the tool that was being invoked
- `tool_args`: Arguments that were passed to the tool

<a id="haystack_integrations.tools.mcp.mcp_tool.MCPClient"></a>

### MCPClient

Abstract base class for MCP clients.

This class defines the common interface and shared functionality for all MCP clients,
regardless of the transport mechanism used.

<a id="haystack_integrations.tools.mcp.mcp_tool.MCPClient.connect"></a>

#### MCPClient.connect

```python
@abstractmethod
async def connect() -> list[types.Tool]
```

Connect to an MCP server.

**Raises**:

- `MCPConnectionError`: If connection to the server fails

**Returns**:

List of available tools on the server

<a id="haystack_integrations.tools.mcp.mcp_tool.MCPClient.call_tool"></a>

#### MCPClient.call\_tool

```python
async def call_tool(tool_name: str, tool_args: dict[str, Any]) -> str
```

Call a tool on the connected MCP server.

**Arguments**:

- `tool_name`: Name of the tool to call
- `tool_args`: Arguments to pass to the tool

**Raises**:

- `MCPConnectionError`: If not connected to an MCP server
- `MCPInvocationError`: If the tool invocation fails

**Returns**:

JSON string representation of the tool invocation result

<a id="haystack_integrations.tools.mcp.mcp_tool.MCPClient.aclose"></a>

#### MCPClient.aclose

```python
async def aclose() -> None
```

Close the connection and clean up resources.

This method ensures all resources are properly released, even if errors occur.

<a id="haystack_integrations.tools.mcp.mcp_tool.StdioClient"></a>

### StdioClient

MCP client that connects to servers using stdio transport.

<a id="haystack_integrations.tools.mcp.mcp_tool.StdioClient.__init__"></a>

#### StdioClient.\_\_init\_\_

```python
def __init__(command: str,
             args: list[str] | None = None,
             env: dict[str, str | Secret] | None = None,
             max_retries: int = 3,
             base_delay: float = 1.0,
             max_delay: float = 30.0) -> None
```

Initialize a stdio MCP client.

**Arguments**:

- `command`: Command to run (e.g., "python", "node")
- `args`: Arguments to pass to the command
- `env`: Environment variables for the command
- `max_retries`: Maximum number of reconnection attempts
- `base_delay`: Base delay for exponential backoff in seconds

<a id="haystack_integrations.tools.mcp.mcp_tool.StdioClient.connect"></a>

#### StdioClient.connect

```python
async def connect() -> list[types.Tool]
```

Connect to an MCP server using stdio transport.

**Raises**:

- `MCPConnectionError`: If connection to the server fails

**Returns**:

List of available tools on the server

<a id="haystack_integrations.tools.mcp.mcp_tool.SSEClient"></a>

### SSEClient

MCP client that connects to servers using SSE transport.

<a id="haystack_integrations.tools.mcp.mcp_tool.SSEClient.__init__"></a>

#### SSEClient.\_\_init\_\_

```python
def __init__(server_info: "SSEServerInfo",
             max_retries: int = 3,
             base_delay: float = 1.0,
             max_delay: float = 30.0) -> None
```

Initialize an SSE MCP client using server configuration.

**Arguments**:

- `server_info`: Configuration object containing URL, token, timeout, etc.
- `max_retries`: Maximum number of reconnection attempts
- `base_delay`: Base delay for exponential backoff in seconds

<a id="haystack_integrations.tools.mcp.mcp_tool.SSEClient.connect"></a>

#### SSEClient.connect

```python
async def connect() -> list[types.Tool]
```

Connect to an MCP server using SSE transport.

Note: If both custom headers and token are provided, custom headers take precedence.

**Raises**:

- `MCPConnectionError`: If connection to the server fails

**Returns**:

List of available tools on the server

<a id="haystack_integrations.tools.mcp.mcp_tool.StreamableHttpClient"></a>

### StreamableHttpClient

MCP client that connects to servers using streamable HTTP transport.

<a id="haystack_integrations.tools.mcp.mcp_tool.StreamableHttpClient.__init__"></a>

#### StreamableHttpClient.\_\_init\_\_

```python
def __init__(server_info: "StreamableHttpServerInfo",
             max_retries: int = 3,
             base_delay: float = 1.0,
             max_delay: float = 30.0) -> None
```

Initialize a streamable HTTP MCP client using server configuration.

**Arguments**:

- `server_info`: Configuration object containing URL, token, timeout, etc.
- `max_retries`: Maximum number of reconnection attempts
- `base_delay`: Base delay for exponential backoff in seconds

<a id="haystack_integrations.tools.mcp.mcp_tool.StreamableHttpClient.connect"></a>

#### StreamableHttpClient.connect

```python
async def connect() -> list[types.Tool]
```

Connect to an MCP server using streamable HTTP transport.

Note: If both custom headers and token are provided, custom headers take precedence.

**Raises**:

- `MCPConnectionError`: If connection to the server fails

**Returns**:

List of available tools on the server

<a id="haystack_integrations.tools.mcp.mcp_tool.MCPServerInfo"></a>

### MCPServerInfo

Abstract base class for MCP server connection parameters.

This class defines the common interface for all MCP server connection types.

<a id="haystack_integrations.tools.mcp.mcp_tool.MCPServerInfo.create_client"></a>

#### MCPServerInfo.create\_client

```python
@abstractmethod
def create_client() -> MCPClient
```

Create an appropriate MCP client for this server info.

**Returns**:

An instance of MCPClient configured with this server info

<a id="haystack_integrations.tools.mcp.mcp_tool.MCPServerInfo.to_dict"></a>

#### MCPServerInfo.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serialize this server info to a dictionary.

**Returns**:

Dictionary representation of this server info

<a id="haystack_integrations.tools.mcp.mcp_tool.MCPServerInfo.from_dict"></a>

#### MCPServerInfo.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "MCPServerInfo"
```

Deserialize server info from a dictionary.

**Arguments**:

- `data`: Dictionary containing serialized server info

**Returns**:

Instance of the appropriate server info class

<a id="haystack_integrations.tools.mcp.mcp_tool.SSEServerInfo"></a>

### SSEServerInfo

Data class that encapsulates SSE MCP server connection parameters.

For authentication tokens containing sensitive data, you can use Secret objects
for secure handling and serialization:

```python
server_info = SSEServerInfo(
    url="https://my-mcp-server.com",
    token=Secret.from_env_var("API_KEY"),
)
```

For custom headers (e.g., non-standard authentication):

```python
# Single custom header with Secret
server_info = SSEServerInfo(
    url="https://my-mcp-server.com",
    headers={"X-API-Key": Secret.from_env_var("API_KEY")},
)

# Multiple headers (mix of Secret and plain strings)
server_info = SSEServerInfo(
    url="https://my-mcp-server.com",
    headers={
        "X-API-Key": Secret.from_env_var("API_KEY"),
        "X-Client-ID": "my-client-id",
    },
)
```

**Arguments**:

- `url`: Full URL of the MCP server (including /sse endpoint)
- `base_url`: Base URL of the MCP server (deprecated, use url instead)
- `token`: Authentication token for the server (optional, generates "Authorization: Bearer `<token>`" header)
- `headers`: Custom HTTP headers (optional, takes precedence over token parameter if provided)
- `timeout`: Connection timeout in seconds

<a id="haystack_integrations.tools.mcp.mcp_tool.SSEServerInfo.base_url"></a>

#### base\_url

deprecated

<a id="haystack_integrations.tools.mcp.mcp_tool.SSEServerInfo.__post_init__"></a>

#### SSEServerInfo.\_\_post\_init\_\_

```python
def __post_init__()
```

Validate that either url or base_url is provided.

<a id="haystack_integrations.tools.mcp.mcp_tool.SSEServerInfo.create_client"></a>

#### SSEServerInfo.create\_client

```python
def create_client() -> MCPClient
```

Create an SSE MCP client.

**Returns**:

Configured MCPClient instance

<a id="haystack_integrations.tools.mcp.mcp_tool.StreamableHttpServerInfo"></a>

### StreamableHttpServerInfo

Data class that encapsulates streamable HTTP MCP server connection parameters.

For authentication tokens containing sensitive data, you can use Secret objects
for secure handling and serialization:

```python
server_info = StreamableHttpServerInfo(
    url="https://my-mcp-server.com",
    token=Secret.from_env_var("API_KEY"),
)
```

For custom headers (e.g., non-standard authentication):

```python
# Single custom header with Secret
server_info = StreamableHttpServerInfo(
    url="https://my-mcp-server.com",
    headers={"X-API-Key": Secret.from_env_var("API_KEY")},
)

# Multiple headers (mix of Secret and plain strings)
server_info = StreamableHttpServerInfo(
    url="https://my-mcp-server.com",
    headers={
        "X-API-Key": Secret.from_env_var("API_KEY"),
        "X-Client-ID": "my-client-id",
    },
)
```

**Arguments**:

- `url`: Full URL of the MCP server (streamable HTTP endpoint)
- `token`: Authentication token for the server (optional, generates "Authorization: Bearer `<token>`" header)
- `headers`: Custom HTTP headers (optional, takes precedence over token parameter if provided)
- `timeout`: Connection timeout in seconds

<a id="haystack_integrations.tools.mcp.mcp_tool.StreamableHttpServerInfo.__post_init__"></a>

#### StreamableHttpServerInfo.\_\_post\_init\_\_

```python
def __post_init__()
```

Validate the URL.

<a id="haystack_integrations.tools.mcp.mcp_tool.StreamableHttpServerInfo.create_client"></a>

#### StreamableHttpServerInfo.create\_client

```python
def create_client() -> MCPClient
```

Create a streamable HTTP MCP client.

**Returns**:

Configured StreamableHttpClient instance

<a id="haystack_integrations.tools.mcp.mcp_tool.StdioServerInfo"></a>

### StdioServerInfo

Data class that encapsulates stdio MCP server connection parameters.

**Arguments**:

- `command`: Command to run (e.g., "python", "node")
- `args`: Arguments to pass to the command
- `env`: Environment variables for the command
For environment variables containing sensitive data, you can use Secret objects
for secure handling and serialization:

```python
server_info = StdioServerInfo(
    command="uv",
    args=["run", "my-mcp-server"],
    env={
        "WORKSPACE_PATH": "/path/to/workspace",  # Plain string
        "API_KEY": Secret.from_env_var("API_KEY"),  # Secret object
    }
)
```

Secret objects will be properly serialized and deserialized without exposing
the secret value, while plain strings will be preserved as-is. Use Secret objects
for sensitive data that needs to be handled securely.

<a id="haystack_integrations.tools.mcp.mcp_tool.StdioServerInfo.create_client"></a>

#### StdioServerInfo.create\_client

```python
def create_client() -> MCPClient
```

Create a stdio MCP client.

**Returns**:

Configured StdioMCPClient instance

<a id="haystack_integrations.tools.mcp.mcp_tool.MCPTool"></a>

### MCPTool

A Tool that represents a single tool from an MCP server.

This implementation uses the official MCP SDK for protocol handling while maintaining
compatibility with the Haystack tool ecosystem.

Response handling:
- Text and image content are supported and returned as JSON strings
- The JSON contains the structured response from the MCP server
- Use json.loads() to parse the response into a dictionary

Example using Streamable HTTP:
```python
import json
from haystack_integrations.tools.mcp import MCPTool, StreamableHttpServerInfo

# Create tool instance
tool = MCPTool(
    name="multiply",
    server_info=StreamableHttpServerInfo(url="http://localhost:8000/mcp")
)

# Use the tool and parse result
result_json = tool.invoke(a=5, b=3)
result = json.loads(result_json)
```

Example using SSE (deprecated):
```python
import json
from haystack.tools import MCPTool, SSEServerInfo

# Create tool instance
tool = MCPTool(
    name="add",
    server_info=SSEServerInfo(url="http://localhost:8000/sse")
)

# Use the tool and parse result
result_json = tool.invoke(a=5, b=3)
result = json.loads(result_json)
```

Example using stdio:
```python
import json
from haystack.tools import MCPTool, StdioServerInfo

# Create tool instance
tool = MCPTool(
    name="get_current_time",
    server_info=StdioServerInfo(command="python", args=["path/to/server.py"])
)

# Use the tool and parse result
result_json = tool.invoke(timezone="America/New_York")
result = json.loads(result_json)
```

<a id="haystack_integrations.tools.mcp.mcp_tool.MCPTool.__init__"></a>

#### MCPTool.\_\_init\_\_

```python
def __init__(name: str,
             server_info: MCPServerInfo,
             description: str | None = None,
             connection_timeout: int = 30,
             invocation_timeout: int = 30,
             eager_connect: bool = False)
```

Initialize the MCP tool.

**Arguments**:

- `name`: Name of the tool to use
- `server_info`: Server connection information
- `description`: Custom description (if None, server description will be used)
- `connection_timeout`: Timeout in seconds for server connection
- `invocation_timeout`: Default timeout in seconds for tool invocations
- `eager_connect`: If True, connect to server during initialization.
If False (default), defer connection until warm_up or first tool use,
whichever comes first.

**Raises**:

- `MCPConnectionError`: If connection to the server fails
- `MCPToolNotFoundError`: If no tools are available or the requested tool is not found
- `TimeoutError`: If connection times out

<a id="haystack_integrations.tools.mcp.mcp_tool.MCPTool.ainvoke"></a>

#### MCPTool.ainvoke

```python
async def ainvoke(**kwargs: Any) -> str
```

Asynchronous tool invocation.

**Arguments**:

- `kwargs`: Arguments to pass to the tool

**Raises**:

- `MCPInvocationError`: If the tool invocation fails
- `TimeoutError`: If the operation times out

**Returns**:

JSON string representation of the tool invocation result

<a id="haystack_integrations.tools.mcp.mcp_tool.MCPTool.warm_up"></a>

#### MCPTool.warm\_up

```python
def warm_up() -> None
```

Connect and fetch the tool schema if eager_connect is turned off.

<a id="haystack_integrations.tools.mcp.mcp_tool.MCPTool.to_dict"></a>

#### MCPTool.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the MCPTool to a dictionary.

The serialization preserves all information needed to recreate the tool,
including server connection parameters and timeout settings. Note that the
active connection is not maintained.

**Returns**:

Dictionary with serialized data in the format:
`{"type": fully_qualified_class_name, "data": {parameters}}`

<a id="haystack_integrations.tools.mcp.mcp_tool.MCPTool.from_dict"></a>

#### MCPTool.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "Tool"
```

Deserializes the MCPTool from a dictionary.

This method reconstructs an MCPTool instance from a serialized dictionary,
including recreating the server_info object. A new connection will be established
to the MCP server during initialization.

**Arguments**:

- `data`: Dictionary containing serialized tool data

**Raises**:

- `None`: Various exceptions if connection fails

**Returns**:

A fully initialized MCPTool instance

<a id="haystack_integrations.tools.mcp.mcp_tool.MCPTool.close"></a>

#### MCPTool.close

```python
def close()
```

Close the tool synchronously.

<a id="haystack_integrations.tools.mcp.mcp_tool.MCPTool.__del__"></a>

#### MCPTool.\_\_del\_\_

```python
def __del__()
```

Cleanup resources when the tool is garbage collected.

<a id="haystack_integrations.tools.mcp.mcp_tool._MCPClientSessionManager"></a>

### \_MCPClientSessionManager

Runs an MCPClient connect/close inside the AsyncExecutor's event loop.

Life-cycle:
  1.  Create the worker to schedule a long-running coroutine in the
      dedicated background loop.
  2.  The coroutine calls *connect* on mcp client; when it has the tool list it fulfils
      a concurrent future so the synchronous thread can continue.
  3.  It then waits on an `asyncio.Event`.
  4.  `stop()` sets the event from any thread. The same coroutine then calls
      *close()* on mcp client and finishes without the dreaded
      `Attempted to exit cancel scope in a different task than it was entered in` error
      thus properly closing the client.

<a id="haystack_integrations.tools.mcp.mcp_tool._MCPClientSessionManager.tools"></a>

#### \_MCPClientSessionManager.tools

```python
def tools() -> list[types.Tool]
```

Return the tool list already collected during startup.

<a id="haystack_integrations.tools.mcp.mcp_tool._MCPClientSessionManager.stop"></a>

#### \_MCPClientSessionManager.stop

```python
def stop() -> None
```

Request the worker to shut down and block until done.

<a id="haystack_integrations.tools.mcp.mcp_toolset"></a>

## Module haystack\_integrations.tools.mcp.mcp\_toolset

<a id="haystack_integrations.tools.mcp.mcp_toolset.MCPToolset"></a>

### MCPToolset

A Toolset that connects to an MCP (Model Context Protocol) server and provides
access to its tools.

MCPToolset dynamically discovers and loads all tools from any MCP-compliant server,
supporting both network-based streaming connections (Streamable HTTP, SSE) and local
process-based stdio connections.
This dual connectivity allows for integrating with both remote and local MCP servers.

Example using MCPToolset in a Haystack Pipeline:
```python
# Prerequisites:
# 1. pip install uvx mcp-server-time  # Install required MCP server and tools
# 2. export OPENAI_API_KEY="your-api-key"  # Set up your OpenAI API key

import os
from haystack import Pipeline
from haystack.components.converters import OutputAdapter
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.tools import ToolInvoker
from haystack.dataclasses import ChatMessage
from haystack_integrations.tools.mcp import MCPToolset, StdioServerInfo

# Create server info for the time service (can also use SSEServerInfo for remote servers)
server_info = StdioServerInfo(command="uvx", args=["mcp-server-time", "--local-timezone=Europe/Berlin"])

# Create the toolset - this will automatically discover all available tools
# You can optionally specify which tools to include
mcp_toolset = MCPToolset(
    server_info=server_info,
    tool_names=["get_current_time"]  # Only include the get_current_time tool
)

# Create a pipeline with the toolset
pipeline = Pipeline()
pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini", tools=mcp_toolset))
pipeline.add_component("tool_invoker", ToolInvoker(tools=mcp_toolset))
pipeline.add_component(
    "adapter",
    OutputAdapter(
        template="{{ initial_msg + initial_tool_messages + tool_messages }}",
        output_type=list[ChatMessage],
        unsafe=True,
    ),
)
pipeline.add_component("response_llm", OpenAIChatGenerator(model="gpt-4o-mini"))
pipeline.connect("llm.replies", "tool_invoker.messages")
pipeline.connect("llm.replies", "adapter.initial_tool_messages")
pipeline.connect("tool_invoker.tool_messages", "adapter.tool_messages")
pipeline.connect("adapter.output", "response_llm.messages")

# Run the pipeline with a user question
user_input = "What is the time in New York? Be brief."
user_input_msg = ChatMessage.from_user(text=user_input)

result = pipeline.run({"llm": {"messages": [user_input_msg]}, "adapter": {"initial_msg": [user_input_msg]}})
print(result["response_llm"]["replies"][0].text)
```

You can also use the toolset via Streamable HTTP to talk to remote servers:
```python
from haystack_integrations.tools.mcp import MCPToolset, StreamableHttpServerInfo

# Create the toolset with streamable HTTP connection
toolset = MCPToolset(
    server_info=StreamableHttpServerInfo(url="http://localhost:8000/mcp"),
    tool_names=["multiply"]  # Optional: only include specific tools
)
# Use the toolset as shown in the pipeline example above
```

Example using SSE (deprecated):
```python
from haystack_integrations.tools.mcp import MCPToolset, SSEServerInfo
from haystack.components.tools import ToolInvoker

# Create the toolset with an SSE connection
sse_toolset = MCPToolset(
    server_info=SSEServerInfo(url="http://some-remote-server.com:8000/sse"),
    tool_names=["add", "subtract"]  # Only include specific tools
)

# Use the toolset as shown in the pipeline example above
```

<a id="haystack_integrations.tools.mcp.mcp_toolset.MCPToolset.__init__"></a>

#### MCPToolset.\_\_init\_\_

```python
def __init__(server_info: MCPServerInfo,
             tool_names: list[str] | None = None,
             connection_timeout: float = 30.0,
             invocation_timeout: float = 30.0,
             eager_connect: bool = False)
```

Initialize the MCP toolset.

**Arguments**:

- `server_info`: Connection information for the MCP server
- `tool_names`: Optional list of tool names to include. If provided, only tools with
matching names will be added to the toolset.
- `connection_timeout`: Timeout in seconds for server connection
- `invocation_timeout`: Default timeout in seconds for tool invocations
- `eager_connect`: If True, connect to server and load tools during initialization.
If False (default), defer connection to warm_up.

**Raises**:

- `MCPToolNotFoundError`: If any of the specified tool names are not found on the server

<a id="haystack_integrations.tools.mcp.mcp_toolset.MCPToolset.warm_up"></a>

#### MCPToolset.warm\_up

```python
def warm_up() -> None
```

Connect and load tools when eager_connect is turned off.

This method is automatically called by ``ToolInvoker.warm_up()`` and ``Pipeline.warm_up()``.
You can also call it directly before using the toolset to ensure all tool schemas
are available without performing a real invocation.

<a id="haystack_integrations.tools.mcp.mcp_toolset.MCPToolset.to_dict"></a>

#### MCPToolset.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serialize the MCPToolset to a dictionary.

**Returns**:

A dictionary representation of the MCPToolset

<a id="haystack_integrations.tools.mcp.mcp_toolset.MCPToolset.from_dict"></a>

#### MCPToolset.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "MCPToolset"
```

Deserialize an MCPToolset from a dictionary.

**Arguments**:

- `data`: Dictionary representation of the MCPToolset

**Returns**:

A new MCPToolset instance

<a id="haystack_integrations.tools.mcp.mcp_toolset.MCPToolset.close"></a>

#### MCPToolset.close

```python
def close()
```

Close the underlying MCP client safely.

