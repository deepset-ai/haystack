---
title: "MCP"
id: integrations-mcp
description: "MCP integration for Haystack"
slug: "/integrations-mcp"
---


## `haystack_integrations.tools.mcp.mcp_tool`

### `AsyncExecutor`

Thread-safe event loop executor for running async code from sync contexts.

#### `get_instance`

```python
get_instance() -> AsyncExecutor
```

Get or create the global singleton executor instance.

#### `__init__`

```python
__init__()
```

Initialize a dedicated event loop

#### `run`

```python
run(coro: Coroutine[Any, Any, Any], timeout: float | None = None) -> Any
```

Run a coroutine in the event loop.

**Parameters:**

- **coro** (<code>Coroutine\[Any, Any, Any\]</code>) – Coroutine to execute
- **timeout** (<code>float | None</code>) – Optional timeout in seconds

**Returns:**

- <code>Any</code> – Result of the coroutine

**Raises:**

- <code>TimeoutError</code> – If execution exceeds timeout

#### `get_loop`

```python
get_loop()
```

Get the event loop.

**Returns:**

- – The event loop

#### `run_background`

```python
run_background(
    coro_factory: Callable[[asyncio.Event], Coroutine[Any, Any, Any]],
    timeout: float | None = None,
) -> tuple[concurrent.futures.Future[Any], asyncio.Event]
```

Schedule `coro_factory` to run in the executor's event loop **without** blocking the
caller thread.

The factory receives an :class:`asyncio.Event` that can be used to cooperatively shut
the coroutine down. The method returns **both** the concurrent future (to observe
completion or failure) and the created *stop_event* so that callers can signal termination.

**Parameters:**

- **coro_factory** (<code>Callable\\[[Event\], Coroutine\[Any, Any, Any\]\]</code>) – A callable receiving the stop_event and returning the coroutine to execute.
- **timeout** (<code>float | None</code>) – Optional timeout while waiting for the stop_event to be created.

**Returns:**

- <code>tuple\[Future\[Any\], Event\]</code> – Tuple `(future, stop_event)`.

#### `shutdown`

```python
shutdown(timeout: float = 2) -> None
```

Shut down the background event loop and thread.

**Parameters:**

- **timeout** (<code>float</code>) – Timeout in seconds for shutting down the event loop

### `MCPError`

Bases: <code>Exception</code>

Base class for MCP-related errors.

#### `__init__`

```python
__init__(message: str) -> None
```

Initialize the MCPError.

**Parameters:**

- **message** (<code>str</code>) – Descriptive error message

### `MCPConnectionError`

Bases: <code>MCPError</code>

Error connecting to MCP server.

#### `__init__`

```python
__init__(
    message: str,
    server_info: MCPServerInfo | None = None,
    operation: str | None = None,
) -> None
```

Initialize the MCPConnectionError.

**Parameters:**

- **message** (<code>str</code>) – Descriptive error message
- **server_info** (<code>MCPServerInfo | None</code>) – Server connection information that was used
- **operation** (<code>str | None</code>) – Name of the operation that was being attempted

### `MCPToolNotFoundError`

Bases: <code>MCPError</code>

Error when a tool is not found on the server.

#### `__init__`

```python
__init__(
    message: str, tool_name: str, available_tools: list[str] | None = None
) -> None
```

Initialize the MCPToolNotFoundError.

**Parameters:**

- **message** (<code>str</code>) – Descriptive error message
- **tool_name** (<code>str</code>) – Name of the tool that was requested but not found
- **available_tools** (<code>list\[str\] | None</code>) – List of available tool names, if known

### `MCPInvocationError`

Bases: <code>ToolInvocationError</code>

Error during tool invocation.

#### `__init__`

```python
__init__(
    message: str, tool_name: str, tool_args: dict[str, Any] | None = None
) -> None
```

Initialize the MCPInvocationError.

**Parameters:**

- **message** (<code>str</code>) – Descriptive error message
- **tool_name** (<code>str</code>) – Name of the tool that was being invoked
- **tool_args** (<code>dict\[str, Any\] | None</code>) – Arguments that were passed to the tool

### `MCPClient`

Bases: <code>ABC</code>

Abstract base class for MCP clients.

This class defines the common interface and shared functionality for all MCP clients,
regardless of the transport mechanism used.

#### `connect`

```python
connect() -> list[types.Tool]
```

Connect to an MCP server.

**Returns:**

- <code>list\[Tool\]</code> – List of available tools on the server

**Raises:**

- <code>MCPConnectionError</code> – If connection to the server fails

#### `call_tool`

```python
call_tool(tool_name: str, tool_args: dict[str, Any]) -> str
```

Call a tool on the connected MCP server.

**Parameters:**

- **tool_name** (<code>str</code>) – Name of the tool to call
- **tool_args** (<code>dict\[str, Any\]</code>) – Arguments to pass to the tool

**Returns:**

- <code>str</code> – JSON string representation of the tool invocation result

**Raises:**

- <code>MCPConnectionError</code> – If not connected to an MCP server
- <code>MCPInvocationError</code> – If the tool invocation fails

#### `aclose`

```python
aclose() -> None
```

Close the connection and clean up resources.

This method ensures all resources are properly released, even if errors occur.

### `StdioClient`

Bases: <code>MCPClient</code>

MCP client that connects to servers using stdio transport.

#### `__init__`

```python
__init__(
    command: str,
    args: list[str] | None = None,
    env: dict[str, str | Secret] | None = None,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> None
```

Initialize a stdio MCP client.

**Parameters:**

- **command** (<code>str</code>) – Command to run (e.g., "python", "node")
- **args** (<code>list\[str\] | None</code>) – Arguments to pass to the command
- **env** (<code>dict\[str, str | Secret\] | None</code>) – Environment variables for the command
- **max_retries** (<code>int</code>) – Maximum number of reconnection attempts
- **base_delay** (<code>float</code>) – Base delay for exponential backoff in seconds

#### `connect`

```python
connect() -> list[types.Tool]
```

Connect to an MCP server using stdio transport.

**Returns:**

- <code>list\[Tool\]</code> – List of available tools on the server

**Raises:**

- <code>MCPConnectionError</code> – If connection to the server fails

### `SSEClient`

Bases: <code>MCPClient</code>

MCP client that connects to servers using SSE transport.

#### `__init__`

```python
__init__(
    server_info: SSEServerInfo,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> None
```

Initialize an SSE MCP client using server configuration.

**Parameters:**

- **server_info** (<code>SSEServerInfo</code>) – Configuration object containing URL, token, timeout, etc.
- **max_retries** (<code>int</code>) – Maximum number of reconnection attempts
- **base_delay** (<code>float</code>) – Base delay for exponential backoff in seconds

#### `connect`

```python
connect() -> list[types.Tool]
```

Connect to an MCP server using SSE transport.

Note: If both custom headers and token are provided, custom headers take precedence.

**Returns:**

- <code>list\[Tool\]</code> – List of available tools on the server

**Raises:**

- <code>MCPConnectionError</code> – If connection to the server fails

### `StreamableHttpClient`

Bases: <code>MCPClient</code>

MCP client that connects to servers using streamable HTTP transport.

#### `__init__`

```python
__init__(
    server_info: StreamableHttpServerInfo,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> None
```

Initialize a streamable HTTP MCP client using server configuration.

**Parameters:**

- **server_info** (<code>StreamableHttpServerInfo</code>) – Configuration object containing URL, token, timeout, etc.
- **max_retries** (<code>int</code>) – Maximum number of reconnection attempts
- **base_delay** (<code>float</code>) – Base delay for exponential backoff in seconds

#### `connect`

```python
connect() -> list[types.Tool]
```

Connect to an MCP server using streamable HTTP transport.

Note: If both custom headers and token are provided, custom headers take precedence.

**Returns:**

- <code>list\[Tool\]</code> – List of available tools on the server

**Raises:**

- <code>MCPConnectionError</code> – If connection to the server fails

### `MCPServerInfo`

Bases: <code>ABC</code>

Abstract base class for MCP server connection parameters.

This class defines the common interface for all MCP server connection types.

#### `create_client`

```python
create_client() -> MCPClient
```

Create an appropriate MCP client for this server info.

**Returns:**

- <code>MCPClient</code> – An instance of MCPClient configured with this server info

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serialize this server info to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary representation of this server info

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> MCPServerInfo
```

Deserialize server info from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary containing serialized server info

**Returns:**

- <code>MCPServerInfo</code> – Instance of the appropriate server info class

### `SSEServerInfo`

Bases: <code>MCPServerInfo</code>

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

**Parameters:**

- **url** (<code>str | None</code>) – Full URL of the MCP server (including /sse endpoint)
- **base_url** (<code>str | None</code>) – Base URL of the MCP server (deprecated, use url instead)
- **token** (<code>str | Secret | None</code>) – Authentication token for the server (optional, generates "Authorization: Bearer `<token>`" header)
- **headers** (<code>dict\[str, str | Secret\] | None</code>) – Custom HTTP headers (optional, takes precedence over token parameter if provided)
- **timeout** (<code>int</code>) – Connection timeout in seconds

#### `create_client`

```python
create_client() -> MCPClient
```

Create an SSE MCP client.

**Returns:**

- <code>MCPClient</code> – Configured MCPClient instance

### `StreamableHttpServerInfo`

Bases: <code>MCPServerInfo</code>

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

**Parameters:**

- **url** (<code>str</code>) – Full URL of the MCP server (streamable HTTP endpoint)
- **token** (<code>str | Secret | None</code>) – Authentication token for the server (optional, generates "Authorization: Bearer `<token>`" header)
- **headers** (<code>dict\[str, str | Secret\] | None</code>) – Custom HTTP headers (optional, takes precedence over token parameter if provided)
- **timeout** (<code>int</code>) – Connection timeout in seconds

#### `create_client`

```python
create_client() -> MCPClient
```

Create a streamable HTTP MCP client.

**Returns:**

- <code>MCPClient</code> – Configured StreamableHttpClient instance

### `StdioServerInfo`

Bases: <code>MCPServerInfo</code>

Data class that encapsulates stdio MCP server connection parameters.

**Parameters:**

- **command** (<code>str</code>) – Command to run (e.g., "python", "node")
- **args** (<code>list\[str\] | None</code>) – Arguments to pass to the command
- **env** (<code>dict\[str, str | Secret\] | None</code>) – Environment variables for the command

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

#### `create_client`

```python
create_client() -> MCPClient
```

Create a stdio MCP client.

**Returns:**

- <code>MCPClient</code> – Configured StdioMCPClient instance

### `MCPTool`

Bases: <code>Tool</code>

A Tool that represents a single tool from an MCP server.

This implementation uses the official MCP SDK for protocol handling while maintaining
compatibility with the Haystack tool ecosystem.

Response handling:

- Text and image content are supported and returned as JSON strings
- The JSON contains the structured response from the MCP server
- Use json.loads() to parse the response into a dictionary

State-mapping support:

- MCPTool supports state-mapping parameters (`outputs_to_string`, `inputs_from_state`, `outputs_to_state`)
- These enable integration with Agent state for automatic parameter injection and output handling
- See the `__init__` method documentation for details on each parameter

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

#### `__init__`

```python
__init__(
    name: str,
    server_info: MCPServerInfo,
    description: str | None = None,
    connection_timeout: int = 30,
    invocation_timeout: int = 30,
    eager_connect: bool = False,
    outputs_to_string: dict[str, Any] | None = None,
    inputs_from_state: dict[str, str] | None = None,
    outputs_to_state: dict[str, dict[str, Any]] | None = None,
)
```

Initialize the MCP tool.

**Parameters:**

- **name** (<code>str</code>) – Name of the tool to use
- **server_info** (<code>MCPServerInfo</code>) – Server connection information
- **description** (<code>str | None</code>) – Custom description (if None, server description will be used)
- **connection_timeout** (<code>int</code>) – Timeout in seconds for server connection
- **invocation_timeout** (<code>int</code>) – Default timeout in seconds for tool invocations
- **eager_connect** (<code>bool</code>) – If True, connect to server during initialization.
  If False (default), defer connection until warm_up or first tool use,
  whichever comes first.
- **outputs_to_string** (<code>dict\[str, Any\] | None</code>) – Optional dictionary defining how tool outputs should be converted into a string.
  If the source is provided only the specified output key is sent to the handler.
  If the source is omitted the whole tool result is sent to the handler.
  Example: `{"source": "docs", "handler": my_custom_function}`
- **inputs_from_state** (<code>dict\[str, str\] | None</code>) – Optional dictionary mapping state keys to tool parameter names.
  Example: `{"repository": "repo"}` maps state's "repository" to tool's "repo" parameter.
- **outputs_to_state** (<code>dict\[str, dict\[str, Any\]\] | None</code>) – Optional dictionary defining how tool outputs map to keys within state as well as
  optional handlers. If the source is provided only the specified output key is sent
  to the handler.
  Example with source: `{"documents": {"source": "docs", "handler": custom_handler}}`
  Example without source: `{"documents": {"handler": custom_handler}}`

**Raises:**

- <code>MCPConnectionError</code> – If connection to the server fails
- <code>MCPToolNotFoundError</code> – If no tools are available or the requested tool is not found
- <code>TimeoutError</code> – If connection times out

#### `ainvoke`

```python
ainvoke(**kwargs: Any) -> str | dict[str, Any]
```

Asynchronous tool invocation.

**Parameters:**

- **kwargs** (<code>Any</code>) – Arguments to pass to the tool

**Returns:**

- <code>str | dict\[str, Any\]</code> – JSON string or dictionary representation of the tool invocation result.
  Returns a dictionary when outputs_to_state is configured to enable state updates.

**Raises:**

- <code>MCPInvocationError</code> – If the tool invocation fails
- <code>TimeoutError</code> – If the operation times out

#### `warm_up`

```python
warm_up() -> None
```

Connect and fetch the tool schema if eager_connect is turned off.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the MCPTool to a dictionary.

The serialization preserves all information needed to recreate the tool,
including server connection parameters, timeout settings, and state-mapping parameters.
Note that the active connection is not maintained.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data in the format:
  `{"type": fully_qualified_class_name, "data": {parameters}}`

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> Tool
```

Deserializes the MCPTool from a dictionary.

This method reconstructs an MCPTool instance from a serialized dictionary,
including recreating the server_info object and state-mapping parameters.
A new connection will be established to the MCP server during initialization.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary containing serialized tool data

**Returns:**

- <code>Tool</code> – A fully initialized MCPTool instance

#### `close`

```python
close()
```

Close the tool synchronously.

## `haystack_integrations.tools.mcp.mcp_toolset`

### `MCPToolset`

Bases: <code>Toolset</code>

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

Example with state configuration for Agent integration:

```python
from haystack_integrations.tools.mcp import MCPToolset, StdioServerInfo

# Create the toolset with per-tool state configuration
# This enables tools to read from and write to the Agent's State
toolset = MCPToolset(
    server_info=StdioServerInfo(command="uvx", args=["mcp-server-git"]),
    tool_names=["git_status", "git_diff", "git_log"],

    # Maps the state key "repository" to the tool parameter "repo_path" for each tool
    inputs_from_state={
        "git_status": {"repository": "repo_path"},
        "git_diff": {"repository": "repo_path"},
        "git_log": {"repository": "repo_path"},
    },
    # Map tool outputs to state keys for each tool
    outputs_to_state={
        "git_status": {"status_result": {"source": "status"}},  # Extract "status" from output
        "git_diff": {"diff_result": {}},  # use full output with default handling
    },
)
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

#### `__init__`

```python
__init__(
    server_info: MCPServerInfo,
    tool_names: list[str] | None = None,
    connection_timeout: float = 30.0,
    invocation_timeout: float = 30.0,
    eager_connect: bool = False,
    inputs_from_state: dict[str, dict[str, str]] | None = None,
    outputs_to_state: dict[str, dict[str, dict[str, Any]]] | None = None,
    outputs_to_string: dict[str, dict[str, Any]] | None = None,
)
```

Initialize the MCP toolset.

**Parameters:**

- **server_info** (<code>MCPServerInfo</code>) – Connection information for the MCP server
- **tool_names** (<code>list\[str\] | None</code>) – Optional list of tool names to include. If provided, only tools with
  matching names will be added to the toolset.
- **connection_timeout** (<code>float</code>) – Timeout in seconds for server connection
- **invocation_timeout** (<code>float</code>) – Default timeout in seconds for tool invocations
- **eager_connect** (<code>bool</code>) – If True, connect to server and load tools during initialization.
  If False (default), defer connection to warm_up.
- **inputs_from_state** (<code>dict\[str, dict\[str, str\]\] | None</code>) – Optional dictionary mapping tool names to their inputs_from_state config.
  Each config maps state keys to tool parameter names.
  Tool names should match available tools from the server; a warning is logged for
  unknown tools. Note: With Haystack >= 2.22.0, parameter names are validated;
  ValueError is raised for invalid parameters. With earlier versions, invalid
  parameters fail at runtime.
  Example: `{"git_status": {"repository": "repo_path"}}`
- **outputs_to_state** (<code>dict\[str, dict\[str, dict\[str, Any\]\]\] | None</code>) – Optional dictionary mapping tool names to their outputs_to_state config.
  Each config defines how tool outputs map to state keys with optional handlers.
  Tool names should match available tools from the server; a warning is logged for
  unknown tools.
  Example: `{"git_status": {"status_result": {"source": "status"}}}`
- **outputs_to_string** (<code>dict\[str, dict\[str, Any\]\] | None</code>) – Optional dictionary mapping tool names to their outputs_to_string config.
  Each config defines how tool outputs are converted to strings.
  Tool names should match available tools from the server; a warning is logged for
  unknown tools.
  Example: `{"git_diff": {"source": "diff", "handler": format_diff}}`

**Raises:**

- <code>MCPToolNotFoundError</code> – If any of the specified tool names are not found on the server
- <code>ValueError</code> – If parameter names in inputs_from_state are invalid (Haystack >= 2.22.0 only)

#### `warm_up`

```python
warm_up() -> None
```

Connect and load tools when eager_connect is turned off.

This method is automatically called by `ToolInvoker.warm_up()` and `Pipeline.warm_up()`.
You can also call it directly before using the toolset to ensure all tool schemas
are available without performing a real invocation.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serialize the MCPToolset to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary representation of the MCPToolset

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> MCPToolset
```

Deserialize an MCPToolset from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary representation of the MCPToolset

**Returns:**

- <code>MCPToolset</code> – A new MCPToolset instance

#### `close`

```python
close()
```

Close the underlying MCP client safely.
