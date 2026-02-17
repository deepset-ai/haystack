---
title: "Data Classes"
id: data-classes-api
description: "Core classes that carry data through the system."
slug: "/data-classes-api"
---


## `ExtractedAnswer`

### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serialize the object to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Serialized dictionary representation of the object.

### `from_dict`

```python
from_dict(data: dict[str, Any]) -> ExtractedAnswer
```

Deserialize the object from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary representation of the object.

**Returns:**

- <code>ExtractedAnswer</code> – Deserialized object.

## `GeneratedAnswer`

### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serialize the object to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Serialized dictionary representation of the object.

### `from_dict`

```python
from_dict(data: dict[str, Any]) -> GeneratedAnswer
```

Deserialize the object from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary representation of the object.

**Returns:**

- <code>GeneratedAnswer</code> – Deserialized object.

## `Breakpoint`

A dataclass to hold a breakpoint for a component.

**Parameters:**

- **component_name** (<code>str</code>) – The name of the component where the breakpoint is set.
- **visit_count** (<code>int</code>) – The number of times the component must be visited before the breakpoint is triggered.
- **snapshot_file_path** (<code>str | None</code>) – Optional path to store a snapshot of the pipeline when the breakpoint is hit.
  This is useful for debugging purposes, allowing you to inspect the state of the pipeline at the time of the
  breakpoint and to resume execution from that point.

### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Convert the Breakpoint to a dictionary representation.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary containing the component name, visit count, and debug path.

### `from_dict`

```python
from_dict(data: dict) -> Breakpoint
```

Populate the Breakpoint from a dictionary representation.

**Parameters:**

- **data** (<code>dict</code>) – A dictionary containing the component name, visit count, and debug path.

**Returns:**

- <code>Breakpoint</code> – An instance of Breakpoint.

## `ToolBreakpoint`

Bases: <code>Breakpoint</code>

A dataclass representing a breakpoint specific to tools used within an Agent component.

Inherits from Breakpoint and adds the ability to target individual tools. If `tool_name` is None,
the breakpoint applies to all tools within the Agent component.

**Parameters:**

- **tool_name** (<code>str | None</code>) – The name of the tool to target within the Agent component. If None, applies to all tools.

### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Convert the Breakpoint to a dictionary representation.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary containing the component name, visit count, and debug path.

### `from_dict`

```python
from_dict(data: dict) -> Breakpoint
```

Populate the Breakpoint from a dictionary representation.

**Parameters:**

- **data** (<code>dict</code>) – A dictionary containing the component name, visit count, and debug path.

**Returns:**

- <code>Breakpoint</code> – An instance of Breakpoint.

## `AgentBreakpoint`

A dataclass representing a breakpoint tied to an Agent’s execution.

This allows for debugging either a specific component (e.g., the chat generator) or a tool used by the agent.
It enforces constraints on which component names are valid for each breakpoint type.

**Parameters:**

- **agent_name** (<code>str</code>) – The name of the agent component in a pipeline where the breakpoint is set.
- **break_point** (<code>Breakpoint | ToolBreakpoint</code>) – An instance of Breakpoint or ToolBreakpoint indicating where to break execution.

**Raises:**

- <code>ValueError</code> – If the component_name is invalid for the given breakpoint type:
- Breakpoint must have component_name='chat_generator'.
- ToolBreakpoint must have component_name='tool_invoker'.

### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Convert the AgentBreakpoint to a dictionary representation.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary containing the agent name and the breakpoint details.

### `from_dict`

```python
from_dict(data: dict) -> AgentBreakpoint
```

Populate the AgentBreakpoint from a dictionary representation.

**Parameters:**

- **data** (<code>dict</code>) – A dictionary containing the agent name and the breakpoint details.

**Returns:**

- <code>AgentBreakpoint</code> – An instance of AgentBreakpoint.

## `AgentSnapshot`

### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Convert the AgentSnapshot to a dictionary representation.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary containing the agent state, timestamp, and breakpoint.

### `from_dict`

```python
from_dict(data: dict) -> AgentSnapshot
```

Populate the AgentSnapshot from a dictionary representation.

**Parameters:**

- **data** (<code>dict</code>) – A dictionary containing the agent state, timestamp, and breakpoint.

**Returns:**

- <code>AgentSnapshot</code> – An instance of AgentSnapshot.

## `PipelineState`

A dataclass to hold the state of the pipeline at a specific point in time.

**Parameters:**

- **component_visits** (<code>dict\[str, int\]</code>) – A dictionary mapping component names to their visit counts.
- **inputs** (<code>dict\[str, Any\]</code>) – The inputs processed by the pipeline at the time of the snapshot.
- **pipeline_outputs** (<code>dict\[str, Any\]</code>) – Dictionary containing the final outputs of the pipeline up to the breakpoint.

### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Convert the PipelineState to a dictionary representation.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary containing the inputs, component visits,
  and pipeline outputs.

### `from_dict`

```python
from_dict(data: dict) -> PipelineState
```

Populate the PipelineState from a dictionary representation.

**Parameters:**

- **data** (<code>dict</code>) – A dictionary containing the inputs, component visits,
  and pipeline outputs.

**Returns:**

- <code>PipelineState</code> – An instance of PipelineState.

## `PipelineSnapshot`

A dataclass to hold a snapshot of the pipeline at a specific point in time.

**Parameters:**

- **original_input_data** (<code>dict\[str, Any\]</code>) – The original input data provided to the pipeline.
- **ordered_component_names** (<code>list\[str\]</code>) – A list of component names in the order they were visited.
- **pipeline_state** (<code>PipelineState</code>) – The state of the pipeline at the time of the snapshot.
- **break_point** (<code>AgentBreakpoint | Breakpoint</code>) – The breakpoint that triggered the snapshot.
- **agent_snapshot** (<code>AgentSnapshot | None</code>) – Optional agent snapshot if the breakpoint is an agent breakpoint.
- **timestamp** (<code>datetime | None</code>) – A timestamp indicating when the snapshot was taken.
- **include_outputs_from** (<code>set\[str\]</code>) – Set of component names whose outputs should be included in the pipeline results.

### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Convert the PipelineSnapshot to a dictionary representation.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary containing the pipeline state, timestamp, breakpoint, agent snapshot, original input data,
  ordered component names, include_outputs_from, and pipeline outputs.

### `from_dict`

```python
from_dict(data: dict) -> PipelineSnapshot
```

Populate the PipelineSnapshot from a dictionary representation.

**Parameters:**

- **data** (<code>dict</code>) – A dictionary containing the pipeline state, timestamp, breakpoint, agent snapshot, original input
  data, ordered component names, include_outputs_from, and pipeline outputs.

## `ByteStream`

Base data class representing a binary object in the Haystack API.

**Parameters:**

- **data** (<code>bytes</code>) – The binary data stored in Bytestream.
- **meta** (<code>dict\[str, Any\]</code>) – Additional metadata to be stored with the ByteStream.
- **mime_type** (<code>str | None</code>) – The mime type of the binary data.

### `to_file`

```python
to_file(destination_path: Path) -> None
```

Write the ByteStream to a file. Note: the metadata will be lost.

**Parameters:**

- **destination_path** (<code>Path</code>) – The path to write the ByteStream to.

### `from_file_path`

```python
from_file_path(filepath: Path, mime_type: str | None = None, meta: dict[str, Any] | None = None, guess_mime_type: bool = False) -> ByteStream
```

Create a ByteStream from the contents read from a file.

**Parameters:**

- **filepath** (<code>Path</code>) – A valid path to a file.
- **mime_type** (<code>str | None</code>) – The mime type of the file.
- **meta** (<code>dict\[str, Any\] | None</code>) – Additional metadata to be stored with the ByteStream.
- **guess_mime_type** (<code>bool</code>) – Whether to guess the mime type from the file.

### `from_string`

```python
from_string(text: str, encoding: str = 'utf-8', mime_type: str | None = None, meta: dict[str, Any] | None = None) -> ByteStream
```

Create a ByteStream encoding a string.

**Parameters:**

- **text** (<code>str</code>) – The string to encode
- **encoding** (<code>str</code>) – The encoding used to convert the string into bytes
- **mime_type** (<code>str | None</code>) – The mime type of the file.
- **meta** (<code>dict\[str, Any\] | None</code>) – Additional metadata to be stored with the ByteStream.

### `to_string`

```python
to_string(encoding: str = 'utf-8') -> str
```

Convert the ByteStream to a string, metadata will not be included.

**Parameters:**

- **encoding** (<code>str</code>) – The encoding used to convert the bytes to a string. Defaults to "utf-8".

**Returns:**

- <code>str</code> – The string representation of the ByteStream.

**Raises:**

- <code>UnicodeDecodeError</code> – If the ByteStream data cannot be decoded with the specified encoding.

### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Convert the ByteStream to a dictionary representation.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with keys 'data', 'meta', and 'mime_type'.

### `from_dict`

```python
from_dict(data: dict[str, Any]) -> ByteStream
```

Create a ByteStream from a dictionary representation.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – A dictionary with keys 'data', 'meta', and 'mime_type'.

**Returns:**

- <code>ByteStream</code> – A ByteStream instance.

## `ChatRole`

Bases: <code>str</code>, <code>Enum</code>

Enumeration representing the roles within a chat.

### `from_str`

```python
from_str(string: str) -> ChatRole
```

Convert a string to a ChatRole enum.

## `TextContent`

The textual content of a chat message.

**Parameters:**

- **text** (<code>str</code>) – The text content of the message.

### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Convert TextContent into a dictionary.

### `from_dict`

```python
from_dict(data: dict[str, Any]) -> TextContent
```

Create a TextContent from a dictionary.

## `ToolCall`

Represents a Tool call prepared by the model, usually contained in an assistant message.

**Parameters:**

- **id** (<code>str | None</code>) – The ID of the Tool call.
- **tool_name** (<code>str</code>) – The name of the Tool to call.
- **arguments** (<code>dict\[str, Any\]</code>) – The arguments to call the Tool with.
- **extra** (<code>dict\[str, Any\] | None</code>) – Dictionary of extra information about the Tool call. Use to store provider-specific
  information. To avoid serialization issues, values should be JSON serializable.

### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Convert ToolCall into a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with keys 'tool_name', 'arguments', 'id', and 'extra'.

### `from_dict`

```python
from_dict(data: dict[str, Any]) -> ToolCall
```

Creates a new ToolCall object from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to build the ToolCall object.

**Returns:**

- <code>ToolCall</code> – The created object.

## `ToolCallResult`

Represents the result of a Tool invocation.

**Parameters:**

- **result** (<code>ToolCallResultContentT</code>) – The result of the Tool invocation.
- **origin** (<code>ToolCall</code>) – The Tool call that produced this result.
- **error** (<code>bool</code>) – Whether the Tool invocation resulted in an error.

### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Converts ToolCallResult into a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with keys 'result', 'origin', and 'error'.

### `from_dict`

```python
from_dict(data: dict[str, Any]) -> ToolCallResult
```

Creates a ToolCallResult from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to build the ToolCallResult object.

**Returns:**

- <code>ToolCallResult</code> – The created object.

## `ReasoningContent`

Represents the optional reasoning content prepared by the model, usually contained in an assistant message.

**Parameters:**

- **reasoning_text** (<code>str</code>) – The reasoning text produced by the model.
- **extra** (<code>dict\[str, Any\]</code>) – Dictionary of extra information about the reasoning content. Use to store provider-specific
  information. To avoid serialization issues, values should be JSON serializable.

### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Convert ReasoningContent into a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with keys 'reasoning_text', and 'extra'.

### `from_dict`

```python
from_dict(data: dict[str, Any]) -> ReasoningContent
```

Creates a new ReasoningContent object from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to build the ReasoningContent object.

**Returns:**

- <code>ReasoningContent</code> – The created object.

## `ChatMessage`

Represents a message in a LLM chat conversation.

Use the `from_assistant`, `from_user`, `from_system`, and `from_tool` class methods to create a ChatMessage.

### `role`

```python
role: ChatRole
```

Returns the role of the entity sending the message.

### `meta`

```python
meta: dict[str, Any]
```

Returns the metadata associated with the message.

### `name`

```python
name: str | None
```

Returns the name associated with the message.

### `texts`

```python
texts: list[str]
```

Returns the list of all texts contained in the message.

### `text`

```python
text: str | None
```

Returns the first text contained in the message.

### `tool_calls`

```python
tool_calls: list[ToolCall]
```

Returns the list of all Tool calls contained in the message.

### `tool_call`

```python
tool_call: ToolCall | None
```

Returns the first Tool call contained in the message.

### `tool_call_results`

```python
tool_call_results: list[ToolCallResult]
```

Returns the list of all Tool call results contained in the message.

### `tool_call_result`

```python
tool_call_result: ToolCallResult | None
```

Returns the first Tool call result contained in the message.

### `images`

```python
images: list[ImageContent]
```

Returns the list of all images contained in the message.

### `image`

```python
image: ImageContent | None
```

Returns the first image contained in the message.

### `files`

```python
files: list[FileContent]
```

Returns the list of all files contained in the message.

### `file`

```python
file: FileContent | None
```

Returns the first file contained in the message.

### `reasonings`

```python
reasonings: list[ReasoningContent]
```

Returns the list of all reasoning contents contained in the message.

### `reasoning`

```python
reasoning: ReasoningContent | None
```

Returns the first reasoning content contained in the message.

### `is_from`

```python
is_from(role: ChatRole | str) -> bool
```

Check if the message is from a specific role.

**Parameters:**

- **role** (<code>ChatRole | str</code>) – The role to check against.

**Returns:**

- <code>bool</code> – True if the message is from the specified role, False otherwise.

### `from_user`

```python
from_user(text: str | None = None, meta: dict[str, Any] | None = None, name: str | None = None, *, content_parts: Sequence[TextContent | str | ImageContent | FileContent] | None = None) -> ChatMessage
```

Create a message from the user.

**Parameters:**

- **text** (<code>str | None</code>) – The text content of the message. Specify this or content_parts.
- **meta** (<code>dict\[str, Any\] | None</code>) – Additional metadata associated with the message.
- **name** (<code>str | None</code>) – An optional name for the participant. This field is only supported by OpenAI.
- **content_parts** (<code>Sequence\[TextContent | str | ImageContent | FileContent\] | None</code>) – A list of content parts to include in the message. Specify this or text.

**Returns:**

- <code>ChatMessage</code> – A new ChatMessage instance.

### `from_system`

```python
from_system(text: str, meta: dict[str, Any] | None = None, name: str | None = None) -> ChatMessage
```

Create a message from the system.

**Parameters:**

- **text** (<code>str</code>) – The text content of the message.
- **meta** (<code>dict\[str, Any\] | None</code>) – Additional metadata associated with the message.
- **name** (<code>str | None</code>) – An optional name for the participant. This field is only supported by OpenAI.

**Returns:**

- <code>ChatMessage</code> – A new ChatMessage instance.

### `from_assistant`

```python
from_assistant(text: str | None = None, meta: dict[str, Any] | None = None, name: str | None = None, tool_calls: list[ToolCall] | None = None, *, reasoning: str | ReasoningContent | None = None) -> ChatMessage
```

Create a message from the assistant.

**Parameters:**

- **text** (<code>str | None</code>) – The text content of the message.
- **meta** (<code>dict\[str, Any\] | None</code>) – Additional metadata associated with the message.
- **name** (<code>str | None</code>) – An optional name for the participant. This field is only supported by OpenAI.
- **tool_calls** (<code>list\[ToolCall\] | None</code>) – The Tool calls to include in the message.
- **reasoning** (<code>str | ReasoningContent | None</code>) – The reasoning content to include in the message.

**Returns:**

- <code>ChatMessage</code> – A new ChatMessage instance.

### `from_tool`

```python
from_tool(tool_result: ToolCallResultContentT, origin: ToolCall, error: bool = False, meta: dict[str, Any] | None = None) -> ChatMessage
```

Create a message from a Tool.

**Parameters:**

- **tool_result** (<code>ToolCallResultContentT</code>) – The result of the Tool invocation.
- **origin** (<code>ToolCall</code>) – The Tool call that produced this result.
- **error** (<code>bool</code>) – Whether the Tool invocation resulted in an error.
- **meta** (<code>dict\[str, Any\] | None</code>) – Additional metadata associated with the message.

**Returns:**

- <code>ChatMessage</code> – A new ChatMessage instance.

### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Converts ChatMessage into a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Serialized version of the object.

### `from_dict`

```python
from_dict(data: dict[str, Any]) -> ChatMessage
```

Creates a new ChatMessage object from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to build the ChatMessage object.

**Returns:**

- <code>ChatMessage</code> – The created object.

### `to_openai_dict_format`

```python
to_openai_dict_format(require_tool_call_ids: bool = True) -> dict[str, Any]
```

Convert a ChatMessage to the dictionary format expected by OpenAI's Chat Completions API.

**Parameters:**

- **require_tool_call_ids** (<code>bool</code>) – If True (default), enforces that each Tool Call includes a non-null `id` attribute.
  Set to False to allow Tool Calls without `id`, which may be suitable for shallow OpenAI-compatible APIs.

**Returns:**

- <code>dict\[str, Any\]</code> – The ChatMessage in the format expected by OpenAI's Chat Completions API.

**Raises:**

- <code>ValueError</code> – If the message format is invalid, or if `require_tool_call_ids` is True and any Tool Call is missing an
  `id` attribute.

### `from_openai_dict_format`

```python
from_openai_dict_format(message: dict[str, Any]) -> ChatMessage
```

Create a ChatMessage from a dictionary in the format expected by OpenAI's Chat API.

NOTE: While OpenAI's API requires `tool_call_id` in both tool calls and tool messages, this method
accepts messages without it to support shallow OpenAI-compatible APIs.
If you plan to use the resulting ChatMessage with OpenAI, you must include `tool_call_id` or you'll
encounter validation errors.

**Parameters:**

- **message** (<code>dict\[str, Any\]</code>) – The OpenAI dictionary to build the ChatMessage object.

**Returns:**

- <code>ChatMessage</code> – The created ChatMessage object.

**Raises:**

- <code>ValueError</code> – If the message dictionary is missing required fields.

## `Document`

Base data class containing some data to be queried.

Can contain text snippets and file paths to images or audios. Documents can be sorted by score and saved
to/from dictionary and JSON.

**Parameters:**

- **id** (<code>str</code>) – Unique identifier for the document. When not set, it's generated based on the Document fields' values.
- **content** (<code>str | None</code>) – Text of the document, if the document contains text.
- **blob** (<code>ByteStream | None</code>) – Binary data associated with the document, if the document has any binary data associated with it.
- **meta** (<code>dict\[str, Any\]</code>) – Additional custom metadata for the document. Must be JSON-serializable.
- **score** (<code>float | None</code>) – Score of the document. Used for ranking, usually assigned by retrievers.
- **embedding** (<code>list\[float\] | None</code>) – dense vector representation of the document.
- **sparse_embedding** (<code>SparseEmbedding | None</code>) – sparse vector representation of the document.

### `to_dict`

```python
to_dict(flatten: bool = True) -> dict[str, Any]
```

Converts Document into a dictionary.

`blob` field is converted to a JSON-serializable type.

**Parameters:**

- **flatten** (<code>bool</code>) – Whether to flatten `meta` field or not. Defaults to `True` to be backward-compatible with Haystack 1.x.

### `from_dict`

```python
from_dict(data: dict[str, Any]) -> Document
```

Creates a new Document object from a dictionary.

The `blob` field is converted to its original type.

### `content_type`

```python
content_type
```

Returns the type of the content for the document.

This is necessary to keep backward compatibility with 1.x.

## `FileContent`

The file content of a chat message.

**Parameters:**

- **base64_data** (<code>str</code>) – A base64 string representing the file.
- **mime_type** (<code>str | None</code>) – The MIME type of the file (e.g. "application/pdf").
  Providing this value is recommended, as most LLM providers require it.
  If not provided, the MIME type is guessed from the base64 string, which can be slow and not always reliable.
- **filename** (<code>str | None</code>) – Optional filename of the file. Some LLM providers use this information.
- **extra** (<code>dict\[str, Any\]</code>) – Dictionary of extra information about the file. Can be used to store provider-specific information.
  To avoid serialization issues, values should be JSON serializable.
- **validation** (<code>bool</code>) – If True (default), a validation process is performed:
- Check whether the base64 string is valid;
- Guess the MIME type if not provided.
  Set to False to skip validation and speed up initialization.

### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Convert FileContent into a dictionary.

### `from_dict`

```python
from_dict(data: dict[str, Any]) -> FileContent
```

Create an FileContent from a dictionary.

### `from_file_path`

```python
from_file_path(file_path: str | Path, *, filename: str | None = None, extra: dict[str, Any] | None = None) -> FileContent
```

Create an FileContent object from a file path.

**Parameters:**

- **file_path** (<code>str | Path</code>) – The path to the file.
- **filename** (<code>str | None</code>) – Optional file name. Some LLM providers use this information. If not provided, the filename is extracted
  from the file path.
- **extra** (<code>dict\[str, Any\] | None</code>) – Dictionary of extra information about the file. Can be used to store provider-specific information.
  To avoid serialization issues, values should be JSON serializable.

**Returns:**

- <code>FileContent</code> – An FileContent object.

### `from_url`

```python
from_url(url: str, *, retry_attempts: int = 2, timeout: int = 10, filename: str | None = None, extra: dict[str, Any] | None = None) -> FileContent
```

Create an FileContent object from a URL. The file is downloaded and converted to a base64 string.

**Parameters:**

- **url** (<code>str</code>) – The URL of the file.
- **retry_attempts** (<code>int</code>) – The number of times to retry to fetch the URL's content.
- **timeout** (<code>int</code>) – Timeout in seconds for the request.
- **filename** (<code>str | None</code>) – Optional filename of the file. Some LLM providers use this information. If not provided, the filename is
  extracted from the URL.
- **extra** (<code>dict\[str, Any\] | None</code>) – Dictionary of extra information about the file. Can be used to store provider-specific information.
  To avoid serialization issues, values should be JSON serializable.

**Returns:**

- <code>FileContent</code> – An FileContent object.

## `ImageContent`

The image content of a chat message.

**Parameters:**

- **base64_image** (<code>str</code>) – A base64 string representing the image.
- **mime_type** (<code>str | None</code>) – The MIME type of the image (e.g. "image/png", "image/jpeg").
  Providing this value is recommended, as most LLM providers require it.
  If not provided, the MIME type is guessed from the base64 string, which can be slow and not always reliable.
- **detail** (<code>Literal['auto', 'high', 'low'] | None</code>) – Optional detail level of the image (only supported by OpenAI). One of "auto", "high", or "low".
- **meta** (<code>dict\[str, Any\]</code>) – Optional metadata for the image.
- **validation** (<code>bool</code>) – If True (default), a validation process is performed:
- Check whether the base64 string is valid;
- Guess the MIME type if not provided;
- Check if the MIME type is a valid image MIME type.
  Set to False to skip validation and speed up initialization.

### `show`

```python
show() -> None
```

Shows the image.

### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Convert ImageContent into a dictionary.

### `from_dict`

```python
from_dict(data: dict[str, Any]) -> ImageContent
```

Create an ImageContent from a dictionary.

### `from_file_path`

```python
from_file_path(file_path: str | Path, *, size: tuple[int, int] | None = None, detail: Literal['auto', 'high', 'low'] | None = None, meta: dict[str, Any] | None = None) -> ImageContent
```

Create an ImageContent object from a file path.

It exposes similar functionality as the `ImageFileToImageContent` component. For PDF to ImageContent conversion,
use the `PDFToImageContent` component.

**Parameters:**

- **file_path** (<code>str | Path</code>) – The path to the image file. PDF files are not supported. For PDF to ImageContent conversion, use the
  `PDFToImageContent` component.
- **size** (<code>tuple\[int, int\] | None</code>) – If provided, resizes the image to fit within the specified dimensions (width, height) while
  maintaining aspect ratio. This reduces file size, memory usage, and processing time, which is beneficial
  when working with models that have resolution constraints or when transmitting images to remote services.
- **detail** (<code>Literal['auto', 'high', 'low'] | None</code>) – Optional detail level of the image (only supported by OpenAI). One of "auto", "high", or "low".
- **meta** (<code>dict\[str, Any\] | None</code>) – Additional metadata for the image.

**Returns:**

- <code>ImageContent</code> – An ImageContent object.

### `from_url`

```python
from_url(url: str, *, retry_attempts: int = 2, timeout: int = 10, size: tuple[int, int] | None = None, detail: Literal['auto', 'high', 'low'] | None = None, meta: dict[str, Any] | None = None) -> ImageContent
```

Create an ImageContent object from a URL. The image is downloaded and converted to a base64 string.

For PDF to ImageContent conversion, use the `PDFToImageContent` component.

**Parameters:**

- **url** (<code>str</code>) – The URL of the image. PDF files are not supported. For PDF to ImageContent conversion, use the
  `PDFToImageContent` component.
- **retry_attempts** (<code>int</code>) – The number of times to retry to fetch the URL's content.
- **timeout** (<code>int</code>) – Timeout in seconds for the request.
- **size** (<code>tuple\[int, int\] | None</code>) – If provided, resizes the image to fit within the specified dimensions (width, height) while
  maintaining aspect ratio. This reduces file size, memory usage, and processing time, which is beneficial
  when working with models that have resolution constraints or when transmitting images to remote services.
- **detail** (<code>Literal['auto', 'high', 'low'] | None</code>) – Optional detail level of the image (only supported by OpenAI). One of "auto", "high", or "low".
- **meta** (<code>dict\[str, Any\] | None</code>) – Additional metadata for the image.

**Returns:**

- <code>ImageContent</code> – An ImageContent object.

**Raises:**

- <code>ValueError</code> – If the URL does not point to an image or if it points to a PDF file.

## `SparseEmbedding`

Class representing a sparse embedding.

**Parameters:**

- **indices** (<code>list\[int\]</code>) – List of indices of non-zero elements in the embedding.
- **values** (<code>list\[float\]</code>) – List of values of non-zero elements in the embedding.

### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Convert the SparseEmbedding object to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Serialized sparse embedding.

### `from_dict`

```python
from_dict(sparse_embedding_dict: dict[str, Any]) -> SparseEmbedding
```

Deserializes the sparse embedding from a dictionary.

**Parameters:**

- **sparse_embedding_dict** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>SparseEmbedding</code> – Deserialized sparse embedding.

## `ToolCallDelta`

Represents a Tool call prepared by the model, usually contained in an assistant message.

**Parameters:**

- **index** (<code>int</code>) – The index of the Tool call in the list of Tool calls.
- **tool_name** (<code>str | None</code>) – The name of the Tool to call.
- **arguments** (<code>str | None</code>) – Either the full arguments in JSON format or a delta of the arguments.
- **id** (<code>str | None</code>) – The ID of the Tool call.
- **extra** (<code>dict\[str, Any\] | None</code>) – Dictionary of extra information about the Tool call. Use to store provider-specific
  information. To avoid serialization issues, values should be JSON serializable.

### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Returns a dictionary representation of the ToolCallDelta.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with keys 'index', 'tool_name', 'arguments', 'id', and 'extra'.

### `from_dict`

```python
from_dict(data: dict[str, Any]) -> ToolCallDelta
```

Creates a ToolCallDelta from a serialized representation.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary containing ToolCallDelta's attributes.

**Returns:**

- <code>ToolCallDelta</code> – A ToolCallDelta instance.

## `ComponentInfo`

The `ComponentInfo` class encapsulates information about a component.

**Parameters:**

- **type** (<code>str</code>) – The type of the component.
- **name** (<code>str | None</code>) – The name of the component assigned when adding it to a pipeline.

### `from_component`

```python
from_component(component: Component) -> ComponentInfo
```

Create a `ComponentInfo` object from a `Component` instance.

**Parameters:**

- **component** (<code>Component</code>) – The `Component` instance.

**Returns:**

- <code>ComponentInfo</code> – The `ComponentInfo` object with the type and name of the given component.

### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Returns a dictionary representation of ComponentInfo.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with keys 'type' and 'name'.

### `from_dict`

```python
from_dict(data: dict[str, Any]) -> ComponentInfo
```

Creates a ComponentInfo from a serialized representation.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary containing ComponentInfo's attributes.

**Returns:**

- <code>ComponentInfo</code> – A ComponentInfo instance.

## `StreamingChunk`

The `StreamingChunk` class encapsulates a segment of streamed content along with associated metadata.

This structure facilitates the handling and processing of streamed data in a systematic manner.

**Parameters:**

- **content** (<code>str</code>) – The content of the message chunk as a string.
- **meta** (<code>dict\[str, Any\]</code>) – A dictionary containing metadata related to the message chunk.
- **component_info** (<code>ComponentInfo | None</code>) – A `ComponentInfo` object containing information about the component that generated the chunk,
  such as the component name and type.
- **index** (<code>int | None</code>) – An optional integer index representing which content block this chunk belongs to.
- **tool_calls** (<code>list\[ToolCallDelta\] | None</code>) – An optional list of ToolCallDelta object representing a tool call associated with the message
  chunk.
- **tool_call_result** (<code>ToolCallResult | None</code>) – An optional ToolCallResult object representing the result of a tool call.
- **start** (<code>bool</code>) – A boolean indicating whether this chunk marks the start of a content block.
- **finish_reason** (<code>FinishReason | None</code>) – An optional value indicating the reason the generation finished.
  Standard values follow OpenAI's convention: "stop", "length", "tool_calls", "content_filter",
  plus Haystack-specific value "tool_call_results".
- **reasoning** (<code>ReasoningContent | None</code>) – An optional ReasoningContent object representing the reasoning content associated
  with the message chunk.

### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Returns a dictionary representation of the StreamingChunk.

**Returns:**

- <code>dict\[str, Any\]</code> – Serialized dictionary representation of the calling object.

### `from_dict`

```python
from_dict(data: dict[str, Any]) -> StreamingChunk
```

Creates a deserialized StreamingChunk instance from a serialized representation.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary containing the StreamingChunk's attributes.

**Returns:**

- <code>StreamingChunk</code> – A StreamingChunk instance.

## `select_streaming_callback`

```python
select_streaming_callback(init_callback: StreamingCallbackT | None, runtime_callback: StreamingCallbackT | None, requires_async: bool) -> StreamingCallbackT | None
```

Picks the correct streaming callback given an optional initial and runtime callback.

The runtime callback takes precedence over the initial callback.

**Parameters:**

- **init_callback** (<code>StreamingCallbackT | None</code>) – The initial callback.
- **runtime_callback** (<code>StreamingCallbackT | None</code>) – The runtime callback.
- **requires_async** (<code>bool</code>) – Whether the selected callback must be async compatible.

**Returns:**

- <code>StreamingCallbackT | None</code> – The selected callback.
