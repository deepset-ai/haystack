---
title: Data Classes
id: data-classess-api
description: Core classes that carry data through the system.
---

<a id="answer"></a>

# Module answer

<a id="answer.ExtractedAnswer"></a>

## ExtractedAnswer

<a id="answer.ExtractedAnswer.to_dict"></a>

#### ExtractedAnswer.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serialize the object to a dictionary.

**Returns**:

Serialized dictionary representation of the object.

<a id="answer.ExtractedAnswer.from_dict"></a>

#### ExtractedAnswer.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "ExtractedAnswer"
```

Deserialize the object from a dictionary.

**Arguments**:

- `data`: Dictionary representation of the object.

**Returns**:

Deserialized object.

<a id="answer.GeneratedAnswer"></a>

## GeneratedAnswer

<a id="answer.GeneratedAnswer.to_dict"></a>

#### GeneratedAnswer.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serialize the object to a dictionary.

**Returns**:

Serialized dictionary representation of the object.

<a id="answer.GeneratedAnswer.from_dict"></a>

#### GeneratedAnswer.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "GeneratedAnswer"
```

Deserialize the object from a dictionary.

**Arguments**:

- `data`: Dictionary representation of the object.

**Returns**:

Deserialized object.

<a id="byte_stream"></a>

# Module byte\_stream

<a id="byte_stream.ByteStream"></a>

## ByteStream

Base data class representing a binary object in the Haystack API.

**Arguments**:

- `data`: The binary data stored in Bytestream.
- `meta`: Additional metadata to be stored with the ByteStream.
- `mime_type`: The mime type of the binary data.

<a id="byte_stream.ByteStream.to_file"></a>

#### ByteStream.to\_file

```python
def to_file(destination_path: Path) -> None
```

Write the ByteStream to a file. Note: the metadata will be lost.

**Arguments**:

- `destination_path`: The path to write the ByteStream to.

<a id="byte_stream.ByteStream.from_file_path"></a>

#### ByteStream.from\_file\_path

```python
@classmethod
def from_file_path(cls,
                   filepath: Path,
                   mime_type: Optional[str] = None,
                   meta: Optional[dict[str, Any]] = None,
                   guess_mime_type: bool = False) -> "ByteStream"
```

Create a ByteStream from the contents read from a file.

**Arguments**:

- `filepath`: A valid path to a file.
- `mime_type`: The mime type of the file.
- `meta`: Additional metadata to be stored with the ByteStream.
- `guess_mime_type`: Whether to guess the mime type from the file.

<a id="byte_stream.ByteStream.from_string"></a>

#### ByteStream.from\_string

```python
@classmethod
def from_string(cls,
                text: str,
                encoding: str = "utf-8",
                mime_type: Optional[str] = None,
                meta: Optional[dict[str, Any]] = None) -> "ByteStream"
```

Create a ByteStream encoding a string.

**Arguments**:

- `text`: The string to encode
- `encoding`: The encoding used to convert the string into bytes
- `mime_type`: The mime type of the file.
- `meta`: Additional metadata to be stored with the ByteStream.

<a id="byte_stream.ByteStream.to_string"></a>

#### ByteStream.to\_string

```python
def to_string(encoding: str = "utf-8") -> str
```

Convert the ByteStream to a string, metadata will not be included.

**Arguments**:

- `encoding`: The encoding used to convert the bytes to a string. Defaults to "utf-8".

**Raises**:

- `None`: UnicodeDecodeError: If the ByteStream data cannot be decoded with the specified encoding.

**Returns**:

The string representation of the ByteStream.

<a id="byte_stream.ByteStream.__repr__"></a>

#### ByteStream.\_\_repr\_\_

```python
def __repr__() -> str
```

Return a string representation of the ByteStream, truncating the data to 100 bytes.

<a id="byte_stream.ByteStream.to_dict"></a>

#### ByteStream.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Convert the ByteStream to a dictionary representation.

**Returns**:

A dictionary with keys 'data', 'meta', and 'mime_type'.

<a id="byte_stream.ByteStream.from_dict"></a>

#### ByteStream.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "ByteStream"
```

Create a ByteStream from a dictionary representation.

**Arguments**:

- `data`: A dictionary with keys 'data', 'meta', and 'mime_type'.

**Returns**:

A ByteStream instance.

<a id="chat_message"></a>

# Module chat\_message

<a id="chat_message.ChatRole"></a>

## ChatRole

Enumeration representing the roles within a chat.

<a id="chat_message.ChatRole.USER"></a>

#### USER

The user role. A message from the user contains only text.

<a id="chat_message.ChatRole.SYSTEM"></a>

#### SYSTEM

The system role. A message from the system contains only text.

<a id="chat_message.ChatRole.ASSISTANT"></a>

#### ASSISTANT

The assistant role. A message from the assistant can contain text and Tool calls. It can also store metadata.

<a id="chat_message.ChatRole.TOOL"></a>

#### TOOL

The tool role. A message from a tool contains the result of a Tool invocation.

<a id="chat_message.ChatRole.from_str"></a>

#### ChatRole.from\_str

```python
@staticmethod
def from_str(string: str) -> "ChatRole"
```

Convert a string to a ChatRole enum.

<a id="chat_message.ToolCall"></a>

## ToolCall

Represents a Tool call prepared by the model, usually contained in an assistant message.

**Arguments**:

- `id`: The ID of the Tool call.
- `tool_name`: The name of the Tool to call.
- `arguments`: The arguments to call the Tool with.

<a id="chat_message.ToolCall.id"></a>

#### id

noqa: A003

<a id="chat_message.ToolCall.to_dict"></a>

#### ToolCall.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Convert ToolCall into a dictionary.

**Returns**:

A dictionary with keys 'tool_name', 'arguments', and 'id'.

<a id="chat_message.ToolCall.from_dict"></a>

#### ToolCall.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "ToolCall"
```

Creates a new ToolCall object from a dictionary.

**Arguments**:

- `data`: The dictionary to build the ToolCall object.

**Returns**:

The created object.

<a id="chat_message.ToolCallResult"></a>

## ToolCallResult

Represents the result of a Tool invocation.

**Arguments**:

- `result`: The result of the Tool invocation.
- `origin`: The Tool call that produced this result.
- `error`: Whether the Tool invocation resulted in an error.

<a id="chat_message.ToolCallResult.to_dict"></a>

#### ToolCallResult.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Converts ToolCallResult into a dictionary.

**Returns**:

A dictionary with keys 'result', 'origin', and 'error'.

<a id="chat_message.ToolCallResult.from_dict"></a>

#### ToolCallResult.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "ToolCallResult"
```

Creates a ToolCallResult from a dictionary.

**Arguments**:

- `data`: The dictionary to build the ToolCallResult object.

**Returns**:

The created object.

<a id="chat_message.TextContent"></a>

## TextContent

The textual content of a chat message.

**Arguments**:

- `text`: The text content of the message.

<a id="chat_message.TextContent.to_dict"></a>

#### TextContent.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Convert TextContent into a dictionary.

<a id="chat_message.TextContent.from_dict"></a>

#### TextContent.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "TextContent"
```

Create a TextContent from a dictionary.

<a id="chat_message.ReasoningContent"></a>

## ReasoningContent

Represents the optional reasoning content prepared by the model, usually contained in an assistant message.

**Arguments**:

- `reasoning_text`: The reasoning text produced by the model.
- `extra`: Dictionary of extra information about the reasoning content. Use to store provider-specific
information. To avoid serialization issues, values should be JSON serializable.

<a id="chat_message.ReasoningContent.to_dict"></a>

#### ReasoningContent.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Convert ReasoningContent into a dictionary.

**Returns**:

A dictionary with keys 'reasoning_text', and 'extra'.

<a id="chat_message.ReasoningContent.from_dict"></a>

#### ReasoningContent.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "ReasoningContent"
```

Creates a new ReasoningContent object from a dictionary.

**Arguments**:

- `data`: The dictionary to build the ReasoningContent object.

**Returns**:

The created object.

<a id="chat_message.ChatMessage"></a>

## ChatMessage

Represents a message in a LLM chat conversation.

Use the `from_assistant`, `from_user`, `from_system`, and `from_tool` class methods to create a ChatMessage.

<a id="chat_message.ChatMessage.__new__"></a>

#### ChatMessage.\_\_new\_\_

```python
def __new__(cls, *args, **kwargs)
```

This method is reimplemented to make the changes to the `ChatMessage` dataclass more visible.

<a id="chat_message.ChatMessage.__getattribute__"></a>

#### ChatMessage.\_\_getattribute\_\_

```python
def __getattribute__(name)
```

This method is reimplemented to make the `content` attribute removal more visible.

<a id="chat_message.ChatMessage.role"></a>

#### ChatMessage.role

```python
@property
def role() -> ChatRole
```

Returns the role of the entity sending the message.

<a id="chat_message.ChatMessage.meta"></a>

#### ChatMessage.meta

```python
@property
def meta() -> dict[str, Any]
```

Returns the metadata associated with the message.

<a id="chat_message.ChatMessage.name"></a>

#### ChatMessage.name

```python
@property
def name() -> Optional[str]
```

Returns the name associated with the message.

<a id="chat_message.ChatMessage.texts"></a>

#### ChatMessage.texts

```python
@property
def texts() -> list[str]
```

Returns the list of all texts contained in the message.

<a id="chat_message.ChatMessage.text"></a>

#### ChatMessage.text

```python
@property
def text() -> Optional[str]
```

Returns the first text contained in the message.

<a id="chat_message.ChatMessage.tool_calls"></a>

#### ChatMessage.tool\_calls

```python
@property
def tool_calls() -> list[ToolCall]
```

Returns the list of all Tool calls contained in the message.

<a id="chat_message.ChatMessage.tool_call"></a>

#### ChatMessage.tool\_call

```python
@property
def tool_call() -> Optional[ToolCall]
```

Returns the first Tool call contained in the message.

<a id="chat_message.ChatMessage.tool_call_results"></a>

#### ChatMessage.tool\_call\_results

```python
@property
def tool_call_results() -> list[ToolCallResult]
```

Returns the list of all Tool call results contained in the message.

<a id="chat_message.ChatMessage.tool_call_result"></a>

#### ChatMessage.tool\_call\_result

```python
@property
def tool_call_result() -> Optional[ToolCallResult]
```

Returns the first Tool call result contained in the message.

<a id="chat_message.ChatMessage.images"></a>

#### ChatMessage.images

```python
@property
def images() -> list[ImageContent]
```

Returns the list of all images contained in the message.

<a id="chat_message.ChatMessage.image"></a>

#### ChatMessage.image

```python
@property
def image() -> Optional[ImageContent]
```

Returns the first image contained in the message.

<a id="chat_message.ChatMessage.reasonings"></a>

#### ChatMessage.reasonings

```python
@property
def reasonings() -> list[ReasoningContent]
```

Returns the list of all reasoning contents contained in the message.

<a id="chat_message.ChatMessage.reasoning"></a>

#### ChatMessage.reasoning

```python
@property
def reasoning() -> Optional[ReasoningContent]
```

Returns the first reasoning content contained in the message.

<a id="chat_message.ChatMessage.is_from"></a>

#### ChatMessage.is\_from

```python
def is_from(role: Union[ChatRole, str]) -> bool
```

Check if the message is from a specific role.

**Arguments**:

- `role`: The role to check against.

**Returns**:

True if the message is from the specified role, False otherwise.

<a id="chat_message.ChatMessage.from_user"></a>

#### ChatMessage.from\_user

```python
@classmethod
def from_user(
    cls,
    text: Optional[str] = None,
    meta: Optional[dict[str, Any]] = None,
    name: Optional[str] = None,
    *,
    content_parts: Optional[Sequence[Union[TextContent, str,
                                           ImageContent]]] = None
) -> "ChatMessage"
```

Create a message from the user.

**Arguments**:

- `text`: The text content of the message. Specify this or content_parts.
- `meta`: Additional metadata associated with the message.
- `name`: An optional name for the participant. This field is only supported by OpenAI.
- `content_parts`: A list of content parts to include in the message. Specify this or text.

**Returns**:

A new ChatMessage instance.

<a id="chat_message.ChatMessage.from_system"></a>

#### ChatMessage.from\_system

```python
@classmethod
def from_system(cls,
                text: str,
                meta: Optional[dict[str, Any]] = None,
                name: Optional[str] = None) -> "ChatMessage"
```

Create a message from the system.

**Arguments**:

- `text`: The text content of the message.
- `meta`: Additional metadata associated with the message.
- `name`: An optional name for the participant. This field is only supported by OpenAI.

**Returns**:

A new ChatMessage instance.

<a id="chat_message.ChatMessage.from_assistant"></a>

#### ChatMessage.from\_assistant

```python
@classmethod
def from_assistant(
        cls,
        text: Optional[str] = None,
        meta: Optional[dict[str, Any]] = None,
        name: Optional[str] = None,
        tool_calls: Optional[list[ToolCall]] = None,
        *,
        reasoning: Optional[Union[str,
                                  ReasoningContent]] = None) -> "ChatMessage"
```

Create a message from the assistant.

**Arguments**:

- `text`: The text content of the message.
- `meta`: Additional metadata associated with the message.
- `name`: An optional name for the participant. This field is only supported by OpenAI.
- `tool_calls`: The Tool calls to include in the message.
- `reasoning`: The reasoning content to include in the message.

**Returns**:

A new ChatMessage instance.

<a id="chat_message.ChatMessage.from_tool"></a>

#### ChatMessage.from\_tool

```python
@classmethod
def from_tool(cls,
              tool_result: str,
              origin: ToolCall,
              error: bool = False,
              meta: Optional[dict[str, Any]] = None) -> "ChatMessage"
```

Create a message from a Tool.

**Arguments**:

- `tool_result`: The result of the Tool invocation.
- `origin`: The Tool call that produced this result.
- `error`: Whether the Tool invocation resulted in an error.
- `meta`: Additional metadata associated with the message.

**Returns**:

A new ChatMessage instance.

<a id="chat_message.ChatMessage.to_dict"></a>

#### ChatMessage.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Converts ChatMessage into a dictionary.

**Returns**:

Serialized version of the object.

<a id="chat_message.ChatMessage.from_dict"></a>

#### ChatMessage.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "ChatMessage"
```

Creates a new ChatMessage object from a dictionary.

**Arguments**:

- `data`: The dictionary to build the ChatMessage object.

**Returns**:

The created object.

<a id="chat_message.ChatMessage.to_openai_dict_format"></a>

#### ChatMessage.to\_openai\_dict\_format

```python
def to_openai_dict_format(
        require_tool_call_ids: bool = True) -> dict[str, Any]
```

Convert a ChatMessage to the dictionary format expected by OpenAI's Chat API.

**Arguments**:

- `require_tool_call_ids`: If True (default), enforces that each Tool Call includes a non-null `id` attribute.
Set to False to allow Tool Calls without `id`, which may be suitable for shallow OpenAI-compatible APIs.

**Raises**:

- `ValueError`: If the message format is invalid, or if `require_tool_call_ids` is True and any Tool Call is missing an
`id` attribute.

**Returns**:

The ChatMessage in the format expected by OpenAI's Chat API.

<a id="chat_message.ChatMessage.from_openai_dict_format"></a>

#### ChatMessage.from\_openai\_dict\_format

```python
@classmethod
def from_openai_dict_format(cls, message: dict[str, Any]) -> "ChatMessage"
```

Create a ChatMessage from a dictionary in the format expected by OpenAI's Chat API.

NOTE: While OpenAI's API requires `tool_call_id` in both tool calls and tool messages, this method
accepts messages without it to support shallow OpenAI-compatible APIs.
If you plan to use the resulting ChatMessage with OpenAI, you must include `tool_call_id` or you'll
encounter validation errors.

**Arguments**:

- `message`: The OpenAI dictionary to build the ChatMessage object.

**Raises**:

- `ValueError`: If the message dictionary is missing required fields.

**Returns**:

The created ChatMessage object.

<a id="document"></a>

# Module document

<a id="document._BackwardCompatible"></a>

## \_BackwardCompatible

Metaclass that handles Document backward compatibility.

<a id="document._BackwardCompatible.__call__"></a>

#### \_BackwardCompatible.\_\_call\_\_

```python
def __call__(cls, *args, **kwargs)
```

Called before Document.__init__, handles legacy fields.

Embedding was stored as NumPy arrays in 1.x, so we convert it to a list of floats.
Other legacy fields are removed.

<a id="document.Document"></a>

## Document

Base data class containing some data to be queried.

Can contain text snippets and file paths to images or audios. Documents can be sorted by score and saved
to/from dictionary and JSON.

**Arguments**:

- `id`: Unique identifier for the document. When not set, it's generated based on the Document fields' values.
- `content`: Text of the document, if the document contains text.
- `blob`: Binary data associated with the document, if the document has any binary data associated with it.
- `meta`: Additional custom metadata for the document. Must be JSON-serializable.
- `score`: Score of the document. Used for ranking, usually assigned by retrievers.
- `embedding`: dense vector representation of the document.
- `sparse_embedding`: sparse vector representation of the document.

<a id="document.Document.__eq__"></a>

#### Document.\_\_eq\_\_

```python
def __eq__(other)
```

Compares Documents for equality.

Two Documents are considered equals if their dictionary representation is identical.

<a id="document.Document.__post_init__"></a>

#### Document.\_\_post\_init\_\_

```python
def __post_init__()
```

Generate the ID based on the init parameters.

<a id="document.Document.to_dict"></a>

#### Document.to\_dict

```python
def to_dict(flatten: bool = True) -> dict[str, Any]
```

Converts Document into a dictionary.

`blob` field is converted to a JSON-serializable type.

**Arguments**:

- `flatten`: Whether to flatten `meta` field or not. Defaults to `True` to be backward-compatible with Haystack 1.x.

<a id="document.Document.from_dict"></a>

#### Document.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "Document"
```

Creates a new Document object from a dictionary.

The `blob` field is converted to its original type.

<a id="document.Document.content_type"></a>

#### Document.content\_type

```python
@property
def content_type()
```

Returns the type of the content for the document.

This is necessary to keep backward compatibility with 1.x.

<a id="image_content"></a>

# Module image\_content

<a id="image_content.ImageContent"></a>

## ImageContent

The image content of a chat message.

**Arguments**:

- `base64_image`: A base64 string representing the image.
- `mime_type`: The MIME type of the image (e.g. "image/png", "image/jpeg").
Providing this value is recommended, as most LLM providers require it.
If not provided, the MIME type is guessed from the base64 string, which can be slow and not always reliable.
- `detail`: Optional detail level of the image (only supported by OpenAI). One of "auto", "high", or "low".
- `meta`: Optional metadata for the image.
- `validation`: If True (default), a validation process is performed:
- Check whether the base64 string is valid;
- Guess the MIME type if not provided;
- Check if the MIME type is a valid image MIME type.
Set to False to skip validation and speed up initialization.

<a id="image_content.ImageContent.__repr__"></a>

#### ImageContent.\_\_repr\_\_

```python
def __repr__() -> str
```

Return a string representation of the ImageContent, truncating the base64_image to 100 bytes.

<a id="image_content.ImageContent.show"></a>

#### ImageContent.show

```python
def show() -> None
```

Shows the image.

<a id="image_content.ImageContent.to_dict"></a>

#### ImageContent.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Convert ImageContent into a dictionary.

<a id="image_content.ImageContent.from_dict"></a>

#### ImageContent.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "ImageContent"
```

Create an ImageContent from a dictionary.

<a id="image_content.ImageContent.from_file_path"></a>

#### ImageContent.from\_file\_path

```python
@classmethod
def from_file_path(cls,
                   file_path: Union[str, Path],
                   *,
                   size: Optional[tuple[int, int]] = None,
                   detail: Optional[Literal["auto", "high", "low"]] = None,
                   meta: Optional[dict[str, Any]] = None) -> "ImageContent"
```

Create an ImageContent object from a file path.

It exposes similar functionality as the `ImageFileToImageContent` component. For PDF to ImageContent conversion,
use the `PDFToImageContent` component.

**Arguments**:

- `file_path`: The path to the image file. PDF files are not supported. For PDF to ImageContent conversion, use the
`PDFToImageContent` component.
- `size`: If provided, resizes the image to fit within the specified dimensions (width, height) while
maintaining aspect ratio. This reduces file size, memory usage, and processing time, which is beneficial
when working with models that have resolution constraints or when transmitting images to remote services.
- `detail`: Optional detail level of the image (only supported by OpenAI). One of "auto", "high", or "low".
- `meta`: Additional metadata for the image.

**Returns**:

An ImageContent object.

<a id="image_content.ImageContent.from_url"></a>

#### ImageContent.from\_url

```python
@classmethod
def from_url(cls,
             url: str,
             *,
             retry_attempts: int = 2,
             timeout: int = 10,
             size: Optional[tuple[int, int]] = None,
             detail: Optional[Literal["auto", "high", "low"]] = None,
             meta: Optional[dict[str, Any]] = None) -> "ImageContent"
```

Create an ImageContent object from a URL. The image is downloaded and converted to a base64 string.

For PDF to ImageContent conversion, use the `PDFToImageContent` component.

**Arguments**:

- `url`: The URL of the image. PDF files are not supported. For PDF to ImageContent conversion, use the
`PDFToImageContent` component.
- `retry_attempts`: The number of times to retry to fetch the URL's content.
- `timeout`: Timeout in seconds for the request.
- `size`: If provided, resizes the image to fit within the specified dimensions (width, height) while
maintaining aspect ratio. This reduces file size, memory usage, and processing time, which is beneficial
when working with models that have resolution constraints or when transmitting images to remote services.
- `detail`: Optional detail level of the image (only supported by OpenAI). One of "auto", "high", or "low".
- `meta`: Additional metadata for the image.

**Raises**:

- `ValueError`: If the URL does not point to an image or if it points to a PDF file.

**Returns**:

An ImageContent object.

<a id="sparse_embedding"></a>

# Module sparse\_embedding

<a id="sparse_embedding.SparseEmbedding"></a>

## SparseEmbedding

Class representing a sparse embedding.

**Arguments**:

- `indices`: List of indices of non-zero elements in the embedding.
- `values`: List of values of non-zero elements in the embedding.

<a id="sparse_embedding.SparseEmbedding.__post_init__"></a>

#### SparseEmbedding.\_\_post\_init\_\_

```python
def __post_init__()
```

Checks if the indices and values lists are of the same length.

Raises a ValueError if they are not.

<a id="sparse_embedding.SparseEmbedding.to_dict"></a>

#### SparseEmbedding.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Convert the SparseEmbedding object to a dictionary.

**Returns**:

Serialized sparse embedding.

<a id="sparse_embedding.SparseEmbedding.from_dict"></a>

#### SparseEmbedding.from\_dict

```python
@classmethod
def from_dict(cls, sparse_embedding_dict: dict[str, Any]) -> "SparseEmbedding"
```

Deserializes the sparse embedding from a dictionary.

**Arguments**:

- `sparse_embedding_dict`: Dictionary to deserialize from.

**Returns**:

Deserialized sparse embedding.

<a id="streaming_chunk"></a>

# Module streaming\_chunk

<a id="streaming_chunk.ToolCallDelta"></a>

## ToolCallDelta

Represents a Tool call prepared by the model, usually contained in an assistant message.

**Arguments**:

- `index`: The index of the Tool call in the list of Tool calls.
- `tool_name`: The name of the Tool to call.
- `arguments`: Either the full arguments in JSON format or a delta of the arguments.
- `id`: The ID of the Tool call.

<a id="streaming_chunk.ToolCallDelta.id"></a>

#### id

noqa: A003

<a id="streaming_chunk.ToolCallDelta.to_dict"></a>

#### ToolCallDelta.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Returns a dictionary representation of the ToolCallDelta.

**Returns**:

A dictionary with keys 'index', 'tool_name', 'arguments', and 'id'.

<a id="streaming_chunk.ToolCallDelta.from_dict"></a>

#### ToolCallDelta.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "ToolCallDelta"
```

Creates a ToolCallDelta from a serialized representation.

**Arguments**:

- `data`: Dictionary containing ToolCallDelta's attributes.

**Returns**:

A ToolCallDelta instance.

<a id="streaming_chunk.ComponentInfo"></a>

## ComponentInfo

The `ComponentInfo` class encapsulates information about a component.

**Arguments**:

- `type`: The type of the component.
- `name`: The name of the component assigned when adding it to a pipeline.

<a id="streaming_chunk.ComponentInfo.from_component"></a>

#### ComponentInfo.from\_component

```python
@classmethod
def from_component(cls, component: Component) -> "ComponentInfo"
```

Create a `ComponentInfo` object from a `Component` instance.

**Arguments**:

- `component`: The `Component` instance.

**Returns**:

The `ComponentInfo` object with the type and name of the given component.

<a id="streaming_chunk.ComponentInfo.to_dict"></a>

#### ComponentInfo.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Returns a dictionary representation of ComponentInfo.

**Returns**:

A dictionary with keys 'type' and 'name'.

<a id="streaming_chunk.ComponentInfo.from_dict"></a>

#### ComponentInfo.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "ComponentInfo"
```

Creates a ComponentInfo from a serialized representation.

**Arguments**:

- `data`: Dictionary containing ComponentInfo's attributes.

**Returns**:

A ComponentInfo instance.

<a id="streaming_chunk.StreamingChunk"></a>

## StreamingChunk

The `StreamingChunk` class encapsulates a segment of streamed content along with associated metadata.

This structure facilitates the handling and processing of streamed data in a systematic manner.

**Arguments**:

- `content`: The content of the message chunk as a string.
- `meta`: A dictionary containing metadata related to the message chunk.
- `component_info`: A `ComponentInfo` object containing information about the component that generated the chunk,
such as the component name and type.
- `index`: An optional integer index representing which content block this chunk belongs to.
- `tool_calls`: An optional list of ToolCallDelta object representing a tool call associated with the message
chunk.
- `tool_call_result`: An optional ToolCallResult object representing the result of a tool call.
- `start`: A boolean indicating whether this chunk marks the start of a content block.
- `finish_reason`: An optional value indicating the reason the generation finished.
Standard values follow OpenAI's convention: "stop", "length", "tool_calls", "content_filter",
plus Haystack-specific value "tool_call_results".

<a id="streaming_chunk.StreamingChunk.to_dict"></a>

#### StreamingChunk.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Returns a dictionary representation of the StreamingChunk.

**Returns**:

Serialized dictionary representation of the calling object.

<a id="streaming_chunk.StreamingChunk.from_dict"></a>

#### StreamingChunk.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "StreamingChunk"
```

Creates a deserialized StreamingChunk instance from a serialized representation.

**Arguments**:

- `data`: Dictionary containing the StreamingChunk's attributes.

**Returns**:

A StreamingChunk instance.

<a id="streaming_chunk.select_streaming_callback"></a>

#### select\_streaming\_callback

```python
def select_streaming_callback(
        init_callback: Optional[StreamingCallbackT],
        runtime_callback: Optional[StreamingCallbackT],
        requires_async: bool) -> Optional[StreamingCallbackT]
```

Picks the correct streaming callback given an optional initial and runtime callback.

The runtime callback takes precedence over the initial callback.

**Arguments**:

- `init_callback`: The initial callback.
- `runtime_callback`: The runtime callback.
- `requires_async`: Whether the selected callback must be async compatible.

**Returns**:

The selected callback.
