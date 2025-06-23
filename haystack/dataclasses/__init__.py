# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "answer": ["Answer", "ExtractedAnswer", "GeneratedAnswer"],
    "byte_stream": ["ByteStream"],
    "chat_message": ["ChatMessage", "ChatRole", "TextContent", "ToolCall", "ToolCallResult"],
    "document": ["Document"],
    "sparse_embedding": ["SparseEmbedding"],
    "state": ["State"],
    "streaming_chunk": [
        "AsyncStreamingCallbackT",
        "ComponentInfo",
        "FinishReason",
        "StreamingCallbackT",
        "StreamingChunk",
        "SyncStreamingCallbackT",
        "ToolCallDelta",
        "select_streaming_callback",
    ],
}

if TYPE_CHECKING:
    from .answer import Answer as Answer
    from .answer import ExtractedAnswer as ExtractedAnswer
    from .answer import GeneratedAnswer as GeneratedAnswer
    from .byte_stream import ByteStream as ByteStream
    from .chat_message import ChatMessage as ChatMessage
    from .chat_message import ChatRole as ChatRole
    from .chat_message import TextContent as TextContent
    from .chat_message import ToolCall as ToolCall
    from .chat_message import ToolCallResult as ToolCallResult
    from .document import Document as Document
    from .sparse_embedding import SparseEmbedding as SparseEmbedding
    from .state import State as State
    from .streaming_chunk import AsyncStreamingCallbackT as AsyncStreamingCallbackT
    from .streaming_chunk import ComponentInfo as ComponentInfo
    from .streaming_chunk import FinishReason as FinishReason
    from .streaming_chunk import StreamingCallbackT as StreamingCallbackT
    from .streaming_chunk import StreamingChunk as StreamingChunk
    from .streaming_chunk import SyncStreamingCallbackT as SyncStreamingCallbackT
    from .streaming_chunk import ToolCallDelta as ToolCallDelta
    from .streaming_chunk import select_streaming_callback as select_streaming_callback
else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
