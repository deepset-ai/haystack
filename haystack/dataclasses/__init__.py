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
        "StreamingChunk",
        "AsyncStreamingCallbackT",
        "StreamingCallbackT",
        "SyncStreamingCallbackT",
        "select_streaming_callback",
    ],
}

if TYPE_CHECKING:
    from .answer import Answer, ExtractedAnswer, GeneratedAnswer
    from .byte_stream import ByteStream
    from .chat_message import ChatMessage, ChatRole, TextContent, ToolCall, ToolCallResult
    from .document import Document
    from .sparse_embedding import SparseEmbedding
    from .state import State
    from .streaming_chunk import (
        AsyncStreamingCallbackT,
        StreamingCallbackT,
        StreamingChunk,
        SyncStreamingCallbackT,
        select_streaming_callback,
    )
else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
