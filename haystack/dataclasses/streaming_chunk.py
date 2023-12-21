from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class StreamingChunk:
    """
    The StreamingChunk class encapsulates a segment of streamed content along with
    associated metadata. This structure facilitates the handling and processing of
    streamed data in a systematic manner.

    :param content: The content of the message chunk as a string.
    :param meta: A dictionary containing metadata related to the message chunk.
    """

    content: str
    meta: Dict[str, Any] = field(default_factory=dict, hash=False)
