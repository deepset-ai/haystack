from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass(frozen=True)
class ByteStream:
    """
    Base data class representing a binary object in the Haystack API.
    """

    data: bytes
    metadata: Dict[str, Any] = field(default_factory=dict, hash=False)
    mime_type: Optional[str] = field(default=None)

    def to_file(self, destination_path: Path):
        with open(destination_path, "wb") as fd:
            fd.write(self.data)

    @classmethod
    def from_file_path(cls, filepath: Path, mime_type: Optional[str] = None) -> "ByteStream":
        """
        Create a ByteStream from the contents read from a file.

        :param filepath: A valid path to a file.
        """
        with open(filepath, "rb") as fd:
            return cls(data=fd.read(), mime_type=mime_type)

    @classmethod
    def from_string(cls, text: str, encoding: str = "utf-8", mime_type: Optional[str] = None) -> "ByteStream":
        """
        Create a ByteStream encoding a string.

        :param text: The string to encode
        :param encoding: The encoding used to convert the string into bytes
        """
        return cls(data=text.encode(encoding), mime_type=mime_type)
