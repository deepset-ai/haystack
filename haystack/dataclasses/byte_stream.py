from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ByteStream:
    """
    Base data class representing a binary object in the Haystack API.
    """

    data: bytes
    meta: Dict[str, Any] = field(default_factory=dict, hash=False)
    mime_type: Optional[str] = field(default=None)
    mime_type_resolution_priority: List[str] = field(default_factory=lambda: ["attribute", "meta"])

    def to_file(self, destination_path: Path):
        """
        Write the ByteStream to a file. Note: the metadata will be lost.

        :param destination_path: The path to write the ByteStream to.
        """
        with open(destination_path, "wb") as fd:
            fd.write(self.data)

    @classmethod
    def from_file_path(
        cls,
        filepath: Path,
        mime_type: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        mime_type_resolution_priority: Optional[List[str]] = None,
    ) -> "ByteStream":
        """
        Create a ByteStream from the contents read from a file.

        :param filepath: A valid path to a file.
        :param mime_type: The mime type of the file.
        :param meta: Additional metadata to be stored with the ByteStream.
        :param mime_type_resolution_priority: The priority order of the mime type resolution
        """
        with open(filepath, "rb") as fd:
            return cls(
                data=fd.read(),
                mime_type=mime_type,
                meta=meta or {},
                mime_type_resolution_priority=mime_type_resolution_priority or ["attribute", "meta"],
            )

    @classmethod
    def from_string(
        cls,
        text: str,
        encoding: str = "utf-8",
        mime_type: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        mime_type_resolution_priority: Optional[List[str]] = None,
    ) -> "ByteStream":
        """
        Create a ByteStream encoding a string.

        :param text: The string to encode
        :param encoding: The encoding used to convert the string into bytes
        :param mime_type: The mime type of the file.
        :param meta: Additional metadata to be stored with the ByteStream.
        :param mime_type_resolution_priority: The priority order of the mime type resolution
        """
        return cls(
            data=text.encode(encoding),
            mime_type=mime_type,
            meta=meta or {},
            mime_type_resolution_priority=mime_type_resolution_priority or ["attribute", "meta"],
        )

    def to_string(self, encoding: str = "utf-8") -> str:
        """
        Convert the ByteStream to a string, metadata will not be included.

        :param encoding: The encoding used to convert the bytes to a string. Defaults to "utf-8".
        :returns: The string representation of the ByteStream.
        :raises: UnicodeDecodeError: If the ByteStream data cannot be decoded with the specified encoding.
        """
        return self.data.decode(encoding)

    @property
    def resolved_mime_type(self) -> Optional[str]:
        """
        Returns the resolved MIME type of the ByteStream based on the `mime_type_resolution_priority` priority.

        :return: The MIME type if available, otherwise `None`.
        :rtype: Optional[str]
        """
        sources = {"meta": self.meta.get("content_type", None), "attribute": self.mime_type}

        for source in self.mime_type_resolution_priority:
            if sources[source]:
                return sources[source]

        return None
