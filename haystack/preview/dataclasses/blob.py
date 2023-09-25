from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class Blob:
    """
    Base data class representing a binary object in the Haystack API.
    """

    data: bytes

    def save(self, destination_path: Path):
        with open(destination_path, "wb") as fd:
            fd.write(self.data)

    @classmethod
    def from_file_path(cls, filepath: Path) -> "Blob":
        with open(filepath, "rb") as fd:
            return cls(data=fd.read())
