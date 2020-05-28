from abc import abstractmethod
from pathlib import Path


class BaseConverter:
    """
    Base class for implementing file converts to transform input documents to text format for indexing in database.
    """

    def __init__(
        self,
        remove_tables: bool = None,
        remove_headers: bool = None,
        remove_whitespace: bool = None,
        remove_empty_lines: bool = None,
        remove_short_lines_length: int = None,
        validate_languages: [str] = None
    ):
        self.remove_tables = remove_tables
        self.remove_headers = remove_headers
        self.remove_whitespace = remove_whitespace
        self.remove_empty_lines = remove_empty_lines
        self.remove_short_lines_length = remove_short_lines_length
        self.validate_languages = validate_languages

    @abstractmethod
    def extract_text(self, file_path: Path) -> str:
        pass
