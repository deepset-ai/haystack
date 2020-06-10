from abc import abstractmethod
from pathlib import Path
from typing import List, Optional


class BaseConverter:
    """
    Base class for implementing file converts to transform input documents to text format for indexing in database.
    """

    def __init__(
        self,
        remove_numeric_tables: Optional[bool] = None,
        remove_header_footer: Optional[bool] = None,
        remove_whitespace: Optional[bool] = None,
        remove_empty_lines: Optional[bool] = None,
        valid_languages: Optional[List[str]] = None,
    ):
        """
        :param remove_numeric_tables: This option uses heuristics to remove numeric rows from the tables.
                                      The tabular structures in documents might be noise for the reader model if it
                                      does not have table parsing capability for finding answers. However, tables
                                      may also have long strings that could possible candidate for searching answers.
                                      The rows containing strings are thus retained in this option.
        :param remove_header_footer: use heuristic to remove footers and headers across different pages by searching
                                     for the longest common string. This heuristic uses exact matches and therefore
                                     works well for footers like "Copyright 2019 by XXX", but won't detect "Page 3 of 4"
                                     or similar.
        :param remove_whitespace: strip whitespaces before or after each line in the text.
        :param remove_empty_lines: remove more than two empty lines in the text.
        :param valid_languages: validate languages from a list of languages specified in the ISO 639-1
                                (https://en.wikipedia.org/wiki/ISO_639-1) format.
                                This option can be used to add test for encoding errors. If the extracted text is
                                not one of the valid languages, then it might likely be encoding error resulting
                                in garbled text.
        """
        self.remove_numeric_tables = remove_numeric_tables
        self.remove_header_footer = remove_header_footer
        self.remove_whitespace = remove_whitespace
        self.remove_empty_lines = remove_empty_lines
        self.valid_languages = valid_languages

    @abstractmethod
    def extract_pages(self, file_path: Path) -> List[str]:
        pass
