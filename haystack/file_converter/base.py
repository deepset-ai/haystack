from abc import abstractmethod
from pathlib import Path
from typing import List, Optional, Dict, Any

import langdetect


class BaseConverter:
    """
    Base class for implementing file converts to transform input documents to text format for ingestion in DocumentStore.
    """

    def __init__(self, remove_numeric_tables: Optional[bool] = None, valid_languages: Optional[List[str]] = None):
        """
        :param remove_numeric_tables: This option uses heuristics to remove numeric rows from the tables.
                                      The tabular structures in documents might be noise for the reader model if it
                                      does not have table parsing capability for finding answers. However, tables
                                      may also have long strings that could possible candidate for searching answers.
                                      The rows containing strings are thus retained in this option.
        :param valid_languages: validate languages from a list of languages specified in the ISO 639-1
                                (https://en.wikipedia.org/wiki/ISO_639-1) format.
                                This option can be used to add test for encoding errors. If the extracted text is
                                not one of the valid languages, then it might likely be encoding error resulting
                                in garbled text.
        """
        self.remove_numeric_tables = remove_numeric_tables
        self.valid_languages = valid_languages

    @abstractmethod
    def convert(self, file_path: Path, meta: Optional[Dict[str, str]]) -> Dict[str, Any]:
        """
        Convert a file to a dictionary containing the text and any associated meta data.

        File converters may extract file meta like name or size. In addition to it, user
        supplied meta data like author, url, external IDs can be supplied as a dictionary.

        :param file_path: path of the file to convert
        :param meta: dictionary of meta data key-value pairs to append in the returned document.
        """
        pass

    def validate_language(self, text: str) -> bool:
        """
        Validate if the language of the text is one of valid languages.
        """
        if not self.valid_languages:
            return True

        try:
            lang = langdetect.detect(text)
        except langdetect.lang_detect_exception.LangDetectException:
            lang = None

        if lang in self.valid_languages:
            return True
        else:
            return False


