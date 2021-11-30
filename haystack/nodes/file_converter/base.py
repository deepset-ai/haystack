from typing import List, Optional, Dict, Any, Union

from abc import abstractmethod
from pathlib import Path
import langdetect

from haystack.nodes.base import BaseComponent


class BaseConverter(BaseComponent):
    """
    Base class for implementing file converts to transform input documents to text format for ingestion in DocumentStore.
    """

    outgoing_edges = 1

    def __init__(
        self,
        remove_numeric_tables: bool = False,
        valid_languages: Optional[List[str]] = None,
    ):
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

        # save init parameters to enable export of component config as YAML
        self.set_config(
            remove_numeric_tables=remove_numeric_tables, valid_languages=valid_languages
        )

        self.remove_numeric_tables = remove_numeric_tables
        self.valid_languages = valid_languages

    @abstractmethod
    def convert(
        self,
        file_path: Path,
        meta: Optional[Dict[str, str]],
        remove_numeric_tables: Optional[bool] = None,
        valid_languages: Optional[List[str]] = None,
        encoding: Optional[str] = "utf-8",
    ) -> List[Dict[str, Any]]:
        """
        Convert a file to a dictionary containing the text and any associated meta data.

        File converters may extract file meta like name or size. In addition to it, user
        supplied meta data like author, url, external IDs can be supplied as a dictionary.

        :param file_path: path of the file to convert
        :param meta: dictionary of meta data key-value pairs to append in the returned document.
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
        :param encoding: Select the file encoding (default is `utf-8`)
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

    def run(self, file_paths: Union[Path, List[Path]],  # type: ignore
            meta: Optional[Union[Dict[str, str], List[Dict[str, str]]]] = None,  # type: ignore
            remove_numeric_tables: Optional[bool] = None,  # type: ignore
            valid_languages: Optional[List[str]] = None):  # type: ignore

        if isinstance(file_paths, Path):
            file_paths = [file_paths]

        if meta is None or isinstance(meta, dict):
            meta = [meta] * len(file_paths)  # type: ignore

        documents: list = []
        for file_path, file_meta in zip(file_paths, meta):
            for doc in self.convert(file_path=file_path,
                                    meta=file_meta,
                                    remove_numeric_tables=remove_numeric_tables,
                                    valid_languages=valid_languages):
                documents.append(doc)

        result = {"documents": documents}
        return result, "output_1"
