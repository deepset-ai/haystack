import logging
import mimetypes
from collections import defaultdict
from pathlib import Path
from typing import List, Union, Optional
from dataclasses import make_dataclass

from haystack.preview import component

logger = logging.getLogger(__name__)


@component
class FileTypeClassifier:
    """
    A component that classifies files based on their MIME types.

    The FileTypeClassifier takes a list of file paths and groups them by their MIME types.
    The list of MIME types to consider is provided during the initialization of the component.

    This component is particularly useful when working with a large number of files, and you
    want to categorize them based on their MIME types.
    """

    @component.input
    def input(self):
        """
        The input data for the FileTypeClassifier.

        It expects a list of file paths. The file paths can be of type str or pathlib.Path.
        """

        class Input:
            paths: List[Union[str, Path]]

        return Input

    @component.output
    def output(self):
        """
        The output data of the FileTypeClassifier.

        It returns a dictionary where the keys are the file mime types, and the values are lists
        of pathlib.Path objects, representing the file paths that match the corresponding mime type.
        """
        return make_dataclass(
            "Output", fields=[(mime_type, Optional[List[Union[str, Path]]], None) for mime_type in self.mime_types]
        )

    def __init__(self, mime_types: List[str]):
        """
        Initialize the FileTypeClassifier.

        :param mime_types: A list of file mime types to consider when classifying
        files (e.g. ["text/plain", "audio/x-wav", "image/jpeg"]).
        """
        if not mime_types:
            raise ValueError("The list of mime types cannot be empty.")

        all_known_mime_types = all(self.is_known_mime_type(mime_type) for mime_type in mime_types)
        if not all_known_mime_types:
            raise ValueError(f"The list of mime types contains unknown mime types: {mime_types}")

        # convert the mime types to the underscore format (e.g. text_plain)
        # otherwise we'll have issues with the dataclass field name convention
        # in the output dataclass
        mime_types = [self.to_underscore_format(mime_type) for mime_type in mime_types]

        # add the "unclassified" mime type to the list of mime types
        mime_types.append("unclassified")
        self.defaults = {"mime_types": mime_types}
        self.mime_types = mime_types

    def run(self, data):
        """
        Run the FileTypeClassifier.

        This method takes the input data, iterates through the provided file paths, checks the file
        mime type of each file, and groups the file paths by their mime types.

        :param data: The input data containing the file paths to classify.
        :return: The output data containing the classified file paths.
        """
        mime_types = defaultdict(list)
        paths: List[Union[str, Path]] = data.paths
        for path in paths:
            if isinstance(path, str):
                path = Path(path)
            mime_type = self.to_underscore_format(self.get_mime_type(path))
            if mime_type in self.mime_types:
                mime_types[mime_type].append(path)
            else:
                mime_types["unclassified"].append(path)

        return self.output(**mime_types)

    def get_mime_type(self, path: Path) -> Optional[str]:
        """
        Get the MIME type of the provided file path.

        :param path: The file path to get the MIME type for.
        :return: The MIME type of the provided file path, or None if the MIME type cannot be determined.
        """
        return mimetypes.guess_type(path.as_posix())[0]

    def to_underscore_format(self, mime_type: str) -> str:
        """
        Convert the provided MIME type to underscore format.
        :param mime_type: The MIME type to convert.
        :return: The converted MIME type or an empty string if the provided MIME type is None.
        """
        if mime_type:
            return "".join(c if c.isalnum() else "_" for c in mime_type)
        return ""

    def is_known_mime_type(self, mime_type: str) -> bool:
        """
        Check if the provided MIME type is a known MIME type.
        :param mime_type: The MIME type to check.
        :return: True if the provided MIME type is a known MIME type, False otherwise.
        """
        return mime_type in mimetypes.types_map.values() or mime_type in mimetypes.common_types.values()
