import logging
import mimetypes
from collections import defaultdict
from pathlib import Path
from typing import List, Union, Optional, Dict, Any

from haystack.preview import component, default_from_dict, default_to_dict

logger = logging.getLogger(__name__)


@component
class FileExtensionClassifier:
    """
    A component that classifies files based on their MIME types read from their file extensions. This component
    does not read the file contents, but rather uses the file extension to determine the MIME type of the file.

    The FileExtensionClassifier takes a list of file paths and groups them by their MIME types.
    The list of MIME types to consider is provided during the initialization of the component.

    This component is particularly useful when working with a large number of files, and you
    want to categorize them based on their MIME types.
    """

    def __init__(self, mime_types: List[str]):
        """
        Initialize the FileExtensionClassifier.

        :param mime_types: A list of file mime types to consider when classifying
        files (e.g. ["text/plain", "audio/x-wav", "image/jpeg"]).
        """
        if not mime_types:
            raise ValueError("The list of mime types cannot be empty.")

        for mime_type in mime_types:
            if not self.is_valid_mime_type_format(mime_type):
                raise ValueError(
                    f"Unknown mime type: '{mime_type}'. Ensure you passed a list of strings in the 'mime_types' parameter"
                )

        component.set_output_types(self, unclassified=List[Path], **{mime_type: List[Path] for mime_type in mime_types})
        self.mime_types = mime_types

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(self, mime_types=self.mime_types)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileExtensionClassifier":
        """
        Deserialize this component from a dictionary.
        """
        return default_from_dict(cls, data)

    def run(self, paths: List[Union[str, Path]]):
        """
        Run the FileExtensionClassifier.

        This method takes the input data, iterates through the provided file paths, checks the file
        mime type of each file, and groups the file paths by their mime types.

        :param paths: The input data containing the file paths to classify.
        :return: The output data containing the classified file paths.
        """
        mime_types = defaultdict(list)
        for path in paths:
            if isinstance(path, str):
                path = Path(path)
            mime_type = self.get_mime_type(path)
            if mime_type in self.mime_types:
                mime_types[mime_type].append(path)
            else:
                mime_types["unclassified"].append(path)

        return mime_types

    def get_mime_type(self, path: Path) -> Optional[str]:
        """
        Get the MIME type of the provided file path.

        :param path: The file path to get the MIME type for.
        :return: The MIME type of the provided file path, or None if the MIME type cannot be determined.
        """
        return mimetypes.guess_type(path.as_posix())[0]

    def is_valid_mime_type_format(self, mime_type: str) -> bool:
        """
        Check if the provided MIME type is in valid format
        :param mime_type: The MIME type to check.
        :return: True if the provided MIME type is a valid MIME type format, False otherwise.
        """
        return mime_type in mimetypes.types_map.values()
