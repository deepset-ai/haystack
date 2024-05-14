# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import mimetypes
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

from haystack import component, logging
from haystack.dataclasses import ByteStream

logger = logging.getLogger(__name__)


@component
class FileTypeRouter:
    """
    Groups a list of data sources by their MIME types.

    FileTypeRouter groups a list of data sources (file paths or byte streams) by their MIME types, allowing
    for flexible routing of files to different components based on their content type. It supports both exact MIME type
    matching and pattern matching using regular expressions.

    For file paths, MIME types are inferred from their extensions, while for byte streams, MIME types are determined from
    the provided metadata. This enables the router to classify a diverse collection of files and data streams for
    specialized processing.

    The router's flexibility is enhanced by the support for regex patterns in the `mime_types` parameter, allowing users
    to specify broad categories (e.g., 'audio/*' or 'text/*') or more specific types with regex patterns. This feature
    is designed to be backward compatible, treating MIME types without regex patterns as exact matches.

    Usage example:
    ```python
    from haystack.components.routers import FileTypeRouter
    from pathlib import Path

    # For exact MIME type matching
    router = FileTypeRouter(mime_types=["text/plain", "application/pdf"])

    # For flexible matching using regex, to handle all audio types
    router_with_regex = FileTypeRouter(mime_types=[r"audio/.*", r"text/plain"])

    sources = [Path("file.txt"), Path("document.pdf"), Path("song.mp3")]
    print(router.run(sources=sources))
    print(router_with_regex.run(sources=sources))

    # Expected output:
    # {'text/plain': [PosixPath('file.txt')], 'application/pdf': [PosixPath('document.pdf')], 'unclassified': [PosixPath('song.mp3')]}
    # {'audio/.*': [PosixPath('song.mp3')], 'text/plain': [PosixPath('file.txt')], 'unclassified': [PosixPath('document.pdf')]}
    ```

    :param mime_types: A list of MIME types or regex patterns to classify the incoming files or data streams.
    """

    def __init__(self, mime_types: List[str]):
        """
        Initialize the FileTypeRouter component.

        :param mime_types: A list of file mime types to consider when routing files
            (e.g. `["text/plain", "audio/x-wav", "image/jpeg"]`).
        """
        if not mime_types:
            raise ValueError("The list of mime types cannot be empty.")

        self.mime_type_patterns = []
        for mime_type in mime_types:
            if not self._is_valid_mime_type_format(mime_type):
                raise ValueError(f"Invalid mime type or regex pattern: '{mime_type}'.")
            pattern = re.compile(mime_type)
            self.mime_type_patterns.append(pattern)

        component.set_output_types(self, unclassified=List[Path], **{mime_type: List[Path] for mime_type in mime_types})
        self.mime_types = mime_types

    def run(self, sources: List[Union[str, Path, ByteStream]]) -> Dict[str, List[Union[ByteStream, Path]]]:
        """
        Categorizes the provided data sources by their MIME types.

        :param sources: A list of file paths or byte streams to categorize.

        :returns: A dictionary where the keys are MIME types (or `"unclassified"`) and the values are lists of data sources.
        """

        mime_types = defaultdict(list)
        for source in sources:
            if isinstance(source, str):
                source = Path(source)
            if isinstance(source, Path):
                mime_type = self._get_mime_type(source)
            elif isinstance(source, ByteStream):
                mime_type = source.mime_type
            else:
                raise ValueError(f"Unsupported data source type: {type(source).__name__}")

            matched = False
            if mime_type:
                for pattern in self.mime_type_patterns:
                    if pattern.fullmatch(mime_type):
                        mime_types[pattern.pattern].append(source)
                        matched = True
                        break
            if not matched:
                mime_types["unclassified"].append(source)

        return dict(mime_types)

    def _get_mime_type(self, path: Path) -> Optional[str]:
        """
        Get the MIME type of the provided file path.

        :param path: The file path to get the MIME type for.

        :returns: The MIME type of the provided file path, or `None` if the MIME type cannot be determined.
        """
        extension = path.suffix.lower()
        mime_type = mimetypes.guess_type(path.as_posix())[0]
        # lookup custom mappings if the mime type is not found
        return self._get_custom_mime_mappings().get(extension, mime_type)

    def _is_valid_mime_type_format(self, mime_type: str) -> bool:
        """
        Checks if the provided MIME type string is a valid regex pattern.

        :param mime_type: The MIME type or regex pattern to validate.
        :raises ValueError: If the mime_type is not a valid regex pattern.
        :returns: Always True because a ValueError is raised for invalid patterns.
        """
        try:
            re.compile(mime_type)
            return True
        except re.error:
            raise ValueError(f"Invalid regex pattern '{mime_type}'.")

    @staticmethod
    def _get_custom_mime_mappings() -> Dict[str, str]:
        """
        Returns a dictionary of custom file extension to MIME type mappings.
        """
        # we add markdown because it is not added by the mimetypes module
        # see https://github.com/python/cpython/pull/17995
        return {".md": "text/markdown", ".markdown": "text/markdown"}
