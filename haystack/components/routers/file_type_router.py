# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import mimetypes
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from haystack import component, default_from_dict, default_to_dict
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream

CUSTOM_MIMETYPES = {
    # we add markdown because it is not added by the mimetypes module
    # see https://github.com/python/cpython/pull/17995
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    # we add msg because it is not added by the mimetypes module
    ".msg": "application/vnd.ms-outlook",
}


@component
class FileTypeRouter:
    """
    Categorizes files or byte streams by their MIME types, helping in context-based routing.

    FileTypeRouter supports both exact MIME type matching and regex patterns.

    For file paths, MIME types come from extensions, while byte streams use metadata.
    You can use regex patterns in the `mime_types` parameter to set broad categories
    (such as 'audio/*' or 'text/*') or specific types.
    MIME types without regex patterns are treated as exact matches.

    ### Usage example

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
    # {'text/plain': [
    #   PosixPath('file.txt')], 'application/pdf': [PosixPath('document.pdf')], 'unclassified': [PosixPath('song.mp3')
    # ]}
    # {'audio/.*': [
    #   PosixPath('song.mp3')], 'text/plain': [PosixPath('file.txt')], 'unclassified': [PosixPath('document.pdf')
    # ]}
    ```
    """

    def __init__(self, mime_types: List[str], additional_mimetypes: Optional[Dict[str, str]] = None):
        """
        Initialize the FileTypeRouter component.

        :param mime_types:
            A list of MIME types or regex patterns to classify the input files or byte streams.
            (for example: `["text/plain", "audio/x-wav", "image/jpeg"]`).

        :param additional_mimetypes:
            A dictionary containing the MIME type to add to the mimetypes package to prevent unsupported or non native
            packages from being unclassified.
            (for example: `{"application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx"}`).
        """
        if not mime_types:
            raise ValueError("The list of mime types cannot be empty.")

        if additional_mimetypes:
            for mime, ext in additional_mimetypes.items():
                mimetypes.add_type(mime, ext)

        self.mime_type_patterns = []
        for mime_type in mime_types:
            try:
                pattern = re.compile(mime_type)
            except re.error:
                raise ValueError(f"Invalid regex pattern '{mime_type}'.")
            self.mime_type_patterns.append(pattern)

        # the actual output type is List[Union[Path, ByteStream]],
        # but this would cause PipelineConnectError with Converters
        component.set_output_types(
            self,
            unclassified=List[Union[str, Path, ByteStream]],
            **dict.fromkeys(mime_types, List[Union[str, Path, ByteStream]]),
        )
        self.mime_types = mime_types
        self._additional_mimetypes = additional_mimetypes

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(self, mime_types=self.mime_types, additional_mimetypes=self._additional_mimetypes)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileTypeRouter":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        return default_from_dict(cls, data)

    def run(
        self,
        sources: List[Union[str, Path, ByteStream]],
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ) -> Dict[str, List[Union[ByteStream, Path]]]:
        """
        Categorize files or byte streams according to their MIME types.

        :param sources:
            A list of file paths or byte streams to categorize.

        :param meta:
            Optional metadata to attach to the sources.
            When provided, the sources are internally converted to ByteStream objects and the metadata is added.
            This value can be a list of dictionaries or a single dictionary.
            If it's a single dictionary, its content is added to the metadata of all ByteStream objects.
            If it's a list, its length must match the number of sources, as they are zipped together.

        :returns: A dictionary where the keys are MIME types (or `"unclassified"`) and the values are lists of data
            sources.
        """

        mime_types = defaultdict(list)
        meta_list = normalize_metadata(meta=meta, sources_count=len(sources))

        for source, meta_dict in zip(sources, meta_list):
            if isinstance(source, str):
                source = Path(source)

            if isinstance(source, Path):
                mime_type = self._get_mime_type(source)
            elif isinstance(source, ByteStream):
                mime_type = source.mime_type
            else:
                raise ValueError(f"Unsupported data source type: {type(source).__name__}")

            # If we have metadata, we convert the source to ByteStream and add the metadata
            if meta_dict:
                source = get_bytestream_from_source(source)
                source.meta.update(meta_dict)

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
        return CUSTOM_MIMETYPES.get(extension, mime_type)
