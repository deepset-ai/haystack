# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64
import os
from pathlib import Path
from typing import Any

from haystack import component, logging
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream, FileContent

logger = logging.getLogger(__name__)


_EMPTY_BYTE_STRING = b""


def _filename_from_meta(meta: dict[str, Any]) -> str | None:
    """
    Get a file name from ByteStream metadata.

    :param meta: The ByteStream metadata.
    :returns: The `file_name` value if set, otherwise the base name of `file_path`, otherwise `None`.
    """
    if file_name := meta.get("file_name"):
        return str(file_name)
    if file_path := meta.get("file_path"):
        return os.path.basename(str(file_path))
    return None


@component
class FileToFileContent:
    """
    Converts files to FileContent objects to be included in ChatMessage objects.

    ### Usage example
    <!-- test-ignore -->
    ```python
    from haystack.components.converters import FileToFileContent

    converter = FileToFileContent()
    sources = ["test/test_files/pdf/react_paper.pdf", "test/test_files/images/haystack-logo.png"]
    file_contents = converter.run(sources=sources)["file_contents"]

    print(file_contents)
    # >> [FileContent(base64_data='...', mime_type='application/pdf', filename='react_paper.pdf', extra={}),
    # >>  FileContent(base64_data='...', mime_type='image/png', filename='haystack-logo.png', extra={})
    # >>]
    ```
    """

    @component.output_types(file_contents=list[FileContent])
    def run(
        self, sources: list[str | Path | ByteStream], *, extra: dict[str, Any] | list[dict[str, Any]] | None = None
    ) -> dict[str, list[FileContent]]:
        """
        Converts files to FileContent objects.

        :param sources:
            List of file paths or ByteStream objects to convert.
            For a ByteStream, the file name is taken from `meta["file_name"]`, or else from the base name of
            `meta["file_path"]`.
        :param extra:
            Optional extra information to attach to the FileContent objects. Can be used to store provider-specific
            information.
            To avoid serialization issues, values should be JSON serializable.
            This value can be a list of dictionaries or a single dictionary.
            If it's a single dictionary, its content is added to the extra of all produced FileContent objects.
            If it's a list, its length must match the number of sources as they're zipped together.

        :returns:
            A dictionary with the following keys:
            - `file_contents`: A list of FileContent objects.
        """
        if not sources:
            return {"file_contents": []}

        file_contents = []

        extra_list = normalize_metadata(extra, sources_count=len(sources))

        for source, extra_dict in zip(sources, extra_list, strict=True):
            if isinstance(source, str):
                source = Path(source)

            try:
                bytestream = get_bytestream_from_source(source, guess_mime_type=True)
            except Exception as e:
                logger.warning("Could not read {source}. Skipping it. Error: {error}", source=source, error=e)
                continue

            filename = source.name if isinstance(source, Path) else _filename_from_meta(bytestream.meta)

            if bytestream.data == _EMPTY_BYTE_STRING:
                logger.warning("File {source} is empty. Skipping it.", source=source)
                continue

            base64_data = base64.b64encode(bytestream.data).decode("utf-8")
            file_content = FileContent(
                base64_data=base64_data, mime_type=bytestream.mime_type, filename=filename, extra=extra_dict
            )
            file_contents.append(file_content)

        return {"file_contents": file_contents}
