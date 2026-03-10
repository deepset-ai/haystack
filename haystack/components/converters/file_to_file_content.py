# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64
from pathlib import Path
from typing import Any

from haystack import component, logging
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream, FileContent

logger = logging.getLogger(__name__)


_EMPTY_BYTE_STRING = b""


@component
class FileToFileContent:
    """
    Converts files to FileContent objects to be included in ChatMessage objects.

    ### Usage example
    ```python
    from haystack.components.converters import FileToFileContent

    converter = FileToFileContent()

    sources = ["document.pdf", "video.mp4"]

    file_contents = converter.run(sources=sources)["file_contents"]
    print(file_contents)

    # [FileContent(base64_data='...',
    #              mime_type='application/pdf',
    #              filename='document.pdf',
    #              extra={}),
    #  ...]
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

            filename = source.name if isinstance(source, Path) else None

            try:
                bytestream = get_bytestream_from_source(source, guess_mime_type=True)
            except Exception as e:
                logger.warning("Could not read {source}. Skipping it. Error: {error}", source=source, error=e)
                continue

            if bytestream.data == _EMPTY_BYTE_STRING:
                logger.warning("File {source} is empty. Skipping it.", source=source)
                continue

            base64_data = base64.b64encode(bytestream.data).decode("utf-8")
            file_content = FileContent(
                base64_data=base64_data, mime_type=bytestream.mime_type, filename=filename, extra=extra_dict
            )
            file_contents.append(file_content)

        return {"file_contents": file_contents}
