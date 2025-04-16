# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from haystack import component, logging
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream
from haystack.dataclasses.chat_message import ImageContent

logger = logging.getLogger(__name__)


@component
class FileToImageContent:
    """
    Converts files to ImageContent objects.
    """

    def __init__(self, detail: Optional[Literal["auto", "high", "low"]] = None):
        self.detail = detail

    @component.output_types(image_contents=List[ImageContent])
    def run(
        self,
        sources: List[Union[str, Path, ByteStream]],
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        detail: Optional[Literal["auto", "high", "low"]] = None,
    ):
        """
        Converts files to ImageContent objects.

        :param sources:
            List of file paths or ByteStream objects to convert.
        :param meta:
            Optional metadata to attach to the documents.
            This value can be a list of dictionaries or a single dictionary.
            If it's a single dictionary, its content is added to the metadata of all produced documents.
            If it's a list, its length must match the number of sources as they're zipped together.
            For ByteStream objects, their `meta` is added to the output documents.
        :param detail:
            The detail level of the image content.
            If not provided, the detail level will be the one set in the constructor.

        :returns:
            A dictionary with the following keys:
            - `image_content`: A list of ImageContent objects.
        """
        detail = detail or self.detail

        image_contents = []

        meta_list = normalize_metadata(meta, sources_count=len(sources))

        for source, metadata in zip(sources, meta_list):
            if isinstance(source, str):
                source = Path(source)

            mime_type = None
            if isinstance(source, Path):
                mime_type = mimetypes.guess_type(source.as_posix())[0]
            elif isinstance(source, ByteStream):
                mime_type = source.mime_type

            try:
                bytestream = get_bytestream_from_source(source)
            except Exception as e:
                logger.warning("Could not read {source}. Skipping it. Error: {error}", source=source, error=e)
                continue
            try:
                # we need base64 here
                base64_image = base64.b64encode(bytestream.data).decode("utf-8")
            except Exception as e:
                logger.warning(
                    "Could not convert file {source}. Skipping it. Error message: {error}", source=source, error=e
                )
                continue

            merged_metadata = {**bytestream.meta, **metadata}

            image_content = ImageContent(
                base64_image=base64_image, mime_type=mime_type, meta=merged_metadata, detail=detail
            )
            image_contents.append(image_content)

        return {"image_contents": image_contents}
