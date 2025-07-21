# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from haystack import component, logging
from haystack.components.converters.image.image_utils import _encode_image_to_base64
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream
from haystack.dataclasses.image_content import ImageContent
from haystack.lazy_imports import LazyImport

with LazyImport(
    "The 'size' parameter is set. "
    "Image resizing will be applied, which requires the Pillow library. "
    "Run 'pip install pillow'"
) as pillow_import:
    import PIL  # pylint: disable=unused-import

logger = logging.getLogger(__name__)


_EMPTY_BYTE_STRING = b""


@component
class ImageFileToImageContent:
    """
    Converts image files to ImageContent objects.

    ### Usage example
    ```python
    from haystack.components.converters.image import ImageFileToImageContent

    converter = ImageFileToImageContent()

    sources = ["image.jpg", "another_image.png"]

    image_contents = converter.run(sources=sources)["image_contents"]
    print(image_contents)

    # [ImageContent(base64_image='...',
    #               mime_type='image/jpeg',
    #               detail=None,
    #               meta={'file_path': 'image.jpg'}),
    #  ...]
    ```
    """

    def __init__(
        self, *, detail: Optional[Literal["auto", "high", "low"]] = None, size: Optional[Tuple[int, int]] = None
    ):
        """
        Create the ImageFileToImageContent component.

        :param detail: Optional detail level of the image (only supported by OpenAI). One of "auto", "high", or "low".
            This will be passed to the created ImageContent objects.
        :param size: If provided, resizes the image to fit within the specified dimensions (width, height) while
            maintaining aspect ratio. This reduces file size, memory usage, and processing time, which is beneficial
            when working with models that have resolution constraints or when transmitting images to remote services.
        """
        self.detail = detail
        self.size = size

        if self.size is not None:
            pillow_import.check()

    @component.output_types(image_contents=List[ImageContent])
    def run(
        self,
        sources: List[Union[str, Path, ByteStream]],
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        *,
        detail: Optional[Literal["auto", "high", "low"]] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, List[ImageContent]]:
        """
        Converts files to ImageContent objects.

        :param sources:
            List of file paths or ByteStream objects to convert.
        :param meta:
            Optional metadata to attach to the ImageContent objects.
            This value can be a list of dictionaries or a single dictionary.
            If it's a single dictionary, its content is added to the metadata of all produced ImageContent objects.
            If it's a list, its length must match the number of sources as they're zipped together.
            For ByteStream objects, their `meta` is added to the output ImageContent objects.
        :param detail:
            Optional detail level of the image (only supported by OpenAI). One of "auto", "high", or "low".
            This will be passed to the created ImageContent objects.
            If not provided, the detail level will be the one set in the constructor.
        :param size: If provided, resizes the image to fit within the specified dimensions (width, height) while
            maintaining aspect ratio. This reduces file size, memory usage, and processing time, which is beneficial
            when working with models that have resolution constraints or when transmitting images to remote services.
            If not provided, the size value will be the one set in the constructor.

        :returns:
            A dictionary with the following keys:
            - `image_contents`: A list of ImageContent objects.
        """
        if not sources:
            return {"image_contents": []}

        resolved_detail = detail or self.detail
        resolved_size = size or self.size

        # Check import
        if resolved_size:
            pillow_import.check()

        image_contents = []

        meta_list = normalize_metadata(meta, sources_count=len(sources))

        for source, metadata in zip(sources, meta_list):
            if isinstance(source, str):
                source = Path(source)

            try:
                bytestream = get_bytestream_from_source(source)
            except Exception as e:
                logger.warning("Could not read {source}. Skipping it. Error: {error}", source=source, error=e)
                continue

            if bytestream.mime_type is None and isinstance(source, Path):
                bytestream.mime_type = mimetypes.guess_type(source.as_posix())[0]

            if bytestream.data == _EMPTY_BYTE_STRING:
                logger.warning("File {source} is empty. Skipping it.", source=source)
                continue

            try:
                inferred_mime_type, base64_image = _encode_image_to_base64(bytestream=bytestream, size=resolved_size)
            except Exception as e:
                logger.warning(
                    "Could not convert file {source}. Skipping it. Error message: {error}", source=source, error=e
                )
                continue

            merged_metadata = {**bytestream.meta, **metadata}
            image_content = ImageContent(
                base64_image=base64_image, mime_type=inferred_mime_type, meta=merged_metadata, detail=resolved_detail
            )
            image_contents.append(image_content)

        return {"image_contents": image_contents}
