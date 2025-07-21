# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from haystack import component, logging
from haystack.components.converters.image.image_utils import _convert_pdf_to_images, pillow_import, pypdfium2_import
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream
from haystack.dataclasses.image_content import ImageContent
from haystack.utils import expand_page_range

logger = logging.getLogger(__name__)


@component
class PDFToImageContent:
    """
    Converts PDF files to ImageContent objects.

    ### Usage example
    ```python
    from haystack.components.converters.image import PDFToImageContent

    converter = PDFToImageContent()

    sources = ["file.pdf", "another_file.pdf"]

    image_contents = converter.run(sources=sources)["image_contents"]
    print(image_contents)

    # [ImageContent(base64_image='...',
    #               mime_type='application/pdf',
    #               detail=None,
    #               meta={'file_path': 'file.pdf', 'page_number': 1}),
    #  ...]
    ```
    """

    def __init__(
        self,
        *,
        detail: Optional[Literal["auto", "high", "low"]] = None,
        size: Optional[Tuple[int, int]] = None,
        page_range: Optional[List[Union[str, int]]] = None,
    ):
        """
        Create the PDFToImageContent component.

        :param detail: Optional detail level of the image (only supported by OpenAI). One of "auto", "high", or "low".
            This will be passed to the created ImageContent objects.
        :param size: If provided, resizes the image to fit within the specified dimensions (width, height) while
            maintaining aspect ratio. This reduces file size, memory usage, and processing time, which is beneficial
            when working with models that have resolution constraints or when transmitting images to remote services.
        :param page_range: List of page numbers and/or page ranges to convert to images. Page numbers start at 1.
            If None, all pages in the PDF will be converted. Pages outside the valid range (1 to number of pages)
            will be skipped with a warning. For example, page_range=[1, 3] will convert only the first and third
            pages of the document. It also accepts printable range strings, e.g.:  ['1-3', '5', '8', '10-12']
            will convert pages 1, 2, 3, 5, 8, 10, 11, 12.
        """
        self.detail = detail
        self.size = size
        self.page_range = page_range
        pypdfium2_import.check()
        pillow_import.check()

    @component.output_types(image_contents=List[ImageContent])
    def run(
        self,
        sources: List[Union[str, Path, ByteStream]],
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        *,
        detail: Optional[Literal["auto", "high", "low"]] = None,
        size: Optional[Tuple[int, int]] = None,
        page_range: Optional[List[Union[str, int]]] = None,
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
        :param size:
            If provided, resizes the image to fit within the specified dimensions (width, height) while
            maintaining aspect ratio. This reduces file size, memory usage, and processing time, which is beneficial
            when working with models that have resolution constraints or when transmitting images to remote services.
            If not provided, the size value will be the one set in the constructor.
        :param page_range:
            List of page numbers and/or page ranges to convert to images. Page numbers start at 1.
            If None, all pages in the PDF will be converted. Pages outside the valid range (1 to number of pages)
            will be skipped with a warning. For example, page_range=[1, 3] will convert only the first and third
            pages of the document. It also accepts printable range strings, e.g.:  ['1-3', '5', '8', '10-12']
            will convert pages 1, 2, 3, 5, 8, 10, 11, 12.
            If not provided, the page_range value will be the one set in the constructor.

        :returns:
            A dictionary with the following keys:
            - `image_contents`: A list of ImageContent objects.
        """
        if not sources:
            return {"image_contents": []}

        resolved_detail = detail or self.detail
        resolved_size = size or self.size
        resolved_page_range = page_range or self.page_range

        expanded_page_range = expand_page_range(resolved_page_range) if resolved_page_range else None

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
            try:
                page_num_and_base64_images = _convert_pdf_to_images(
                    bytestream=bytestream, page_range=expanded_page_range, size=resolved_size, return_base64=True
                )
            except Exception as e:
                logger.warning(
                    "Could not convert file {source}. Skipping it. Error message: {error}", source=source, error=e
                )
                continue

            merged_metadata = {**bytestream.meta, **metadata}

            for page_number, image in page_num_and_base64_images:
                per_page_metadata = {**merged_metadata, "page_number": page_number}
                # we already know that image is a string because we set return_base64=True but mypy doesn't know that
                assert isinstance(image, str)
                image_contents.append(
                    ImageContent(
                        base64_image=image, mime_type="image/jpeg", meta=per_page_metadata, detail=resolved_detail
                    )
                )

        return {"image_contents": image_contents}
