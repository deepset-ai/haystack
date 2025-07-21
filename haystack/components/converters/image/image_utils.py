# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64
import mimetypes
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict, Union

from typing_extensions import NotRequired

from haystack import logging
from haystack.dataclasses import ByteStream, Document
from haystack.dataclasses.image_content import IMAGE_MIME_TYPES, MIME_TO_FORMAT
from haystack.lazy_imports import LazyImport

with LazyImport("Run 'pip install pypdfium2'") as pypdfium2_import:
    from pypdfium2 import PdfDocument

with LazyImport("Run 'pip install pillow'") as pillow_import:
    from PIL import Image as PILImage
    from PIL.Image import Image
    from PIL.ImageFile import ImageFile


logger = logging.getLogger(__name__)


def _encode_image_to_base64(
    bytestream: ByteStream, size: Optional[Tuple[int, int]] = None
) -> Tuple[Optional[str], str]:
    """
    Encode an image from a ByteStream into a base64-encoded string.

    Optionally resize the image before encoding to improve performance for downstream processing.

    :param bytestream: ByteStream containing the image data.
    :param size: If provided, resizes the image to fit within the specified dimensions (width, height) while
        maintaining aspect ratio. This reduces file size, memory usage, and processing time, which is beneficial
        when working with models that have resolution constraints or when transmitting images to remote services.

    :returns:
        A tuple (mime_type, base64_str), where:
        - mime_type (Optional[str]): The mime type of the encoded image, determined from the original data or image
          content. Can be None if the mime type cannot be reliably identified.
        - base64_str (str): The base64-encoded string representation of the (optionally resized) image.
    """
    if size is None:
        if bytestream.mime_type is None:
            logger.warning(
                "No mime type provided for the image. "
                "This may cause compatibility issues with downstream systems requiring a specific mime type. "
                "Please provide a mime type for the image."
            )
        return bytestream.mime_type, base64.b64encode(bytestream.data).decode("utf-8")

    # Check the import
    pillow_import.check()

    # Load the image
    if bytestream.mime_type and bytestream.mime_type in MIME_TO_FORMAT:
        formats = [MIME_TO_FORMAT[bytestream.mime_type]]
    else:
        formats = None
    image: "ImageFile" = PILImage.open(BytesIO(bytestream.data), formats=formats)

    # NOTE: We prefer the format returned by PIL
    inferred_mime_type = image.get_format_mimetype() or bytestream.mime_type

    # Downsize the image in place
    if size is not None:
        # Set reducing_gap=None to disable multi-step shrink; better quality.
        # https://pillow.readthedocs.io/en/latest/reference/Image.html#PIL.Image.Image.thumbnail
        image.thumbnail(size=size, reducing_gap=None)

    # Convert the image to base64 string
    if not inferred_mime_type:
        logger.warning(
            "Could not determine mime type for image. Defaulting to 'image/jpeg'. "
            "Consider providing a mime_type parameter."
        )
        inferred_mime_type = "image/jpeg"
    return inferred_mime_type, _encode_pil_image_to_base64(image=image, mime_type=inferred_mime_type)


def _encode_pil_image_to_base64(image: Union["Image", "ImageFile"], mime_type: str = "image/jpeg") -> str:
    """
    Convert a PIL Image object to a base64-encoded string.

    Automatically converts images with transparency to RGB if saving as JPEG.

    :param image: A PIL Image or ImageFile object to encode.
    :param mime_type: The MIME type to use when encoding the image. Defaults to "image/jpeg".
    :returns:
        Base64-encoded string representing the image.
    """
    # Check the import
    pillow_import.check()

    # Convert image to RGB if it has an alpha channel and we are saving as JPEG
    if (mime_type == "image/jpeg" or mime_type == "image/jpg") and (
        image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info)
    ):
        image = image.convert("RGB")

    buffered = BytesIO()
    form = MIME_TO_FORMAT.get(mime_type)
    if form is None:
        logger.warning("Could not determine format for mime type {mime_type}. Defaulting to JPEG.", mime_type=mime_type)
        form = "JPEG"
    image.save(buffered, format=form)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _convert_pdf_to_images(
    *,
    bytestream: ByteStream,
    return_base64: bool = False,
    page_range: Optional[List[int]] = None,
    size: Optional[Tuple[int, int]] = None,
) -> Union[List[Tuple[int, "Image"]], List[Tuple[int, str]]]:
    """
    Convert a PDF file into a list of PIL Image objects or base64-encoded images.

    Checks PDF dimensions and adjusts size constraints based on aspect ratio.

    :param bytestream: ByteStream object containing the PDF data
    :param return_base64: If True, return base64-encoded images instead of PIL images.
    :param page_range: List of page numbers and/or page ranges to convert to images. Page numbers start at 1.
        If None, all pages in the PDF will be converted. Pages outside the valid range (1 to number of pages)
        will be skipped with a warning. For example, page_range=[1, 3] will convert only the first and third
        pages of the document. It also accepts printable range strings, e.g.:  ['1-3', '5', '8', '10-12']
        will convert pages 1, 2, 3, 5, 8, 10, 11, 12.
    :param size: If provided, resizes the image to fit within the specified dimensions (width, height) while
        maintaining aspect ratio. This reduces file size, memory usage, and processing time, which is beneficial
        when working with models that have resolution constraints or when transmitting images to remote services.

    :returns:
        A list of tuples, each tuple containing the page number and the PIL Image object or base64-encoded image string.
    """

    pypdfium2_import.check()
    pillow_import.check()

    try:
        pdf = PdfDocument(BytesIO(bytestream.data))
    except Exception as e:
        logger.warning(
            "Could not read PDF file {file_path}. Skipping it. Error: {error}",
            file_path=bytestream.meta.get("file_path"),
            error=e,
        )
        return []

    num_pages = len(pdf)
    if num_pages == 0:
        logger.warning("PDF file is empty: {file_path}", file_path=bytestream.meta.get("file_path"))
        pdf.close()
        return []

    all_pdf_images = []

    resolved_page_range = page_range or range(1, num_pages + 1)

    for page_number in resolved_page_range:
        if page_number < 1 or page_number > num_pages:
            logger.warning("Page {page_number} is out of range for the PDF file. Skipping it.", page_number=page_number)
            continue

        # Get dimensions of the page
        page = pdf[max(page_number - 1, 0)]  # Adjust for 0-based indexing
        _, _, width, height = page.get_mediabox()

        target_resolution_dpi = 300.0

        # From pypdfium2 docs: scale (float) – A factor scaling the number of pixels per PDF canvas unit. This defines
        # the resolution of the image. To convert a DPI value to a scale factor, multiply it by the size of 1 canvas
        # unit in inches (usually 1/72in).
        # https://pypdfium2.readthedocs.io/en/stable/python_api.html#pypdfium2._helpers.page.PdfPage.render
        target_scale = target_resolution_dpi / 72.0

        # Calculate potential pixels for target_dpi
        pixels_for_target_scale = width * height * target_scale**2

        pil_max_pixels = PILImage.MAX_IMAGE_PIXELS or int(1024 * 1024 * 1024 // 4 // 3)
        # 90% of PIL's default limit to prevent borderline cases
        pixel_limit = pil_max_pixels * 0.9

        scale = target_scale
        if pixels_for_target_scale > pixel_limit:
            logger.info(
                "Large PDF detected ({pixels:.2f} pixels). Resizing the image to fit the pixel limit.",
                pixels=pixels_for_target_scale,
            )
            scale = (pixel_limit / (width * height)) ** 0.5

        pdf_bitmap = page.render(scale=scale)

        image: "Image" = pdf_bitmap.to_pil()
        pdf_bitmap.close()
        if size is not None:
            # Set reducing_gap=None to disable multi-step shrink; better quality.
            # https://pillow.readthedocs.io/en/latest/reference/Image.html#PIL.Image.Image.thumbnail
            image.thumbnail(size=size, reducing_gap=None)

        all_pdf_images.append((page_number, image))

    pdf.close()

    if return_base64:
        return [
            (page_number, _encode_pil_image_to_base64(image, mime_type="image/jpeg"))
            for page_number, image in all_pdf_images
        ]

    return all_pdf_images


class _ImageSourceInfo(TypedDict):
    path: Path
    mime_type: Optional[str]
    page_number: NotRequired[int]  # Only present for PDF documents


def _extract_image_sources_info(
    documents: List[Document], file_path_meta_field: str, root_path: str
) -> List[_ImageSourceInfo]:
    """
    Extracts the image source information from the documents.

    :param documents: List of documents to extract image source information from.
    :param file_path_meta_field: The metadata field in the Document that contains the file path to the image or PDF.
    :param root_path: The root directory path where document files are located.

    :returns:
        A list of _ImageSourceInfo dictionaries, each containing the path and type of the image.
        If the image is a PDF, the dictionary also contains the page number.
    :raises ValueError: If the document is missing the file_path_meta_field key in its metadata, the file path is
        invalid, the MIME type is not supported, or the page number is missing for a PDF document.
    """
    images_source_info: List[_ImageSourceInfo] = []
    for doc in documents:
        file_path = doc.meta.get(file_path_meta_field)
        if file_path is None:
            raise ValueError(
                f"Document with ID '{doc.id}' is missing the '{file_path_meta_field}' key in its metadata."
                f" Please ensure that the documents you are trying to convert have this key set."
            )

        resolved_file_path = Path(root_path, file_path)
        if not resolved_file_path.is_file():
            raise ValueError(
                f"Document with ID '{doc.id}' has an invalid file path '{resolved_file_path}'. "
                f"Please ensure that the documents you are trying to convert have valid file paths."
            )

        mime_type = doc.meta.get("mime_type") or mimetypes.guess_type(resolved_file_path)[0]
        if mime_type not in IMAGE_MIME_TYPES:
            raise ValueError(
                f"Document with file path '{resolved_file_path}' has an unsupported MIME type '{mime_type}'. "
                f"Please ensure that the documents you are trying to convert are of the supported "
                f"types: {', '.join(IMAGE_MIME_TYPES)}."
            )

        image_info: _ImageSourceInfo = {"path": resolved_file_path, "mime_type": mime_type}

        # If mimetype is PDF we also need the page number to be able to convert the right page
        if mime_type == "application/pdf":
            page_number = doc.meta.get("page_number")
            if page_number is None:
                raise ValueError(
                    f"Document with ID '{doc.id}' comes from the PDF file '{resolved_file_path}' but is missing "
                    f"the 'page_number' key in its metadata. Please ensure that PDF documents you are trying to "
                    f"convert have this key set."
                )
            image_info["page_number"] = page_number

        images_source_info.append(image_info)

    return images_source_info


class _PDFPageInfo(TypedDict):
    doc_idx: int
    path: Path
    page_number: int


def _batch_convert_pdf_pages_to_images(
    *, pdf_page_infos: List[_PDFPageInfo], return_base64: bool = False, size: Optional[Tuple[int, int]] = None
) -> Union[Dict[int, str], Dict[int, "Image"]]:
    """
    Converts selected PDF pages to images, returning a mapping from document indices to images (PIL or base64).

    Pages are grouped by file path to ensure each PDF is opened and processed only once for efficiency.

    :param pdf_page_infos: List of _PDFPageInfo dictionaries with doc_idx, path, and page_number.
    :param size: Optional tuple of width and height to resize the images to.
    :param return_base64: If True, return base64 encoded images instead of PIL images.

    :returns: Dictionary mapping document indices to images (PIL.Image or base64 string).
    """
    if not pdf_page_infos:
        return {}

    page_infos_by_pdf_path = defaultdict(list)
    for page_info in pdf_page_infos:
        page_infos_by_pdf_path[page_info["path"]].append(page_info)

    converted_images_by_doc_index = {}

    for pdf_path, page_infos_for_pdf in page_infos_by_pdf_path.items():
        page_numbers_to_convert = [info["page_number"] for info in page_infos_for_pdf]
        bytestream = ByteStream.from_file_path(pdf_path)

        converted_pages = _convert_pdf_to_images(
            bytestream=bytestream, return_base64=return_base64, page_range=page_numbers_to_convert, size=size
        )

        # Map results back to document indices
        page_number_to_image = dict(converted_pages)
        for page_info in page_infos_for_pdf:
            converted_images_by_doc_index[page_info["doc_idx"]] = page_number_to_image[page_info["page_number"]]

    # mypy is not able to infer that we match the declared return type
    return converted_images_by_doc_index  # type: ignore[return-value]
