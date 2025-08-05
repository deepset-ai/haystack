# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, Optional, Union

import filetype

from haystack import logging
from haystack.lazy_imports import LazyImport
from haystack.utils import is_in_jupyter

with LazyImport("The 'show' method requires the 'PIL' library. Run 'pip install pillow'") as pillow_import:
    from PIL import Image

logger = logging.getLogger(__name__)

# NOTE: We have to rely on this since our util functions are using the bytestream object.
#      We could change this to use the file path instead, where the file extension is used to determine the format.
# This is a mapping of image formats to their MIME types.
# from PIL import Image
# Image.init()  # <- Must force all plugins to initialize to get this mapping
# print(Image.MIME)
FORMAT_TO_MIME = {
    "BMP": "image/bmp",
    "DIB": "image/bmp",
    "PCX": "image/x-pcx",
    "EPS": "application/postscript",
    "GIF": "image/gif",
    "PNG": "image/png",
    "JPEG2000": "image/jp2",
    "ICNS": "image/icns",
    "ICO": "image/x-icon",
    "JPEG": "image/jpeg",
    "MPEG": "video/mpeg",
    "TIFF": "image/tiff",
    "MPO": "image/mpo",
    "PALM": "image/palm",
    "PDF": "application/pdf",
    "PPM": "image/x-portable-anymap",
    "PSD": "image/vnd.adobe.photoshop",
    "SGI": "image/sgi",
    "TGA": "image/x-tga",
    "WEBP": "image/webp",
    "XBM": "image/xbm",
    "XPM": "image/xpm",
}
MIME_TO_FORMAT = {v: k for k, v in FORMAT_TO_MIME.items()}
# Adding some common MIME types that are not in the PIL mapping
MIME_TO_FORMAT["image/jpg"] = "JPEG"

IMAGE_MIME_TYPES = set(MIME_TO_FORMAT.keys())


@dataclass
class ImageContent:
    """
    The image content of a chat message.

    :param base64_image: A base64 string representing the image.
    :param mime_type: The MIME type of the image (e.g. "image/png", "image/jpeg").
        Providing this value is recommended, as most LLM providers require it.
        If not provided, the MIME type is guessed from the base64 string, which can be slow and not always reliable.
    :param detail: Optional detail level of the image (only supported by OpenAI). One of "auto", "high", or "low".
    :param meta: Optional metadata for the image.
    :param validation: If True (default), a validation process is performed:
        - Check whether the base64 string is valid;
        - Guess the MIME type if not provided;
        - Check if the MIME type is a valid image MIME type.
        Set to False to skip validation and speed up initialization.
    """

    base64_image: str
    mime_type: Optional[str] = None
    detail: Optional[Literal["auto", "high", "low"]] = None
    meta: dict[str, Any] = field(default_factory=dict)
    validation: bool = True

    def __post_init__(self):
        if not self.validation:
            return

        try:
            decoded_image = base64.b64decode(self.base64_image, validate=True)
        except Exception as e:
            raise ValueError("The base64 string is not valid") from e

        # mime_type is an important information, so we try to guess it if not provided
        if not self.mime_type:
            guess = filetype.guess(decoded_image)
            if guess:
                self.mime_type = guess.mime
            else:
                msg = (
                    "Failed to guess the MIME type of the image. Omitting the MIME type may result in "
                    "processing errors or incorrect handling of the image by LLM providers."
                )
                logger.warning(msg)

        if self.mime_type and self.mime_type not in IMAGE_MIME_TYPES:
            raise ValueError(f"{self.mime_type} is not a valid image MIME type.")

    def __repr__(self) -> str:
        """
        Return a string representation of the ImageContent, truncating the base64_image to 100 bytes.
        """
        fields = []

        truncated_data = self.base64_image[:100] + "..." if len(self.base64_image) > 100 else self.base64_image
        fields.append(f"base64_image={truncated_data!r}")
        fields.append(f"mime_type={self.mime_type!r}")
        fields.append(f"detail={self.detail!r}")
        fields.append(f"meta={self.meta!r}")
        fields_str = ", ".join(fields)
        return f"{self.__class__.__name__}({fields_str})"

    def show(self) -> None:
        """
        Shows the image.
        """
        pillow_import.check()
        image_bytes = BytesIO(base64.b64decode(self.base64_image))
        image = Image.open(image_bytes)

        if is_in_jupyter():
            # ipython is not a core dependency so we cannot import it at the module level
            from IPython.display import display

            display(image)
        else:
            image.show()

    def to_dict(self) -> dict[str, Any]:
        """
        Convert ImageContent into a dictionary.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImageContent":
        """
        Create an ImageContent from a dictionary.
        """
        return ImageContent(**data)

    @classmethod
    def from_file_path(
        cls,
        file_path: Union[str, Path],
        *,
        size: Optional[tuple[int, int]] = None,
        detail: Optional[Literal["auto", "high", "low"]] = None,
        meta: Optional[dict[str, Any]] = None,
    ) -> "ImageContent":
        """
        Create an ImageContent object from a file path.

        It exposes similar functionality as the `ImageFileToImageContent` component. For PDF to ImageContent conversion,
        use the `PDFToImageContent` component.

        :param file_path:
            The path to the image file. PDF files are not supported. For PDF to ImageContent conversion, use the
            `PDFToImageContent` component.
        :param size:
            If provided, resizes the image to fit within the specified dimensions (width, height) while
            maintaining aspect ratio. This reduces file size, memory usage, and processing time, which is beneficial
            when working with models that have resolution constraints or when transmitting images to remote services.
        :param detail:
            Optional detail level of the image (only supported by OpenAI). One of "auto", "high", or "low".
        :param meta:
            Additional metadata for the image.

        :returns:
            An ImageContent object.
        """
        # to avoid a circular import
        from haystack.components.converters.image import ImageFileToImageContent

        converter = ImageFileToImageContent(size=size, detail=detail)
        result = converter.run(sources=[file_path], meta=[meta] if meta else None)
        return result["image_contents"][0]

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        retry_attempts: int = 2,
        timeout: int = 10,
        size: Optional[tuple[int, int]] = None,
        detail: Optional[Literal["auto", "high", "low"]] = None,
        meta: Optional[dict[str, Any]] = None,
    ) -> "ImageContent":
        """
        Create an ImageContent object from a URL. The image is downloaded and converted to a base64 string.

        For PDF to ImageContent conversion, use the `PDFToImageContent` component.

        :param url:
            The URL of the image. PDF files are not supported. For PDF to ImageContent conversion, use the
            `PDFToImageContent` component.
        :param retry_attempts:
            The number of times to retry to fetch the URL's content.
        :param timeout:
            Timeout in seconds for the request.
        :param size:
            If provided, resizes the image to fit within the specified dimensions (width, height) while
            maintaining aspect ratio. This reduces file size, memory usage, and processing time, which is beneficial
            when working with models that have resolution constraints or when transmitting images to remote services.
        :param detail:
            Optional detail level of the image (only supported by OpenAI). One of "auto", "high", or "low".
        :param meta:
            Additional metadata for the image.

        :raises ValueError:
            If the URL does not point to an image or if it points to a PDF file.

        :returns:
            An ImageContent object.
        """
        # to avoid circular imports
        from haystack.components.converters.image import ImageFileToImageContent
        from haystack.components.fetchers.link_content import LinkContentFetcher

        fetcher = LinkContentFetcher(raise_on_failure=True, retry_attempts=retry_attempts, timeout=timeout)
        bytestream = fetcher.run(urls=[url])["streams"][0]

        if bytestream.mime_type not in IMAGE_MIME_TYPES:
            msg = f"The URL does not point to an image. The MIME type of the URL is {bytestream.mime_type}."
            raise ValueError(msg)

        if bytestream.mime_type == "application/pdf":
            raise ValueError(
                "PDF files are not supported. "
                "For PDF to ImageContent conversion, use the `PDFToImageContent` component."
            )

        converter = ImageFileToImageContent(size=size, detail=detail)
        result = converter.run(sources=[bytestream], meta=[meta] if meta else None)
        return result["image_contents"][0]
