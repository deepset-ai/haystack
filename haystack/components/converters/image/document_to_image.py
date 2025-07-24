# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Literal, Optional, Tuple

from haystack import Document, component, logging
from haystack.components.converters.image.image_utils import (
    _batch_convert_pdf_pages_to_images,
    _encode_image_to_base64,
    _extract_image_sources_info,
    _PDFPageInfo,
    pillow_import,
    pypdfium2_import,
)
from haystack.dataclasses import ByteStream
from haystack.dataclasses.image_content import ImageContent

logger = logging.getLogger(__name__)


@component
class DocumentToImageContent:
    """
    Converts documents sourced from PDF and image files into ImageContents.

    This component processes a list of documents and extracts visual content from supported file formats, converting
    them into ImageContents that can be used for multimodal AI tasks. It handles both direct image files and PDF
    documents by extracting specific pages as images.

    Documents are expected to have metadata containing:
    - The `file_path_meta_field` key with a valid file path that exists when combined with `root_path`
    - A supported image format (MIME type must be one of the supported image types)
    - For PDF files, a `page_number` key specifying which page to extract

    ### Usage example
        ```python
        from haystack import Document
        from haystack.components.converters.image.document_to_image import DocumentToImageContent

        converter = DocumentToImageContent(
            file_path_meta_field="file_path",
            root_path="/data/files",
            detail="high",
            size=(800, 600)
        )

        documents = [
            Document(content="Optional description of image.jpg", meta={"file_path": "image.jpg"}),
            Document(content="Text content of page 1 of doc.pdf", meta={"file_path": "doc.pdf", "page_number": 1})
        ]

        result = converter.run(documents)
        image_contents = result["image_contents"]
        # [ImageContent(
        #    base64_image='/9j/4A...', mime_type='image/jpeg', detail='high', meta={'file_path': 'image.jpg'}
        #  ),
        #  ImageContent(
        #    base64_image='/9j/4A...', mime_type='image/jpeg', detail='high',
        #    meta={'page_number': 1, 'file_path': 'doc.pdf'}
        #  )]
        ```
    """

    def __init__(
        self,
        *,
        file_path_meta_field: str = "file_path",
        root_path: Optional[str] = None,
        detail: Optional[Literal["auto", "high", "low"]] = None,
        size: Optional[Tuple[int, int]] = None,
    ):
        """
        Initialize the DocumentToImageContent component.

        :param file_path_meta_field: The metadata field in the Document that contains the file path to the image or PDF.
        :param root_path: The root directory path where document files are located. If provided, file paths in
            document metadata will be resolved relative to this path. If None, file paths are treated as absolute paths.
        :param detail: Optional detail level of the image (only supported by OpenAI). Can be "auto", "high", or "low".
            This will be passed to the created ImageContent objects.
        :param size: If provided, resizes the image to fit within the specified dimensions (width, height) while
            maintaining aspect ratio. This reduces file size, memory usage, and processing time, which is beneficial
            when working with models that have resolution constraints or when transmitting images to remote services.
        """
        pillow_import.check()
        pypdfium2_import.check()

        self.file_path_meta_field = file_path_meta_field
        self.root_path = root_path or ""
        self.detail = detail
        self.size = size

    @component.output_types(image_contents=List[Optional[ImageContent]])
    def run(self, documents: List[Document]) -> Dict[str, List[Optional[ImageContent]]]:
        """
        Convert documents with image or PDF sources into ImageContent objects.

        This method processes the input documents, extracting images from supported file formats and converting them
        into ImageContent objects.

        :param documents: A list of documents to process. Each document should have metadata containing at minimum
            a 'file_path_meta_field' key. PDF documents additionally require a 'page_number' key to specify which
            page to convert.

        :returns:
            Dictionary containing one key:
            - "image_contents": ImageContents created from the processed documents. These contain base64-encoded image
                data and metadata. The order corresponds to order of input documents.
        :raises ValueError:
            If any document is missing the required metadata keys, has an invalid file path, or has an unsupported
            MIME type. The error message will specify which document and what information is missing or incorrect.
        """
        if not documents:
            return {"image_contents": []}

        images_source_info = _extract_image_sources_info(
            documents=documents, file_path_meta_field=self.file_path_meta_field, root_path=self.root_path
        )

        image_contents: List[Optional[ImageContent]] = [None] * len(documents)

        pdf_page_infos: List[_PDFPageInfo] = []

        for doc_idx, image_source_info in enumerate(images_source_info):
            mime_type = image_source_info["mime_type"]
            path = image_source_info["path"]
            if mime_type == "application/pdf":
                # Store PDF documents for later processing
                page_number = image_source_info.get("page_number")
                assert page_number is not None  # checked in _extract_image_sources_info but mypy doesn't know that
                pdf_page_info: _PDFPageInfo = {"doc_idx": doc_idx, "path": path, "page_number": page_number}
                pdf_page_infos.append(pdf_page_info)
            else:
                # Process images directly
                bytestream = ByteStream.from_file_path(filepath=path, mime_type=mime_type)
                _, base64_image = _encode_image_to_base64(bytestream=bytestream, size=self.size)
                image_contents[doc_idx] = ImageContent(
                    base64_image=base64_image,
                    mime_type=mime_type,
                    detail=self.detail,
                    meta={"file_path": documents[doc_idx].meta[self.file_path_meta_field]},
                )

        # efficiently convert PDF pages to images: each PDF is opened and processed only once
        pdf_page_infos_by_doc_idx: Dict[int, _PDFPageInfo] = {
            pdf_page_info["doc_idx"]: pdf_page_info for pdf_page_info in pdf_page_infos
        }
        pdf_images_by_doc_idx = _batch_convert_pdf_pages_to_images(
            pdf_page_infos=pdf_page_infos, size=self.size, return_base64=True
        )
        for doc_idx, base64_pdf_image in pdf_images_by_doc_idx.items():
            meta = {
                "file_path": documents[doc_idx].meta[self.file_path_meta_field],
                "page_number": pdf_page_infos_by_doc_idx[doc_idx]["page_number"],
            }
            # we know that base64_pdf_image is a string because we set return_base64=True but mypy doesn't know that
            assert isinstance(base64_pdf_image, str)
            image_contents[doc_idx] = ImageContent(
                base64_image=base64_pdf_image, mime_type="image/jpeg", detail=self.detail, meta=meta
            )

        none_image_contents_doc_ids = [
            documents[doc_idx].id for doc_idx, image_content in enumerate(image_contents) if image_content is None
        ]
        if none_image_contents_doc_ids:
            logger.warning(
                "Conversion failed for some documents. Their output will be None. "
                f"Document IDs: {none_image_contents_doc_ids}"
            )

        return {"image_contents": image_contents}
