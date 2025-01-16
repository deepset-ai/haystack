# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import io
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream
from haystack.lazy_imports import LazyImport

with LazyImport("Run 'pip install pypdf'") as pypdf_import:
    from pypdf import PdfReader


logger = logging.getLogger(__name__)


class PyPDFExtractionMode(Enum):
    """
    The mode to use for extracting text from a PDF.
    """

    PLAIN = "plain"
    LAYOUT = "layout"

    def __str__(self) -> str:
        """
        Convert a PyPDFExtractionMode enum to a string.
        """
        return self.value

    @staticmethod
    def from_str(string: str) -> "PyPDFExtractionMode":
        """
        Convert a string to a PyPDFExtractionMode enum.
        """
        enum_map = {e.value: e for e in PyPDFExtractionMode}
        mode = enum_map.get(string)
        if mode is None:
            msg = f"Unknown extraction mode '{string}'. Supported modes are: {list(enum_map.keys())}"
            raise ValueError(msg)
        return mode


@component
class PyPDFToDocument:
    """
    Converts PDF files to documents your pipeline can query.

    This component uses the PyPDF library.
    You can attach metadata to the resulting documents.

    ### Usage example

    ```python
    from haystack.components.converters.pypdf import PyPDFToDocument

    converter = PyPDFToDocument()
    results = converter.run(sources=["sample.pdf"], meta={"date_added": datetime.now().isoformat()})
    documents = results["documents"]
    print(documents[0].content)
    # 'This is a text from the PDF file.'
    ```
    """

    def __init__(
        self,
        *,
        extraction_mode: Union[str, PyPDFExtractionMode] = PyPDFExtractionMode.PLAIN,
        plain_mode_orientations: tuple = (0, 90, 180, 270),
        plain_mode_space_width: float = 200.0,
        layout_mode_space_vertically: bool = True,
        layout_mode_scale_weight: float = 1.25,
        layout_mode_strip_rotated: bool = True,
        layout_mode_font_height_weight: float = 1.0,
        store_full_path: bool = False,
    ):
        """
        Create an PyPDFToDocument component.

        :param extraction_mode:
            The mode to use for extracting text from a PDF.
            Layout mode is an experimental mode that adheres to the rendered layout of the PDF.
        :param plain_mode_orientations:
            Tuple of orientations to look for when extracting text from a PDF in plain mode.
            Ignored if `extraction_mode` is `PyPDFExtractionMode.LAYOUT`.
        :param plain_mode_space_width:
            Forces default space width if not extracted from font.
            Ignored if `extraction_mode` is `PyPDFExtractionMode.LAYOUT`.
        :param layout_mode_space_vertically:
            Whether to include blank lines inferred from y distance + font height.
            Ignored if `extraction_mode` is `PyPDFExtractionMode.PLAIN`.
        :param layout_mode_scale_weight:
            Multiplier for string length when calculating weighted average character width.
            Ignored if `extraction_mode` is `PyPDFExtractionMode.PLAIN`.
        :param layout_mode_strip_rotated:
            Layout mode does not support rotated text. Set to `False` to include rotated text anyway.
            If rotated text is discovered, layout will be degraded and a warning will be logged.
            Ignored if `extraction_mode` is `PyPDFExtractionMode.PLAIN`.
        :param layout_mode_font_height_weight:
            Multiplier for font height when calculating blank line height.
            Ignored if `extraction_mode` is `PyPDFExtractionMode.PLAIN`.
        :param store_full_path:
            If True, the full path of the file is stored in the metadata of the document.
            If False, only the file name is stored.
        """
        pypdf_import.check()

        self.store_full_path = store_full_path

        if isinstance(extraction_mode, str):
            extraction_mode = PyPDFExtractionMode.from_str(extraction_mode)
        self.extraction_mode = extraction_mode
        self.plain_mode_orientations = plain_mode_orientations
        self.plain_mode_space_width = plain_mode_space_width
        self.layout_mode_space_vertically = layout_mode_space_vertically
        self.layout_mode_scale_weight = layout_mode_scale_weight
        self.layout_mode_strip_rotated = layout_mode_strip_rotated
        self.layout_mode_font_height_weight = layout_mode_font_height_weight

    def to_dict(self):
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            extraction_mode=str(self.extraction_mode),
            plain_mode_orientations=self.plain_mode_orientations,
            plain_mode_space_width=self.plain_mode_space_width,
            layout_mode_space_vertically=self.layout_mode_space_vertically,
            layout_mode_scale_weight=self.layout_mode_scale_weight,
            layout_mode_strip_rotated=self.layout_mode_strip_rotated,
            layout_mode_font_height_weight=self.layout_mode_font_height_weight,
            store_full_path=self.store_full_path,
        )

    @classmethod
    def from_dict(cls, data):
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary with serialized data.

        :returns:
            Deserialized component.
        """
        return default_from_dict(cls, data)

    def _default_convert(self, reader: "PdfReader") -> str:
        texts = []
        for page in reader.pages:
            texts.append(
                page.extract_text(
                    orientations=self.plain_mode_orientations,
                    extraction_mode=self.extraction_mode.value,
                    space_width=self.plain_mode_space_width,
                    layout_mode_space_vertically=self.layout_mode_space_vertically,
                    layout_mode_scale_weight=self.layout_mode_scale_weight,
                    layout_mode_strip_rotated=self.layout_mode_strip_rotated,
                    layout_mode_font_height_weight=self.layout_mode_font_height_weight,
                )
            )
        text = "\f".join(texts)
        return text

    @component.output_types(documents=List[Document])
    def run(
        self,
        sources: List[Union[str, Path, ByteStream]],
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ):
        """
        Converts PDF files to documents.

        :param sources:
            List of file paths or ByteStream objects to convert.
        :param meta:
            Optional metadata to attach to the documents.
            This value can be a list of dictionaries or a single dictionary.
            If it's a single dictionary, its content is added to the metadata of all produced documents.
            If it's a list, its length must match the number of sources, as they are zipped together.
            For ByteStream objects, their `meta` is added to the output documents.

        :returns:
            A dictionary with the following keys:
            - `documents`: A list of converted documents.
        """
        documents = []
        meta_list = normalize_metadata(meta, sources_count=len(sources))

        for source, metadata in zip(sources, meta_list):
            try:
                bytestream = get_bytestream_from_source(source)
            except Exception as e:
                logger.warning("Could not read {source}. Skipping it. Error: {error}", source=source, error=e)
                continue
            try:
                pdf_reader = PdfReader(io.BytesIO(bytestream.data))
                text = self._default_convert(pdf_reader)
            except Exception as e:
                logger.warning(
                    "Could not read {source} and convert it to Document, skipping. {error}", source=source, error=e
                )
                continue

            if text is None or text.strip() == "":
                logger.warning(
                    "PyPDFToDocument could not extract text from the file {source}. Returning an empty document.",
                    source=source,
                )

            merged_metadata = {**bytestream.meta, **metadata}

            if not self.store_full_path and (file_path := bytestream.meta.get("file_path")):
                merged_metadata["file_path"] = os.path.basename(file_path)
            document = Document(content=text, meta=merged_metadata)
            documents.append(document)

        return {"documents": documents}
