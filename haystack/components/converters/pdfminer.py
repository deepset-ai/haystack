# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from haystack import Document, component, logging
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream
from haystack.lazy_imports import LazyImport

with LazyImport("Run 'pip install pdfminer.six'") as pdfminer_import:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LAParams, LTTextContainer

logger = logging.getLogger(__name__)


@component
class PDFMinerToDocument:
    """
    Converts PDF files to Documents.

    Uses `pdfminer` compatible converters to convert PDF files to Documents. https://pdfminersix.readthedocs.io/en/latest/

    Usage example:
    ```python
    from haystack.components.converters.pdfminer import PDFMinerToDocument

    converter = PDFMinerToDocument()
    results = converter.run(sources=["sample.pdf"], meta={"date_added": datetime.now().isoformat()})
    documents = results["documents"]
    print(documents[0].content)
    # 'This is a text from the PDF file.'
    ```
    """

    def __init__(
        self,
        line_overlap: float = 0.5,
        char_margin: float = 2.0,
        line_margin: float = 0.5,
        word_margin: float = 0.1,
        boxes_flow: Optional[float] = 0.5,
        detect_vertical: bool = True,
        all_texts: bool = False,
    ) -> None:
        """
        Create a PDFMinerToDocument component.

        :param line_overlap:
            This parameter determines whether two characters are considered to be on
            the same line based on the amount of overlap between them.
            The overlap is calculated relative to the minimum height of both characters.
        :param char_margin:
            Determines whether two characters are part of the same line based on the distance between them.
            If the distance is less than the margin specified, the characters are considered to be on the same line.
            The margin is calculated relative to the width of the character.
        :param word_margin:
            Determines whether two characters on the same line are part of the same word
            based on the distance between them. If the distance is greater than the margin specified,
            an intermediate space will be added between them to make the text more readable.
            The margin is calculated relative to the width of the character.
        :param line_margin:
            This parameter determines whether two lines are part of the same paragraph based on
            the distance between them. If the distance is less than the margin specified,
            the lines are considered to be part of the same paragraph.
            The margin is calculated relative to the height of a line.
        :param boxes_flow:
            This parameter determines the importance of horizontal and vertical position when
            determining the order of text boxes. A value between -1.0 and +1.0 can be set,
            with -1.0 indicating that only horizontal position matters and +1.0 indicating
            that only vertical position matters. Setting the value to 'None' will disable advanced
            layout analysis, and text boxes will be ordered based on the position of their bottom left corner.
        :param detect_vertical:
            This parameter determines whether vertical text should be considered during layout analysis.
        :param all_texts:
            If layout analysis should be performed on text in figures.
        """

        pdfminer_import.check()

        self.layout_params = LAParams(
            line_overlap=line_overlap,
            char_margin=char_margin,
            line_margin=line_margin,
            word_margin=word_margin,
            boxes_flow=boxes_flow,
            detect_vertical=detect_vertical,
            all_texts=all_texts,
        )

    def __converter(self, extractor) -> Document:
        """
        Extracts text from PDF pages then convert the text into Documents

        :param extractor:
            Python generator that yields PDF pages.

        :returns:
            PDF text converted to Haystack Document
        """
        pages = []
        for page in extractor:
            text = ""
            for container in page:
                # Keep text only
                if isinstance(container, LTTextContainer):
                    text += container.get_text()
            pages.append(text)

        # Add a page delimiter
        concat = "\f".join(pages)

        return Document(content=concat)

    @component.output_types(documents=List[Document])
    def run(
        self,
        sources: List[Union[str, Path, ByteStream]],
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ):
        """
        Converts PDF files to Documents.

        :param sources:
            List of PDF file paths or ByteStream objects.
        :param meta:
            Optional metadata to attach to the Documents.
            This value can be either a list of dictionaries or a single dictionary.
            If it's a single dictionary, its content is added to the metadata of all produced Documents.
            If it's a list, the length of the list must match the number of sources, because the two lists will be zipped.
            If `sources` contains ByteStream objects, their `meta` will be added to the output Documents.

        :returns:
            A dictionary with the following keys:
            - `documents`: Created Documents
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
                pdf_reader = extract_pages(io.BytesIO(bytestream.data), laparams=self.layout_params)
                document = self.__converter(pdf_reader)
            except Exception as e:
                logger.warning(
                    "Could not read {source} and convert it to Document, skipping. {error}", source=source, error=e
                )
                continue

            merged_metadata = {**bytestream.meta, **metadata}
            document.meta = merged_metadata
            documents.append(document)

        return {"documents": documents}
