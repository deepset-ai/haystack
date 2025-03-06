# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import io
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from haystack import Document, component, logging
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream
from haystack.lazy_imports import LazyImport

with LazyImport("Run 'pip install pdfminer.six'") as pdfminer_import:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LAParams, LTTextContainer

logger = logging.getLogger(__name__)

CID_PATTERN = r"\(cid:\d+\)"  # regex pattern to detect CID characters


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

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        line_overlap: float = 0.5,
        char_margin: float = 2.0,
        line_margin: float = 0.5,
        word_margin: float = 0.1,
        boxes_flow: Optional[float] = 0.5,
        detect_vertical: bool = True,
        all_texts: bool = False,
        store_full_path: bool = False,
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
        :param store_full_path:
            If True, the full path of the file is stored in the metadata of the document.
            If False, only the file name is stored.
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
        self.store_full_path = store_full_path
        self.cid_pattern = re.compile(CID_PATTERN)

    @staticmethod
    def _converter(lt_page_objs: Iterator) -> str:
        """
        Extracts text from PDF pages then converts the text into a single str

        :param lt_page_objs:
            Python generator that yields PDF pages.

        :returns:
            PDF text converted to single str
        """
        pages = []
        for page in lt_page_objs:
            text = ""
            for container in page:
                # Keep text only
                if isinstance(container, LTTextContainer):
                    container_text = container.get_text()
                    if container_text:
                        text += "\n\n"
                    text += container_text
            pages.append(text)

        # Add a page delimiter
        delimited_pages = "\f".join(pages)

        return delimited_pages

    def detect_undecoded_cid_characters(self, text: str) -> Dict[str, Any]:
        """
        Look for character sequences of CID, i.e.: characters that haven't been properly decoded from their CID format.

        This is useful to detect if the text extractor is not able to extract the text correctly, e.g. if the PDF uses
        non-standard fonts.

        A PDF font may include a ToUnicode map (mapping from character code to Unicode) to support operations like
        searching strings or copy & paste in a PDF viewer. This map immediately provides the mapping the text extractor
        needs. If that map is not available the text extractor cannot decode the CID characters and will return them
        as is.

        see: https://pdfminersix.readthedocs.io/en/latest/faq.html#why-are-there-cid-x-values-in-the-textual-output

        :param: text: The text to check for undecoded CID characters
        :returns:
            A dictionary containing detection results
        """

        matches = re.findall(self.cid_pattern, text)
        total_chars = len(text)
        cid_chars = sum(len(match) for match in matches)
        percentage = (cid_chars / total_chars * 100) if total_chars > 0 else 0

        return {"total_chars": total_chars, "cid_chars": cid_chars, "percentage": round(percentage, 2)}

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
            If it's a list, the length of the list must match the number of sources, because the two lists will
            be zipped.
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
                pages = extract_pages(io.BytesIO(bytestream.data), laparams=self.layout_params)
                text = self._converter(pages)
            except Exception as e:
                logger.warning(
                    "Could not read {source} and convert it to Document, skipping. {error}", source=source, error=e
                )
                continue

            if text is None or text.strip() == "":
                logger.warning(
                    "PDFMinerToDocument could not extract text from the file {source}. Returning an empty document.",
                    source=source,
                )

            merged_metadata = {**bytestream.meta, **metadata}

            if not self.store_full_path and (file_path := bytestream.meta.get("file_path")):
                merged_metadata["file_path"] = os.path.basename(file_path)

            analysis = self.detect_undecoded_cid_characters(text)

            if analysis["percentage"] > 0:
                logger.warning(
                    "Detected {cid_chars} undecoded CID characters in {total_chars} characters"
                    " ({percentage}%) in {source}.",
                    cid_chars=analysis["cid_chars"],
                    total_chars=analysis["total_chars"],
                    percentage=analysis["percentage"],
                    source=source,
                )

            document = Document(content=text, meta=merged_metadata)
            documents.append(document)

        return {"documents": documents}
