# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import io
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.converters.utils import LinkFormat, get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream
from haystack.lazy_imports import LazyImport

with LazyImport("Run 'pip install pdfminer.six'") as pdfminer_import:
    from pdfminer.converter import PDFPageAggregator
    from pdfminer.layout import LAParams, LTTextContainer
    from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
    from pdfminer.pdfpage import PDFPage
    from pdfminer.pdftypes import resolve1

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
    from datetime import datetime

    converter = PDFMinerToDocument()
    results = converter.run(
        sources=["test/test_files/pdf/sample_pdf_1.pdf"], meta={"date_added": datetime.now().isoformat()}
    )

    print(results["documents"][0].content)
    # >> 'This is a text from the PDF file.'
    ```
    """

    def __init__(
        self,
        line_overlap: float = 0.5,
        char_margin: float = 2.0,
        line_margin: float = 0.5,
        word_margin: float = 0.1,
        boxes_flow: float | None = 0.5,
        detect_vertical: bool = True,
        all_texts: bool = False,
        store_full_path: bool = False,
        link_format: str | LinkFormat = LinkFormat.NONE,
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
        :param link_format:
            The format used for the hyperlinks found in the PDF link annotations.
            The links of a page are appended at the end of that page's text, one per line. PDF link annotations
            carry no anchor text, so the address is used as the link text as well. Can be either:
            `LinkFormat.MARKDOWN` or `"markdown"` to get `[address](address)`,
            `LinkFormat.PLAIN` or `"plain"` to get `address (address)`,
            `LinkFormat.NONE` or `"none"` to get text without links.
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
        self.link_format = LinkFormat.from_str(link_format) if isinstance(link_format, str) else link_format
        self.cid_pattern = re.compile(CID_PATTERN)

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            line_overlap=self.layout_params.line_overlap,
            char_margin=self.layout_params.char_margin,
            line_margin=self.layout_params.line_margin,
            word_margin=self.layout_params.word_margin,
            boxes_flow=self.layout_params.boxes_flow,
            detect_vertical=self.layout_params.detect_vertical,
            all_texts=self.layout_params.all_texts,
            store_full_path=self.store_full_path,
            link_format=str(self.link_format),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PDFMinerToDocument":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary with serialized data.

        :returns:
            Deserialized component.
        """
        if "link_format" in data.get("init_parameters", {}):
            data["init_parameters"]["link_format"] = LinkFormat.from_str(data["init_parameters"]["link_format"])
        return default_from_dict(cls, data)

    def _iter_pages(self, fp: io.BytesIO) -> Iterator[tuple[Any, Any]]:
        """
        Lazily yields the pages of a PDF as `(LTPage, PDFPage)` pairs.

        The pages are processed one at a time so that only a single page layout is kept in memory.

        :param fp:
            The PDF file to read.

        :returns:
            An iterator over `(page layout object, page object)` pairs.
        """
        rsrcmgr = PDFResourceManager(caching=True)
        device = PDFPageAggregator(rsrcmgr, laparams=self.layout_params)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for pdf_page in PDFPage.get_pages(fp, caching=True):
            interpreter.process_page(pdf_page)
            yield device.get_result(), pdf_page

    def _convert_page(self, lt_page: Any, pdf_page: Any) -> str:
        """
        Extracts text from a single PDF page.

        :param lt_page:
            PDF page layout object (LTPage).
        :param pdf_page:
            PDF page (PDFPage).

        :returns:
            PDF text from the single page converted to str
        """
        text = ""
        for container in lt_page:
            # Keep text only
            if isinstance(container, LTTextContainer):
                container_text = container.get_text()
                if container_text:
                    text += "\n\n"
                text += container_text

        if self.link_format != LinkFormat.NONE and getattr(pdf_page, "annots", None):
            page_links = []
            annots = resolve1(pdf_page.annots)
            if annots:
                for annot in annots:
                    try:
                        annot_obj = resolve1(annot)
                        if (
                            isinstance(annot_obj, dict)
                            and annot_obj.get("Subtype")
                            and resolve1(annot_obj.get("Subtype")).name == "Link"
                        ):
                            a = annot_obj.get("A")
                            if a:
                                a_obj = resolve1(a)
                                if (
                                    isinstance(a_obj, dict)
                                    and a_obj.get("S")
                                    and resolve1(a_obj.get("S")).name == "URI"
                                ):
                                    uri = a_obj.get("URI")
                                    if uri:
                                        # Decode bytes if needed (pdfminer sometimes returns bytes for strings)
                                        uri_str = uri.decode("utf-8") if isinstance(uri, bytes) else str(uri)
                                        if self.link_format == LinkFormat.MARKDOWN:
                                            page_links.append(f"[{uri_str}]({uri_str})")
                                        else:  # PLAIN
                                            page_links.append(f"{uri_str} ({uri_str})")
                    except Exception:
                        logger.debug("Skipping malformed annotation")
                        continue
            if page_links:
                text += "\n\n" + "\n".join(page_links)

        return text

    def detect_undecoded_cid_characters(self, text: str) -> dict[str, Any]:
        """
        Look for character sequences of CID, i.e.: characters that haven't been properly decoded from their CID format.

        This is useful to detect if the text extractor is not able to extract the text correctly, e.g. if the PDF uses
        non-standard fonts.

        A PDF font may include a ToUnicode map (mapping from character code to Unicode) to support operations like
        searching strings or copy & paste in a PDF viewer. This map immediately provides the mapping the text extractor
        needs. If that map is not available the text extractor cannot decode the CID characters and will return them
        as is.

        see: https://pdfminersix.readthedocs.io/en/latest/faq.html#why-are-there-cid-x-values-in-the-textual-output

        :param text: The text to check for undecoded CID characters
        :returns:
            A dictionary containing detection results
        """

        matches = re.findall(self.cid_pattern, text)
        total_chars = len(text)
        cid_chars = sum(len(match) for match in matches)
        percentage = (cid_chars / total_chars * 100) if total_chars > 0 else 0

        return {"total_chars": total_chars, "cid_chars": cid_chars, "percentage": round(percentage, 2)}

    @component.output_types(documents=list[Document])
    def run(
        self, sources: list[str | Path | ByteStream], meta: dict[str, Any] | list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
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

        for source, metadata in zip(sources, meta_list, strict=True):
            try:
                bytestream = get_bytestream_from_source(source)
            except Exception as e:
                logger.warning("Could not read {source}. Skipping it. Error: {error}", source=source, error=e)
                continue
            try:
                fp = io.BytesIO(bytestream.data)
                text = "\f".join(self._convert_page(lt_page, pdf_page) for lt_page, pdf_page in self._iter_pages(fp))
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
