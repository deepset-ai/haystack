# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from haystack import Document, component, logging
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream
from haystack.lazy_imports import LazyImport

logger = logging.getLogger(__name__)

with LazyImport("Run 'pip install python-docx'") as docx_import:
    import docx
    from docx.document import Document as DocxDocument
    from docx.text.paragraph import Paragraph


@dataclass
class DOCXMetadata:
    """
    Describes the metadata of Docx file.

    :param author: The author
    :param category: The category
    :param comments: The comments
    :param content_status: The content status
    :param created: The creation date (ISO formatted string)
    :param identifier: The identifier
    :param keywords: Available keywords
    :param language: The language of the document
    :param last_modified_by: User who last modified the document
    :param last_printed: The last printed date (ISO formatted string)
    :param modified: The last modification date (ISO formatted string)
    :param revision: The revision number
    :param subject: The subject
    :param title: The title
    :param version: The version
    """

    author: str
    category: str
    comments: str
    content_status: str
    created: Optional[str]
    identifier: str
    keywords: str
    language: str
    last_modified_by: str
    last_printed: Optional[str]
    modified: Optional[str]
    revision: int
    subject: str
    title: str
    version: str


@component
class DOCXToDocument:
    """
    Converts DOCX files to Documents.

    Uses `python-docx` library to convert the DOCX file to a document.
    This component does not preserve page breaks in the original document.

    Usage example:
    ```python
    from haystack.components.converters.docx import DOCXToDocument

    converter = DOCXToDocument()
    results = converter.run(sources=["sample.docx"], meta={"date_added": datetime.now().isoformat()})
    documents = results["documents"]
    print(documents[0].content)
    # 'This is a text from the DOCX file.'
    ```
    """

    def __init__(self):
        """
        Create a DOCXToDocument component.
        """
        docx_import.check()

    @component.output_types(documents=List[Document])
    def run(
        self,
        sources: List[Union[str, Path, ByteStream]],
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ):
        """
        Converts DOCX files to Documents.

        :param sources:
            List of file paths or ByteStream objects.
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
        meta_list = normalize_metadata(meta=meta, sources_count=len(sources))

        for source, metadata in zip(sources, meta_list):
            try:
                bytestream = get_bytestream_from_source(source)
            except Exception as e:
                logger.warning("Could not read {source}. Skipping it. Error: {error}", source=source, error=e)
                continue
            try:
                file = docx.Document(io.BytesIO(bytestream.data))
                paragraphs = self._extract_paragraphs_with_page_breaks(file.paragraphs)
                text = "\n".join(paragraphs)
            except Exception as e:
                logger.warning(
                    "Could not read {source} and convert it to a DOCX Document, skipping. Error: {error}",
                    source=source,
                    error=e,
                )
                continue

            docx_metadata = self._get_docx_metadata(document=file)
            merged_metadata = {**bytestream.meta, **metadata, "docx": docx_metadata}
            document = Document(content=text, meta=merged_metadata)
            documents.append(document)

        return {"documents": documents}

    def _extract_paragraphs_with_page_breaks(self, paragraphs: List["Paragraph"]) -> List[str]:
        """
        Extracts paragraphs from a DOCX file, including page breaks.

        Page breaks (both soft and hard page breaks) are not automatically extracted by python-docx as '\f' chars.
        This means we need to add them in ourselves, as done here. This allows the correct page number
        to be associated with each document if the file contents are split, e.g. by DocumentSplitter.

        :param paragraphs:
            List of paragraphs from a DOCX file.

        :returns:
            List of strings (paragraph text fields) with all page breaks added in as '\f' characters.
        """
        paragraph_texts = []
        for para in paragraphs:
            if para.contains_page_break:
                para_text_w_page_breaks = ""
                # Usually, just 1 page break exists, but could be more if paragraph is really long, so we loop over them
                for pb_index, page_break in enumerate(para.rendered_page_breaks):
                    # Can only extract text from first paragraph page break, unfortunately
                    if pb_index == 0:
                        if page_break.preceding_paragraph_fragment:
                            para_text_w_page_breaks += page_break.preceding_paragraph_fragment.text
                        para_text_w_page_breaks += "\f"
                        if page_break.following_paragraph_fragment:
                            # following_paragraph_fragment contains all text for remainder of paragraph.
                            # However, if the remainder of the paragraph spans multiple page breaks, it won't include
                            # those later page breaks so we have to add them at end of text in the `else` block below.
                            # This is not ideal, but this case should be very rare and this is likely good enough.
                            para_text_w_page_breaks += page_break.following_paragraph_fragment.text
                    else:
                        para_text_w_page_breaks += "\f"

                paragraph_texts.append(para_text_w_page_breaks)
            else:
                paragraph_texts.append(para.text)

        return paragraph_texts

    def _get_docx_metadata(self, document: "DocxDocument") -> DOCXMetadata:
        """
        Get all relevant data from the 'core_properties' attribute from a DOCX Document.

        :param document:
            The DOCX Document you want to extract metadata from

        :returns:
            A `DOCXMetadata` dataclass all the relevant fields from the 'core_properties'
        """
        return DOCXMetadata(
            author=document.core_properties.author,
            category=document.core_properties.category,
            comments=document.core_properties.comments,
            content_status=document.core_properties.content_status,
            created=document.core_properties.created.isoformat() if document.core_properties.created else None,
            identifier=document.core_properties.identifier,
            keywords=document.core_properties.keywords,
            language=document.core_properties.language,
            last_modified_by=document.core_properties.last_modified_by,
            last_printed=document.core_properties.last_printed.isoformat()
            if document.core_properties.last_printed
            else None,
            modified=document.core_properties.modified.isoformat() if document.core_properties.modified else None,
            revision=document.core_properties.revision,
            subject=document.core_properties.subject,
            title=document.core_properties.title,
            version=document.core_properties.version,
        )
