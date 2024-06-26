# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import io
from dataclasses import dataclass
from datetime import datetime
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


@dataclass
class DOCXMetadata:
    """
    Describes the metadata of Docx file.

    :param author: The author
    :param category: The category
    :param comments: The comments
    :param content_status: The content status
    :param created: The creation date
    :param identifier: The identifier
    :param keywords: Available keywords
    :param language: The language of the document
    :param last_modified_by: The last modified by user date
    :param last_printed: The last printed date
    :param modified: The last modification date
    :param revision: The revision number
    :param subject: The subject
    :param title: The title
    :param version: The version
    """

    author: str
    category: str
    comments: str
    content_status: str
    created: Optional[datetime]
    identifier: str
    keywords: str
    language: str
    last_modified_by: str
    last_printed: Optional[datetime]
    modified: Optional[datetime]
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
                paragraphs = [para.text for para in file.paragraphs]
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
            created=document.core_properties.created,
            identifier=document.core_properties.identifier,
            keywords=document.core_properties.keywords,
            language=document.core_properties.language,
            last_modified_by=document.core_properties.last_modified_by,
            last_printed=document.core_properties.last_printed,
            modified=document.core_properties.modified,
            revision=document.core_properties.revision,
            subject=document.core_properties.subject,
            title=document.core_properties.title,
            version=document.core_properties.version,
        )
