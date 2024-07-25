# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import io
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from haystack import Document, component, logging
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream
from haystack.lazy_imports import LazyImport

with LazyImport("Run 'pip install tika'") as tika_import:
    from tika import parser as tika_parser

logger = logging.getLogger(__name__)


class TikaXHTMLParser(HTMLParser):
    # Use the built-in HTML parser with minimum dependencies
    def __init__(self):
        tika_import.check()
        self.ingest = True
        self.page = ""
        self.pages: List[str] = []
        super(TikaXHTMLParser, self).__init__()

    def handle_starttag(self, tag, attrs):
        """Handle Start Tag"""
        # find page div
        pagediv = [value for attr, value in attrs if attr == "class" and value == "page"]
        if tag == "div" and pagediv:
            self.ingest = True

    def handle_endtag(self, tag):
        """Handle End Tag"""
        # close page div, or a single page without page div, save page and open a new page
        if (tag == "div" or tag == "body") and self.ingest:
            self.ingest = False
            # restore words hyphened to the next line
            self.pages.append(self.page.replace("-\n", ""))
            self.page = ""

    def handle_data(self, data):
        """Handle Data"""
        if self.ingest:
            self.page += data


@component
class TikaDocumentConverter:
    """
    Converts files of different types to Documents.

    This component uses [Apache Tika](https://tika.apache.org/) for parsing the files and, therefore,
    requires a running Tika server.
    For more options on running Tika,
    see the [official documentation](https://github.com/apache/tika-docker/blob/main/README.md#usage).

    Usage example:
    ```python
    from haystack.components.converters.tika import TikaDocumentConverter

    converter = TikaDocumentConverter()
    results = converter.run(
        sources=["sample.docx", "my_document.rtf", "archive.zip"],
        meta={"date_added": datetime.now().isoformat()}
    )
    documents = results["documents"]
    print(documents[0].content)
    # 'This is a text from the docx file.'
    ```
    """

    def __init__(self, tika_url: str = "http://localhost:9998/tika"):
        """
        Create a TikaDocumentConverter component.

        :param tika_url:
            Tika server URL.
        """
        tika_import.check()
        self.tika_url = tika_url

    @component.output_types(documents=List[Document])
    def run(
        self,
        sources: List[Union[str, Path, ByteStream]],
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ):
        """
        Converts files to Documents.

        :param sources:
            List of HTML file paths or ByteStream objects.
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
                parsed = tika_parser.from_buffer(
                    io.BytesIO(bytestream.data), serverEndpoint=self.tika_url, xmlContent=True
                )
                parser = TikaXHTMLParser()
                parser.feed(parsed["content"])
            except Exception as conversion_e:
                logger.warning(
                    "Failed to extract text from {source}. Skipping it. Error: {error}",
                    source=source,
                    error=conversion_e,
                )
                continue

            # Old Processing Code from Haystack 1.X Tika integration
            cleaned_pages = []
            # TODO investigate title of document appearing in the first extracted page
            for page in parser.pages:
                lines = page.splitlines()
                cleaned_lines = list(lines)

                page = "\n".join(cleaned_lines)
                cleaned_pages.append(page)
            text = "\f".join(cleaned_pages)
            merged_metadata = {**bytestream.meta, **metadata}
            document = Document(content=text, meta=merged_metadata)
            documents.append(document)
        return {"documents": documents}
