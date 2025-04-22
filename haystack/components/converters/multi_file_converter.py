# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

from haystack import Pipeline, super_component
from haystack.components.converters import (
    CSVToDocument,
    DOCXToDocument,
    HTMLToDocument,
    JSONConverter,
    PPTXToDocument,
    PyPDFToDocument,
    TextFileToDocument,
    XLSXToDocument,
)
from haystack.components.joiners import DocumentJoiner
from haystack.components.routers import FileTypeRouter


class ConverterMimeType(str, Enum):
    CSV = "text/csv"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    HTML = "text/html"
    JSON = "application/json"
    MD = "text/markdown"
    TEXT = "text/plain"
    PDF = "application/pdf"
    PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


@super_component
class MultiFileConverter:
    """
    A file converter that handles conversion of multiple file types.

    The MultiFileConverter handles the following file types:
    - CSV
    - DOCX
    - HTML
    - JSON
    - MD
    - TEXT
    - PDF (no OCR)
    - PPTX
    - XLSX

    Usage example:
    ```
    from haystack.super_components.converters import MultiFileConverter

    converter = MultiFileConverter()
    converter.run(sources=["test.txt", "test.pdf"], meta={})
    ```
    """

    def __init__(self, encoding: str = "utf-8", json_content_key: str = "content") -> None:
        """
        Initialize the MultiFileConverter.

        :param encoding: The encoding to use when reading files.
        :param json_content_key: The key to use in a content field in a document when converting JSON files.
        """
        self.encoding = encoding
        self.json_content_key = json_content_key

        # initialize components
        router = FileTypeRouter(
            mime_types=[mime_type.value for mime_type in ConverterMimeType],
            # Ensure common extensions are registered. Tests on Windows fail otherwise.
            additional_mimetypes={
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
                "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
            },
        )

        # Create pipeline and add components
        pp = Pipeline()

        # We use type ignore here to avoid type checking errors
        # This is due to how the run method within the Component protocol is defined
        pp.add_component("router", router)  # type: ignore[arg-type]
        pp.add_component("docx", DOCXToDocument(link_format="markdown"))  # type: ignore[arg-type]
        pp.add_component(
            "html",
            HTMLToDocument(  # type: ignore[arg-type]
                extraction_kwargs={"output_format": "markdown", "include_tables": True, "include_links": True}
            ),
        )
        pp.add_component("json", JSONConverter(content_key=self.json_content_key))  # type: ignore[arg-type]
        pp.add_component("md", TextFileToDocument(encoding=self.encoding))  # type: ignore[arg-type]
        pp.add_component("text", TextFileToDocument(encoding=self.encoding))  # type: ignore[arg-type]
        pp.add_component("pdf", PyPDFToDocument())  # type: ignore[arg-type]
        pp.add_component("pptx", PPTXToDocument())  # type: ignore[arg-type]
        pp.add_component("xlsx", XLSXToDocument())  # type: ignore[arg-type]
        pp.add_component("joiner", DocumentJoiner())  # type: ignore[arg-type]
        pp.add_component("csv", CSVToDocument(encoding=self.encoding))  # type: ignore[arg-type]

        for mime_type in ConverterMimeType:
            pp.connect(f"router.{mime_type.value}", str(mime_type).lower().rsplit(".", maxsplit=1)[-1])

        pp.connect("docx.documents", "joiner.documents")
        pp.connect("html.documents", "joiner.documents")
        pp.connect("json.documents", "joiner.documents")
        pp.connect("md.documents", "joiner.documents")
        pp.connect("text.documents", "joiner.documents")
        pp.connect("pdf.documents", "joiner.documents")
        pp.connect("pptx.documents", "joiner.documents")

        pp.connect("csv.documents", "joiner.documents")
        pp.connect("xlsx.documents", "joiner.documents")

        self.pipeline = pp
        self.output_mapping = {"joiner.documents": "documents", "router.unclassified": "unclassified"}
        self.input_mapping = {"sources": ["router.sources"], "meta": ["router.meta"]}
