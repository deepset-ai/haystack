# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Any, Dict

from haystack import Pipeline, component, default_from_dict, default_to_dict
from haystack.components.converters import (
    CSVToDocument,
    DOCXToDocument,
    HTMLToDocument,
    JSONConverter,
    MarkdownToDocument,
    PPTXToDocument,
    PyPDFToDocument,
    TextFileToDocument,
    XLSXToDocument,
)
from haystack.components.joiners import DocumentJoiner
from haystack.components.routers import FileTypeRouter
from haystack.core.super_component import SuperComponent


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


@component
class MultiFileConverter(SuperComponent):
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

    Usage:
    ```
    from haystack.super_components.converters import MultiFileConverter

    converter = MultiFileConverter()
    converter.run(sources=["test.txt", "test.pdf"], meta={})
    ```
    """

    def __init__(  # noqa: PLR0915
        self, encoding: str = "utf-8", json_content_key: str = "content"
    ) -> None:
        """
        Initialize the MultiFileConverter.

        :param encoding: The encoding to use when reading files.
        :param json_content_key: The key to use as content-field in a document when converting json-files.
        """
        self.encoding = encoding
        self.json_content_key = json_content_key

        # initialize components
        router = FileTypeRouter(
            mime_types=[
                ConverterMimeType.CSV.value,
                ConverterMimeType.DOCX.value,
                ConverterMimeType.HTML.value,
                ConverterMimeType.JSON.value,
                ConverterMimeType.MD.value,
                ConverterMimeType.TEXT.value,
                ConverterMimeType.PDF.value,
                ConverterMimeType.PPTX.value,
                ConverterMimeType.XLSX.value,
            ],
            # Ensure common extensions are registered. Tests on Windows fail otherwise.
            additional_mimetypes={
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
                "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
            },
        )

        csv = CSVToDocument(encoding=self.encoding)
        docx = DOCXToDocument()
        html = HTMLToDocument()
        json = JSONConverter(content_key=self.json_content_key)
        md = MarkdownToDocument()
        txt = TextFileToDocument(encoding=self.encoding)
        pdf = PyPDFToDocument()
        pptx = PPTXToDocument()
        xlsx = XLSXToDocument()

        joiner = DocumentJoiner()

        # Create pipeline and add components
        pp = Pipeline()

        pp.add_component("router", router)

        pp.add_component("docx", docx)
        pp.add_component("html", html)
        pp.add_component("json", json)
        pp.add_component("md", md)
        pp.add_component("txt", txt)
        pp.add_component("pdf", pdf)
        pp.add_component("pptx", pptx)
        pp.add_component("xlsx", xlsx)
        pp.add_component("joiner", joiner)
        pp.add_component("csv", csv)

        pp.connect(f"router.{ConverterMimeType.CSV.value}", "csv")
        pp.connect(f"router.{ConverterMimeType.DOCX.value}", "docx")
        pp.connect(f"router.{ConverterMimeType.HTML.value}", "html")
        pp.connect(f"router.{ConverterMimeType.JSON.value}", "json")
        pp.connect(f"router.{ConverterMimeType.MD.value}", "md")
        pp.connect(f"router.{ConverterMimeType.TEXT.value}", "txt")
        pp.connect(f"router.{ConverterMimeType.PDF.value}", "pdf")
        pp.connect(f"router.{ConverterMimeType.PPTX.value}", "pptx")
        pp.connect(f"router.{ConverterMimeType.XLSX.value}", "xlsx")

        pp.connect("docx.documents", "joiner.documents")
        pp.connect("html.documents", "joiner.documents")
        pp.connect("json.documents", "joiner.documents")
        pp.connect("md.documents", "joiner.documents")
        pp.connect("txt.documents", "joiner.documents")
        pp.connect("pdf.documents", "joiner.documents")
        pp.connect("pptx.documents", "joiner.documents")

        pp.connect("csv.documents", "joiner.documents")
        pp.connect("xlsx.documents", "joiner.documents")

        output_mapping = {"joiner.documents": "documents", "router.unclassified": "unclassified"}
        input_mapping = {"sources": ["router.sources"], "meta": ["router.meta"]}

        super(MultiFileConverter, self).__init__(
            pipeline=pp, output_mapping=output_mapping, input_mapping=input_mapping
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this instance to a dictionary.
        """
        return default_to_dict(self, encoding=self.encoding, json_content_key=self.json_content_key)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiFileConverter":
        """
        Load this instance from a dictionary.
        """
        return default_from_dict(cls, data)
