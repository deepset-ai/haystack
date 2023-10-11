import io
import logging
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

from haystack.preview.dataclasses import ByteStream
from haystack.preview.lazy_imports import LazyImport
from haystack.preview import Document, component, default_to_dict, default_from_dict

with LazyImport("Run 'pip install pypdf'") as pypdf_import:
    from pypdf import PdfReader


logger = logging.getLogger(__name__)


@component
class PyPDFToDocument:
    """
    Converts a PDF file to a Document.
    """

    def __init__(self, id_hash_keys: Optional[List[str]] = None):
        """
        Initializes the PyPDFToDocument component.

        :param id_hash_keys: Generate the Document ID from a custom list of strings that refer to the Document's
            attributes. Default: `None`
        """
        pypdf_import.check()
        self.id_hash_keys = id_hash_keys or []

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        :return: The dictionary containing the component's data.
        """
        return default_to_dict(self, id_hash_keys=self.id_hash_keys)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PyPDFToDocument":
        """
        Deserialize this component from a dictionary.
        :param data: The dictionary containing the component's data.
        :return: The component instance.
        """
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(self, sources: List[Union[str, Path, ByteStream]], id_hash_keys: Optional[List[str]] = None):
        """
        Converts PDF files to Documents.

        :param sources: A list of PDF data sources
        :param id_hash_keys: Generate the Document ID from a custom list of strings that refer to the Document's
            attributes. Default: `None`
        """
        id_hash_keys = id_hash_keys or self.id_hash_keys
        documents = []
        for source in sources:
            try:
                text = self._read_pdf_file(source)
            except Exception as e:
                logger.warning("Could not read %s. Skipping it. Error message: %s", source, e)
                continue

            document = Document(text=text, id_hash_keys=id_hash_keys)
            documents.append(document)

        return {"documents": documents}

    def _read_pdf_file(self, source: Union[str, Path, ByteStream]) -> str:
        """
        Extracts content from the given PDF source.
        :param source:  PDF file data source
        :return: The extracted text.
        """
        if isinstance(source, (str, Path)):
            pdf_reader = PdfReader(str(source))
        elif isinstance(source, ByteStream):
            pdf_reader = PdfReader(io.BytesIO(source.data))
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

        text = "".join(extracted_text for page in pdf_reader.pages if (extracted_text := page.extract_text()))

        return text
