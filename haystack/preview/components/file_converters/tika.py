import logging
from pathlib import Path
from typing import List, Union

from haystack.preview.lazy_imports import LazyImport
from haystack.preview import component, Document


with LazyImport("Run 'pip install tika'") as tika_import:
    from tika import parser as tika_parser

logger = logging.getLogger(__name__)


@component
class TikaDocumentConverter:
    """
    A component for converting files of different types (pdf, docx, html, etc.) to Documents.
    This component uses [Apache Tika](https://tika.apache.org/) for parsing the files and, therefore,
    requires a running Tika server.
    """

    def __init__(self, tika_url: str = "http://localhost:9998/tika"):
        """
        Create a TikaDocumentConverter component.

        :param tika_url: URL of the Tika server. Default: `"http://localhost:9998/tika"`
        """
        tika_import.check()
        self.tika_url = tika_url

    @component.output_types(documents=List[Document])
    def run(self, paths: List[Union[str, Path]]):
        """
        Convert files to Documents.

        :param paths: A list of paths to the files to convert.
        """

        documents = []
        for path in paths:
            path = Path(path)
            try:
                parsed_file = tika_parser.from_file(path.as_posix(), self.tika_url)
                extracted_text = parsed_file["content"]
                if not extracted_text:
                    logger.warning("Skipping file at '%s' as Tika was not able to extract any content.", str(path))
                    continue
                document = Document(content=extracted_text)
                documents.append(document)
            except Exception as e:
                logger.error("Could not convert file at '%s' to Document. Error: %s", str(path), e)

        return {"documents": documents}
