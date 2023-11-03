import logging
from pathlib import Path
from typing import List, Union

from haystack.preview import Document, component
from haystack.preview.dataclasses import ByteStream


logger = logging.getLogger(__name__)


@component
class TextFileToDocument:
    """
    A component for converting a text file to a Document.
    """

    def __init__(self, encoding: str = "utf-8"):
        """
        Create a TextFileToDocument component.

        :param encoding: The encoding of the text files. Default: `"utf-8"`
        """
        self.encoding = encoding

    @component.output_types(documents=List[Document])
    def run(self, streams: List[Union[str, Path, ByteStream]]):
        """
        Convert text files to Documents.

        :param streams: A list of paths to text files or ByteStream representing such files.
        """
        documents = []
        for stream in streams:
            if isinstance(stream, (Path, str)):
                if not Path(stream).exists():
                    logger.warning("File %s does not exist. Skipping it.", stream)
                    continue
                try:
                    stream = ByteStream.from_file_path(stream)
                    stream.metadata["file_path"] = str(stream)
                except Exception as e:
                    logger.warning("Could not read file %s. Skipping it. Error message: %s", stream, e)
                    continue

            document = Document(content=stream.data)
            document.meta = stream.metadata
            documents.append(document)

        return {"documents": documents}
