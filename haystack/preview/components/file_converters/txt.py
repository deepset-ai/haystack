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

        :param encoding: The default encoding of the text files. Default: `"utf-8"`.
            Note that if the encoding is specified in the metadata of a ByteStream,
            it will override this default.
        """
        self.encoding = encoding

    @component.output_types(documents=List[Document])
    def run(self, streams: List[Union[str, Path, ByteStream]]):
        """
        Convert text files to Documents.

        :param streams: A list of paths to text files or ByteStream objects.
            Note that if an encoding is specified in the metadata of a ByteStream,
            it will override the component's default.
        :return: A dictionary containing the converted documents.
        """
        documents = []
        for stream in streams:
            if isinstance(stream, (Path, str)):
                if not Path(stream).exists():
                    logger.warning("File %s does not exist. Skipping it.", stream)
                    continue
                try:
                    path = stream
                    stream = ByteStream.from_file_path(Path(stream))
                    stream.metadata["file_path"] = str(path)
                except Exception as e:
                    logger.warning("Could not convert file %s. Skipping it. Error message: %s", stream, e)
                    continue
            try:
                encoding = stream.metadata.get("encoding", self.encoding)
                document = Document(content=stream.data.decode(encoding))
                document.meta = stream.metadata
                documents.append(document)
            except Exception as e:
                logger.warning("Could not convert file %s. Skipping it. Error message: %s", stream, e)

        return {"documents": documents}
