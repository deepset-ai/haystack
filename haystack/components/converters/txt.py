import logging
from pathlib import Path
from typing import List, Union

from haystack import Document, component
from haystack.dataclasses import ByteStream
from haystack.components.converters.utils import get_bytestream_from_source


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
    def run(self, sources: List[Union[str, Path, ByteStream]]):
        """
        Convert text files to Documents.

        :param streams: A list of paths to text files or ByteStream objects.
            Note that if an encoding is specified in the metadata of a ByteStream,
            it will override the component's default.
        :return: A dictionary containing the converted documents.
        """
        documents = []
        for source in sources:
            try:
                bytestream = get_bytestream_from_source(source)
            except Exception as e:
                logger.warning("Could not read %s. Skipping it. Error: %s", source, e)
                continue
            try:
                encoding = bytestream.metadata.get("encoding", self.encoding)
                document = Document(content=bytestream.data.decode(encoding))
                document.meta = bytestream.metadata
                documents.append(document)
            except Exception as e:
                source_str = str(source) if len(str(source)) < 100 else str(source)[:100] + "..."
                logger.warning("Could not convert file %s. Skipping it. Error message: %s", source_str, e)

        return {"documents": documents}
    
    def _get_bytestream_from_source(self, source: Union[str, Path, ByteStream]) -> ByteStream:
        if isinstance(source, ByteStream):
            return source
        if isinstance(source, (str, Path)):
            bs = ByteStream.from_file_path(Path(source))
            bs.metadata["file_path"] = str(source)
            return bs
        raise ValueError(f"Unsupported source type {type(source)}")
