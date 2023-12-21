import logging
from pathlib import Path
from typing import List, Union, Dict, Any, Optional

from haystack import Document, component
from haystack.dataclasses import ByteStream
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata


logger = logging.getLogger(__name__)


@component
class TextFileToDocument:
    """
    A component for converting a text file to a Document.

    Usage example:
    ```python
    from haystack.components.converters.txt import TextFileToDocument

    converter = TextFileToDocument()
    results = converter.run(sources=["sample.txt"])
    documents = results["documents"]
    print(documents[0].content)
    # 'This is the content from the txt file.'
    ```
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
    def run(
        self,
        sources: List[Union[str, Path, ByteStream]],
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ):
        """
        Convert text files to Documents.

        :param sources: A list of paths to text files or ByteStream objects.
          Note that if an encoding is specified in the metadata of a ByteStream,
          it will override the component's default.
        :param meta: Optional metadata to attach to the Documents.
          This value can be either a list of dictionaries or a single dictionary.
          If it's a single dictionary, its content is added to the metadata of all produced Documents.
          If it's a list, the length of the list must match the number of sources, because the two lists will be zipped.
          Defaults to `None`.
        :return: A dictionary containing a list of Document objects under the 'documents' key.
        """
        documents = []

        meta_list = normalize_metadata(meta, sources_count=len(sources))

        for source, metadata in zip(sources, meta_list):
            try:
                bytestream = get_bytestream_from_source(source)
            except Exception as e:
                logger.warning("Could not read %s. Skipping it. Error: %s", source, e)
                continue
            try:
                encoding = bytestream.meta.get("encoding", self.encoding)
                text = bytestream.data.decode(encoding)
            except Exception as e:
                logger.warning("Could not convert file %s. Skipping it. Error message: %s", source, e)
                continue

            merged_metadata = {**bytestream.meta, **metadata}
            document = Document(content=text, meta=merged_metadata)
            documents.append(document)

        return {"documents": documents}
