from pathlib import Path
from typing import List, Optional, Union

from haystack import component, logging
from haystack.components.converters.utils import get_bytestream_from_source
from haystack.dataclasses.document import Document

logger = logging.getLogger(__name__)


@component
class ImageToDocument:
    """
    Converts an image to a document.

    In its current form, this is just a placeholder. I imagine it to incorporate some/all image processing utilities.
    """

    @component.output_types(documents=List[Document])
    def run(self, sources: List[Union[str, Path]], texts: Optional[List[str]] = None):
        """
        Converts an image to a document.
        """
        documents = []

        if texts and len(texts) != len(sources):
            raise ValueError("The length of the texts list must match the number of sources.")
        if not texts:
            texts = [None] * len(sources)

        for source, text in zip(sources, texts):
            try:
                bytestream = get_bytestream_from_source(source)

                # maybe we should check if this is an image

                # we should guess the mime type

                # we should also store the media type (in Document? In ByteStream?)
                bytestream.meta["media_type"] = "image"
            except Exception as e:
                logger.warning("Could not read {source}. Skipping it. Error: {error}", source=source, error=e)
                continue
            document = Document(content=text, blob=bytestream)
            documents.append(document)

        return {"documents": documents}
