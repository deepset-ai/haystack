import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from haystack import Document, component, logging
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream
from haystack.lazy_imports import LazyImport

with LazyImport("Run 'pip install python-oxmsg'") as oxmsg_import:
    from oxmsg import Message, recipient


logger = logging.getLogger(__name__)


@component
class MSGToDocument:
    """
    Converts .msg files to Documents.

    ### Usage example

    ```python
    from haystack.components.converters.msg import MSGToDocument

    converter = MSGToDocument()
    results = converter.run(sources=["sample.msg"], meta={"date_added": datetime.now().isoformat()})
    documents = results["documents"]
    print(documents[0].content)
    ```
    """

    def __init__(self) -> None:
        """
        Creates a ONMSGToDocument component.

        """
        oxmsg_import.check()

    @staticmethod
    def _create_recipient_str(recip: recipient.Recipient) -> str:
        return str(recip.name) + " " + str(recip.email_address)

    def _convert(self, file_content: io.BytesIO) -> str:
        """
        Converts the MSG file to markdown.
        """
        msg = Message.load(file_content)

        md = "# Email Headers\n\n"
        recipients_str = ",".join(MSGToDocument._create_recipient_str(r) for r in msg.recipients)
        md += f"From : {msg.sender}\nTo : {recipients_str}\nSubject : {msg.subject}\n"
        md += "\n# Body\n\n"
        md += msg.body
        return md

    @component.output_types(documents=List[Document])
    def run(
        self,
        sources: List[Union[str, Path, ByteStream]],
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ) -> Dict[str, List[Document]]:
        """
        Converts MSG files to Documents.

        :param sources:
            List of file paths or ByteStream objects.
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
        meta_list = normalize_metadata(meta, sources_count=len(sources))

        for source, metadata in zip(sources, meta_list):
            try:
                bytestream = get_bytestream_from_source(source)
            except Exception as e:
                logger.warning("Could not read {source}. Skipping it. Error: {error}", source=source, error=e)
                continue
            try:
                text = self._convert(io.BytesIO(bytestream.data))
            except Exception as e:
                logger.warning(
                    "Could not read {source} and convert it to Document, skipping. {error}", source=source, error=e
                )
                continue

            merged_metadata = {**bytestream.meta, **metadata}

            documents.append(Document(content=text, meta=merged_metadata))

        return {"documents": documents}
