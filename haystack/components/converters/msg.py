# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import io
import os
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

    def __init__(self, store_full_path: bool = False) -> None:
        """
        Creates a MSGToDocument component.

        :param store_full_path:
            If True, the full path of the file is stored in the metadata of the document.
            If False, only the file name is stored.
        """
        oxmsg_import.check()
        self.store_full_path = store_full_path

    @staticmethod
    def _is_encrypted(msg: Message) -> bool:
        return "encrypted" in msg.message_headers.get("Content-Type", "")

    @staticmethod
    def _create_recipient_str(recip: recipient.Recipient) -> str:
        recip_str = ""
        if recip.name != "":
            recip_str += f"{recip.name} "
        if recip.email_address != "":
            recip_str += f"{recip.email_address}"
        return recip_str

    def _convert(self, file_content: io.BytesIO) -> str:
        """
        Converts the MSG file to markdown.
        """
        msg = Message.load(file_content)
        if self._is_encrypted(msg):
            raise ValueError("The MSG file is encrypted and cannot be read.")

        txt = ""

        # Sender
        if msg.sender is not None:
            txt += f"From: {msg.sender}\n"

        # To
        recipients_str = ",".join(self._create_recipient_str(r) for r in msg.recipients)
        if recipients_str != "":
            txt += f"To: {recipients_str}\n"

        # CC
        cc_header = None
        if msg.message_headers.get("Cc") is not None:
            cc_header = msg.message_headers.get("Cc")
        elif msg.message_headers.get("CC") is not None:
            cc_header = msg.message_headers.get("CC")

        if cc_header is not None:
            txt += f"Cc: {cc_header}\n"

        # BCC
        bcc_header = None
        if msg.message_headers.get("Bcc") is not None:
            bcc_header = msg.message_headers.get("Bcc")
        elif msg.message_headers.get("BCC") is not None:
            bcc_header = msg.message_headers.get("BCC")

        if bcc_header is not None:
            txt += f"Bcc: {bcc_header}\n"

        # Subject
        if msg.subject != "":
            txt += f"Subject: {msg.subject}\n"

        # Body
        if msg.body is not None:
            txt += "\n" + msg.body

        return txt

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

            if not self.store_full_path and "file_path" in bytestream.meta:
                file_path = bytestream.meta["file_path"]
                merged_metadata["file_path"] = os.path.basename(file_path)

            documents.append(Document(content=text, meta=merged_metadata))

        return {"documents": documents}
