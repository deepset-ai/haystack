# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import io
import os
from pathlib import Path
from typing import Any

from haystack import Document, component, logging
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream
from haystack.lazy_imports import LazyImport

with LazyImport("Run 'pip install python-oxmsg'") as oxmsg_import:
    from oxmsg import Message, recipient
    from oxmsg.domain.constants import PID_RECIPIENT_TYPE


logger = logging.getLogger(__name__)

_MAPI_TO = 1
_MAPI_CC = 2
_MAPI_BCC = 3
_MAPI_RESEND_FLAGS = 0x10000000 | 0x80000000


@component
class MSGToDocument:
    """
    Converts Microsoft Outlook .msg files into Haystack Documents.

    This component extracts email metadata (such as sender, recipients, CC, BCC, subject) and body content from .msg
    files and converts them into structured Haystack Documents. Additionally, any file attachments within the .msg
    file are extracted as ByteStream objects.

    To recipients are listed separately from Cc and Bcc recipients. Cc and Bcc values come from transport headers
    when available, otherwise from the corresponding entries in the recipient table. Header values replace rather
    than supplement the corresponding recipient-table values. Originator entries and recipients with missing or
    unsupported types are omitted from the recipient lines.

    ### Example Usage

    ```python
    from haystack.components.converters.msg import MSGToDocument
    from datetime import datetime

    converter = MSGToDocument()
    results = converter.run(sources=["test/test_files/msg/sample.msg"], meta={"date_added": datetime.now().isoformat()})
    documents = results["documents"]
    attachments = results["attachments"]
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
    def _is_encrypted(msg: "Message") -> bool:
        """
        Determines whether the provided MSG file is encrypted.

        :param msg: The MSG file as a parsed Message object.
        :returns: True if the MSG file is encrypted, otherwise False.
        """
        return "encrypted" in msg.message_headers.get("Content-Type", "")

    @staticmethod
    def _create_recipient_str(recip: "recipient.Recipient") -> str:
        """
        Formats a recipient's name and email into a single string.

        :param recip: A recipient object extracted from the MSG file.
        :returns: A formatted string combining the recipient's name and email address.
        """
        recip_str = ""
        if recip.name != "":
            recip_str += f"{recip.name} "
        if recip.email_address != "":
            recip_str += f"{recip.email_address}"
        return recip_str

    def _convert(self, file_content: io.BytesIO) -> tuple[str, list[ByteStream]]:
        """
        Converts the MSG file content into text and extracts any attachments.

        :param file_content: The MSG file content as a binary stream.
        :returns: A tuple containing the extracted email text and a list of ByteStream objects for attachments.
        :raises ValueError: If the MSG file is encrypted and cannot be read.
        """
        msg = Message.load(file_content)
        if self._is_encrypted(msg):
            raise ValueError("The MSG file is encrypted and cannot be read.")

        txt = ""

        # Sender
        if msg.sender is not None:
            txt += f"From: {msg.sender}\n"

        recipients_by_type: dict[int, list[str]] = {_MAPI_TO: [], _MAPI_CC: [], _MAPI_BCC: []}
        for recip in msg.recipients:
            recipient_type = recip.properties.int_prop_value(PID_RECIPIENT_TYPE)
            if recipient_type is None:
                continue
            # Resend flags affect delivery, not the recipient's To/Cc/Bcc classification.
            recipient_type &= ~_MAPI_RESEND_FLAGS
            if recipient_type in recipients_by_type:
                recipients_by_type[recipient_type].append(self._create_recipient_str(recip))

        # To
        recipients_str = ",".join(recipients_by_type[_MAPI_TO])
        if recipients_str != "":
            txt += f"To: {recipients_str}\n"

        # CC
        cc_header = msg.message_headers.get("Cc") or msg.message_headers.get("CC")
        if cc_header is None and recipients_by_type[_MAPI_CC]:
            cc_header = ",".join(recipients_by_type[_MAPI_CC])
        if cc_header is not None:
            txt += f"Cc: {cc_header}\n"

        # BCC
        bcc_header = msg.message_headers.get("Bcc") or msg.message_headers.get("BCC")
        if bcc_header is None and recipients_by_type[_MAPI_BCC]:
            bcc_header = ",".join(recipients_by_type[_MAPI_BCC])
        if bcc_header is not None:
            txt += f"Bcc: {bcc_header}\n"

        # Subject
        if msg.subject != "":
            txt += f"Subject: {msg.subject}\n"

        # Body
        if msg.body is not None:
            txt += "\n" + msg.body

        # attachments
        attachments = [
            ByteStream(
                data=attachment.file_bytes, meta={"file_path": attachment.file_name}, mime_type=attachment.mime_type
            )
            for attachment in msg.attachments
            if attachment.file_bytes is not None
        ]

        return txt, attachments

    @component.output_types(documents=list[Document], attachments=list[ByteStream])
    def run(
        self, sources: list[str | Path | ByteStream], meta: dict[str, Any] | list[dict[str, Any]] | None = None
    ) -> dict[str, list[Document] | list[ByteStream]]:
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
            - `documents`: Created Documents.
            - `attachments`: Created ByteStream objects from file attachments.
        """
        if len(sources) == 0:
            return {"documents": [], "attachments": []}

        documents = []
        all_attachments = []
        meta_list = normalize_metadata(meta, sources_count=len(sources))

        for source, metadata in zip(sources, meta_list, strict=True):
            try:
                bytestream = get_bytestream_from_source(source)
            except Exception as e:
                logger.warning("Could not read {source}. Skipping it. Error: {error}", source=source, error=e)
                continue
            try:
                text, attachments = self._convert(io.BytesIO(bytestream.data))
            except Exception as e:
                logger.warning(
                    "Could not read {source} and convert it to Document, skipping. {error}", source=source, error=e
                )
                continue

            merged_metadata = {**bytestream.meta, **metadata}

            if not self.store_full_path and "file_path" in bytestream.meta:
                merged_metadata["file_path"] = os.path.basename(bytestream.meta["file_path"])

            documents.append(Document(content=text, meta=merged_metadata))
            for attachment in attachments:
                attachment_meta = {**merged_metadata, "file_path": attachment.meta["file_path"]}
                if "file_path" in merged_metadata:
                    attachment_meta["parent_file_path"] = merged_metadata["file_path"]
                all_attachments.append(
                    ByteStream(data=attachment.data, meta=attachment_meta, mime_type=attachment.mime_type)
                )

        return {"documents": documents, "attachments": all_attachments}
