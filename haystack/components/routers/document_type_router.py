# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import mimetypes
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

from haystack import component
from haystack.dataclasses import Document
from haystack.utils.misc import _guess_mime_type


@component
class DocumentTypeRouter:
    """
    Routes documents by their MIME types.

    DocumentTypeRouter is used to dynamically route documents within a pipeline based on their MIME types.
    It supports exact MIME type matches and regex patterns.

    MIME types can be extracted directly from document metadata or inferred from file paths using standard or
    user-supplied MIME type mappings.

    ### Usage example

    ```python
    from haystack.components.routers import DocumentTypeRouter
    from haystack.dataclasses import Document

    docs = [
        Document(content="Example text", meta={"file_path": "example.txt"}),
        Document(content="Another document", meta={"mime_type": "application/pdf"}),
        Document(content="Unknown type")
    ]

    router = DocumentTypeRouter(
        mime_type_meta_field="mime_type",
        file_path_meta_field="file_path",
        mime_types=["text/plain", "application/pdf"]
    )

    result = router.run(documents=docs)
    print(result)
    ```

    Expected output:
    ```python
    {
        "text/plain": [Document(...)],
        "application/pdf": [Document(...)],
        "unclassified": [Document(...)]
    }
    ```
    """

    def __init__(
        self,
        *,
        mime_types: list[str],
        mime_type_meta_field: Optional[str] = None,
        file_path_meta_field: Optional[str] = None,
        additional_mimetypes: Optional[dict[str, str]] = None,
    ) -> None:
        """
        Initialize the DocumentTypeRouter component.

        :param mime_types:
            A list of MIME types or regex patterns to classify the input documents.
            (for example: `["text/plain", "audio/x-wav", "image/jpeg"]`).
        :param mime_type_meta_field:
            Optional name of the metadata field that holds the MIME type.
        :param file_path_meta_field:
            Optional name of the metadata field that holds the file path. Used to infer the MIME type if
            `mime_type_meta_field` is not provided or missing in a document.
        :param additional_mimetypes:
            Optional dictionary mapping MIME types to file extensions to enhance or override the standard
            `mimetypes` module. Useful when working with uncommon or custom file types.
            For example: `{"application/vnd.custom-type": ".custom"}`.

        :raises ValueError: If `mime_types` is empty or if both `mime_type_meta_field` and `file_path_meta_field` are
            not provided.
        """
        if not mime_types:
            raise ValueError("The list of mime types cannot be empty.")

        if mime_type_meta_field is None and file_path_meta_field is None:
            raise ValueError(
                "At least one of 'mime_type_meta_field' or 'file_path_meta_field' must be provided to determine MIME "
                "types."
            )
        self.mime_type_meta_field = mime_type_meta_field
        self.file_path_meta_field = file_path_meta_field

        if additional_mimetypes:
            for mime, ext in additional_mimetypes.items():
                mimetypes.add_type(mime, ext)

        self._mime_type_patterns = []
        for mime_type in mime_types:
            try:
                pattern = re.compile(mime_type)
            except re.error:
                raise ValueError(f"Invalid regex pattern '{mime_type}'.")
            self._mime_type_patterns.append(pattern)

        component.set_output_types(self, unclassified=list[Document], **dict.fromkeys(mime_types, list[Document]))
        self.mime_types = mime_types
        self.additional_mimetypes = additional_mimetypes

    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """
        Categorize input documents into groups based on their MIME type.

        MIME types can either be directly available in document metadata or derived from file paths using the
        standard Python `mimetypes` module and custom mappings.

        :param documents:
            A list of documents to be categorized.

        :returns:
            A dictionary where the keys are MIME types (or `"unclassified"`) and the values are lists of documents.
        """
        mime_types = defaultdict(list)

        for doc in documents:
            mime_type = doc.meta.get(self.mime_type_meta_field) if self.mime_type_meta_field else None
            file_path = doc.meta.get(self.file_path_meta_field) if self.file_path_meta_field else None

            if mime_type is None and file_path:
                # if mime_type is not provided, try to guess it from the file path
                mime_type = _guess_mime_type(Path(file_path))

            matched = False
            if mime_type:
                for pattern in self._mime_type_patterns:
                    if pattern.fullmatch(mime_type):
                        mime_types[pattern.pattern].append(doc)
                        matched = True
                        break
            if not matched:
                mime_types["unclassified"].append(doc)

        return dict(mime_types)
