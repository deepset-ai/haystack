# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

from haystack import Document, component


@component
class EmptyDocumentRemover:
    """
    Removes empty documents from a list of documents.
    """

    def __init__(self) -> None:
        """
        Initialize the EmptyDocumentRemover.
        """

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        """
        Removes the empty documents.

        :param documents: List of Documents to clean.

        :returns: A dictionary with the following key:
            - `documents`: List of cleaned Documents.

        :raises TypeError: if documents is not a list of Document.
        """
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            raise TypeError("DocumentCleaner expects a List of Documents as input.")

        cleaned_docs: List[Document] = []
        for doc in documents:
            if doc.content is not None:
                cleaned_docs.append(doc)

        return {"documents": cleaned_docs}
