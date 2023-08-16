from typing import List, Optional

from haystack.preview import component, Document
from haystack.preview.document_stores import DocumentStore, DuplicatePolicy


@component
class DocumentWriter:
    """
    A component for writing documents to a DocumentStore.
    """

    def __init__(self, document_store: DocumentStore, policy: DuplicatePolicy = DuplicatePolicy.FAIL):
        """
        Create a DocumentWriter component.

        :param policy: The policy to use when encountering duplicate documents (default is DuplicatePolicy.FAIL).
        """
        self.document_store = document_store
        self.policy = policy

    def run(self, documents: List[Document], policy: Optional[DuplicatePolicy] = None):
        """
        Run DocumentWriter on the given input data.

        :param documents: A list of documents to write to the store.
        :param policy: The policy to use when encountering duplicate documents.
        :return: None

        :raises ValueError: If the specified document store is not found.
        """
        if policy is None:
            policy = self.policy

        self.document_store.write_documents(documents=documents, policy=policy)
        return {}
