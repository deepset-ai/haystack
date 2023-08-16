from typing import List, Optional

from haystack.preview import component, Document
from haystack.preview.document_stores import DocumentStoreAwareMixin, DocumentStore, DuplicatePolicy


@component
class DocumentWriter(DocumentStoreAwareMixin):
    """
    A component for writing documents to a DocumentStore.
    """

    supported_document_stores = [DocumentStore]  # type: ignore

    def __init__(self, policy: DuplicatePolicy = DuplicatePolicy.FAIL):
        """
        Create a DocumentWriter component.

        :param policy: The policy to use when encountering duplicate documents (default is DuplicatePolicy.FAIL).
        """
        self.policy = policy

    def run(self, documents: List[Document], policy: Optional[DuplicatePolicy] = None):
        """
        Run DocumentWriter on the given input data.

        :param documents: A list of documents to write to the store.
        :param policy: The policy to use when encountering duplicate documents.
        :return: None

        :raises ValueError: If the specified document store is not found.
        """
        if not self.document_store:
            raise ValueError(
                "DocumentWriter needs a DocumentStore to run: set the DocumentStore instance to the self.document_store attribute."
            )

        if policy is None:
            policy = self.policy

        self.document_store.write_documents(documents=documents, policy=policy)
        return {}
