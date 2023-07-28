from typing import List

from haystack.preview import component, Document
from haystack.preview.document_stores import StoreAwareMixin, Store, DuplicatePolicy


@component
class DocumentWriter(StoreAwareMixin):
    """
    A component for writing documents to a DocumentStore.
    """

    supported_stores = [Store]  # type: ignore

    @component.input
    def input(self):
        class Input:
            """
            Input data for the DocumentWriter component.

            :param documents: A list of documents to write to the store.
            :param policy: The policy to use when encountering duplicate documents.
            """

            documents: List[Document]
            policy: DuplicatePolicy

        return Input

    @component.output
    def output(self):
        class Output:
            """
            Output data of the DocumentWriter component.
            """

            ...

        return Output

    def __init__(self, policy: DuplicatePolicy = DuplicatePolicy.FAIL):
        """
        Create a DocumentWriter component.

        :param policy: The policy to use when encountering duplicate documents (default is DuplicatePolicy.FAIL).
        """
        self.defaults = {"policy": policy}

    def run(self, data):
        """
        Run DocumentWriter on the given input data.

        :param data: The input data for DocumentWriter.
        :return: None

        :raises ValueError: If the specified document store is not found or is not a Store instance.
        """
        self.store: Store

        if not self.store:
            raise ValueError("DocumentWriter needs a store to run: set the store instance to the self.store attribute")

        self.store.write_documents(documents=data.documents, policy=data.policy)
        return self.output()
