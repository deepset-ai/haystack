from abc import abstractmethod


class BaseDocumentStore:
    """
    Base class for implementing DataStores.
    """

    @abstractmethod
    def write_documents(self, documents):
        pass

    @abstractmethod
    def get_document_by_id(self, id):
        pass

    @abstractmethod
    def get_document_ids_by_tag(self, tag):
        pass

    @abstractmethod
    def get_document_count(self):
        pass


