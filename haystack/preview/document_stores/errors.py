class DocumentStore(Exception):
    pass


class FilterError(DocumentStore):
    pass


class DuplicateDocumentError(DocumentStore):
    pass


class MissingDocumentError(DocumentStore):
    pass


class StoreDeserializationError(StoreError):
    pass
