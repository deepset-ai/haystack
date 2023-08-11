class DocumentStoreError(Exception):
    pass


class FilterError(DocumentStoreError):
    pass


class DuplicateDocumentError(DocumentStoreError):
    pass


class MissingDocumentError(DocumentStoreError):
    pass


class DocumentStoreDeserializationError(DocumentStoreError):
    pass
