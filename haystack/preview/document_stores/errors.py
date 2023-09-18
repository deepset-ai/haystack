class DocumentStoreError(Exception):
    pass


class DuplicateDocumentError(DocumentStoreError):
    pass


class MissingDocumentError(DocumentStoreError):
    pass
