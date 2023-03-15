class StoreError(Exception):
    pass


class DuplicateDocumentError(StoreError):
    pass


class MissingDocumentError(StoreError):
    pass
