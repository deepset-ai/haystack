class StoreError(Exception):
    pass


class FilterError(StoreError):
    pass


class DuplicateDocumentError(StoreError):
    pass


class MissingDocumentError(StoreError):
    pass


class StoreDeserializationError(StoreError):
    pass
