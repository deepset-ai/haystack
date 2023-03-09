class StoreError(Exception):
    pass


class DuplicateError(StoreError):
    pass


class MissingItemError(StoreError):
    pass
