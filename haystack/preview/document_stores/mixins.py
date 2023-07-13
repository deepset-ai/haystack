from typing import List, Type


from haystack.preview.document_stores.protocols import Store


class StoreAwareMixin:
    """
    Signals to the pipeline that this component will need a store added to their input data.
    To use this mixin you must specify which document stores to support by setting a value to `supported_stores`.
    To support any document store, set it to `[Store]`.
    """

    supported_stores: List[Type[Store]]  # type: ignore # (see https://github.com/python/mypy/issues/4717)
