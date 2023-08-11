from typing import Any, Dict, Optional, Tuple, Type, List

from haystack.preview.dataclasses import Document
from haystack.preview.document_stores import store, Store, DuplicatePolicy


def store_class(
    name: str,
    documents: Optional[List[Document]] = None,
    documents_count: Optional[int] = None,
    bases: Optional[Tuple[type, ...]] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Type[Store]:
    """
    Utility function to create a Store class with the given name and list of documents.

    If `documents` is set but `documents_count` is not, `documents_count` will be the length
    of `documents`.
    If both are set explicitly they don't influence each other.

    `write_documents()` and `delete_documents()` are no-op.
    You can override them using `extra_fields`.

    ### Usage

    Create a store class that returns no documents:
    ```python
    MyFakeStore = store_class("MyFakeComponent")
    store = MyFakeStore()
    assert store.documents_count() == 0
    assert store.filter_documents() == []
    ```

    Create a store class that returns a single document:
    ```python
    doc = Document(id="fake_id", content="Fake content")
    MyFakeStore = store_class("MyFakeComponent", documents=[doc])
    store = MyFakeStore()
    assert store.documents_count() == 1
    assert store.filter_documents() == [doc]
    ```

    Create a store class that returns no document but returns a custom count:
    ```python
    MyFakeStore = store_class("MyFakeComponent", documents_count=100)
    store = MyFakeStore()
    assert store.documents_count() == 100
    assert store.filter_documents() == []
    ```

    Create a store class that returns a document and a custom count:
    ```python
    doc = Document(id="fake_id", content="Fake content")
    MyFakeStore = store_class("MyFakeComponent", documents=[doc], documents_count=100)
    store = MyFakeStore()
    assert store.documents_count() == 100
    assert store.filter_documents() == [doc]
    ```

    Create a store class with a custom base class:
    ```python
    MyFakeStore = store_class(
        "MyFakeStore",
        bases=(MyBaseClass,)
    )
    store = MyFakeStore()
    assert isinstance(store, MyBaseClass)
    ```

    Create a store class with an extra field `my_field`:
    ```python
    MyFakeStore = store_class(
        "MyFakeStore",
        extra_fields={"my_field": 10}
    )
    store = MyFakeStore()
    assert store.my_field == 10
    ```
    """

    if documents is not None and documents_count is None:
        documents_count = len(documents)
    elif documents_count is None:
        documents_count = 0

    def count_documents(self) -> int:
        return documents_count

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        if documents is not None:
            return documents
        return []

    def write_documents(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.FAIL) -> None:
        return

    def delete_documents(self, document_ids: List[str]) -> None:
        return

    fields = {
        "count_documents": count_documents,
        "filter_documents": filter_documents,
        "write_documents": write_documents,
        "delete_documents": delete_documents,
    }

    if extra_fields is not None:
        fields = {**fields, **extra_fields}

    if bases is None:
        bases = (object,)

    cls = type(name, bases, fields)
    return store(cls)
