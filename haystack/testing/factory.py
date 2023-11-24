from typing import Any, Dict, Optional, Tuple, Type, List, Union

from haystack.preview import default_to_dict, default_from_dict
from haystack.preview.dataclasses import Document
from haystack.preview.document_stores import document_store, DocumentStore, DuplicatePolicy


def document_store_class(
    name: str,
    documents: Optional[List[Document]] = None,
    documents_count: Optional[int] = None,
    bases: Optional[Tuple[type, ...]] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Type[DocumentStore]:
    """
    Utility function to create a DocumentStore class with the given name and list of documents.

    If `documents` is set but `documents_count` is not, `documents_count` will be the length
    of `documents`.
    If both are set explicitly they don't influence each other.

    `write_documents()` and `delete_documents()` are no-op.
    You can override them using `extra_fields`.

    ### Usage

    Create a DocumentStore class that returns no documents:
    ```python
    MyFakeStore = document_store_class("MyFakeComponent")
    document_store = MyFakeStore()
    assert document_store.documents_count() == 0
    assert document_store.filter_documents() == []
    ```

    Create a DocumentStore class that returns a single document:
    ```python
    doc = Document(id="fake_id", text="Fake content")
    MyFakeStore = document_store_class("MyFakeComponent", documents=[doc])
    document_store = MyFakeStore()
    assert document_store.documents_count() == 1
    assert document_store.filter_documents() == [doc]
    ```

    Create a DocumentStore class that returns no document but returns a custom count:
    ```python
    MyFakeStore = document_store_class("MyFakeComponent", documents_count=100)
    document_store = MyFakeStore()
    assert document_store.documents_count() == 100
    assert document_store.filter_documents() == []
    ```

    Create a DocumentStore class that returns a document and a custom count:
    ```python
    doc = Document(id="fake_id", text="Fake content")
    MyFakeStore = document_store_class("MyFakeComponent", documents=[doc], documents_count=100)
    document_store = MyFakeStore()
    assert document_store.documents_count() == 100
    assert document_store.filter_documents() == [doc]
    ```

    Create a DocumentStore class with a custom base class:
    ```python
    MyFakeStore = document_store_class(
        "MyFakeStore",
        bases=(MyBaseClass,)
    )
    document_store = MyFakeStore()
    assert isinstance(store, MyBaseClass)
    ```

    Create a DocumentStore class with an extra field `my_field`:
    ```python
    MyFakeStore = document_store_class(
        "MyFakeStore",
        extra_fields={"my_field": 10}
    )
    document_store = MyFakeStore()
    assert document_store.my_field == 10
    ```
    """
    if documents is not None and documents_count is None:
        documents_count = len(documents)
    elif documents_count is None:
        documents_count = 0

    def count_documents(self) -> Union[int, None]:
        return documents_count

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        if documents is not None:
            return documents
        return []

    def write_documents(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.FAIL) -> None:
        return

    def delete_documents(self, document_ids: List[str]) -> None:
        return

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(self)

    fields = {
        "count_documents": count_documents,
        "filter_documents": filter_documents,
        "write_documents": write_documents,
        "delete_documents": delete_documents,
        "to_dict": to_dict,
        "from_dict": classmethod(default_from_dict),
    }

    if extra_fields is not None:
        fields = {**fields, **extra_fields}

    if bases is None:
        bases = (object,)

    cls = type(name, bases, fields)
    return document_store(cls)
