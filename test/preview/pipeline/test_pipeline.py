from typing import Any, Optional, Dict, List

import pytest

from haystack.preview import Pipeline, component, store, NoSuchStoreError, Document
from haystack.preview.pipeline import NotAStoreError
from haystack.preview.document_stores import StoreAwareMixin, DuplicatePolicy, Store


# Note: we're using a real class instead of a mock because mocks don't play too well with protocols.
@store
class MockStore:
    def count_documents(self) -> int:
        return 0

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        return []

    def write_documents(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.FAIL) -> None:
        return None

    def delete_documents(self, document_ids: List[str]) -> None:
        return None


@pytest.mark.unit
def test_add_store():
    store_1 = MockStore()
    store_2 = MockStore()
    pipe = Pipeline()

    pipe.add_store(name="first_store", store=store_1)
    pipe.add_store(name="second_store", store=store_2)
    assert pipe._stores.get("first_store") == store_1
    assert pipe._stores.get("second_store") == store_2


@pytest.mark.unit
def test_add_store_wrong_object():
    pipe = Pipeline()

    with pytest.raises(NotAStoreError, match="'str' is not decorated with @store,"):
        pipe.add_store(name="store", store="I'm surely not a Store object!")


@pytest.mark.unit
def test_list_stores():
    store_1 = MockStore()
    store_2 = MockStore()
    pipe = Pipeline()

    pipe.add_store(name="first_store", store=store_1)
    pipe.add_store(name="second_store", store=store_2)

    assert pipe.list_stores() == ["first_store", "second_store"]


@pytest.mark.unit
def test_get_store():
    store_1 = MockStore()
    store_2 = MockStore()
    pipe = Pipeline()

    pipe.add_store(name="first_store", store=store_1)
    pipe.add_store(name="second_store", store=store_2)

    assert pipe.get_store("first_store") == store_1
    assert pipe.get_store("second_store") == store_2


@pytest.mark.unit
def test_get_store_wrong_name():
    store_1 = MockStore()
    pipe = Pipeline()

    with pytest.raises(NoSuchStoreError):
        pipe.get_store("first_store")

    pipe.add_store(name="first_store", store=store_1)
    assert pipe.get_store("first_store") == store_1

    with pytest.raises(NoSuchStoreError):
        pipe.get_store("third_store")


@pytest.mark.unit
def test_add_component_store_aware_component_receives_one_docstore():
    store_1 = MockStore()
    store_2 = MockStore()

    @component
    class MockComponent(StoreAwareMixin):
        supported_stores = [Store]

        @component.output_types(value=int)
        def run(self, value: int):
            return {"value": value}

    mock = MockComponent()
    pipe = Pipeline()
    pipe.add_store(name="first_store", store=store_1)
    pipe.add_store(name="second_store", store=store_2)
    pipe.add_component("component", mock, store="first_store")
    assert mock.store == store_1
    assert pipe.run(data={"component": {"value": 1}}) == {"component": {"value": 1}}


@pytest.mark.unit
def test_add_component_store_aware_component_receives_no_docstore():
    store_1 = MockStore()
    store_2 = MockStore()

    @component
    class MockComponent(StoreAwareMixin):
        supported_stores = [Store]

        @component.output_types(value=int)
        def run(self, value: int):
            return {"value": value}

    pipe = Pipeline()
    pipe.add_store(name="first_store", store=store_1)
    pipe.add_store(name="second_store", store=store_2)

    with pytest.raises(ValueError, match="Component 'component' needs a store."):
        pipe.add_component("component", MockComponent())


@pytest.mark.unit
def test_non_store_aware_component_receives_one_docstore():
    store_1 = MockStore()
    store_2 = MockStore()

    @component
    class MockComponent:
        supported_stores = [Store]

        @component.output_types(value=int)
        def run(self, value: int):
            return {"value": value}

    pipe = Pipeline()
    pipe.add_store(name="first_store", store=store_1)
    pipe.add_store(name="second_store", store=store_2)

    with pytest.raises(ValueError, match="Component 'component' doesn't support stores."):
        pipe.add_component("component", MockComponent(), store="first_store")


@pytest.mark.unit
def test_add_component_store_aware_component_receives_wrong_docstore_name():
    store_1 = MockStore()
    store_2 = MockStore()

    @component
    class MockComponent(StoreAwareMixin):
        supported_stores = [Store]

        @component.output_types(value=int)
        def run(self, value: int):
            return {"value": value}

    pipe = Pipeline()
    pipe.add_store(name="first_store", store=store_1)
    pipe.add_store(name="second_store", store=store_2)

    with pytest.raises(NoSuchStoreError, match="Store named 'wrong_store' not found."):
        pipe.add_component("component", MockComponent(), store="wrong_store")


@pytest.mark.unit
def test_add_component_store_aware_component_receives_correct_docstore_type():
    store_1 = MockStore()
    store_2 = MockStore()

    @component
    class MockComponent(StoreAwareMixin):
        supported_stores = [MockStore]

        @component.output_types(value=int)
        def run(self, value: int):
            return {"value": value}

    mock = MockComponent()
    pipe = Pipeline()
    pipe.add_store(name="first_store", store=store_1)
    pipe.add_store(name="second_store", store=store_2)

    pipe.add_component("component", mock, store="second_store")
    assert mock.store == store_2


@pytest.mark.unit
def test_add_component_store_aware_component_is_reused():
    store_1 = MockStore()
    store_2 = MockStore()

    @component
    class MockComponent(StoreAwareMixin):
        supported_stores = [MockStore]

        @component.output_types(value=int)
        def run(self, value: int):
            return {"value": value}

    mock = MockComponent()
    pipe = Pipeline()
    pipe.add_store(name="first_store", store=store_1)
    pipe.add_store(name="second_store", store=store_2)

    pipe.add_component("component", mock, store="second_store")

    with pytest.raises(ValueError, match="Reusing components with stores is not supported"):
        pipe.add_component("component2", mock, store="second_store")

    with pytest.raises(ValueError, match="Reusing components with stores is not supported"):
        pipe.add_component("component2", mock, store="first_store")

    assert mock.store == store_2


@pytest.mark.unit
def test_add_component_store_aware_component_receives_subclass_of_correct_docstore_type():
    class MockStoreSubclass(MockStore):
        ...

    store_1 = MockStoreSubclass()
    store_2 = MockStore()

    @component
    class MockComponent(StoreAwareMixin):
        supported_stores = [MockStore]

        @component.output_types(value=int)
        def run(self, value: int):
            return {"value": value}

    mock = MockComponent()
    mock2 = MockComponent()
    pipe = Pipeline()
    pipe.add_store(name="first_store", store=store_1)
    pipe.add_store(name="second_store", store=store_2)

    pipe.add_component("component", mock, store="first_store")
    assert mock.store == store_1
    pipe.add_component("component2", mock2, store="second_store")
    assert mock2.store == store_2


@pytest.mark.unit
def test_add_component_store_aware_component_does_not_check_supported_stores():
    class SomethingElse:
        ...

    @component
    class MockComponent(StoreAwareMixin):
        supported_stores = [SomethingElse]

        @component.output_types(value=int)
        def run(self, value: int):
            return {"value": value}

    MockComponent()


@pytest.mark.unit
def test_add_component_store_aware_component_receives_wrong_docstore_type():
    store_1 = MockStore()
    store_2 = MockStore()

    class MockStore2:
        def count_documents(self) -> int:
            return 0

        def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
            return []

        def write_documents(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.FAIL) -> None:
            return None

        def delete_documents(self, document_ids: List[str]) -> None:
            return None

    @component
    class MockComponent(StoreAwareMixin):
        supported_stores = [MockStore2]

        @component.output_types(value=int)
        def run(self, value: int):
            return {"value": value}

    mock = MockComponent()
    pipe = Pipeline()
    pipe.add_store(name="first_store", store=store_1)
    pipe.add_store(name="second_store", store=store_2)

    with pytest.raises(ValueError, match="is not compatible with this component"):
        pipe.add_component("component", mock, store="second_store")
