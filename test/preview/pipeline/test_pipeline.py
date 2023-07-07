from typing import Any, Optional, Dict, List
from unittest.mock import MagicMock

from dataclasses import dataclass

import pytest

from haystack.preview import Pipeline, component, NoSuchStoreError, Document, ComponentInput, ComponentOutput
from haystack.preview.document_stores import StoreAwareMixin, Store, DuplicatePolicy


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
    with pytest.raises(NoSuchStoreError):
        pipe.get_store("third_store")


@pytest.mark.unit
def test_add_component_store_aware_component_receives_one_docstore():
    store_1 = MockStore()
    store_2 = MockStore()

    @component
    class MockComponent(StoreAwareMixin):
        _supported_stores = [Any]

        @dataclass
        class Input(ComponentInput):
            value: int

        @dataclass
        class Output(ComponentOutput):
            value: int

        def run(self, data: Input) -> Output:
            return MockComponent.Output(value=data.value)

    mock = MockComponent()
    pipe = Pipeline()
    pipe.add_store(name="first_store", store=store_1)
    pipe.add_store(name="second_store", store=store_2)
    pipe.add_component("component", mock, store="first_store")
    assert mock.store == store_1
    assert pipe.run(data={"component": MockComponent.Input(value=1)}) == {"component": MockComponent.Output(value=1)}


@pytest.mark.unit
def test_add_component_store_aware_component_receives_no_docstore():
    store_1 = MockStore()
    store_2 = MockStore()

    @component
    class MockComponent(StoreAwareMixin):
        _supported_stores = [Any]

        @dataclass
        class Input(ComponentInput):
            value: int

        @dataclass
        class Output(ComponentOutput):
            value: int

        def run(self, data: Input) -> Output:
            return MockComponent.Output(value=data.value)

    pipe = Pipeline()
    pipe.add_store(name="first_store", store=store_1)
    pipe.add_store(name="second_store", store=store_2)

    with pytest.raises(ValueError, match="Component 'component' needs a store."):
        pipe.add_component("component", MockComponent())


@pytest.mark.unit
def test_add_component_non_store_aware_component_receives_one_docstore():
    store_1 = MockStore()
    store_2 = MockStore()

    @component
    class MockComponent:
        _supported_stores = [Any]

        @dataclass
        class Input(ComponentInput):
            value: int

        @dataclass
        class Output(ComponentOutput):
            value: int

        def run(self, data: Input) -> Output:
            return MockComponent.Output(value=data.value)

    pipe = Pipeline()
    pipe.add_store(name="first_store", store=store_1)
    pipe.add_store(name="second_store", store=store_2)

    with pytest.raises(ValueError, match="Component 'component' doesn't support stores."):
        pipe.add_component("component", MockComponent(), store="first_store")


@pytest.mark.unit
def test_store_aware_component_receives_wrong_docstore_name():
    store_1 = MockStore()
    store_2 = MockStore()

    @component
    class MockComponent(StoreAwareMixin):
        _supported_stores = [Any]

        @dataclass
        class Input(ComponentInput):
            value: int

        @dataclass
        class Output(ComponentOutput):
            value: int

        def run(self, data: Input) -> Output:
            return MockComponent.Output(value=data.value)

    pipe = Pipeline()
    pipe.add_store(name="first_store", store=store_1)
    pipe.add_store(name="second_store", store=store_2)

    with pytest.raises(NoSuchStoreError, match="Store named 'wrong_store' not found."):
        pipe.add_component("component", MockComponent(), store="wrong_store")


@pytest.mark.unit
def test_store_aware_component_receives_correct_docstore_type():
    store_1 = MockStore()
    store_2 = MockStore()

    @component
    class MockComponent(StoreAwareMixin):
        _supported_stores = [MockStore]

        @dataclass
        class Input(ComponentInput):
            value: int

        @dataclass
        class Output(ComponentOutput):
            value: int

        def run(self, data: Input) -> Output:
            return MockComponent.Output(value=data.value)

    mock = MockComponent()
    pipe = Pipeline()
    pipe.add_store(name="first_store", store=store_1)
    pipe.add_store(name="second_store", store=store_2)

    pipe.add_component("component", mock, store="second_store")
    assert mock.store == store_2


@pytest.mark.unit
def test_store_aware_component_receives_not_a_docstore():
    store_1 = MockStore()
    store_2 = MockStore()

    class MockStore2:
        ...

    @component
    class MockComponent(StoreAwareMixin):
        _supported_stores = [MockStore2]

        @dataclass
        class Input(ComponentInput):
            value: int

        @dataclass
        class Output(ComponentOutput):
            value: int

        def run(self, data: Input) -> Output:
            return MockComponent.Output(value=data.value)

    mock = MockComponent()
    pipe = Pipeline()
    pipe.add_store(name="first_store", store=store_1)
    pipe.add_store(name="second_store", store=store_2)

    with pytest.raises(ValueError, match="is not compatible with this component"):
        pipe.add_component("component", mock, store="second_store")


@pytest.mark.unit
def test_store_aware_component_receives_wrong_docstore_type():
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
        _supported_stores = [MockStore2]

        @dataclass
        class Input(ComponentInput):
            value: int

        @dataclass
        class Output(ComponentOutput):
            value: int

        def run(self, data: Input) -> Output:
            return MockComponent.Output(value=data.value)

    mock = MockComponent()
    pipe = Pipeline()
    pipe.add_store(name="first_store", store=store_1)
    pipe.add_store(name="second_store", store=store_2)

    with pytest.raises(ValueError, match="is not compatible with this component"):
        pipe.add_component("component", mock, store="second_store")
