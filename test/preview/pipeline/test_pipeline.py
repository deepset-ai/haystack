from typing import Dict, Any
from dataclasses import dataclass

import pytest

from haystack.preview import Pipeline, component, NoSuchStoreError, ComponentInput, ComponentOutput
from haystack.preview.document_stores import StoreMixin, MultiStoreMixin


class MockStore:
    ...


@pytest.mark.unit
def test_pipeline_store_api():
    store_1 = MockStore()
    store_2 = MockStore()
    pipe = Pipeline()

    pipe.add_store(name="first_store", store=store_1)
    pipe.add_store(name="second_store", store=store_2)

    assert pipe.list_stores() == ["first_store", "second_store"]

    assert pipe.get_store("first_store") == store_1
    assert pipe.get_store("second_store") == store_2
    with pytest.raises(NoSuchStoreError):
        pipe.get_store("third_store")


@pytest.mark.unit
def test_pipeline_storemixin_one_existing_docstore():
    store_1 = MockStore()
    store_2 = MockStore()

    @component
    class MockComponent(StoreMixin):
        @dataclass
        class Input(ComponentInput):
            value: int

        @dataclass
        class Output(ComponentOutput):
            value: int

        def run(self, data: Input) -> Output:
            assert self.store == store_1
            return MockComponent.Output(value=data.value)

    pipe = Pipeline()
    pipe.add_store(name="first_store", store=store_1)
    pipe.add_store(name="second_store", store=store_2)
    pipe.add_component("component", MockComponent(), stores=["first_store"])
    assert pipe.run(data={"component": MockComponent.Input(value=1)}) == {"component": MockComponent.Output(value=1)}


@pytest.mark.unit
def test_pipeline_storemixin_no_docstore():
    store_1 = MockStore()
    store_2 = MockStore()

    @component
    class MockComponent(StoreMixin):
        @dataclass
        class Input(ComponentInput):
            value: int

        @dataclass
        class Output(ComponentOutput):
            value: int

        def run(self, data: Input) -> Output:
            assert data.store == store_1
            return MockComponent.Output(value=data.value)

    pipe = Pipeline()
    pipe.add_store(name="first_store", store=store_1)
    pipe.add_store(name="second_store", store=store_2)

    with pytest.raises(ValueError, match="Component 'component' needs exactly one store."):
        pipe.add_component("component", MockComponent())


@pytest.mark.unit
def test_pipeline_storemixin_many_existing_docstores():
    store_1 = MockStore()
    store_2 = MockStore()

    @component
    class MockComponent(StoreMixin):
        @dataclass
        class Input(ComponentInput):
            value: int

        @dataclass
        class Output(ComponentOutput):
            value: int

        def run(self, data: Input) -> Output:
            assert data.store == store_1
            return MockComponent.Output(value=data.value)

    pipe = Pipeline()
    pipe.add_store(name="first_store", store=store_1)
    pipe.add_store(name="second_store", store=store_2)

    with pytest.raises(ValueError, match="Component 'component' needs exactly one store."):
        pipe.add_component("component", MockComponent(), stores=["first_store", "second_store"])


@pytest.mark.unit
def test_pipeline_storemixin_one_non_existing_docstore():
    store_1 = MockStore()
    store_2 = MockStore()

    @component
    class MockComponent(StoreMixin):
        @dataclass
        class Input(ComponentInput):
            value: int

        @dataclass
        class Output(ComponentOutput):
            value: int

        def run(self, data: Input) -> Output:
            assert data.store == store_1
            return MockComponent.Output(value=data.value)

    pipe = Pipeline()
    pipe.add_store(name="first_store", store=store_1)
    pipe.add_store(name="second_store", store=store_2)

    with pytest.raises(NoSuchStoreError, match="Store named 'wrong_store' not found."):
        pipe.add_component("component", MockComponent(), stores=["wrong_store"])


@pytest.mark.unit
def test_pipeline_storesmixin_one_existing_docstore():
    store_1 = MockStore()
    store_2 = MockStore()

    @component
    class MockComponent(MultiStoreMixin):
        @dataclass
        class Input(ComponentInput):
            value: int

        @dataclass
        class Output(ComponentOutput):
            value: int

        def run(self, data: Input) -> Output:
            assert self.stores == {"first_store": store_1}
            return MockComponent.Output(value=data.value)

    pipe = Pipeline()
    pipe.add_store(name="first_store", store=store_1)
    pipe.add_store(name="second_store", store=store_2)
    pipe.add_component("component", MockComponent(), stores=["first_store"])
    assert pipe.run(data={"component": MockComponent.Input(value=1)}) == {"component": MockComponent.Output(value=1)}


@pytest.mark.unit
def test_pipeline_storesmixin_no_docstore():
    store_1 = MockStore()
    store_2 = MockStore()

    @component
    class MockComponent(MultiStoreMixin):
        @dataclass
        class Input(ComponentInput):
            value: int

        @dataclass
        class Output(ComponentOutput):
            value: int

        def run(self, data: Input) -> Output:
            assert self.stores == {}
            return MockComponent.Output(value=data.value)

    pipe = Pipeline()
    pipe.add_store(name="first_store", store=store_1)
    pipe.add_store(name="second_store", store=store_2)
    pipe.add_component("component", MockComponent())
    assert pipe.run(data={"component": MockComponent.Input(value=1)}) == {"component": MockComponent.Output(value=1)}


@pytest.mark.unit
def test_pipeline_storesmixin_many_existing_docstores():
    store_1 = MockStore()
    store_2 = MockStore()

    @component
    class MockComponent(MultiStoreMixin):
        @dataclass
        class Input(ComponentInput):
            value: int

        @dataclass
        class Output(ComponentOutput):
            value: int

        def run(self, data: Input) -> Output:
            assert self.stores == {"first_store": store_1, "second_store": store_2}
            return MockComponent.Output(value=data.value)

    pipe = Pipeline()
    pipe.add_store(name="first_store", store=store_1)
    pipe.add_store(name="second_store", store=store_2)
    pipe.add_component("component", MockComponent(), stores=["first_store", "second_store"])


@pytest.mark.unit
def test_pipeline_storemixin_one_non_existing_docstore_alone():
    store_1 = MockStore()
    store_2 = MockStore()

    @component
    class MockComponent(MultiStoreMixin):
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
        pipe.add_component("component", MockComponent(), stores=["wrong_store"])


@pytest.mark.unit
def test_pipeline_storemixin_one_non_existing_docstore_among_existing_ones():
    store_1 = MockStore()
    store_2 = MockStore()

    @component
    class MockComponent(MultiStoreMixin):
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
        pipe.add_component("component", MockComponent(), stores=["first_store", "wrong_store"])
