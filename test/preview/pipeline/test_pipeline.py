from dataclasses import dataclass

import pytest

from haystack.preview import Pipeline, component, NoSuchStoreError, ComponentInput, ComponentOutput
from haystack.preview.document_stores import StoreMixin, MultiStoreMixin


class MockStore:
    ...


@pytest.mark.unit
def test_pipeline_store_add_list_get():
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
def test_pipeline_component_expects_one_docstore_receives_one_docstore():
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
            return MockComponent.Output(value=data.value)

    mock = MockComponent()
    pipe = Pipeline()
    pipe.add_store(name="first_store", store=store_1)
    pipe.add_store(name="second_store", store=store_2)
    pipe.add_component("component", mock, stores=["first_store"])
    assert mock.store == store_1
    assert pipe.run(data={"component": MockComponent.Input(value=1)}) == {"component": MockComponent.Output(value=1)}


@pytest.mark.unit
def test_pipeline_component_expects_one_docstore_receives_no_docstore():
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
            return MockComponent.Output(value=data.value)

    pipe = Pipeline()
    pipe.add_store(name="first_store", store=store_1)
    pipe.add_store(name="second_store", store=store_2)

    with pytest.raises(ValueError, match="Component 'component' needs exactly one store."):
        pipe.add_component("component", MockComponent())


@pytest.mark.unit
def test_pipeline_component_expects_one_docstore_receives_many_docstores():
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
            return MockComponent.Output(value=data.value)

    pipe = Pipeline()
    pipe.add_store(name="first_store", store=store_1)
    pipe.add_store(name="second_store", store=store_2)

    with pytest.raises(ValueError, match="Component 'component' needs exactly one store."):
        pipe.add_component("component", MockComponent(), stores=["first_store", "second_store"])


@pytest.mark.unit
def test_pipeline_component_expects_one_docstore_receives_wrong_docstore():
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
            return MockComponent.Output(value=data.value)

    pipe = Pipeline()
    pipe.add_store(name="first_store", store=store_1)
    pipe.add_store(name="second_store", store=store_2)

    with pytest.raises(NoSuchStoreError, match="Store named 'wrong_store' not found."):
        pipe.add_component("component", MockComponent(), stores=["wrong_store"])


@pytest.mark.unit
def test_pipeline_component_expects_many_docstores_receives_one_docstore():
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

    mock = MockComponent()
    pipe = Pipeline()
    pipe.add_store(name="first_store", store=store_1)
    pipe.add_store(name="second_store", store=store_2)
    pipe.add_component("component", mock, stores=["first_store"])

    assert mock.stores == {"first_store": store_1}
    assert pipe.run(data={"component": MockComponent.Input(value=1)}) == {"component": MockComponent.Output(value=1)}


@pytest.mark.unit
def test_pipeline_component_expects_many_docstores_receives_no_docstore():
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

    mock = MockComponent()
    pipe = Pipeline()
    pipe.add_store(name="first_store", store=store_1)
    pipe.add_store(name="second_store", store=store_2)
    pipe.add_component("component", mock)
    mock.stores = {}
    assert pipe.run(data={"component": MockComponent.Input(value=1)}) == {"component": MockComponent.Output(value=1)}


@pytest.mark.unit
def test_pipeline_component_expects_many_docstores_receives_many_docstores():
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

    mock = MockComponent()
    pipe = Pipeline()
    pipe.add_store(name="first_store", store=store_1)
    pipe.add_store(name="second_store", store=store_2)
    pipe.add_component("component", mock, stores=["first_store", "second_store"])
    assert mock.stores == {"first_store": store_1, "second_store": store_2}


@pytest.mark.unit
def test_pipeline_component_expects_many_docstores_receives_one_wrong_docstore():
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
def test_pipeline_component_expects_many_docstores_receives_non_existing_docstore_among_existing_ones():
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
