from typing import Dict, Any
from dataclasses import dataclass

import pytest

from haystack.preview import Pipeline, component, NoSuchStoreError, ComponentInput, ComponentOutput


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
def test_pipeline_stores_in_params():
    store_1 = MockStore()
    store_2 = MockStore()

    @component
    class MockComponent:
        @dataclass
        class Input(ComponentInput):
            value: int
            stores: Dict[str, Any]

        @dataclass
        class Output(ComponentOutput):
            value: int

        def run(self, data: Input) -> Output:
            assert data.stores == {"first_store": store_1, "second_store": store_2}
            return MockComponent.Output(value=data.value)

    pipe = Pipeline()
    pipe.add_component("component", MockComponent())

    pipe.add_store(name="first_store", store=store_1)
    pipe.add_store(name="second_store", store=store_2)

    assert pipe.run(data={"component": MockComponent.Input(value=1)}) == {"component": MockComponent.Output(value=1)}
