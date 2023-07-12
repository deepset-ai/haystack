from typing import Dict, Any
from dataclasses import dataclass

import pytest

from haystack.preview import (
    Pipeline,
    component,
    NoSuchStoreError,
    ComponentInput,
    ComponentOutput,
    marshal_pipelines,
    unmarshal_pipelines,
)


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


@pytest.mark.unit
def test_marshal_pipeline():
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
    pipe.add_component("component1", MockComponent())
    pipe.add_component("component2", MockComponent())
    pipe.connect("component1", "component2")

    pipe.add_store(name="first_store", store=store_1)
    pipe.add_store(name="second_store", store=store_2)

    marshalled = marshal_pipelines({"test_pipe": pipe})
    assert marshalled == {
        "pipelines": {
            "test_pipe": {
                "metadata": {},
                "max_loops_allowed": 100,
                "components": {
                    "component1": {"type": "MockComponent", "init_parameters": {}},
                    "component2": {"type": "MockComponent", "init_parameters": {}},
                },
                "connections": [("component1", "component2", "value/value")],
                "stores": {
                    "first_store": {"type": "MockStore", "init_parameters": {}},
                    "second_store": {"type": "MockStore", "init_parameters": {}},
                },
            },
            "dependencies": ["test_pipeline", "canals"],
        }
    }
