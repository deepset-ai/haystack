from typing import Dict, Any

import pytest

from haystack.preview import Pipeline, node, NoSuchStoreError


class MockStore:
    pass


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

    @node
    class MockNode:
        def __init__(self):
            self.inputs = ["value"]
            self.outputs = ["value"]
            self.init_parameters = {}

        def run(self, name: str, data: Dict[str, Any], parameters: Dict[str, Dict[str, Any]]):
            assert name in parameters.keys()
            assert "stores" in parameters[name].keys()
            assert parameters[name]["stores"] == {"first_store": store_1, "second_store": store_2}
            return ({"value": None}, parameters or {})

    pipe = Pipeline()
    pipe.add_node("node", MockNode())

    pipe.add_store(name="first_store", store=store_1)
    pipe.add_store(name="second_store", store=store_2)

    pipe.run(data={"value": None})
