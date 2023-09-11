from typing import Optional, List

import logging

from haystack.preview import Pipeline, component
from haystack.preview.components.batch.batch_processor import BatchProcessor
from haystack.preview.components.batch.batch_creator import BatchCreator


logging.basicConfig(level=logging.DEBUG)


def test_batch_and_non_batch_pipeline_components():
    @component
    class Addition:
        @component.output_types(result=int)
        def run(self, value: int, add: Optional[int] = 1):
            return {"result": value + add}

    @component
    class BatchAddition:
        @component.output_types(results=List[int])
        def run(self, values: List[int], add: Optional[int] = 1):
            return {"results": [value + add for value in values]}

    pipe = Pipeline(max_loops_allowed=4)
    pipe.add_component(name="initial_addition", instance=Addition())
    pipe.add_component(name="final_addition", instance=Addition())
    pipe.add_component(name="batch_addition", instance=BatchAddition())
    pipe.add_component(name="batch_creator", instance=BatchCreator(expected_type=int, max_batch_size=3))
    pipe.add_component(name="batch_processor", instance=BatchProcessor(expected_type=int))
    pipe.add_component(name="batch_output", instance=BatchCreator(expected_type=int, max_batch_size=3))

    pipe.connect("initial_addition.result", "batch_creator.item")
    pipe.connect("batch_creator.batch", "batch_addition.values")
    pipe.connect("batch_addition.results", "batch_processor.new_batch")
    pipe.connect("batch_processor.item", "final_addition.value")
    pipe.connect("batch_processor.current_batch", "batch_processor.current_batch")
    pipe.connect("final_addition.result", "batch_output.item")

    pipe.draw("test.jpg", engine="graphviz")

    assert pipe.run({"initial_addition": {"value": 1}}) == {}
    assert pipe.run({"initial_addition": {"value": 2}}) == {}
    output = pipe.run({"initial_addition": {"value": 3}})
    assert output == {"batch_output": {"batch": [4, 5, 6]}}
    assert pipe.get_component("batch_output").batch == []


if __name__ == "__main__":
    test_batch_and_non_batch_pipeline_components()
