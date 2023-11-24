from typing import Optional

import pytest

from haystack.preview import Pipeline, component


@component
class TestComponent:
    def __init__(self, an_init_param: Optional[str] = None):
        pass

    @component.output_types(value=str)
    def run(self, input_: str):
        return {"value": input_}


@pytest.fixture
def pipeline():
    return Pipeline()


@pytest.mark.unit
def test_pipeline_dumps(pipeline, test_files_path):
    pipeline.add_component("Comp1", TestComponent("Foo"))
    pipeline.add_component("Comp2", TestComponent())
    pipeline.connect("Comp1.value", "Comp2.input_")
    pipeline.max_loops_allowed = 99
    result = pipeline.dumps()
    with open(f"{test_files_path}/yaml/test_pipeline.yaml", "r") as f:
        assert f.read() == result


@pytest.mark.unit
def test_pipeline_loads(test_files_path):
    with open(f"{test_files_path}/yaml/test_pipeline.yaml", "r") as f:
        pipeline = Pipeline.loads(f.read())
        assert pipeline.max_loops_allowed == 99
        assert isinstance(pipeline.get_component("Comp1"), TestComponent)
        assert isinstance(pipeline.get_component("Comp2"), TestComponent)


@pytest.mark.unit
def test_pipeline_dump(pipeline, test_files_path, tmp_path):
    pipeline.add_component("Comp1", TestComponent("Foo"))
    pipeline.add_component("Comp2", TestComponent())
    pipeline.connect("Comp1.value", "Comp2.input_")
    pipeline.max_loops_allowed = 99
    with open(tmp_path / "out.yaml", "w") as f:
        pipeline.dump(f)
    # re-open and ensure it's the same data as the test file
    with open(f"{test_files_path}/yaml/test_pipeline.yaml", "r") as test_f, open(tmp_path / "out.yaml", "r") as f:
        assert f.read() == test_f.read()


@pytest.mark.unit
def test_pipeline_load(test_files_path):
    with open(f"{test_files_path}/yaml/test_pipeline.yaml", "r") as f:
        pipeline = Pipeline.load(f)
        assert pipeline.max_loops_allowed == 99
        assert isinstance(pipeline.get_component("Comp1"), TestComponent)
        assert isinstance(pipeline.get_component("Comp2"), TestComponent)
