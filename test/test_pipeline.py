from typing import Optional

import pytest

from haystack import Pipeline, component


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


def test_pipeline_dumps(pipeline, test_files_path):
    pipeline.add_component("Comp1", TestComponent("Foo"))
    pipeline.add_component("Comp2", TestComponent())
    pipeline.connect("Comp1.value", "Comp2.input_")
    pipeline.max_loops_allowed = 99
    result = pipeline.dumps()
    with open(f"{test_files_path}/yaml/test_pipeline.yaml", "r") as f:
        assert f.read() == result


def test_pipeline_loads(test_files_path):
    with open(f"{test_files_path}/yaml/test_pipeline.yaml", "r") as f:
        pipeline = Pipeline.loads(f.read())
        assert pipeline.max_loops_allowed == 99
        assert isinstance(pipeline.get_component("Comp1"), TestComponent)
        assert isinstance(pipeline.get_component("Comp2"), TestComponent)


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


def test_pipeline_load(test_files_path):
    with open(f"{test_files_path}/yaml/test_pipeline.yaml", "r") as f:
        pipeline = Pipeline.load(f)
        assert pipeline.max_loops_allowed == 99
        assert isinstance(pipeline.get_component("Comp1"), TestComponent)
        assert isinstance(pipeline.get_component("Comp2"), TestComponent)


@pytest.mark.unit
def test_pipeline_resolution_simple_input():
    @component
    class Hello:
        @component.output_types(output=str)
        def run(self, word: str):
            """
            Takes a string in input and returns "Hello, <string>!"
            in output.
            """
            return {"output": f"Hello, {word}!"}

    pipeline = Pipeline()
    pipeline.add_component("hello", Hello())
    pipeline.add_component("hello2", Hello())

    pipeline.connect("hello.output", "hello2.word")
    result = pipeline.run(data={"hello": {"word": "world"}})
    assert result == {"hello2": {"output": "Hello, Hello, world!!"}}

    result = pipeline.run(data={"word": "world"})
    assert result == {"hello2": {"output": "Hello, Hello, world!!"}}


def test_pipeline_resolution_wrong_input_name(caplog):
    @component
    class Hello:
        @component.output_types(output=str)
        def run(self, who: str):
            """
            Takes a string in input and returns "Hello, <string>!"
            in output.
            """
            return {"output": f"Hello, {who}!"}

    pipeline = Pipeline()
    pipeline.add_component("hello", Hello())
    pipeline.add_component("hello2", Hello())

    pipeline.connect("hello.output", "hello2.who")

    # test case with nested component inputs
    with pytest.raises(ValueError):
        pipeline.run(data={"hello": {"non_existing_input": "world"}})

    # test case with flat component inputs
    with pytest.raises(ValueError):
        pipeline.run(data={"non_existing_input": "world"})

    # important to check that the warning is logged for UX purposes, leave it here
    assert "were not matched to any component" in caplog.text


def test_pipeline_resolution_with_mixed_correct_and_incorrect_input_names(caplog):
    @component
    class Hello:
        @component.output_types(output=str)
        def run(self, who: str):
            """
            Takes a string in input and returns "Hello, <string>!"
            in output.
            """
            return {"output": f"Hello, {who}!"}

    pipeline = Pipeline()
    pipeline.add_component("hello", Hello())
    pipeline.add_component("hello2", Hello())

    pipeline.connect("hello.output", "hello2.who")

    # test case with nested component inputs
    # this will raise ValueError because hello component does not have an input named "non_existing_input"
    # even though it has an input named "who"
    with pytest.raises(ValueError):
        pipeline.run(data={"hello": {"non_existing_input": "world", "who": "world"}})

    # test case with flat component inputs
    # this will not raise ValueError because the input "who" will be resolved to the correct component
    # and we'll log a warning for the input "non_existing_input" which was not resolved
    result = pipeline.run(data={"non_existing_input": "world", "who": "world"})
    assert result == {"hello2": {"output": "Hello, Hello, world!!"}}

    # important to check that the warning is logged for UX purposes, leave it here
    assert "were not matched to any component" in caplog.text


def test_pipeline_resolution_duplicate_input_names_across_components():
    @component
    class Hello:
        @component.output_types(output=str)
        def run(self, who: str, what: str):
            return {"output": f"Hello {who} {what}!"}

    pipe = Pipeline()
    pipe.add_component("hello", Hello())
    pipe.add_component("hello2", Hello())

    pipe.connect("hello.output", "hello2.who")

    result = pipe.run(data={"what": "Haystack", "who": "world"})
    assert result == {"hello2": {"output": "Hello Hello world Haystack! Haystack!"}}

    resolved, _ = pipe._prepare_component_input_data(data={"what": "Haystack", "who": "world"})

    # why does hello2 have only one input? Because who of hello2 is inserted from hello.output
    assert resolved == {"hello": {"what": "Haystack", "who": "world"}, "hello2": {"what": "Haystack"}}
