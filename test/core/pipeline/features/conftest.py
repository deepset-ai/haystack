from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any, Set, Union
from pathlib import Path
import re
import pytest
import asyncio
from pandas import DataFrame

from pytest_bdd import when, then, parsers

from haystack import Pipeline, AsyncPipeline

PIPELINE_NAME_REGEX = re.compile(r"\[(.*)\]")


@pytest.fixture(params=[AsyncPipeline, Pipeline])
def pipeline_class(request):
    """
    A parametrized fixture that will yield AsyncPipeline for one test run
    and Pipeline for the next test run.
    """
    return request.param


@dataclass
class PipelineRunData:
    """
    Holds the inputs and expected outputs for a single Pipeline run.
    """

    inputs: Dict[str, Any]
    include_outputs_from: Set[str] = field(default_factory=set)
    expected_outputs: Dict[str, Any] = field(default_factory=dict)
    expected_component_calls: Dict[Tuple[str, int], Dict[str, Any]] = field(default_factory=dict)


@dataclass
class _PipelineResult:
    """
    Holds the outputs and the run order of a single Pipeline run.
    """

    outputs: Dict[str, Any]
    component_calls: Dict[Tuple[str, int], Dict[str, Any]] = field(default_factory=dict)


@when("I run the Pipeline", target_fixture="pipeline_result")
def run_pipeline(
    pipeline_data: Tuple[Union[AsyncPipeline, Pipeline], List[PipelineRunData]], spying_tracer
) -> Union[List[Tuple[_PipelineResult, PipelineRunData]], Exception]:
    if isinstance(pipeline_data[0], AsyncPipeline):
        return run_async_pipeline(pipeline_data, spying_tracer)
    else:
        return run_sync_pipeline(pipeline_data, spying_tracer)


def run_async_pipeline(
    pipeline_data: Tuple[Union[AsyncPipeline], List[PipelineRunData]], spying_tracer
) -> Union[List[Tuple[_PipelineResult, PipelineRunData]], Exception]:
    """
    Attempts to run a pipeline with the given inputs.
    `pipeline_data` is a tuple that must contain:
    * A Pipeline instance
    * The data to run the pipeline with

    If successful returns a tuple of the run outputs and the expected outputs.
    In case an exceptions is raised returns that.
    """
    pipeline, pipeline_run_data = pipeline_data[0], pipeline_data[1]

    results: List[_PipelineResult] = []

    async def run_inner(data, include_outputs_from):
        """Wrapper function to call pipeline.run_async method with required params."""
        return await pipeline.run_async(data=data.inputs, include_outputs_from=include_outputs_from)

    for data in pipeline_run_data:
        try:
            outputs = asyncio.run(run_inner(data, data.include_outputs_from))

            component_calls = {
                (span.tags["haystack.component.name"], span.tags["haystack.component.visits"]): span.tags[
                    "haystack.component.input"
                ]
                for span in spying_tracer.spans
                if "haystack.component.name" in span.tags and "haystack.component.visits" in span.tags
            }
            results.append(_PipelineResult(outputs=outputs, component_calls=component_calls))
            spying_tracer.spans.clear()
        except Exception as e:
            return e

    return [e for e in zip(results, pipeline_run_data)]


def run_sync_pipeline(
    pipeline_data: Tuple[Pipeline, List[PipelineRunData]], spying_tracer
) -> Union[List[Tuple[_PipelineResult, PipelineRunData]], Exception]:
    """
    Attempts to run a pipeline with the given inputs.
    `pipeline_data` is a tuple that must contain:
    * A Pipeline instance
    * The data to run the pipeline with

    If successful returns a tuple of the run outputs and the expected outputs.
    In case an exceptions is raised returns that.
    """
    pipeline, pipeline_run_data = pipeline_data[0], pipeline_data[1]

    results: List[_PipelineResult] = []

    for data in pipeline_run_data:
        try:
            outputs = pipeline.run(data=data.inputs, include_outputs_from=data.include_outputs_from)

            component_calls = {
                (span.tags["haystack.component.name"], span.tags["haystack.component.visits"]): span.tags[
                    "haystack.component.input"
                ]
                for span in spying_tracer.spans
                if "haystack.component.name" in span.tags and "haystack.component.visits" in span.tags
            }
            results.append(_PipelineResult(outputs=outputs, component_calls=component_calls))
            spying_tracer.spans.clear()
        except Exception as e:
            return e
    return [e for e in zip(results, pipeline_run_data)]


@then("draw it to file")
def draw_pipeline(pipeline_data: Tuple[Pipeline, List[PipelineRunData]], request):
    """
    Draw the pipeline to a file with the same name as the test.
    """
    if m := PIPELINE_NAME_REGEX.search(request.node.name):
        name = m.group(1).replace(" ", "_")
        pipeline = pipeline_data[0]
        graphs_dir = Path(request.config.rootpath) / "test_pipeline_graphs"
        graphs_dir.mkdir(exist_ok=True)
        pipeline.draw(graphs_dir / f"{name}.png")


@then("it should return the expected result")
def check_pipeline_result(pipeline_result: List[Tuple[_PipelineResult, PipelineRunData]]):
    for res, data in pipeline_result:
        compare_outputs_with_dataframes(res.outputs, data.expected_outputs)


@then("components are called with the expected inputs")
def check_component_calls(pipeline_result: List[Tuple[_PipelineResult, PipelineRunData]]):
    for res, data in pipeline_result:
        assert compare_outputs_with_dataframes(res.component_calls, data.expected_component_calls)


@then(parsers.parse("it must have raised {exception_class_name}"))
def check_pipeline_raised(pipeline_result: Exception, exception_class_name: str):
    assert pipeline_result.__class__.__name__ == exception_class_name


def compare_outputs_with_dataframes(actual: Dict, expected: Dict) -> bool:
    """
    Compare two component_calls or pipeline outputs dictionaries where values may contain DataFrames.
    """
    assert actual.keys() == expected.keys()

    for key in actual:
        actual_data = actual[key]
        expected_data = expected[key]

        assert actual_data.keys() == expected_data.keys()

        for data_key in actual_data:
            actual_value = actual_data[data_key]
            expected_value = expected_data[data_key]

            if isinstance(actual_value, DataFrame) and isinstance(expected_value, DataFrame):
                assert actual_value.equals(expected_value)
            else:
                assert actual_value == expected_value

    return True
