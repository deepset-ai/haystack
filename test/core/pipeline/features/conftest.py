from typing import Tuple, List, Dict, Any

from pytest_bdd import when, then, parsers

from haystack import Pipeline


PipelineData = Tuple[Pipeline, List[Dict[str, Any]], List[Dict[str, Any]], List[List[str]]]
PipelineResult = Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[List[str]], List[List[str]]]


@when("I run the Pipeline", target_fixture="pipeline_result")
def run_pipeline(pipeline_data: PipelineData, spying_tracer):
    """
    Attempts to run a pipeline with the given inputs.
    `pipeline_data` is a tuple that must contain:
    * A Pipeline instance
    * The Pipeline inputs
    * The expected outputs

    Optionally it can contain:
    * The expected order of execution

    If successful returns a tuple of the run outputs and the expected outputs.
    In case an exceptions is raised returns that.
    """
    pipeline, inputs, expected_outputs = pipeline_data[0], pipeline_data[1], pipeline_data[2]
    expected_order = []
    if len(pipeline_data) == 4:
        expected_order = pipeline_data[3]

    if not isinstance(inputs, list):
        inputs = [inputs]
        expected_outputs = [expected_outputs]
        expected_order = [expected_order]

    results = []
    run_orders = []

    for i in inputs:
        try:
            res = pipeline.run(i)
            run_order = [
                span.tags["haystack.component.name"]
                for span in spying_tracer.spans
                if "haystack.component.name" in span.tags
            ]
            results.append(res)
            run_orders.append(run_order)
            spying_tracer.spans.clear()
        except Exception as e:
            return e
    return results, expected_outputs, run_orders, expected_order


@then("it should return the expected result")
def check_pipeline_result(pipeline_result: PipelineResult):
    assert pipeline_result[0] == pipeline_result[1]


@then("components ran in the expected order")
def check_pipeline_run_order(pipeline_result: PipelineResult):
    assert pipeline_result[2] == pipeline_result[3]


@then(parsers.parse("it must have raised {exception_class_name}"))
def check_pipeline_raised(pipeline_result: Exception, exception_class_name: str):
    assert pipeline_result.__class__.__name__ == exception_class_name
