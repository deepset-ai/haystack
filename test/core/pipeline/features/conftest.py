from typing import Tuple, Dict, Any

from pytest_bdd import when, then, parsers

from haystack import Pipeline


@when("I run the Pipeline", target_fixture="pipeline_result")
def run_pipeline(pipeline_data: Tuple[Pipeline, Dict[str, Any], Dict[str, Any]]):
    """
    Attempts to run a pipeline with the given inputs.
    `pipeline_data` is a tuple that contains:
    * A Pipeline instance
    * The Pipeline inputs
    * The expected outputs

    If successful returns a tuple of the run outputs and the expected outputs.
    In case an exceptions is raised returns that.
    """
    pipeline, inputs, expected_outputs = pipeline_data[0], pipeline_data[1], pipeline_data[2]
    try:
        return pipeline.run(inputs), expected_outputs
    except Exception as e:
        return e


@then("it should return the expected result")
def check_pipeline_result(pipeline_result):
    assert pipeline_result[0] == pipeline_result[1]


@then(parsers.parse("it must have raised {exception_class_name}"))
def check_pipeline_raised(pipeline_result, exception_class_name):
    assert pipeline_result.__class__.__name__ == exception_class_name
