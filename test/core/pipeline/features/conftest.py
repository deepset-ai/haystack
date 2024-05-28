from pytest_bdd import when, then, parsers


@when("I run the Pipeline", target_fixture="pipeline_result")
def run_pipeline(pipeline_data, spying_tracer):
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
    if len(pipeline_data) == 4:
        expected_order = pipeline_data[3]

    try:
        res = pipeline.run(inputs)
        run_order = [
            span.tags["haystack.component.name"]
            for span in spying_tracer.spans
            if "haystack.component.name" in span.tags
        ]
        return res, expected_outputs, run_order, expected_order
    except Exception as e:
        return e


@then("it should return the expected result")
def check_pipeline_result(pipeline_result):
    assert pipeline_result[0] == pipeline_result[1]


@then("components ran in the expected order")
def check_pipeline_run_order(pipeline_result):
    assert pipeline_result[2] == pipeline_result[3]


@then(parsers.parse("it must have raised {exception_class_name}"))
def check_pipeline_raised(pipeline_result, exception_class_name):
    assert pipeline_result.__class__.__name__ == exception_class_name
