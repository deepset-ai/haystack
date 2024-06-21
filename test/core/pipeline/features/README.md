# `Pipeline.run()` behavioural tests

This module contains all behavioural tests for `Pipeline.run()`.

`pipeline_run.feature` contains the definition of the tests using a subset of the [Gherkin language](https://cucumber.io/docs/gherkin/). It's not the full language because we're using `pytest-bdd` and it doesn't implement it in full, but it's good enough for our use case. For more info see the [project `README.md`](https://github.com/pytest-dev/pytest-bdd).

There are two cases covered by these tests:

1. `Pipeline.run()` returns some output
2. `Pipeline.run()` raises an exception

### Correct Pipeline

In the first case to add a new test you need add a new entry in the `Examples` of the `Running a correct Pipeline` scenario outline and create the corresponding step that creates the `Pipeline` you need to test.

For example to add a test for a linear `Pipeline` I add a new `that is linear` kind in `pipeline_run.feature`.

```gherkin
    Scenario Outline: Running a correct Pipeline
        Given a pipeline <kind>
        When I run the Pipeline
        Then it should return the expected result

        Examples:
        | kind |
        | that has no components |
        | that is linear |
```

Then define a new `pipeline_that_is_linear` function in `test_run.py`.
The function must be decorated with `@given` and return a tuple containing the `Pipeline` instance and a list of `PipelineRunData` instances.
`PipelineRunData` is a dataclass that stores all the information necessary to verify the `Pipeline` ran as expected.
The `@given` arguments must be the full step name, `"a pipeline that is linear"` in this case, and `target_fixture` must be set to `"pipeline_data"`.

```python
@given("a pipeline that is linear", target_fixture="pipeline_data")
def pipeline_that_is_linear():
    pipeline = Pipeline()
    pipeline.add_component("first_addition", AddFixedValue(add=2))
    pipeline.add_component("second_addition", AddFixedValue())
    pipeline.add_component("double", Double())
    pipeline.connect("first_addition", "double")
    pipeline.connect("double", "second_addition")

    return (
        pipeline,
        [
            PipelineRunData(
                inputs={"first_addition": {"value": 1}},
                expected_outputs={"second_addition": {"result": 7}},
                expected_run_order=["first_addition", "double", "second_addition"],
            )
        ],
    )
```

Some kinds of `Pipeline`s require multiple runs to verify they work correctly, for example those with multiple branches.
For this reason we can return a list of `PipelineRunData`, we'll run the `Pipeline` for each instance.
For example, we could test two different runs of the same pipeline like this:

```python
@given("a pipeline that is linear", target_fixture="pipeline_data")
def pipeline_that_is_linear():
    pipeline = Pipeline()
    pipeline.add_component("first_addition", AddFixedValue(add=2))
    pipeline.add_component("second_addition", AddFixedValue())
    pipeline.add_component("double", Double())
    pipeline.connect("first_addition", "double")
    pipeline.connect("double", "second_addition")

    return (
        pipeline,
        [
            PipelineRunData(
                inputs={"first_addition": {"value": 1}},
                include_outputs_from=set(),
                expected_outputs={"second_addition": {"result": 7}},
                expected_run_order=["first_addition", "double", "second_addition"],
            ),
            PipelineRunData(
                inputs={"first_addition": {"value": 100}},
                include_outputs_from=set(),
                expected_outputs={"first_addition": {"value": 206}},
                expected_run_order=["first_addition", "double", "second_addition"],
            ),
        ],
    )
```

### Bad Pipeline

The second case is similar to the first one.
In this case we test that a `Pipeline` with an infinite loop raises `PipelineMaxLoops`.

```gherkin
    Scenario Outline: Running a bad Pipeline
        Given a pipeline <kind>
        When I run the Pipeline
        Then it must have raised <exception>

        Examples:
        | kind | exception |
        | that has an infinite loop | PipelineMaxLoops |
```

In a similar way as first case we need to defined a new `pipeline_that_has_an_infinite_loop` function in `test_run.py`, with some small differences.
The only difference from the first case is the last value returned by the function, we just omit the expected outputs and the expected run order.

```python
@given("a pipeline that has an infinite loop", target_fixture="pipeline_data")
def pipeline_that_has_an_infinite_loop():
    def custom_init(self):
        component.set_input_type(self, "x", int)
        component.set_input_type(self, "y", int, 1)
        component.set_output_types(self, a=int, b=int)

    FakeComponent = component_class("FakeComponent", output={"a": 1, "b": 1}, extra_fields={"__init__": custom_init})
    pipe = Pipeline(max_loops_allowed=1)
    pipe.add_component("first", FakeComponent())
    pipe.add_component("second", FakeComponent())
    pipe.connect("first.a", "second.x")
    pipe.connect("second.b", "first.y")
    return pipe, [PipelineRunData({"first": {"x": 1}})]
```

## Why?

As the time of writing, tests that invoke `Pipeline.run()` are scattered between different files with very little clarity on what they are intended to test - the only indicators are the name of each test itself and the name of their parent module. This makes it difficult to understand which behaviours are being tested, if they are tested redundantly or if they work correctly.

The introduction of the Gherkin file allows for a single "source of truth" that enumerates (ideally, in an exhaustive manner) all the behaviours of the pipeline execution logic that we wish to test. This intermediate mapping of behaviours to actual test cases is meant to provide an overview of the latter and reduce the cognitive overhead of understanding them. When writing new tests, we now "tag" them with a specific behavioural parameter that's specified in a Gherkin scenario.

This tag and behavioural parameter mapping is meant to be 1 to 1, meaning each "Given" step must map to one and only one function. If multiple function are marked with `@given("step name")` the last declaration will override all the previous ones. So it's important to verify that there are no other existing steps with the same name when adding a new one.

While one could functionally do the same with well-defined test names and detailed comments on what is being tested, it would still lack the overview that the above approach provides. It's also extensible in that new scenarios with different behaviours can be introduced easily (e.g: for `async` pipeline execution logic).

Apart from the above, the newly introduced harness ensures that all behavioural pipeline tests return a structured result, which simplifies checking of side-effects.
