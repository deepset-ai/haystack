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
The function must be decorated with `@given` and return a tuple containing the `Pipeline` instance, the `Pipeline.run()` inputs, the expected output and the expected Components run order, in this exact order.
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
        {"first_addition": {"value": 1}},
        {"second_addition": {"result": 7}},
        ["first_addition", "double", "second_addition"],
    )
```

### Bad Pipeline

The second case is similar to the first one, but we can also specify the expected exception.
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
The only difference from the first case is the last value returned by the function, in this case we return the expected exception class.

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
    return pipe, {"first": {"x": 1}}, PipelineMaxLoops
```

## Why?

As the time I'm writing this `Pipeline.run()` tests are scattered between different files, with close to no explanation on what they're testing, apart from the name of the test function or file.

This makes it hard to know all the cases we're testing, if some are tested multiple times, and if the test actually does what it says.

The Gherkin file adds some overhead as it forces us to add a sort of "tag" for every test function, that most of the times is going to be really similar to the test function name. It also creates some indirection as one needs to know all the steps to add a new test.

Though it gives also some benefits. All the cases we're testing are grouped up and can be checked together. The devs are forced to write in natural language the type of Pipeline they're testing. They're also forced to properly test all "parts" of `Pipeline.run()`, its outputs and the order it runs the Components.

We can do all the above with some custom `pytest` logic that generates tests from a list of functions that return `Pipeline`s. Though that requires some extra code that needs to be maintained, something that we get for free with `pytest-bdd`. Also the tests are not grouped up together, making it harder to understand and reason about the cases we support.

Also in the future if we need to test new `run()` functions -- say an `async run` -- we can easily reuse the existing `given` steps that create the `Pipeline` just by adding a new scenario that test for a different type of run.
