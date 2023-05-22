from canals.pipeline import Pipeline
from canals.pipeline.sockets import InputSocket, OutputSocket
from canals.pipeline.validation import find_pipeline_inputs, find_pipeline_outputs
from test.sample_components import Double, AddFixedValue, Sum, Parity


def test_find_pipeline_input_no_input():
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")
    pipe.connect("comp2", "comp1")

    assert find_pipeline_inputs(pipe.graph) == {"comp1": [], "comp2": []}


def test_find_pipeline_input_one_input():
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")

    assert find_pipeline_inputs(pipe.graph) == {
        "comp1": [InputSocket(name="value", type=int, variadic=False)],
        "comp2": [],
    }


def test_find_pipeline_input_two_inputs_same_component():
    pipe = Pipeline()
    pipe.add_component("comp1", AddFixedValue())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")

    assert find_pipeline_inputs(pipe.graph) == {
        "comp1": [
            InputSocket(name="value", type=int, variadic=False),
            InputSocket(name="add", type=int, variadic=False),
        ],
        "comp2": [],
    }


def test_find_pipeline_input_some_inputs_different_components():
    pipe = Pipeline()
    pipe.add_component("comp1", AddFixedValue())
    pipe.add_component("comp2", Double())
    pipe.add_component("comp3", AddFixedValue())
    pipe.connect("comp1", "comp3")
    pipe.connect("comp2", "comp3.add")

    assert find_pipeline_inputs(pipe.graph) == {
        "comp1": [
            InputSocket(name="value", type=int, variadic=False),
            InputSocket(name="add", type=int, variadic=False),
        ],
        "comp2": [InputSocket(name="value", type=int, variadic=False)],
        "comp3": [],
    }


def test_find_pipeline_input_variadic_nodes_in_the_pipeline():
    pipe = Pipeline()
    pipe.add_component("comp1", AddFixedValue())
    pipe.add_component("comp2", Double())
    pipe.add_component("comp3", Sum())
    pipe.connect("comp1", "comp3")
    pipe.connect("comp2", "comp3")

    assert find_pipeline_inputs(pipe.graph) == {
        "comp1": [
            InputSocket(name="value", type=int, variadic=False),
            InputSocket(name="add", type=int, variadic=False),
        ],
        "comp2": [InputSocket(name="value", type=int, variadic=False)],
        "comp3": [InputSocket(name="values", type=int, variadic=True)],
    }


def test_find_pipeline_output_no_output():
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")
    pipe.connect("comp2", "comp1")

    assert find_pipeline_outputs(pipe.graph) == {}


def test_find_pipeline_output_one_output():
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")

    assert find_pipeline_outputs(pipe.graph) == {"comp2": [OutputSocket(name="value", type=int)]}


def test_find_pipeline_some_outputs_same_component():
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Parity())
    pipe.connect("comp1", "comp2")

    assert find_pipeline_outputs(pipe.graph) == {
        "comp2": [OutputSocket(name="even", type=int), OutputSocket(name="odd", type=int)]
    }


def test_find_pipeline_some_outputs_different_components():
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Parity())
    pipe.add_component("comp3", Double())
    pipe.connect("comp1", "comp2")
    pipe.connect("comp1", "comp3")

    assert find_pipeline_outputs(pipe.graph) == {
        "comp2": [OutputSocket(name="even", type=int), OutputSocket(name="odd", type=int)],
        "comp3": [
            OutputSocket(name="value", type=int),
        ],
    }
