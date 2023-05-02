import logging
import inspect

from canals.errors import ComponentError
from canals.component.save_init_params import save_init_parameters


logger = logging.getLogger(__name__)


def component(class_):
    """
    Marks a class as a component.
    Any class decorated with `@component` can be used by a Pipeline.

    All components MUST follow the contract below.
    This docstring is the source of truth for components contract.

    ```python
    def __init__(self, [... components init parameters ...]):
    ```
    Mandatory method.

    Components should have an `__init__` method where they define:

    - `self.inputs = [<expected_input_connection_name(s)>]`:
        A list with all the connections they can possibly receive input from

    - `self.outputs = [<expected_output_connection_name(s)>]`:
        A list with the connections they might possibly produce as output

    - `self.init_parameters = {<init parameters>}`:
        Any state they wish to be persisted when they are saved.
        These values will be given to the `__init__` method of a new instance
        when the pipeline is loaded.
        Note that by default the `@component` decorator saves the arguments
        automatically. Components can assume that the dictionary exists and
        can alter its content in the `__init__` method if needed.

    Components should take only "basic" Python types as parameters of their
    `__init__` function, or iterables and dictionaries containing only such values.
    Anything else (objects, functions, etc) will raise an exception at init time.

    _(TODO explain how to use classes and functions in init. In the meantime see
    `test/components/test_accumulate.py`)_

    If components want to let users customize their input and output connections (be it
    the connection name, the connection count, etc...) they should provide properly
    named init parameters:

    - `input: str` or `inputs: List[str]` (always with proper defaults)
    - `output: str` or `outputs: List[str]` (always with proper defaults)

    All the rest is going to be interpreted as a regular init parameter that
    has nothing to do with the component connections.

    The `__init__` must be extrememly lightweight, because it's a frequent
    operation during the construction and validation of the pipeline. If a component
    has some heavy state to initialize (models, backends, etc...) refer to the
    `warm_up()` method.

    ```
    def warm_up(self):
    ```
    Optional method.

    This method is called by Pipeline before the graph execution.
    Make sure to avoid double-initializations, because Pipeline will not keep
    track of which components it called `warm_up()` on.

    ```
    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Dict[str, Any]],
    ):
    ```
    Mandatory method.

    This is the method where the main functionality of the component should be carried out.
    It's called by `Pipeline.run()`, which passes the following parameters to it:

    - `name: str`: the name of the component. Allows the component to find its own parameters in
        the `parameters` dictionary (see below).

    - `data: List[Tuple[str, Any]]`: the input data.
        Pipeline guarantees that the following assert always passes:

        `assert self.inputs == [name for name, value in data]`

        which means that:
        - `data` is of the same length as `self.inputs`.
        - `data` contains one tuple for each string stored in `self.inputs`.
        - no guarantee is given on the values of these tuples: notably, if there was a
            decision component upstream, some values might be `None`.

        For example, if a component declares `self.inputs = ["value", "value"]` (think of a
        `Sum` component), `data` might look like:

        `[("value", 1), ("value", 10)]`

        `[("value", None), ("value", 10)]`

        `[("value", None), ("value", None)]`

        `[("value", 1), ("value", ["something", "unexpected"])]`

        but it will never look like:

        `[("value", 1), ("value", 10), ("value", 100)]`

        `[("value": 15)]`

        `[("value": 15), ("unexpected", 10)]`

    - `parameters: Dict[str, Dict[str, Any]]`: a dictionary of dictionaries with all
        the parameters for all components.
        Note that all components have access to all parameters for all other components: this
        might come handy to components like `Agent`s, that want to influence the behavior
        of components downstream.
        Components can access their own parameters using `name`, but they must **not** assume
        their name is present in the dictionary.
        Therefore, the best way to get the parameters is with
        `my_parameters = parameters.get(name, {})`

    Pipeline expect the output of this function to be a tuple of two dictionaries.
    The first item is a dictionary that represents the output and it should always
    abide to the following format:

    `{output_name: output_value for output_name in <subset of self.expected_output>}`

    Which means that:
    - Components are not forced to produce output on all the expected outputs: for example,
        components taking a decision, like classifiers, can produce output on a subset of
        the expected output connections and Pipeline will figure out the rest.
    - Components must not add any key in the data dictionary that is not present in `self.outputs`.

    The second item of the tuple is the `parameters` dictionary. This allows component to
    propagate downstream any change they might have done to the `parameters` dictionary.

    Args:
        class_: the class that Canals should use as a component.
        serializable: whether to check, at init time, if the component can be saved with
        `save_pipelines()`.

    Returns:
        A class that can be recognized by Canals as a component.

    Raises:
        ComponentError: if the class provided has no `run()` method.
    """
    logger.debug("Registering %s as a component", class_)

    # '__canals_component__' is used to distinguish components from regular classes.
    # Its value is set to the desired component name: normally it is the class name, but it can technically be customized.
    class_.__canals_component__ = class_.__name__

    # Check for run()
    if not hasattr(class_, "run"):
        raise ComponentError(f"{class_.__name__} must have a 'run()' method. See the docs for more information.")
    run_signature = inspect.signature(class_.run)

    # Check the run() signature for keyword variadic arguments
    if any(run_signature.parameters[param].kind == inspect.Parameter.VAR_KEYWORD for param in run_signature.parameters):
        raise ComponentError(
            f"{class_.__name__} can't have variadic keyword arguments like **kwargs in its 'run()' method."
        )

    # Check the run() signature for missing types in positional variadic arguments
    if any(
        run_signature.parameters[param].kind == inspect.Parameter.VAR_POSITIONAL
        and run_signature.parameters[param].annotation == inspect.Parameter.empty
        for param in run_signature.parameters
    ):
        raise ComponentError(
            f"{class_.__name__} must type also variadic arguments like *args in its 'run()' method.\n"
            "Hint: variadic arguments are typed in singular form, so if your component takes an arbitrary number of "
            "strings, the signature of run() should look like 'def run(self, *my_strings: str) -> Output:'."
        )

    # Check the run() signature for other missing types
    missing_types = [
        parameter
        for parameter in list(run_signature.parameters)[1:]  # First is 'self' and it doesn't matter.
        if run_signature.parameters[parameter].annotation == inspect.Parameter.empty
    ]
    if missing_types:
        raise ComponentError(
            f"{class_.__name__}.run() must declare types for all its parameters, "
            f"but these parameters are not typed: {', '.join(missing_types)}."
        )

    # Check for the return types
    if run_signature.return_annotation == inspect.Parameter.empty:
        if not hasattr(class_, "output_type"):
            raise ComponentError(f"{class_.__name__}.run() must declare the type of its return value.")

    # Automatically registers all the init parameters in an instance attribute called `init_parameters`.
    # See `save_init_parameters()`.
    class_.__init__ = save_init_parameters(class_.__init__)

    return class_
